#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Customer end-to-end test: N phone calls through pipecat-mirai, as a customer runs them.

Each call is the bot a customer builds from our README:

  fake LLM (a Hindi/Hinglish reply streamed token by token, ~40 tokens/s)
    -> MiraiTTSService (WebSocket; or --tts http: MiraiHttpTTSService), 8 kHz pipeline
    -> FastAPIWebsocketTransport + TwilioFrameSerializer (8 kHz mu-law),
       apply_output_lead(transport, --lead)
    -> a phone simulator that plays the media at exactly real time, like a
       carrier, and records what the caller hears (one WAV per call).

--defaults builds the bot the way a customer who only installed the package
writes it: MiraiTTSService(api_key=key, voice=voice) (or
MiraiHttpTTSService(api_key=key, voice=voice)) and nothing else, and no
apply_output_lead. base_url/url are passed only with --standin, --url or
--http-url: the library's defaults point at Mirai's production endpoint.

Turn loop per call: the bot replies, the caller "speaks" for 2-6 s, repeat for
--duration. The first reply (the greeting) starts 1-2 s after the call is
answered, or the moment it connects with --greeting-now (on by default with
--defaults); its TTFB is reported apart from the later turns'. --barge-in makes
that share of replies get interrupted mid-way (InterruptionFrame, so Pipecat
sends the provider a "clear"); with --greeting-now the greeting never is.

The calls start --stagger seconds apart from one shared instant (0: all at
once). --seed-call S first runs one S-second call on each bot process and
waits --seed-gap seconds, so the burst meets a server that has already served
something; the seed calls are reported apart. --stall-every/--stall-ms block
each bot's event loop now and then, like a busy customer server. The whole run
is capped at --max-runtime seconds: the calls are hung up and the bots killed.

Bots run in --procs worker processes (uvicorn + FastAPI, like a customer's
server); the phone runs in this process, so a stalled bot cannot distort the
"carrier" clock. All processes are on one host and share time.monotonic().

    # stand-ins (tests/fake_mirai_ws.py for --tts ws, an HTTP fake for --tts http), no key:
    uv run --with uvicorn python benchmarks/customer-e2e/run.py --standin --calls 12 --out /tmp/e2e
    # what a customer gets from the package's defaults (production endpoint):
    uv run --with uvicorn python benchmarks/customer-e2e/run.py --defaults --tts http --calls 10 \\
        --procs 1 --stagger 0 --seed-call 10 --duration 100 --barge-in 0.2 --key-file key.txt \\
        --out results/x
    # a given endpoint:
    uv run --with uvicorn python benchmarks/customer-e2e/run.py \\
        --url ws://localhost:8130/v1/audio/speech/stream --key-file key.txt \\
        --calls 12 --duration 100 --out results/e2e

Prints a per-call table, an overall summary and PASS/FAIL flags, and writes
result.json and callNN.wav (seedN.wav for seed calls) to --out.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import random
import re
import resource
import signal
import socket
import subprocess
import sys
import threading
import time
import wave
from http import HTTPStatus
from pathlib import Path

import numpy as np

SR = 8000
REPO = Path(__file__).resolve().parents[2]
LOCAL_WS_URL = "ws://127.0.0.1:8130/v1/audio/speech/stream"

REPLIES = [
    "नमस्ते, मैं Mirai से बोल रही हूँ। आपका order कल शाम तक deliver हो जाएगा। क्या उस समय घर पर कोई रहेगा?",
    "जी बिल्कुल, मैं आपकी मदद कर सकती हूँ। आपकी appointment शनिवार सुबह ग्यारह बजे के लिए confirm है।",
    "हमारी team आपसे payment के बारे में बात करना चाहती थी। क्या अभी दो मिनट बात हो सकती है? ज़्यादा समय नहीं लगेगा।",
    "ठीक है, मैंने सारी details note कर ली हैं। हमारी team आपसे जल्द ही संपर्क करेगी।",
    "आपका refund तीन से पाँच working days में आपके account में आ जाएगा। "
    "अगर कोई दिक्कत हो तो आप हमें दोबारा call कर सकते हैं।",
    "क्या आप अपने area का pincode बता सकते हैं? उससे मैं नज़दीकी store ढूँढ दूँगी।",
    "आपके plan में हर महीने दो सौ GB data मिलता है। अभी आपका साठ प्रतिशत data बचा हुआ है।",
    "आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
]


def tokens(text):
    """Split a reply like an LLM streams it: a word and its trailing space at a time."""
    return re.findall(r"\S+\s*", text)


def sentences(text):
    return [s.strip() for s in re.findall(r".+?(?:[।?!.]+(?=\s|$)|$)", text) if s.strip()]


def cpu_seconds():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def pct(values, q):
    """Nearest-rank percentile: the smallest value with at least q% of the values at or below it.

    With ~10 values p95 is the maximum; every percentile in this report is computed this way.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


async def loop_lag(samples, interval=0.05):
    """Record how late a 50 ms sleep wakes up: the event loop's stalls."""
    while True:
        t = time.monotonic()
        await asyncio.sleep(interval)
        samples.append(time.monotonic() - t - interval)


def lag_stats(samples):
    if not samples:
        return {}
    ms = [s * 1000 for s in samples]
    return {"lag_p99_ms": round(pct(ms, 99), 1), "lag_max_ms": round(max(ms), 1)}


def write_json(path, obj):
    """Write ``obj`` so that a reader never sees half a file."""
    tmp = Path(f"{path}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False))
    os.replace(tmp, path)


def exit_after(seconds):
    """End this (child) process after ``seconds``, even if the parent that should stop it is gone."""
    timer = threading.Timer(seconds, os._exit, args=(3,))
    timer.daemon = True
    timer.start()


def tone(seconds, rate, hz=220.0):
    t = np.arange(int(round(seconds * rate))) / rate
    return (np.sin(2 * np.pi * hz * t) * 8000).astype("<i2").tobytes()


def standin_seconds(text):
    """Audio length the stand-ins speak for ``text`` (~14 characters a second)."""
    return max(0.8, len(text) / 14)


# ------------------------------------------------------------------------------------ stand-ins
def standin_main(a):
    """tests/fake_mirai_ws.py on a fixed port, paced like Mirai (150 ms first byte, then
    --standin-speed times real time; below 1.0 the caller must hear gaps)."""
    exit_after(a.max_runtime + 5)
    sys.path.insert(0, str(REPO / "tests"))
    from fake_mirai_ws import FakeMiraiWS
    from websockets.asyncio.server import serve

    capacity = {}
    for reply in REPLIES[: a.standin_capacity]:
        capacity[sentences(reply)[1]] = 0.5  # each refused once, retry after 0.5 s
    fake = FakeMiraiWS(
        seconds=standin_seconds,
        reads=(1280,),  # 80 ms of 8 kHz audio per frame
        pace=0.08 / a.standin_speed,
        first_byte=0.15,
        capacity=capacity,
    )

    async def main():
        lag = []
        _lagger = asyncio.create_task(loop_lag(lag))
        t0, c0 = time.monotonic(), cpu_seconds()

        def write():
            billed = [len(s) for _, _, s in fake.spoken]
            stats = {
                "cpu_s": round(cpu_seconds() - c0, 2),
                "wall_s": round(time.monotonic() - t0, 2),
                "sentences": len(billed),
                "chars": sum(billed),
                "cancels": len(fake.messages("cancel")),
                "connections": len(fake.conns),
                **lag_stats(lag),
            }
            write_json(Path(a.out, "standin.json"), stats)

        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, write)
        async with serve(fake._handler, "127.0.0.1", a.port, max_size=None):
            print("ready", flush=True)
            while True:
                await asyncio.sleep(2)
                write()

    asyncio.run(main())


def standin_http_main(a):
    """Mirai's HTTP API on a fixed port: ``POST /v1/audio/speech`` streams 16-bit PCM at the
    requested ``sample_rate`` (chunked, kept alive) and ``GET /v1/models`` answers 200.

    Paced like the WebSocket stand-in: 150 ms to the first byte, then 80 ms pieces at
    --standin-speed times real time. It refuses nothing (the HTTP service doesn't retry).
    """
    exit_after(a.max_runtime + 5)
    pace = 0.08 / a.standin_speed
    stats = {"requests": 0, "sentences": 0, "chars": 0, "cut_short": 0, "models": 0, "connections": 0}

    def respond(writer, status, body):
        head = (
            f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n"
            f"content-type: application/json\r\ncontent-length: {len(body)}\r\n\r\n"
        )
        writer.write(head.encode() + body)

    async def speak(reader, writer, req):
        text = str(req.get("input", ""))
        rate = int(req.get("sample_rate") or 48000)
        audio = tone(standin_seconds(text), rate)
        piece = max(1, int(rate * 0.08)) * 2  # 80 ms
        stats["requests"] += 1
        await asyncio.sleep(0.15)  # synthesis, up to the first byte
        writer.write(
            (
                "HTTP/1.1 200 OK\r\ncontent-type: audio/pcm\r\n"
                f"x-sample-rate: {rate}\r\nx-audio-encoding: pcm_s16le\r\n"
                "transfer-encoding: chunked\r\n\r\n"
            ).encode()
        )
        try:
            for pos in range(0, len(audio), piece):
                if reader.at_eof():  # the client hung up: an interruption
                    raise ConnectionResetError("client closed the connection")
                chunk = audio[pos : pos + piece]
                writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                await writer.drain()
                await asyncio.sleep(pace)
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except ConnectionError:
            stats["cut_short"] += 1
            raise
        stats["sentences"] += 1
        stats["chars"] += len(text)

    async def handle(reader, writer):
        stats["connections"] += 1
        try:
            while line := await reader.readline():
                method, target, _ = line.decode("latin-1").split(" ", 2)
                headers = {}
                while (h := await reader.readline()) not in (b"\r\n", b""):
                    k, _, v = h.decode("latin-1").partition(":")
                    headers[k.strip().lower()] = v.strip()
                body = await reader.readexactly(int(headers.get("content-length", "0")))
                path = target.split("?", 1)[0]
                if not headers.get("authorization", "").startswith("Bearer "):
                    respond(writer, 401, b'{"error": {"code": "unauthorized", "message": "no API key"}}')
                elif method == "GET" and path.endswith("/models"):
                    stats["models"] += 1
                    respond(writer, 200, b'{"object": "list", "data": [{"id": "mira-tts"}]}')
                elif method == "POST" and path.endswith("/audio/speech"):
                    await speak(reader, writer, json.loads(body))
                else:
                    respond(writer, 404, b'{"error": {"code": "not_found", "message": "no such route"}}')
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            writer.close()

    async def main():
        lag = []
        _lagger = asyncio.create_task(loop_lag(lag))
        t0, c0 = time.monotonic(), cpu_seconds()

        def write():
            write_json(
                Path(a.out, "standin_http.json"),
                {
                    "cpu_s": round(cpu_seconds() - c0, 2),
                    "wall_s": round(time.monotonic() - t0, 2),
                    **stats,
                    **lag_stats(lag),
                },
            )

        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, write)
        server = await asyncio.start_server(handle, "127.0.0.1", a.port)
        async with server:
            print("ready", flush=True)
            while True:
                await asyncio.sleep(2)
                write()

    asyncio.run(main())


# ------------------------------------------------------------------------------------ bot
def package_version(module):
    version = getattr(module, "__version__", None)
    if version:
        return version
    try:
        from importlib.metadata import version as dist_version

        return dist_version("pipecat-mirai")
    except Exception:
        return None


def bot_main(a):
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import FastAPI, WebSocket
    from loguru import logger
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        EndFrame,
        ErrorFrame,
        InterruptionFrame,
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        MetricsFrame,
        TextFrame,
    )
    from pipecat.metrics.metrics import TTFBMetricsData, TTSUsageMetricsData
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
    from pipecat.workers.runner import WorkerRunner

    import pipecat_mirai

    # Only names pipecat-mirai 0.3.0 has too: this harness runs against both releases.
    from pipecat_mirai import apply_output_lead

    # 0.5.0 renamed the HTTP service MiraiHttpTTSService (MiraiTTSService is now the
    # WebSocket one); before that, MiraiTTSService was HTTP. Works with either.
    MiraiHttpTTS = getattr(pipecat_mirai, "MiraiHttpTTSService", None) or pipecat_mirai.MiraiTTSService
    MiraiWsTTS = pipecat_mirai.MiraiWebsocketTTSService

    exit_after(a.max_runtime + 5)
    logger.remove()
    logger.add(sys.stderr, level=a.log_level)
    key = os.environ.get("MIRAI_API_KEY", "")
    lag = []
    stalls = {"count": 0, "total_ms": 0.0, "max_ms": 0.0}
    live, outputs = {}, {}  # label -> call record / transport output, while the call runs
    info = {
        "index": a.index,
        "mode": "defaults" if a.defaults else "explicit",
        "pipecat_mirai_version": package_version(pipecat_mirai),
        "pipecat_mirai_path": str(Path(pipecat_mirai.__file__).parent),
        "tts": a.tts,
        "endpoint": None,
        "harness_lead_secs": None if a.defaults else a.lead,
    }
    t0, c0 = time.monotonic(), cpu_seconds()

    class Tap(FrameProcessor):
        """Notes errors, metrics and the bot stopping, in either direction."""

        def __init__(self, call):
            super().__init__()
            self.call = call
            self.stopped = asyncio.Event()

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            now = time.monotonic()
            if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
                self.call["errors"].append((now, str(frame.error)[:300]))
            elif isinstance(frame, MetricsFrame):
                for d in frame.data:
                    if isinstance(d, TTSUsageMetricsData):
                        self.call["billed"].append((now, d.value))
                    elif isinstance(d, TTFBMetricsData) and d.value > 0:
                        self.call["ttfb_pipecat"].append((now, d.value))
            elif isinstance(frame, BotStoppedSpeakingFrame) and direction == FrameDirection.UPSTREAM:
                self.stopped.set()
            await self.push_frame(frame, direction)

    def make_tts():
        if a.defaults:
            # Exactly what a customer writes after `pip install pipecat-mirai`. The URL only when
            # one was given: the library's default is the production endpoint.
            if a.tts == "http":
                extra = {"base_url": a.http_url} if a.http_url else {}
                return MiraiHttpTTS(api_key=key, voice=a.voice, **extra)
            extra = {"url": a.url} if a.url else {}
            return MiraiWsTTS(api_key=key, voice=a.voice, **extra)
        if a.tts == "http":
            return MiraiHttpTTS(api_key=key, base_url=a.http_url, voice=a.voice)
        return MiraiWsTTS(api_key=key, url=a.url, voice=a.voice)

    def note_lead(call, output):
        """The lead the transport ended up with: the harness's (explicit mode) or the library's own."""
        if output is not None:
            call["output_lead_secs"] = getattr(output, "_mirai_output_lead_secs", None)
            call["output_pacing_patched"] = "_write_audio_sleep" in vars(output)

    def write_call(call):
        write_json(Path(a.out, f"bot_call{call['label']}.json"), call)

    def pool_stats():
        stats_fn = getattr(pipecat_mirai, "shared_connection_stats", None)  # 0.3.1 and later
        if stats_fn is None:
            return None
        try:
            return json.loads(json.dumps(stats_fn(), default=str))
        except Exception as exc:  # an API in flux must not break the run
            return {"error": repr(exc)}

    def write_stats():
        proc = {
            **info,
            "cpu_s": round(cpu_seconds() - c0, 2),
            "wall_s": round(time.monotonic() - t0, 2),
            "stalls": stalls["count"],
            "stalled_ms": round(stalls["total_ms"]),
            "stall_max_ms": round(stalls["max_ms"]),
            **lag_stats(lag),
        }
        pool = pool_stats()
        if pool is not None:
            proc["pool"] = pool
        write_json(Path(a.out, f"bot_proc{a.index}.json"), proc)

    def dump():
        """SIGUSR1 from the parent at the end of the run: write everything now. A call still
        running here was cut short by --max-runtime; its record is written as it stands."""
        now = time.monotonic()
        for label, call in list(live.items()):
            call["cut_at"] = now
            note_lead(call, outputs.get(label))
            write_call(call)
        write_stats()

    async def stats_loop():
        while True:
            write_stats()
            await asyncio.sleep(2)

    async def stall_loop():
        """A busy customer server: now and then the event loop does nothing else for a while."""
        lo, hi = a.stall_ms
        rng = random.Random(a.seed * 7919 + a.index)
        while True:
            await asyncio.sleep(a.stall_every * rng.uniform(0.7, 1.3))
            ms = rng.uniform(lo, hi)
            time.sleep(ms / 1000)
            stalls["count"] += 1
            stalls["total_ms"] += ms
            stalls["max_ms"] = max(stalls["max_ms"], ms)

    @asynccontextmanager
    async def lifespan(_app):
        tasks = [asyncio.create_task(loop_lag(lag)), asyncio.create_task(stats_loop())]
        if a.stall_every > 0:
            tasks.append(asyncio.create_task(stall_loop()))
        asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, dump)
        yield
        for task in tasks:
            task.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        t_connected = time.monotonic()
        await websocket.accept()
        await websocket.receive_text()
        start = json.loads(await websocket.receive_text())["start"]
        params = start["customParameters"]
        n = int(params["call"])
        label = params.get("label") or f"{n:02d}"
        duration = float(params.get("duration", a.duration))
        rng = random.Random(a.seed * 1000 + n)
        call = {
            "call": n,
            "label": label,
            "duration": duration,
            "greeting_now": a.greeting_now,
            "t_connected": t_connected,
            "turns": [],
            "errors": [],
            "billed": [],
            "ttfb_pipecat": [],
            "audio_starts": [],
            "retries": 0,
        }
        live[label] = call
        transport = FastAPIWebsocketTransport(
            websocket,
            FastAPIWebsocketParams(
                audio_in_enabled=False,
                audio_out_enabled=True,
                add_wav_header=False,
                serializer=TwilioFrameSerializer(
                    start["streamSid"], params=TwilioFrameSerializer.InputParams(auto_hang_up=False)
                ),
            ),
        )
        if not a.defaults:
            apply_output_lead(transport, a.lead)
        tts = make_tts()
        is_ws = isinstance(tts, MiraiWsTTS)
        info["endpoint"] = info["endpoint"] or getattr(tts, "_url" if is_ws else "_speech_url", None)
        if is_ws and hasattr(tts, "_on_audio_start") and hasattr(tts, "_retry_later"):
            on_audio_start, retry_later = tts._on_audio_start, tts._retry_later

            async def audio_start(event):
                await on_audio_start(event)
                if tts._sentence is not None and tts._sentence.turn is not None:
                    call["audio_starts"].append((time.monotonic(), event.get("text") or ""))

            async def retry(*args, **kwargs):
                call["retries"] += 1
                await retry_later(*args, **kwargs)

            tts._on_audio_start, tts._retry_later = audio_start, retry
        up, down = Tap(call), Tap(call)
        output = outputs[label] = transport.output()
        worker = PipelineWorker(
            Pipeline([transport.input(), up, tts, down, output]),
            params=PipelineParams(audio_out_sample_rate=SR, enable_metrics=True, enable_usage_metrics=True),
        )

        async def turns():
            if not a.greeting_now:
                await asyncio.sleep(1.0 + rng.uniform(0, 1.0))  # the call is answered
            end = time.monotonic() + duration
            k = 0
            while time.monotonic() < end:
                reply = REPLIES[(n + k) % len(REPLIES)]
                # With --greeting-now the greeting is always heard out: its TTFB is the point.
                barge = rng.random() < a.barge_in and not (a.greeting_now and k == 0)
                barge_after = rng.uniform(1.2, 3.0)
                up.stopped.clear()
                turn = {"k": k, "reply": reply, "barge": barge, "t_first_token": time.monotonic()}
                # Recorded from its start, so a call cut by --max-runtime still has this turn.
                turn["text_sent"] = ""
                call["turns"].append(turn)
                await worker.queue_frame(LLMFullResponseStartFrame())
                for tok in tokens(reply):
                    if barge and time.monotonic() - turn["t_first_token"] >= barge_after:
                        break
                    await worker.queue_frame(TextFrame(tok))
                    turn["text_sent"] += tok
                    await asyncio.sleep(1 / a.tokens_per_sec)
                if barge:
                    await asyncio.sleep(max(0, turn["t_first_token"] + barge_after - time.monotonic()))
                    turn["t_barge"] = time.monotonic()
                    await worker.queue_frame(InterruptionFrame())
                else:
                    await worker.queue_frame(LLMFullResponseEndFrame())
                    try:
                        await asyncio.wait_for(up.stopped.wait(), timeout=30)
                    except TimeoutError:
                        turn["no_stop"] = True
                turn["t_done"] = time.monotonic()
                k += 1
                await asyncio.sleep(rng.uniform(2.0, 6.0))  # the caller speaks
            # After this, Pipecat's output sends its end-of-call silence: not part of any reply.
            call["t_end"] = time.monotonic()
            await worker.queue_frame(EndFrame())

        task = asyncio.create_task(turns())
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        try:
            await runner.run()
        finally:
            task.cancel()
            note_lead(call, output)
            live.pop(label, None)
            outputs.pop(label, None)
            write_call(call)

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning", lifespan="on")


# ------------------------------------------------------------------------------------ phone
_ULAW = np.zeros(256, dtype=np.int16)
for _i in range(256):
    _u = ~_i & 0xFF
    _mag = (((_u & 0x0F) << 3) + 0x84) << ((_u >> 4) & 0x07)
    _ULAW[_i] = -(_mag - 0x84) if _u & 0x80 else _mag - 0x84


async def phone_call(n, label, port, at, duration, results):
    """One caller: connect at ``at`` (time.monotonic()), then record what arrives until hang-up.

    ``results[label]`` fills in as the call runs, so a call cut short still has its events.
    """
    import websockets

    rec = results[label] = {"events": []}
    events = rec["events"]
    await asyncio.sleep(max(0.0, at - time.monotonic()))
    deadline = time.monotonic() + duration + 60
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None, close_timeout=2) as ws:
            rec["t_connect"] = time.monotonic()
            await ws.send(json.dumps({"event": "connected"}))
            start = {
                "streamSid": f"MZ{label:0>30}",
                "callSid": f"CA{label:0>30}",
                "customParameters": {"call": str(n), "label": label, "duration": str(duration)},
            }
            await ws.send(json.dumps({"event": "start", "start": start}))
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except TimeoutError:
                    continue
                now = time.monotonic()
                msg = json.loads(raw)
                if msg.get("event") == "media":
                    pcm = _ULAW[np.frombuffer(base64.b64decode(msg["media"]["payload"]), dtype=np.uint8)]
                    events.append((now, "media", pcm))
                elif msg.get("event") == "clear":
                    events.append((now, "clear", None))
    except Exception as exc:  # the bot hung up (EndFrame) or the connection failed
        if not events:
            rec["phone_error"] = repr(exc)


def playout(media, stop_at=None, jb=0.060, tick=0.020):
    """Play ``media`` [(arrival, pcm)] at real time from a jitter buffer, like a carrier.

    Returns (start time, played pcm, gaps, silence_ms): a gap is a tick with too
    little audio while more of this reply was still to come.
    """
    if not media:
        return None, np.zeros(0, np.int16), 0, 0
    nf = int(tick * SR)
    t = media[0][0] + jb
    start, out, q, qi = t, [], [], 0
    gaps = inserted = 0
    in_gap = False
    while True:
        if stop_at is not None and t > stop_at:
            break
        while qi < len(media) and media[qi][0] <= t:
            q.append(media[qi][1])
            qi += 1
        need, got = nf, []
        while need and q:
            take = q[0][:need]
            got.append(take)
            need -= len(take)
            q[0] = q[0][len(take) :]
            if not len(q[0]):
                q.pop(0)
        seg = np.concatenate(got) if got else np.zeros(0, np.int16)
        more = qi < len(media) or bool(q)
        if len(seg) < nf and more:
            inserted += nf - len(seg)
            seg = np.concatenate([seg, np.zeros(nf - len(seg), np.int16)])
            gaps += 0 if in_gap else 1
            in_gap = True
        else:
            in_gap = False
        out.append(seg)
        t += tick
        if not more:
            break
    return start, np.concatenate(out) if out else np.zeros(0, np.int16), gaps, round(inserted / SR * 1000)


def analyse(events, bot, tts, t_connect=None):
    media = [(t, p) for t, kind, p in events if kind == "media"]
    clears = [t for t, kind, _ in events if kind == "clear"]
    turns = bot.get("turns", [])
    origin = turns[0]["t_first_token"] if turns else (media[0][0] if media else 0)
    wav = np.zeros(int(SR * ((media[-1][0] - origin) + 10 if media else 1)), np.int16)
    rows = []
    for i, turn in enumerate(turns):
        lo = turn["t_first_token"]
        # The last turn ends where the call did, or where --max-runtime cut it.
        end = bot.get("t_end", bot.get("cut_at", float("inf")))
        hi = turns[i + 1]["t_first_token"] if i + 1 < len(turns) else end
        m = [x for x in media if lo <= x[0] < hi]
        clear = next((c for c in clears if lo <= c < hi), None)
        before = [x for x in m if clear is None or x[0] <= clear]
        after_ms = sum(len(p) for t, p in m if clear is not None and t > clear) / SR * 1000
        start, pcm, gaps, silence = playout(before, stop_at=clear)
        if start is not None:
            at = int((start - origin) * SR)
            wav[at : at + len(pcm)] = pcm[: max(0, len(wav) - at)]
        heard = [s for t, s in bot.get("audio_starts", []) if lo <= t < hi]
        order_ok = None
        if tts == "ws":
            pos, order_ok = 0, True
            for s in heard:
                at = turn["reply"].find(s.strip(), pos)
                if at < 0:
                    order_ok = False
                    break
                pos = at + len(s.strip())
        billed = sum(v for t, v in bot.get("billed", []) if lo <= t < hi)
        first = before[0][0] if before else None
        rows.append(
            {
                "k": turn["k"],
                "barge": turn["barge"],
                "ttfb_ms": round((first - lo) * 1000) if first is not None else None,
                # The greeting only: from the phone's connect to the first audio it received.
                "from_connect_ms": (
                    round((first - t_connect) * 1000)
                    if turn["k"] == 0 and first is not None and t_connect is not None
                    else None
                ),
                "speech_s": round(len(pcm) / SR, 2),
                "gaps": gaps,
                "silence_ms": silence,
                "after_clear_ms": round(after_ms),
                "cleared": clear is not None,
                "order_ok": order_ok,
                "chars_sent": len(turn.get("text_sent", "")),
                "chars_billed": billed,
                "no_stop": turn.get("no_stop", False),
                # Still running when --max-runtime cut the call: not a reply that went unheard.
                "cut": "t_done" not in turn and "cut_at" in bot,
            }
        )
    return rows, wav


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, deadline, proc, log, timeout=60):
    """Wait for ``proc`` to listen on ``port``; if it dies first, stop with the end of its log."""
    end = min(time.monotonic() + timeout, deadline)
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        if proc.poll() is not None:
            tail = "".join(Path(log).read_text(errors="replace").splitlines(keepends=True)[-15:])
            raise SystemExit(f"{log} exited ({proc.returncode}) before listening on port {port}:\n{tail}")
        time.sleep(0.2)
    raise SystemExit(f"nothing listening on port {port} (gave up after {timeout} s or --max-runtime)")


def http_from_ws(url):
    return re.sub(r"^ws", "http", url).replace("/audio/speech/stream", "")


def ms_range(text):
    try:
        lo, hi = (float(x) for x in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected LO,HI in milliseconds; got {text!r}") from None
    if not 0 <= lo <= hi:
        raise argparse.ArgumentTypeError(f"need 0 <= LO <= HI; got {text!r}")
    return lo, hi


def main(a):
    run_start = time.monotonic()
    deadline = run_start + a.max_runtime
    if a.greeting_now is None:
        a.greeting_now = a.defaults
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    me = [sys.executable, str(Path(__file__).resolve())]
    procs = []
    env = dict(os.environ)

    def spawn(args, name):
        """Start a child; returns (process, log path)."""
        path = out / f"{name}.log"
        with open(path, "w") as log:
            procs.append(
                subprocess.Popen(
                    me + args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env
                )
            )
        return procs[-1], path

    def signal_all(sig):
        for p in list(procs):
            try:
                os.killpg(p.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass

    # The hard cap: whatever the parent is doing at --max-runtime, every child dies then.
    watchdog = threading.Timer(a.max_runtime, signal_all, args=(signal.SIGKILL,))
    watchdog.daemon = True
    watchdog.start()
    guard = {"hit": False}
    common = ["--out", a.out, "--max-runtime", str(a.max_runtime)]
    try:
        if a.standin:
            ws_port, http_port = free_port(), free_port()
            knobs = ["--standin-capacity", str(a.standin_capacity), "--standin-speed", str(a.standin_speed)]
            ws_fake = spawn(["_standin", "--port", str(ws_port), *knobs, *common], "standin")
            http_fake = spawn(["_standin_http", "--port", str(http_port), *knobs, *common], "standin_http")
            wait_port(ws_port, deadline, *ws_fake)
            wait_port(http_port, deadline, *http_fake)
            a.url = f"ws://127.0.0.1:{ws_port}/v1/audio/speech/stream"
            a.http_url = f"http://127.0.0.1:{http_port}/v1"
            key = "standin"
        else:
            key = Path(a.key_file).read_text().strip() if a.key_file else os.environ.get("MIRAI_API_KEY", "")
            if not key:
                raise SystemExit("pass --key-file or set MIRAI_API_KEY")
            if not a.defaults:
                a.url = a.url or LOCAL_WS_URL
                a.http_url = a.http_url or http_from_ws(a.url)
            elif a.tts == "http" and a.url and not a.http_url:
                a.http_url = http_from_ws(a.url)
            # --defaults with neither: the bots pass no URL, so the library's default is used.
        env["MIRAI_API_KEY"] = key  # to the bots by environment, not argv (ps shows argv)
        ports = [free_port() for _ in range(a.procs)]
        bots = []
        for i, port in enumerate(ports):
            args = [
                "_bot",
                "--index",
                str(i),
                "--port",
                str(port),
                "--tts",
                a.tts,
                "--voice",
                a.voice,
                "--lead",
                str(a.lead),
                "--duration",
                str(a.duration),
                "--barge-in",
                str(a.barge_in),
                "--tokens-per-sec",
                str(a.tokens_per_sec),
                "--seed",
                str(a.seed),
                "--log-level",
                a.log_level,
                "--stall-every",
                str(a.stall_every),
                "--stall-ms",
                "{:g},{:g}".format(*a.stall_ms),
                "--greeting-now" if a.greeting_now else "--no-greeting-now",
                *common,
            ]
            if a.defaults:
                args.append("--defaults")
            if a.url:
                args += ["--url", a.url]
            if a.http_url:
                args += ["--http-url", a.http_url]
            bots.append(spawn(args, f"bot{i}"))
        for port, bot in zip(ports, bots, strict=True):
            wait_port(port, deadline, *bot)
        endpoint = (a.http_url if a.tts == "http" else a.url) or "the library's default (production)"
        print(
            f"{a.calls} calls, {a.procs} bot process(es), tts={a.tts}, "
            f"{'--defaults' if a.defaults else f'explicit, lead {a.lead} s'}, {a.duration:.0f} s, "
            f"endpoint={endpoint}",
            flush=True,
        )
        setup = time.monotonic() - run_start
        seed = a.seed_call + 12 + a.seed_gap if a.seed_call > 0 else 0
        burst = 1 + a.stagger * (a.calls - 1) + (0 if a.greeting_now else 2) + a.duration + 14
        planned = setup + seed + burst
        if planned > a.max_runtime:
            print(
                f"WARNING: this run needs about {planned:.0f} s, more than --max-runtime "
                f"{a.max_runtime:.0f} s; calls still running then are cut (raise --max-runtime "
                "or lower --duration)",
                flush=True,
            )

        results = {}
        lag = []
        t_phones, c0 = time.monotonic(), cpu_seconds()

        async def phones():
            lagger = asyncio.create_task(loop_lag(lag))
            try:
                if a.seed_call > 0:
                    at = time.monotonic() + 0.5
                    seeds = (
                        phone_call(-(i + 1), f"seed{i}", port, at, a.seed_call, results)
                        for i, port in enumerate(ports)
                    )
                    await asyncio.gather(*seeds)
                    await asyncio.sleep(a.seed_gap)
                t0 = time.monotonic() + 1.0  # one shared start: --stagger 0 starts every call at once
                await asyncio.gather(
                    *(
                        phone_call(n, f"{n:02d}", ports[n % a.procs], t0 + n * a.stagger, a.duration, results)
                        for n in range(a.calls)
                    )
                )
            finally:
                lagger.cancel()

        async def guarded():
            try:
                await asyncio.wait_for(phones(), timeout=max(1.0, deadline - time.monotonic() - 4))
            except TimeoutError:
                guard["hit"] = True
                print(
                    f"WARNING: --max-runtime {a.max_runtime:.0f} s reached: hanging up and stopping the bots",
                    flush=True,
                )

        asyncio.run(guarded())
        wall = time.monotonic() - t_phones
        phone_cpu = cpu_seconds() - c0
        signal_all(signal.SIGUSR1)  # every child writes its final stats (and any cut call) now
        time.sleep(1.0)
    finally:
        signal_all(signal.SIGTERM)
        end = time.monotonic() + 5
        for p in procs:
            try:
                p.wait(timeout=max(0.1, end - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        watchdog.cancel()

    report(a, out, results, wall, phone_cpu, lag, guard["hit"])


def call_report(a, out, label, n, res, wav_name):
    botf = out / f"bot_call{label}.json"
    bot = json.loads(botf.read_text()) if botf.exists() else {}
    rows, wav = analyse(res.get("events", []), bot, a.tts, res.get("t_connect"))
    with wave.open(str(out / wav_name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(wav.tobytes())
    ttfbs = [r["ttfb_ms"] for r in rows if r["ttfb_ms"] is not None]
    later = [r["ttfb_ms"] for r in rows if r["k"] > 0 and r["ttfb_ms"] is not None]
    greeting = next((r for r in rows if r["k"] == 0), None)
    speech = sum(r["speech_s"] for r in rows)
    silence = sum(r["silence_ms"] for r in rows)
    c = {
        "call": n,
        "label": label,
        "turns": len(rows),
        "barge_ins": sum(r["barge"] for r in rows),
        "silent_replies": sum(1 for r in rows if not r["barge"] and not r["cut"] and r["ttfb_ms"] is None),
        "ttfb_p50_ms": pct(ttfbs, 50),
        "ttfb_p95_ms": pct(ttfbs, 95),
        "greeting_ttfb_ms": greeting["ttfb_ms"] if greeting else None,
        "greeting_from_connect_ms": greeting["from_connect_ms"] if greeting else None,
        "later_ttfb_p50_ms": pct(later, 50),
        "later_ttfb_p95_ms": pct(later, 95),
        "later_ttfb_max_ms": max(later) if later else None,
        "gaps": sum(r["gaps"] for r in rows),
        "silence_ms": silence,
        "stretch_pct": round(100 * silence / 1000 / speech, 2) if speech else 0.0,
        "after_clear_ms": sum(r["after_clear_ms"] for r in rows),
        "order_errors": sum(1 for r in rows if r["order_ok"] is False),
        "errors": len(bot.get("errors", [])) + (1 if "phone_error" in res or not bot else 0),
        "retries": bot.get("retries", 0),
        "chars_sent": sum(r["chars_sent"] for r in rows),
        "chars_billed": sum(r["chars_billed"] for r in rows),
        "completed": "t_end" in bot,
        "output_lead_secs": bot.get("output_lead_secs"),
        "output_pacing_patched": bot.get("output_pacing_patched"),
        "error_samples": [e for _, e in bot.get("errors", [])][:3]
        + ([res["phone_error"]] if "phone_error" in res else []),
        "turn_rows": rows,
    }
    return c, rows


def report(a, out, results, wall, phone_cpu, lag, guard_hit):
    calls, all_rows = [], []
    for n in range(a.calls):
        label = f"{n:02d}"
        c, rows = call_report(a, out, label, n, results.get(label, {}), f"call{label}.wav")
        calls.append(c)
        all_rows += rows
    seeds = []
    if a.seed_call > 0:
        for i in range(a.procs):
            label = f"seed{i}"
            seeds.append(call_report(a, out, label, -(i + 1), results.get(label, {}), f"{label}.wav")[0])

    cols = [
        "call",
        "turns",
        "barge_ins",
        "greeting_ttfb_ms",
        "later_ttfb_p50_ms",
        "later_ttfb_p95_ms",
        "later_ttfb_max_ms",
        "ttfb_p50_ms",
        "ttfb_p95_ms",
        "gaps",
        "silence_ms",
        "stretch_pct",
        "after_clear_ms",
        "order_errors",
        "errors",
        "retries",
        "chars_sent",
        "chars_billed",
    ]
    print("\n" + " | ".join(cols))
    for c in calls:
        print(" | ".join(str(c[k]) for k in cols))
    for c in seeds:
        print(f"{c['label']} (not in the stats) | " + " | ".join(str(c[k]) for k in cols[1:]))

    ttfbs = [r["ttfb_ms"] for r in all_rows if r["ttfb_ms"] is not None]
    greetings = [c["greeting_ttfb_ms"] for c in calls if c["greeting_ttfb_ms"] is not None]
    later = [r["ttfb_ms"] for r in all_rows if r["k"] > 0 and r["ttfb_ms"] is not None]
    from_connect = [c["greeting_from_connect_ms"] for c in calls if c["greeting_from_connect_ms"] is not None]
    procs = [json.loads(p.read_text()) for p in sorted(out.glob("bot_proc*.json"))]
    bot_cpu = sum(p["cpu_s"] for p in procs)
    cores = os.cpu_count() or 1
    client_cpu_pct = round(100 * (bot_cpu + phone_cpu) / wall / cores, 1)
    standin_file = out / ("standin_http.json" if a.tts == "http" else "standin.json")
    standin = json.loads(standin_file.read_text()) if standin_file.exists() else None
    speech = sum(r["speech_s"] for r in all_rows)
    versions = sorted({str(p.get("pipecat_mirai_version")) for p in procs})
    endpoints = sorted({str(p.get("endpoint")) for p in procs if p.get("endpoint")})
    summary = {
        "mode": "defaults" if a.defaults else "explicit",
        "pipecat_mirai_version": versions[0] if len(versions) == 1 else versions or None,
        "pipecat_mirai_path": procs[0].get("pipecat_mirai_path") if procs else None,
        "tts": a.tts,
        "endpoint": endpoints[0] if len(endpoints) == 1 else endpoints or None,
        "harness_output_lead_secs": None if a.defaults else a.lead,
        "output_lead_on_transport": sorted({c["output_lead_secs"] for c in calls}, key=str),
        "greeting_now": a.greeting_now,
        "calls": a.calls,
        "procs": a.procs,
        "stagger_s": a.stagger,
        "duration_s": a.duration,
        "seed_call_s": a.seed_call,
        "seed_gap_s": a.seed_gap if a.seed_call > 0 else None,
        "turns": len(all_rows),
        "barge_ins": sum(r["barge"] for r in all_rows),
        "ttfb_p50_ms": pct(ttfbs, 50),
        "ttfb_p95_ms": pct(ttfbs, 95),
        "ttfb_max_ms": max(ttfbs) if ttfbs else None,
        "greeting_ttfb_p50_ms": pct(greetings, 50),
        "greeting_ttfb_p95_ms": pct(greetings, 95),
        "greeting_ttfb_max_ms": max(greetings) if greetings else None,
        "later_ttfb_p50_ms": pct(later, 50),
        "later_ttfb_p95_ms": pct(later, 95),
        "later_ttfb_max_ms": max(later) if later else None,
        "greeting_from_connect_p50_ms": pct(from_connect, 50),
        "greeting_from_connect_p95_ms": pct(from_connect, 95),
        "greeting_from_connect_max_ms": max(from_connect) if from_connect else None,
        "gaps": sum(c["gaps"] for c in calls),
        "silence_ms": sum(c["silence_ms"] for c in calls),
        "stretch_pct": round(100 * sum(c["silence_ms"] for c in calls) / 1000 / speech, 3) if speech else 0.0,
        "after_clear_ms": sum(c["after_clear_ms"] for c in calls),
        "order_errors": sum(c["order_errors"] for c in calls),
        "silent_replies": sum(c["silent_replies"] for c in calls),
        "errors": sum(c["errors"] for c in calls),
        "retries": sum(c["retries"] for c in calls),
        "chars_sent": sum(c["chars_sent"] for c in calls),
        "chars_billed": sum(c["chars_billed"] for c in calls),
        "speech_heard_s": round(speech, 1),
        "stall_every_s": a.stall_every or None,
        "stall_ms": list(a.stall_ms) if a.stall_every > 0 else None,
        "bot_stalls": sum(p.get("stalls", 0) for p in procs),
        "bot_stalled_ms": sum(p.get("stalled_ms", 0) for p in procs),
        "max_runtime_s": a.max_runtime,
        "max_runtime_hit": guard_hit,
        "calls_cut": sum(1 for c in calls if not c["completed"]),
        "seed_calls": [
            {
                k: c[k]
                for k in (
                    "label",
                    "turns",
                    "greeting_ttfb_ms",
                    "greeting_from_connect_ms",
                    "later_ttfb_max_ms",
                    "gaps",
                    "errors",
                    "error_samples",
                )
            }
            for c in seeds
        ],
        "client_cpu_pct_of_host": client_cpu_pct,
        "cores": cores,
        "bot_procs": procs,
        "phone": {"cpu_s": round(phone_cpu, 2), "wall_s": round(wall, 1), **lag_stats(lag)},
        "standin": standin,
    }
    flags = {
        "no_errors": summary["errors"] == 0,
        "no_audible_gaps": summary["gaps"] == 0,
        "no_audio_after_clear": summary["after_clear_ms"] == 0,
        "sentences_in_order": summary["order_errors"] == 0,
        "every_reply_heard": summary["silent_replies"] == 0,
        "client_cpu_under_60pct": client_cpu_pct < 60,
        "within_max_runtime": not guard_hit,
    }
    (out / "result.json").write_text(
        json.dumps({"summary": summary, "flags": flags, "calls": calls, "seed_calls": seeds}, indent=1)
    )
    print(
        "\nsummary:",
        json.dumps({k: v for k, v in summary.items() if k not in ("bot_procs",)}, ensure_ascii=False),
    )
    print("bot processes:", json.dumps(procs))
    print_summary(summary, procs, len(greetings), len(later), len(ttfbs))
    for name, ok in flags.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    for c in calls + seeds:
        for e in c["error_samples"]:
            print(f"  call {c['label']} error: {e}")
    print(f"\nwrote {out}/result.json and {out}/callNN.wav")
    if not all(flags.values()):
        sys.exit(1)


def print_summary(s, procs, n_greeting, n_later, n_all):
    def trio(name):
        return f"p50 {s[name + '_p50_ms']} ms, p95 {s[name + '_p95_ms']} ms, max {s[name + '_max_ms']} ms"

    lead = (
        "none applied by the harness"
        if s["mode"] == "defaults"
        else f"apply_output_lead({s['harness_output_lead_secs']})"
    )
    stagger = "all at once" if not s["stagger_s"] else f"{s['stagger_s']} s apart"
    print()
    print(
        f"mode: {s['mode']}, pipecat-mirai {s['pipecat_mirai_version']} (as the bot reports it), "
        f"tts={s['tts']}, endpoint={s['endpoint']}"
    )
    print(f"output lead: {lead}; seen on the transport: {s['output_lead_on_transport']}")
    print(
        f"calls: {s['calls']} ({stagger}) on {s['procs']} bot process(es), {s['duration_s']:g} s of turns, "
        f"greeting at connect: {'yes' if s['greeting_now'] else 'no (1-2 s answer delay)'}"
    )
    for seed in s["seed_calls"]:
        print(
            f"seed call {seed['label']} ({s['seed_call_s']:g} s, then {s['seed_gap_s']:g} s idle, not in the "
            f"stats): greeting TTFB {seed['greeting_ttfb_ms']} ms, errors {seed['errors']}"
        )
    print(f"TTFB (LLM first token -> first audio at the phone), all turns (n={n_all}): {trio('ttfb')}")
    print(f"TTFB, greeting (turn 0, n={n_greeting}): {trio('greeting_ttfb')}")
    print(f"TTFB, later turns (n={n_later}): {trio('later_ttfb')}")
    print(f"greeting, phone connected -> first audio: {trio('greeting_from_connect')}")
    print(
        f"audible gaps: {s['gaps']} ({s['silence_ms']} ms of silence inserted, stretch {s['stretch_pct']}%)"
    )
    print(
        f"audio after a clear: {s['after_clear_ms']} ms; sentences out of order: {s['order_errors']}; "
        f"silent replies: {s['silent_replies']}"
    )
    print(
        f"errors: {s['errors']}; capacity retries: {s['retries']}; "
        f"chars sent/billed: {s['chars_sent']}/{s['chars_billed']}"
    )
    if s["stall_every_s"]:
        print(
            f"bot stalls (every ~{s['stall_every_s']:g} s, {s['stall_ms'][0]:g}-{s['stall_ms'][1]:g} ms): "
            f"{s['bot_stalls']}, {s['bot_stalled_ms']} ms in all"
        )
    else:
        print("bot stalls: off")
    lag_p99 = max((p.get("lag_p99_ms", 0) for p in procs), default=None)
    lag_max = max((p.get("lag_max_ms", 0) for p in procs), default=None)
    print(
        f"bot event-loop lag: p99 {lag_p99} ms, max {lag_max} ms; "
        f"client CPU {s['client_cpu_pct_of_host']}% of {s['cores']} cores"
    )
    if s["max_runtime_hit"]:
        print(
            f"WARNING: cut at --max-runtime {s['max_runtime_s']:g} s; {s['calls_cut']} call(s) did not "
            "finish and are reported as far as they got"
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--url",
        default=None,
        help=f"streaming TTS URL (default: {LOCAL_WS_URL}; with --defaults, the library's own)",
    )
    p.add_argument("--http-url", default=None, help="base URL for --tts http (default: derived from --url)")
    p.add_argument("--key-file", help="file holding the API key (default: $MIRAI_API_KEY)")
    p.add_argument("--tts", choices=["ws", "http"], default="ws")
    p.add_argument(
        "--defaults",
        action="store_true",
        help="build the TTS service from api_key and voice only, as a customer does; no output lead",
    )
    p.add_argument("--calls", type=int, default=12)
    p.add_argument("--procs", type=int, default=1, help="bot processes (calls are split across them)")
    p.add_argument("--duration", type=float, default=100, help="seconds of turns per call")
    p.add_argument("--barge-in", type=float, default=0.2, help="share of replies interrupted mid-way")
    p.add_argument("--tokens-per-sec", type=float, default=40)
    p.add_argument("--lead", type=float, default=0.4, help="apply_output_lead seconds (not with --defaults)")
    p.add_argument("--voice", default="shruti")
    p.add_argument("--stagger", type=float, default=0.25, help="seconds between call starts (0: all at once)")
    p.add_argument(
        "--greeting-now",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="start the greeting the moment the call connects (default: on with --defaults)",
    )
    p.add_argument(
        "--seed-call",
        type=float,
        default=0,
        metavar="SECS",
        help="first run one call of SECS seconds on each bot process (0: off)",
    )
    p.add_argument("--seed-gap", type=float, default=5, help="seconds between the seed call(s) and the burst")
    p.add_argument(
        "--stall-every",
        type=float,
        default=0,
        metavar="SECS",
        help="block each bot's event loop about every SECS seconds (±30%%; 0: off)",
    )
    p.add_argument(
        "--stall-ms", type=ms_range, default=(150.0, 300.0), metavar="LO,HI", help="length of each stall"
    )
    p.add_argument(
        "--max-runtime", type=float, default=150, help="seconds; the run is cut and the bots killed then"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-level", default="WARNING")
    p.add_argument("--standin", action="store_true", help="run against local stand-ins (WS and HTTP)")
    p.add_argument("--standin-capacity", type=int, default=2, help="WS stand-in: sentences refused once")
    p.add_argument("--standin-speed", type=float, default=1.6, help="stand-in: delivery speed x real time")
    p.add_argument("--out", required=True)
    roles = {"_bot": bot_main, "_standin": standin_main, "_standin_http": standin_http_main}
    if len(sys.argv) > 1 and sys.argv[1] in roles:
        role = sys.argv.pop(1)
        p.add_argument("--index", type=int, default=0)
        p.add_argument("--port", type=int, required=True)
        roles[role](p.parse_args())
    else:
        main(p.parse_args())
