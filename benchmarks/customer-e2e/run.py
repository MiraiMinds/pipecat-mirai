#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Customer end-to-end test: N phone calls through pipecat-mirai, as a customer runs them.

Each call is the bot a customer builds from our README:

  fake LLM (a Hindi/Hinglish reply streamed token by token, ~40 tokens/s)
    -> MiraiWebsocketTTSService (or --tts http: MiraiTTSService), 8 kHz pipeline
    -> FastAPIWebsocketTransport + TwilioFrameSerializer (8 kHz mu-law),
       apply_output_lead(transport, 0.4)
    -> a phone simulator that plays the media at exactly real time, like a
       carrier, and records what the caller hears (one WAV per call).

Turn loop per call: the bot replies, the caller "speaks" for 2-6 s, repeat for
--duration. --barge-in makes that share of replies get interrupted mid-way
(InterruptionFrame, so Pipecat sends the provider a "clear").

Bots run in --procs worker processes (uvicorn + FastAPI, like a customer's
server); the phone runs in this process, so a stalled bot cannot distort the
"carrier" clock. All processes are on one host and share time.monotonic().

    # stand-in server (tests/fake_mirai_ws.py), no key needed:
    uv run --with uvicorn python benchmarks/customer-e2e/run.py --standin --calls 12 --out /tmp/e2e
    # the real endpoint:
    uv run --with uvicorn python benchmarks/customer-e2e/run.py \\
        --url ws://localhost:8130/v1/audio/speech/stream --key-file key.txt \\
        --calls 12 --duration 180 --out results/e2e

Prints a per-call table, an overall summary and PASS/FAIL flags, and writes
result.json and callNN.wav to --out.
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import resource
import signal
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

SR = 8000
REPO = Path(__file__).resolve().parents[2]

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


async def loop_lag(samples, interval=0.05):
    """Record how late a 50 ms sleep wakes up: the event loop's stalls."""
    while True:
        t = time.monotonic()
        await asyncio.sleep(interval)
        samples.append(time.monotonic() - t - interval)


def lag_stats(samples):
    if not samples:
        return {}
    a = np.array(samples) * 1000
    return {"lag_p99_ms": round(float(np.percentile(a, 99)), 1), "lag_max_ms": round(float(a.max()), 1)}


# ------------------------------------------------------------------------------------ stand-in
def standin_main(a):
    """tests/fake_mirai_ws.py on a fixed port, paced like Mirai (150 ms first byte, then
    --standin-speed times real time; below 1.0 the caller must hear gaps)."""
    sys.path.insert(0, str(REPO / "tests"))
    from fake_mirai_ws import FakeMiraiWS
    from websockets.asyncio.server import serve

    capacity = {}
    for reply in REPLIES[: a.standin_capacity]:
        capacity[sentences(reply)[1]] = 0.5  # each refused once, retry after 0.5 s
    fake = FakeMiraiWS(
        seconds=lambda s: max(0.8, len(s) / 14),
        reads=(1280,),  # 80 ms of 8 kHz audio per frame
        pace=0.08 / a.standin_speed,
        first_byte=0.15,
        capacity=capacity,
    )

    async def main():
        lag = []
        asyncio.create_task(loop_lag(lag))
        t0, c0 = time.monotonic(), cpu_seconds()
        async with serve(fake._handler, "127.0.0.1", a.port, max_size=None):
            print("ready", flush=True)
            while True:
                await asyncio.sleep(2)
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
                Path(a.out, "standin.json").write_text(json.dumps(stats))

    asyncio.run(main())


# ------------------------------------------------------------------------------------ bot
def bot_main(a):
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

    from pipecat_mirai import MiraiTTSService, MiraiWebsocketTTSService, apply_output_lead

    logger.remove()
    logger.add(sys.stderr, level=a.log_level)
    key = os.environ.get("MIRAI_API_KEY", "")
    app = FastAPI()
    lag = []
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
        if a.tts == "http":
            return MiraiTTSService(api_key=key, base_url=a.http_url, voice=a.voice)
        return MiraiWebsocketTTSService(api_key=key, url=a.url, voice=a.voice)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        start_background()
        await websocket.accept()
        await websocket.receive_text()
        start = json.loads(await websocket.receive_text())["start"]
        n = int(start["customParameters"]["call"])
        rng = random.Random(a.seed * 1000 + n)
        call = {"call": n, "turns": [], "errors": [], "billed": [], "ttfb_pipecat": [], "audio_starts": []}
        call["retries"] = 0
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
        apply_output_lead(transport, a.lead)
        tts = make_tts()
        if isinstance(tts, MiraiWebsocketTTSService):
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
        worker = PipelineWorker(
            Pipeline([transport.input(), up, tts, down, transport.output()]),
            params=PipelineParams(audio_out_sample_rate=SR, enable_metrics=True, enable_usage_metrics=True),
        )

        async def turns():
            await asyncio.sleep(1.0 + rng.uniform(0, 1.0))  # the call is answered
            end = time.monotonic() + a.duration
            k = 0
            while time.monotonic() < end:
                reply = REPLIES[(n + k) % len(REPLIES)]
                barge = rng.random() < a.barge_in
                barge_after = rng.uniform(1.2, 3.0)
                up.stopped.clear()
                turn = {"k": k, "reply": reply, "barge": barge, "t_first_token": time.monotonic()}
                await worker.queue_frame(LLMFullResponseStartFrame())
                sent = ""
                for tok in tokens(reply):
                    if barge and time.monotonic() - turn["t_first_token"] >= barge_after:
                        break
                    await worker.queue_frame(TextFrame(tok))
                    sent += tok
                    await asyncio.sleep(1 / a.tokens_per_sec)
                turn["text_sent"] = sent
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
                call["turns"].append(turn)
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
            Path(a.out, f"bot_call{n:02d}.json").write_text(json.dumps(call, ensure_ascii=False))

    async def stats():
        while True:
            await asyncio.sleep(2)
            Path(a.out, f"bot_proc{a.index}.json").write_text(
                json.dumps(
                    {
                        "cpu_s": round(cpu_seconds() - c0, 2),
                        "wall_s": round(time.monotonic() - t0, 2),
                        **lag_stats(lag),
                    }
                )
            )

    background = []

    def start_background():  # on the first call, in uvicorn's loop
        if not background:
            background.extend([asyncio.create_task(loop_lag(lag)), asyncio.create_task(stats())])

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


# ------------------------------------------------------------------------------------ phone
_ULAW = np.zeros(256, dtype=np.int16)
for _i in range(256):
    _u = ~_i & 0xFF
    _mag = (((_u & 0x0F) << 3) + 0x84) << ((_u >> 4) & 0x07)
    _ULAW[_i] = -(_mag - 0x84) if _u & 0x80 else _mag - 0x84


async def phone_call(a, n, port, results):
    import websockets

    await asyncio.sleep(n * a.stagger)
    events = []
    deadline = time.monotonic() + a.duration + 60
    try:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws", max_size=None) as ws:
            await ws.send(json.dumps({"event": "connected"}))
            start = {
                "streamSid": f"MZ{n:030d}",
                "callSid": f"CA{n:030d}",
                "customParameters": {"call": str(n)},
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
            results[n] = {"phone_error": repr(exc)}
            return
    results[n] = {"events": events}


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


def analyse(n, events, bot, tts):
    media = [(t, p) for t, kind, p in events if kind == "media"]
    clears = [t for t, kind, _ in events if kind == "clear"]
    turns = bot.get("turns", [])
    origin = turns[0]["t_first_token"] if turns else (media[0][0] if media else 0)
    wav = np.zeros(int(SR * ((media[-1][0] - origin) + 10 if media else 1)), np.int16)
    rows = []
    for i, turn in enumerate(turns):
        lo = turn["t_first_token"]
        hi = turns[i + 1]["t_first_token"] if i + 1 < len(turns) else bot.get("t_end", float("inf"))
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
        rows.append(
            {
                "k": turn["k"],
                "barge": turn["barge"],
                "ttfb_ms": round((before[0][0] - lo) * 1000) if before else None,
                "speech_s": round(len(pcm) / SR, 2),
                "gaps": gaps,
                "silence_ms": silence,
                "after_clear_ms": round(after_ms),
                "cleared": clear is not None,
                "order_ok": order_ok,
                "chars_sent": len(turn.get("text_sent", "")),
                "chars_billed": billed,
                "no_stop": turn.get("no_stop", False),
            }
        )
    return rows, wav


def pct(values, q):
    return round(float(np.percentile(values, q))) if values else None


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_port(port, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise SystemExit(f"nothing listening on port {port} after {timeout} s")


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    me = [sys.executable, str(Path(__file__).resolve())]
    procs = []

    env = dict(os.environ)

    def spawn(args, name):
        log = open(out / f"{name}.log", "w")
        procs.append(
            subprocess.Popen(me + args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
        )

    try:
        if a.standin:
            port = free_port()
            spawn(
                [
                    "_standin",
                    "--port",
                    str(port),
                    "--out",
                    a.out,
                    "--standin-capacity",
                    str(a.standin_capacity),
                    "--standin-speed",
                    str(a.standin_speed),
                ],
                "standin",
            )
            wait_port(port)
            a.url = f"ws://127.0.0.1:{port}/v1/audio/speech/stream"
            key = "standin"
        else:
            key = Path(a.key_file).read_text().strip() if a.key_file else os.environ.get("MIRAI_API_KEY", "")
            if not key:
                raise SystemExit("pass --key-file or set MIRAI_API_KEY")
        env["MIRAI_API_KEY"] = key  # to the bots by environment, not argv (ps shows argv)
        http_url = a.http_url or re.sub(r"^ws", "http", a.url).replace("/audio/speech/stream", "")
        ports = [free_port() for _ in range(a.procs)]
        for i, port in enumerate(ports):
            spawn(
                [
                    "_bot",
                    "--index",
                    str(i),
                    "--port",
                    str(port),
                    "--tts",
                    a.tts,
                    "--url",
                    a.url,
                    "--http-url",
                    http_url,
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
                    "--out",
                    a.out,
                ],
                f"bot{i}",
            )
        for port in ports:
            wait_port(port)
        print(
            f"{a.calls} calls, {a.procs} bot process(es), tts={a.tts}, {a.duration:.0f} s, url={a.url}",
            flush=True,
        )

        results = {}
        lag = []
        t0, c0 = time.monotonic(), cpu_seconds()

        async def phones():
            lagger = asyncio.create_task(loop_lag(lag))
            await asyncio.gather(*(phone_call(a, n, ports[n % a.procs], results) for n in range(a.calls)))
            lagger.cancel()

        asyncio.run(phones())
        wall = time.monotonic() - t0
        phone_cpu = cpu_seconds() - c0
        time.sleep(2.5)  # let the bots write their last stats
    finally:
        for p in procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)

    report(a, out, results, wall, phone_cpu, lag)


def report(a, out, results, wall, phone_cpu, lag):
    calls, all_rows = [], []
    for n in range(a.calls):
        botf = out / f"bot_call{n:02d}.json"
        bot = json.loads(botf.read_text()) if botf.exists() else {}
        res = results.get(n, {})
        rows, wav = analyse(n, res.get("events", []), bot, a.tts)
        with wave.open(str(out / f"call{n:02d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(wav.tobytes())
        ttfbs = [r["ttfb_ms"] for r in rows if r["ttfb_ms"] is not None]
        speech = sum(r["speech_s"] for r in rows)
        silence = sum(r["silence_ms"] for r in rows)
        c = {
            "call": n,
            "turns": len(rows),
            "barge_ins": sum(r["barge"] for r in rows),
            "silent_replies": sum(1 for r in rows if not r["barge"] and r["ttfb_ms"] is None),
            "ttfb_p50_ms": pct(ttfbs, 50),
            "ttfb_p95_ms": pct(ttfbs, 95),
            "gaps": sum(r["gaps"] for r in rows),
            "silence_ms": silence,
            "stretch_pct": round(100 * silence / 1000 / speech, 2) if speech else 0.0,
            "after_clear_ms": sum(r["after_clear_ms"] for r in rows),
            "order_errors": sum(1 for r in rows if r["order_ok"] is False),
            "errors": len(bot.get("errors", [])) + (1 if "phone_error" in res or not bot else 0),
            "retries": bot.get("retries", 0),
            "chars_sent": sum(r["chars_sent"] for r in rows),
            "chars_billed": sum(r["chars_billed"] for r in rows),
            "error_samples": [e for _, e in bot.get("errors", [])][:3]
            + ([res["phone_error"]] if "phone_error" in res else []),
            "turn_rows": rows,
        }
        calls.append(c)
        all_rows += rows

    cols = [
        "call",
        "turns",
        "barge_ins",
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

    ttfbs = [r["ttfb_ms"] for r in all_rows if r["ttfb_ms"] is not None]
    procs = [json.loads(p.read_text()) for p in sorted(out.glob("bot_proc*.json"))]
    bot_cpu = sum(p["cpu_s"] for p in procs)
    cores = os.cpu_count() or 1
    client_cpu_pct = round(100 * (bot_cpu + phone_cpu) / wall / cores, 1)
    standin = json.loads((out / "standin.json").read_text()) if (out / "standin.json").exists() else None
    speech = sum(r["speech_s"] for r in all_rows)
    summary = {
        "calls": a.calls,
        "procs": a.procs,
        "tts": a.tts,
        "turns": len(all_rows),
        "barge_ins": sum(r["barge"] for r in all_rows),
        "ttfb_p50_ms": pct(ttfbs, 50),
        "ttfb_p95_ms": pct(ttfbs, 95),
        "ttfb_max_ms": max(ttfbs) if ttfbs else None,
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
    }
    (out / "result.json").write_text(
        json.dumps({"summary": summary, "flags": flags, "calls": calls}, indent=1)
    )
    print(
        "\nsummary:",
        json.dumps({k: v for k, v in summary.items() if k not in ("bot_procs",)}, ensure_ascii=False),
    )
    print("bot processes:", json.dumps(procs))
    for name, ok in flags.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(
        f"TTFB (LLM first token -> first audio at the phone): p50 {summary['ttfb_p50_ms']} ms, "
        f"p95 {summary['ttfb_p95_ms']} ms, max {summary['ttfb_max_ms']} ms"
    )
    for c in calls:
        for e in c["error_samples"]:
            print(f"  call {c['call']} error: {e}")
    print(f"\nwrote {out}/result.json and {out}/callNN.wav")
    if not all(flags.values()):
        sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="ws://127.0.0.1:8130/v1/audio/speech/stream", help="streaming TTS URL")
    p.add_argument("--http-url", default=None, help="base URL for --tts http (default: derived from --url)")
    p.add_argument("--key-file", help="file holding the API key (default: $MIRAI_API_KEY)")
    p.add_argument("--tts", choices=["ws", "http"], default="ws")
    p.add_argument("--calls", type=int, default=12)
    p.add_argument("--procs", type=int, default=1, help="bot processes (calls are split across them)")
    p.add_argument("--duration", type=float, default=180, help="seconds of turns per call")
    p.add_argument("--barge-in", type=float, default=0.2, help="share of replies interrupted mid-way")
    p.add_argument("--tokens-per-sec", type=float, default=40)
    p.add_argument("--lead", type=float, default=0.4, help="apply_output_lead seconds (README default)")
    p.add_argument("--voice", default="shruti")
    p.add_argument("--stagger", type=float, default=0.25, help="seconds between call starts")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-level", default="WARNING")
    p.add_argument("--standin", action="store_true", help="run against tests/fake_mirai_ws.py")
    p.add_argument("--standin-capacity", type=int, default=2, help="stand-in: sentences refused once")
    p.add_argument("--standin-speed", type=float, default=1.6, help="stand-in: delivery speed x real time")
    p.add_argument("--out", required=True)
    if len(sys.argv) > 1 and sys.argv[1] in ("_bot", "_standin"):
        role = sys.argv.pop(1)
        p.add_argument("--index", type=int, default=0)
        p.add_argument("--port", type=int, required=True)
        args = p.parse_args()
        bot_main(args) if role == "_bot" else standin_main(args)
    else:
        main(p.parse_args())
