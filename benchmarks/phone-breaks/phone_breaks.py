#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Measure audible breaks on phone calls: a Pipecat bot vs a simulated phone provider.

Three processes, so that a stall in the bot cannot stall the "phone":

  standin   Optional stand-in for the Mirai API. Streams a 48 kHz WAV with Mirai's
            measured delivery pattern (first byte ~150 ms carrying 240 ms of audio,
            ~1.5 s of audio within 0.5 s, then 160 ms pieces at ~1.1x real time).
            Use it to run without an API key; point --tts-url at the real API
            otherwise.
  bot       A Pipecat backend: FastAPI websocket endpoint speaking the Twilio
            Media Streams protocol, Pipeline([transport.input(), MiraiTTSService,
            transport.output()]) at 8 kHz. Each call speaks the script's lines at
            fixed offsets. --stall-every/--stall-ms block the event loop the way
            synchronous work in a real bot does. --lead applies apply_output_lead.
  phone     The provider: opens N calls, plays received audio in real time from a
            60 ms jitter buffer and counts every moment it had nothing to play.

    python phone_breaks.py standin --wav speech48k.wav &
    python phone_breaks.py bot --lead 0 &          # stock Pipecat pacing
    python phone_breaks.py phone --calls 3 --out results/stock_c3
"""

import argparse
import asyncio
import audioop
import base64
import json
import os
import random
import time
import wave

import numpy as np

SR = 8000
SCRIPT = [
    (0.0, "नमस्ते, मैं Mirai से बोल रही हूँ। क्या मेरी बात Amit जी से हो रही है?"),
    (10.8, "आपका order कल शाम तक deliver हो जाएगा — क्या उस समय घर पर कोई उपलब्ध रहेगा?"),
    (26.0, "हमारी team आपसे payment के बारे में बात करना चाहती थी — क्या अभी दो मिनट बात हो सकती है?"),
    (40.3, "आपकी appointment शनिवार सुबह ग्यारह बजे के लिए confirm है।"),
    (50.9, "क्या आप अपने area का pincode बता सकते हैं?"),
    (59.4, "ठीक है, मैंने सारी details note कर ली हैं — हमारी team आपसे जल्द ही संपर्क करेगी।"),
    (72.5, "आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!"),
]


# --------------------------------------------------------------------------- standin
def standin(a):
    from aiohttp import web

    with wave.open(a.wav) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1 and w.getsampwidth() == 2, (
            "need 48 kHz mono s16"
        )
        pcm = w.readframes(w.getnframes())
    bps = 96000

    async def speech(req):
        body = await req.json()
        seconds = min(len(pcm) / bps, max(1.0, len(body.get("input", "")) / 14))
        audio = pcm[: int(seconds * bps) // 2 * 2]
        await asyncio.sleep(random.gauss(0.150, 0.015))
        resp = web.StreamResponse(headers={"Content-Type": "audio/pcm", "X-Sample-Rate": "48000"})
        await resp.prepare(req)
        chunk, pos, t0 = int(0.160 * bps), int(0.240 * bps), time.monotonic()
        await resp.write(audio[:pos])
        while pos < len(audio) and pos < 1.5 * bps and time.monotonic() - t0 < 0.5:
            await asyncio.sleep(0.040)
            await resp.write(audio[pos : pos + chunk])
            pos += chunk
        nxt = time.monotonic()
        while pos < len(audio):
            nxt += 0.160 / 1.10
            await asyncio.sleep(max(0, nxt - time.monotonic()))
            await resp.write(audio[pos : pos + chunk])
            pos += chunk
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/audio/speech", speech)
    web.run_app(app, host="127.0.0.1", port=a.port, print=None)


# --------------------------------------------------------------------------- bot
def bot(a):
    import uvicorn
    from fastapi import FastAPI, WebSocket
    from pipecat.frames.frames import EndFrame, TTSSpeakFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.serializers.twilio import TwilioFrameSerializer
    from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
    from pipecat.workers.runner import WorkerRunner

    from pipecat_mirai import MiraiTTSService, apply_output_lead

    lo, hi = (float(x) for x in a.stall_ms.split(","))
    script_lines = SCRIPT
    if a.texts:
        lines = [line.strip() for line in open(a.texts, encoding="utf-8") if line.strip()]
        script_lines = [(offset, text) for (offset, _), text in zip(SCRIPT, lines, strict=False)]
    app = FastAPI()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.receive_text()
        start = json.loads(await websocket.receive_text())["start"]
        rng = random.Random(int(start["customParameters"]["call"]))
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
        tts = MiraiTTSService(
            api_key=os.getenv("MIRAI_API_KEY", "standin"), base_url=a.tts_url, voice=a.voice
        )
        worker = PipelineWorker(
            Pipeline([transport.input(), tts, transport.output()]),
            params=PipelineParams(audio_out_sample_rate=SR),
        )
        stop = asyncio.Event()

        async def busy():  # per-call CPU work, plus occasional event-loop stalls
            while not stop.is_set():
                ms = a.work_ms
                if a.stall_every and rng.random() < 0.020 / a.stall_every:
                    ms += rng.uniform(lo, hi)
                end = time.perf_counter() + ms / 1000
                while time.perf_counter() < end:
                    pass
                await asyncio.sleep(0.020)

        async def script():
            t0 = time.monotonic()
            for offset, text in script_lines:
                await asyncio.sleep(max(0, t0 + offset - time.monotonic()))
                await worker.queue_frame(TTSSpeakFrame(text))
            await asyncio.sleep(max(0, t0 + SCRIPT[-1][0] + 9 - time.monotonic()))
            await worker.queue_frame(EndFrame())

        tasks = [asyncio.create_task(busy()), asyncio.create_task(script())]
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        try:
            await runner.run()
        finally:
            stop.set()
            for t in tasks:
                t.cancel()

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


# --------------------------------------------------------------------------- phone
def playout(events, jb=0.060, frame=0.020):
    """Real-time playout of received audio; returns caller-side audio and per-turn stats."""
    media = [(t, np.frombuffer(b, dtype=np.int16)) for t, kind, b in events if kind == "media"]
    clears = [t for t, kind, _ in events if kind == "clear"]
    if not media:
        return np.zeros(0, np.int16), []
    turns, cur = [], [media[0]]
    for m in media[1:]:
        if m[0] - cur[-1][0] > 1.0:
            turns.append(cur)
            cur = [m]
        else:
            cur.append(m)
    turns.append(cur)
    origin = media[0][0]
    out = np.zeros(int(SR * (media[-1][0] - origin + 30)), dtype=np.int16)
    nf = int(frame * SR)
    stats = []
    for arr in turns:
        q, qi, t = [], 0, arr[0][0] + jb
        cursor = int((t - origin) * SR)
        played = inserted = dropped = breaks = 0
        in_break = False
        while True:
            while qi < len(arr) and arr[qi][0] <= t:
                q.append(arr[qi][1])
                qi += 1
            for c in clears:
                if t - frame < c <= t:
                    dropped += sum(len(x) for x in q)
                    q = []
            need, got = nf, []
            while need and q:
                take = q[0][:need]
                got.append(take)
                need -= len(take)
                q[0] = q[0][len(take) :]
                if not len(q[0]):
                    q.pop(0)
            seg = np.concatenate(got) if got else np.zeros(0, np.int16)
            if cursor + nf <= len(out):
                out[cursor : cursor + len(seg)] = seg
            played += len(seg)
            if len(seg) < nf and (qi < len(arr) or q):
                inserted += nf - len(seg)
                breaks += 0 if in_break else 1
                in_break = True
            else:
                in_break = False
            cursor += nf
            t += frame
            if qi >= len(arr) and not q:
                break
        stats.append(
            dict(
                played_s=round(played / SR, 2),
                inserted_ms=round(inserted / SR * 1000),
                dropped_ms=round(dropped / SR * 1000),
                breaks=breaks,
            )
        )
    return out, stats


async def phone_call(a, n, results):
    import websockets

    await asyncio.sleep(n * 0.3)
    events = []
    async with websockets.connect(a.ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(
            json.dumps(
                {
                    "event": "start",
                    "start": {
                        "streamSid": f"MZ{n:030d}",
                        "callSid": f"CA{n:030d}",
                        "customParameters": {"call": str(n)},
                    },
                }
            )
        )
        end = time.monotonic() + SCRIPT[-1][0] + 12
        while time.monotonic() < end:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            except TimeoutError:
                continue
            except Exception:
                break
            now = time.monotonic()
            if msg.get("event") == "media":
                events.append((now, "media", audioop.ulaw2lin(base64.b64decode(msg["media"]["payload"]), 2)))
            elif msg.get("event") == "clear":
                events.append((now, "clear", None))
    audio, stats = playout(events)
    results[n] = stats
    if n == 0:
        with wave.open(f"{a.out}/caller_hears_call0.wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(audio.tobytes())


async def phone(a):
    os.makedirs(a.out, exist_ok=True)
    results = {}
    await asyncio.gather(*(phone_call(a, n, results) for n in range(a.calls)))
    rows = [s for v in results.values() for s in v]
    played = sum(r["played_s"] for r in rows)
    summary = dict(
        calls=a.calls,
        turns=len(rows),
        breaks_per_turn=round(sum(r["breaks"] for r in rows) / max(1, len(rows)), 2),
        silence_inserted_ms_per_turn=round(sum(r["inserted_ms"] for r in rows) / max(1, len(rows))),
        speech_stretched_pct=round(100 * sum(r["inserted_ms"] for r in rows) / 1000 / max(1e-9, played), 2),
    )
    json.dump(dict(summary=summary, calls=results), open(f"{a.out}/result.json", "w"), indent=1)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("standin")
    s.add_argument("--wav", required=True, help="48 kHz mono 16-bit speech, 10 s or longer")
    s.add_argument("--port", type=int, default=18080)
    b = sub.add_parser("bot")
    b.add_argument("--port", type=int, default=8765)
    b.add_argument("--tts-url", default="http://127.0.0.1:18080/v1")
    b.add_argument("--voice", default="shruti")
    b.add_argument("--texts", help="optional file with one line per bot turn (replaces the built-in script)")
    b.add_argument("--lead", type=float, default=0.0, help="apply_output_lead seconds; 0 = stock Pipecat")
    b.add_argument("--work-ms", type=float, default=1.0, help="steady CPU work per call every 20 ms")
    b.add_argument(
        "--stall-every", type=float, default=3.0, help="mean seconds between stalls per call; 0 = none"
    )
    b.add_argument("--stall-ms", default="50,250")
    f = sub.add_parser("phone")
    f.add_argument("--ws-url", default="ws://127.0.0.1:8765/ws")
    f.add_argument("--calls", type=int, default=3)
    f.add_argument("--out", required=True)
    a = p.parse_args()
    {"standin": standin, "bot": bot}.get(a.cmd, lambda a: asyncio.run(phone(a)))(a)
