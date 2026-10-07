#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Burst start: K pipelines start at the same instant, as in a customer's load test.

Each call is a Pipecat pipeline with MiraiTTSService (--tts http) or
MiraiWebsocketTTSService (--tts ws) at 8 kHz. It speaks a greeting the moment
it has started, then --sentences more, each after a pause. The calls run in
--procs fresh processes (K/procs each), and every process starts its calls at
one shared instant, after its own start-up:

    --variant baseline   start-up does nothing; services as constructed by 0.3.0
                         (on 0.3.1, shared_pool=False)
    --variant prewarm    start-up awaits pipecat_mirai.prewarm() for this
                         process's calls (HTTP connections or sockets)
    --variant default    0.3.1 defaults only: no prewarm() call, nothing configured.
                         Use --rounds 2 to see a worker's first (cold) burst and a
                         later burst after --round-gap seconds idle.

Per call it records, for the greeting, the time from the call's start to the
first audio byte (the handshake a WebSocket service pays at start is in
here) and Pipecat's TTFB metric; for later sentences, Pipecat's TTFB metric.

    unset VIRTUAL_ENV
    uv run python benchmarks/burst-start/burst.py --tts http --calls 12 --procs 1 \\
        --variant prewarm --key-file key.txt --out results/burst-http-prewarm

Keep production load small: at most --calls TTS requests are in flight at once.
"""

import argparse
import asyncio
import inspect
import json
import os
import random
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

GREETING = "नमस्ते, मैं Mirai से बोल रही हूँ। क्या अभी दो मिनट बात हो सकती है?"
LATER = [
    "जी बिल्कुल, आपकी appointment शनिवार सुबह ग्यारह बजे के लिए confirm है।",
    "आपका order कल शाम तक deliver हो जाएगा।",
    "क्या आप अपने area का pincode बता सकते हैं?",
    "आपका refund तीन से पाँच working days में आ जाएगा।",
    "ठीक है, मैंने सारी details note कर ली हैं।",
    "आपके समय के लिए धन्यवाद। आपका दिन शुभ हो!",
]


# ------------------------------------------------------------------------------------ worker
def worker_main(a):
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level=a.log_level)

    from pipecat.frames.frames import (
        EndFrame,
        ErrorFrame,
        MetricsFrame,
        StartFrame,
        TTSAudioRawFrame,
        TTSSpeakFrame,
        TTSStoppedFrame,
    )
    from pipecat.metrics.metrics import TTFBMetricsData
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
    from pipecat.workers.runner import WorkerRunner

    import pipecat_mirai
    from pipecat_mirai import MiraiTTSService, MiraiWebsocketTTSService

    key = os.environ["MIRAI_API_KEY"]
    ws_url = a.ws_url or a.base_url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")
    if not a.ws_url:
        ws_url += "/audio/speech/stream"
    cls = MiraiTTSService if a.tts == "http" else MiraiWebsocketTTSService
    extra = {}
    if a.variant == "baseline" and "shared_pool" in inspect.signature(cls.__init__).parameters:
        extra["shared_pool"] = False  # 0.3.0 behaviour on 0.3.1

    def make_tts():
        if a.tts == "http":
            return MiraiTTSService(api_key=key, base_url=a.base_url, voice=a.voice, **extra)
        return MiraiWebsocketTTSService(api_key=key, url=ws_url, voice=a.voice, **extra)

    class Tap(FrameProcessor):
        def __init__(self, call):
            super().__init__()
            self.call = call
            self.started = asyncio.Event()
            self.stopped = asyncio.Event()
            self.turn = None

        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            now = time.monotonic()
            if isinstance(frame, ErrorFrame) and direction == FrameDirection.UPSTREAM:
                self.call["errors"].append(str(frame.error)[:300])
            elif direction == FrameDirection.DOWNSTREAM:
                if isinstance(frame, StartFrame):
                    self.call["t_started"] = now
                    self.started.set()
                elif isinstance(frame, MetricsFrame) and self.turn is not None:
                    for d in frame.data:
                        if isinstance(d, TTFBMetricsData) and d.value > 0 and "ttfb" not in self.turn:
                            self.turn["ttfb"] = d.value
                            self.turn["t_first_byte"] = now
                elif isinstance(frame, TTSAudioRawFrame) and self.turn is not None:
                    self.turn.setdefault("t_first_audio", now)
                elif isinstance(frame, TTSStoppedFrame):
                    self.stopped.set()
            await self.push_frame(frame, direction)

    async def one_call(n, t0, rng):
        call = {"call": n, "errors": [], "turns": []}
        up, down = Tap(call), Tap(call)
        tts = make_tts()
        worker = PipelineWorker(
            Pipeline([up, tts, down]),
            params=PipelineParams(audio_out_sample_rate=8000, enable_metrics=True),
        )
        texts = [GREETING] + [LATER[(n + i) % len(LATER)] for i in range(a.sentences)]

        async def turns():
            await down.started.wait()
            for k, text in enumerate(texts):
                if k:
                    await asyncio.sleep(rng.uniform(a.pause_min, a.pause_max))  # the caller talks
                turn = {"k": k, "t_queued": time.monotonic()}
                down.turn = turn
                down.stopped.clear()
                await worker.queue_frame(TTSSpeakFrame(text))
                try:
                    await asyncio.wait_for(down.stopped.wait(), 20)
                except TimeoutError:
                    turn["timeout"] = True
                call["turns"].append(turn)
            await worker.queue_frame(EndFrame())

        task = asyncio.create_task(turns())
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        try:
            await asyncio.wait_for(runner.run(), 120)
        finally:
            task.cancel()
        # Relative to the shared start instant (t0), in ms.
        rel = {}
        for turn in call["turns"]:
            for f in ("t_queued", "t_first_byte", "t_first_audio"):
                if f in turn:
                    turn[f] = round((turn[f] - t0) * 1000, 1)
            if "ttfb" in turn:
                turn["ttfb"] = round(turn["ttfb"] * 1000, 1)
        if "t_started" in call:
            rel["started_ms"] = round((call.pop("t_started") - t0) * 1000, 1)
        call.update(rel)
        return call

    async def main():
        t_proc = time.monotonic()
        prewarmed = None
        if a.variant == "prewarm":
            t = time.monotonic()
            r = await pipecat_mirai.prewarm(
                api_key=key,
                base_url=a.base_url,
                connections=a.calls if a.tts == "http" else 0,
                websocket=a.calls if a.tts == "ws" else 0,
                websocket_url=ws_url,
            )
            prewarmed = {
                "http": r.http_connections,
                "ws": r.websockets,
                "errors": r.errors,
                "ms": round((time.monotonic() - t) * 1000),
            }
        print("ready " + json.dumps({"prewarm": prewarmed}), flush=True)
        line = await asyncio.get_running_loop().run_in_executor(None, sys.stdin.readline)
        t0 = float(line.split()[1])
        await asyncio.sleep(max(0.0, t0 - time.monotonic()))
        out = []
        for rnd in range(a.rounds):
            if rnd:
                await asyncio.sleep(a.round_gap)  # the worker sits idle between load-test rounds
                t0 = time.monotonic() + 0.2
                await asyncio.sleep(0.2)
            rng = random.Random(a.seed * 100 + a.index + rnd * 7919)
            calls = await asyncio.gather(
                *(one_call(a.index * 1000 + rnd * 100 + i, t0, rng) for i in range(a.calls)),
                return_exceptions=True,
            )
            for c in calls:
                if isinstance(c, BaseException):
                    c = {"errors": [repr(c)], "turns": []}
                c["round"] = rnd
                out.append(c)
        stats = None
        if hasattr(pipecat_mirai, "shared_connection_stats"):
            stats = pipecat_mirai.shared_connection_stats()
        print(
            "result "
            + json.dumps(
                {
                    "index": a.index,
                    "version": pipecat_mirai.__version__,
                    "prewarm": prewarmed,
                    "startup_ms": round((t0 - t_proc) * 1000),
                    "pool": stats,
                    "calls": out,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if hasattr(pipecat_mirai, "close_shared_connections"):
            await pipecat_mirai.close_shared_connections()

    asyncio.run(main())


# ------------------------------------------------------------------------------------ parent
def tcp_rtt(url, n=5):
    """TCP connect time to the API host, for context."""
    u = urlparse(url)
    port = u.port or (443 if u.scheme in ("https", "wss") else 80)
    out = []
    for _ in range(n):
        t = time.monotonic()
        try:
            with socket.create_connection((u.hostname, port), timeout=5):
                out.append((time.monotonic() - t) * 1000)
        except OSError:
            pass
        time.sleep(0.1)
    return round(statistics.median(out), 1) if out else None


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    # nearest-rank percentile: honest with a dozen samples
    rank = max(1, -(-q * len(values) // 100))
    return round(values[int(rank) - 1])


def summarize(values):
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "max": pct(values, 100)}


def main(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    key = Path(a.key_file).read_text().strip() if a.key_file else os.environ.get("MIRAI_API_KEY", "")
    if not key:
        raise SystemExit("pass --key-file or set MIRAI_API_KEY")
    env = dict(os.environ, MIRAI_API_KEY=key)  # by environment, never argv
    if a.calls % a.procs:
        raise SystemExit("--calls must be a multiple of --procs")
    per = a.calls // a.procs
    rtt = tcp_rtt(a.base_url)
    me = [sys.executable, str(Path(__file__).resolve()), "_worker"]
    procs = []
    for i in range(a.procs):
        args = [
            "--index", str(i), "--calls", str(per), "--tts", a.tts, "--variant", a.variant,
            "--base-url", a.base_url, "--voice", a.voice, "--sentences", str(a.sentences),
            "--pause-min", str(a.pause_min), "--pause-max", str(a.pause_max), "--seed", str(a.seed),
            "--log-level", a.log_level, "--out", a.out,
            "--rounds", str(a.rounds), "--round-gap", str(a.round_gap),
        ]  # fmt: skip
        if a.ws_url:
            args += ["--ws-url", a.ws_url]
        log = open(out / f"worker{i}.log", "w")
        procs.append(
            subprocess.Popen(
                me + args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env
            )
        )
    ready = []
    for p in procs:
        line = p.stdout.readline()
        if not line.startswith("ready"):
            raise SystemExit(f"a worker failed to start: {line!r}; see {out}/worker*.log")
        ready.append(json.loads(line.split(" ", 1)[1]))
    t0 = time.monotonic() + 0.5  # every process starts its calls at this instant
    for p in procs:
        p.stdin.write(f"go {t0}\n")
        p.stdin.flush()
    results = []
    for p in procs:
        for line in p.stdout:
            if line.startswith("result "):
                results.append(json.loads(line[len("result ") :]))
        p.wait(timeout=180 + a.rounds * (120 + a.round_gap))

    calls = [c for r in results for c in r["calls"]]
    per_round = {}
    for rnd in range(a.rounds):
        g, lt = [], []
        for c in calls:
            if c.get("round", 0) != rnd:
                continue
            for t in c.get("turns", []):
                if t["k"] == 0 and "t_first_byte" in t:
                    g.append(t["t_first_byte"])
                elif t["k"] and "ttfb" in t:
                    lt.append(t["ttfb"])
        per_round[rnd] = {"greeting_from_call_start_ms": summarize(g), "later_ttfb_ms": summarize(lt)}
    greet_start, greet_ttfb, later_ttfb, started, errors, timeouts = [], [], [], [], [], 0
    for c in calls:
        errors += c.get("errors", [])
        if "started_ms" in c:
            started.append(c["started_ms"])
        for t in c.get("turns", []):
            timeouts += bool(t.get("timeout"))
            if t["k"] == 0:
                if "t_first_byte" in t:
                    greet_start.append(t["t_first_byte"])
                if "ttfb" in t:
                    greet_ttfb.append(t["ttfb"])
            elif "ttfb" in t:
                later_ttfb.append(t["ttfb"])
    summary = {
        "tts": a.tts,
        "variant": a.variant,
        "calls": a.calls,
        "procs": a.procs,
        "version": sorted({r["version"] for r in results}),
        "tcp_connect_ms": rtt,
        "prewarm": [r["prewarm"] for r in results],
        "pipeline_started_ms": summarize(started),
        "greeting_from_call_start_ms": summarize(greet_start),
        "greeting_ttfb_ms": summarize(greet_ttfb),
        "later_ttfb_ms": summarize(later_ttfb),
        "per_round": per_round,
        "errors": len(errors),
        "error_samples": errors[:5],
        "timeouts": timeouts,
        "pool_at_end": [r["pool"] for r in results],
    }
    (out / "result.json").write_text(json.dumps({"summary": summary, "workers": results}, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tts", choices=["http", "ws"], default="http")
    p.add_argument("--variant", choices=["baseline", "prewarm", "default"], default="prewarm")
    p.add_argument("--rounds", type=int, default=1, help="bursts per worker process")
    p.add_argument("--round-gap", type=float, default=20.0, help="idle seconds between bursts")
    p.add_argument("--calls", type=int, default=12, help="pipelines in all (K)")
    p.add_argument("--procs", type=int, default=1, help="processes the calls are split across")
    p.add_argument("--base-url", default="https://sandbox.voice.miraiminds.co/v1")
    p.add_argument("--ws-url", default=None, help="default: --base-url as wss://.../audio/speech/stream")
    p.add_argument("--key-file", help="file holding the API key (default: $MIRAI_API_KEY)")
    p.add_argument("--voice", default="shruti")
    p.add_argument("--sentences", type=int, default=4, help="sentences after the greeting")
    p.add_argument("--pause-min", type=float, default=1.5)
    p.add_argument("--pause-max", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--log-level", default="WARNING")
    p.add_argument("--out", required=True)
    if len(sys.argv) > 1 and sys.argv[1] == "_worker":
        sys.argv.pop(1)
        p.add_argument("--index", type=int, default=0)
        worker_main(p.parse_args())
    else:
        main(p.parse_args())
