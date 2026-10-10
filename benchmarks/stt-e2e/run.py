#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Streaming STT end-to-end test: N phone-quality calls against Mirai's streaming transcription socket.

Each call is one WebSocket on ``/v1/audio/transcriptions/stream`` (8 kHz
PCM16 by default, manual endpointing, as the voice workers and a Pipecat
pipeline with a VAD use it), driven the way a carrier delivers audio:

  speech_start + 0.5 s of pre-roll, the utterance in 60 ms pieces at exactly
  real time, speech_end, then the next utterance after a pause.

and records, per call: connect -> session.begin, speech_end -> transcript.final
(the number the caller feels), partial transcripts (count, time to the first,
gaps between them), missing finals, errors and the event loop's lag in the
process that drove it. ``--endpointing vad`` streams everything (silence too)
and measures from Mirai's own ``vad.speech_end`` instead.

The audio is synthetic by default (speech-like voiced bursts and pauses, built
in: the transcripts are meaningless but every utterance gets a final, which is
what latency needs). ``--audio-dir DIR`` plays real audio instead: each
``name.wav`` (mono, 16-bit; resampled to ``--sample-rate`` if it differs) with
an optional ``name.json`` sidecar ``{"utterances": [[start_s, end_s], ...]}``;
without a sidecar, utterances are found by energy.

Calls run in ``--procs`` worker processes, ``--calls`` calls in each (so
``procs x calls`` sockets at once), started ``--stagger`` seconds apart.

    # a local stand-in (tests/fake_mirai_stt.py), no key:
    uv run python benchmarks/stt-e2e/run.py --standin --procs 2 --calls 5 --duration 20 --out /tmp/stt
    # Mirai's API, then the same through its edge:
    uv run python benchmarks/stt-e2e/run.py --key-file key.txt --no-edge --procs 4 --calls 5 \\
        --out results/stt-api
    uv run python benchmarks/stt-e2e/run.py --key-file key.txt --edge --procs 4 --calls 5 \\
        --out results/stt-edge

Prints a per-call table, a summary and PASS/FAIL flags; writes result.json and
callNN.json (every latency, raw) to --out; exits 1 if any flag fails.
"""

import argparse
import asyncio
import base64
import json
import math
import os
import random
import resource
import subprocess
import sys
import time
import wave
from collections import deque
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_URL = "wss://sandbox.voice.miraiminds.co/v1/audio/transcriptions/stream"
CHUNK_SECS = 0.06
PREROLL_SECS = 0.5
PING_SECS = 20.0


def pct(values, q):
    """Nearest-rank percentile: the smallest value with at least q% of the values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def stats(values):
    return {
        "n": len(values),
        "p50": pct(values, 50),
        "p95": pct(values, 95),
        "p99": pct(values, 99),
        "max": max(values) if values else None,
    }


def fmt(value, unit="", digits=1):
    return "-" if value is None else f"{value:.{digits}f}{unit}"


# ---------- audio ----------


def synth(duration: float, rate: int, rng: random.Random):
    """Speech-like audio for ``duration`` s and the utterances in it: [(start_s, end_s)]."""
    n = int(duration * rate)
    pcm = np.zeros(n, dtype=np.float64)
    pcm += np.array([rng.gauss(0, 1) for _ in range(256)] * (n // 256 + 1))[:n] * 15.0  # a quiet floor
    utterances = []
    t = rng.uniform(0.4, 1.0)
    while True:
        length = rng.uniform(1.0, 3.0)
        if t + length > duration - 0.3:
            break
        utterances.append((round(t, 3), round(t + length, 3)))
        a, b = int(t * rate), int((t + length) * rate)
        x = np.arange(b - a) / rate
        f0 = rng.uniform(110, 220)
        glide = 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * x)
        phase = 2 * np.pi * np.cumsum(f0 * glide) / rate
        voiced = sum(np.sin(h * phase) / h for h in range(1, 14))
        syllable = rng.uniform(3.5, 5.0)
        envelope = 0.55 + 0.45 * np.sin(2 * np.pi * syllable * x + rng.uniform(0, 6)) ** 2
        edge = np.minimum(1.0, np.minimum(x, x[-1] - x) / 0.04)
        pcm[a:b] += voiced * envelope * edge * 2200
        t += length + rng.uniform(0.7, 2.0)
    return np.clip(pcm, -32768, 32767).astype("<i2"), utterances


def find_utterances(pcm: np.ndarray, rate: int):
    """Utterances by energy: 30 ms frames above a threshold, merged across pauses under 0.35 s."""
    step = int(rate * 0.03)
    frames = len(pcm) // step
    rms = np.sqrt((pcm[: frames * step].astype(np.float64).reshape(frames, step) ** 2).mean(axis=1))
    loud = rms > max(150.0, float(np.percentile(rms, 90)) * 0.1)
    out, start, quiet = [], None, 0
    for i, is_loud in enumerate(loud):
        if is_loud:
            start = i if start is None else start
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet * 0.03 >= 0.35:
                out.append((start * 0.03, (i - quiet + 1) * 0.03 + 0.1))
                start, quiet = None, 0
    if start is not None:
        out.append((start * 0.03, frames * 0.03))
    return [(round(s, 3), round(e, 3)) for s, e in out if e - s >= 0.3]


def load_wav(path: Path, rate: int):
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise SystemExit(f"{path}: need 16-bit PCM")
        channels, src_rate = w.getnchannels(), w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0].copy()
    if src_rate != rate:
        import soxr

        pcm = soxr.resample(pcm, src_rate, rate, quality="HQ").astype("<i2")
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        utterances = [tuple(u) for u in json.loads(sidecar.read_text())["utterances"]]
    else:
        utterances = find_utterances(pcm, rate)
    return pcm, utterances


def call_audio(index: int, cfg):
    """The audio and utterances of call ``index``."""
    rate = cfg["sample_rate"]
    if cfg["audio_dir"]:
        files = sorted(Path(cfg["audio_dir"]).glob("*.wav"))
        if not files:
            raise SystemExit(f"no .wav files in {cfg['audio_dir']}")
        pcm, utterances = load_wav(files[index % len(files)], rate)
    else:
        pcm, utterances = synth(cfg["duration"], rate, random.Random(cfg["seed"] * 1000 + index))
    limit = cfg["duration"] if cfg["duration"] > 0 else len(pcm) / rate
    pcm = pcm[: int(limit * rate)]
    return pcm, [(s, e) for s, e in utterances if e <= len(pcm) / rate]


# ---------- one call ----------


async def run_call(index: int, cfg, url: str, headers: dict, started_at: float):
    import websockets

    rate = cfg["sample_rate"]
    manual = cfg["endpointing"] == "manual"
    pcm, utterances = call_audio(index, cfg)
    raw = pcm.tobytes()
    query = {
        "language_code": cfg["language"],
        "model": cfg["model"],
        "encoding": "linear16",
        "sample_rate": rate,
        "endpointing": cfg["endpointing"],
    }
    result = {
        "call": index,
        "url": url,
        "utterances_sent": 0,
        "finals": 0,
        "empty_finals": 0,
        "errors": [],
        "connect_ms": None,
        "begin_ms": None,
        "final_ms": [],
        "first_partial_ms": [],
        "partial_gap_ms": [],
        "partials": 0,
        "audio_s": round(len(pcm) / 2 / rate, 3),
        "utterance_s": round(sum(e - s for s, e in utterances), 3),
        "session": None,
        "billed": None,
    }
    await asyncio.sleep(max(0.0, started_at - time.monotonic()))
    t0 = time.monotonic()
    try:
        ws = await websockets.connect(
            f"{url}?{urlencode(query)}",
            additional_headers=headers,
            max_size=None,
            open_timeout=cfg["connect_timeout"],
            compression=None,
        )
    except Exception as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        result["errors"].append(
            f"connect: HTTP {status}" if status else f"connect: {type(exc).__name__}: {exc}"
        )
        return result
    result["connect_ms"] = (time.monotonic() - t0) * 1000
    try:
        first = json.loads(await asyncio.wait_for(ws.recv(), cfg["connect_timeout"]))
    except Exception as exc:
        result["errors"].append(f"session.begin: {type(exc).__name__}: {exc}")
        await ws.close()
        return result
    result["begin_ms"] = (time.monotonic() - t0) * 1000
    if first.get("event") != "session.begin":
        result["errors"].append(f"first event was {first.get('event')!r}: {first}")
        await ws.close()
        return result
    result["session"] = first.get("session_id")

    ends: deque[float] = deque()  # reference times (monotonic) of utterances awaiting a final
    starts: deque[float] = deque()  # speech_start sent, for the first partial
    last_partial: dict[int, float] = {}
    session_end = asyncio.Event()

    async def reader():
        try:
            async for message in ws:
                now = time.monotonic()
                event = json.loads(message)
                kind = event.get("event")
                if kind == "transcript.partial":
                    result["partials"] += 1
                    idx = event.get("utterance_idx", -1)
                    if idx in last_partial:
                        result["partial_gap_ms"].append((now - last_partial[idx]) * 1000)
                    elif starts:
                        result["first_partial_ms"].append((now - starts[0]) * 1000)
                    last_partial[idx] = now
                elif kind == "transcript.final":
                    result["finals"] += 1
                    if not event.get("text"):
                        result["empty_finals"] += 1
                    if ends:
                        result["final_ms"].append((now - ends.popleft()) * 1000)
                    if starts:
                        starts.popleft()
                elif kind == "vad.speech_start":
                    starts.append(now)
                elif kind == "vad.speech_end":
                    ends.append(now)
                elif kind == "session.end":
                    result["billed"] = {
                        k: event.get(k) for k in ("audio_seconds_billed", "cost_paise", "total_utterances")
                    }
                    session_end.set()
                elif kind == "error":
                    result["errors"].append(f"{event.get('code')}: {event.get('message')}")
                    if event.get("is_fatal"):
                        session_end.set()
        except Exception as exc:
            if not session_end.is_set():
                result["errors"].append(f"socket: {type(exc).__name__}: {exc}")
        finally:
            session_end.set()

    read_task = asyncio.create_task(reader())
    step = int(rate * CHUNK_SECS) * 2
    base = time.monotonic()
    last_sent = base

    async def send(event: dict):
        nonlocal last_sent
        await ws.send(json.dumps(event))
        last_sent = time.monotonic()

    async def audio(chunk: bytes):
        await send({"event": "audio_input", "audio": base64.b64encode(chunk).decode()})

    async def until(t: float):
        """Wait for ``t`` seconds into the call, pinging if it takes long."""
        while True:
            left = base + t - time.monotonic()
            if left <= 0:
                return
            nap = min(left, max(0.01, PING_SECS - (time.monotonic() - last_sent)))
            await asyncio.sleep(nap)
            if time.monotonic() - last_sent >= PING_SECS:
                await send({"event": "ping"})

    try:
        if manual:
            for start, end in utterances:
                if session_end.is_set():
                    break
                await until(start)
                pre = max(0, int((start - PREROLL_SECS) * rate) * 2)
                a = int(start * rate) * 2
                await send({"event": "speech_start"})
                starts.append(time.monotonic())
                if a > pre:
                    await audio(raw[pre:a])
                pos = a
                b = int(end * rate) * 2
                while pos < b:
                    nxt = min(pos + step, b)
                    await until(nxt / 2 / rate)
                    await audio(raw[pos:nxt])
                    pos = nxt
                await send({"event": "speech_end"})
                ends.append(time.monotonic())
                result["utterances_sent"] += 1
        else:
            pos = 0
            while pos < len(raw) and not session_end.is_set():
                nxt = min(pos + step, len(raw))
                await until(nxt / 2 / rate)
                await audio(raw[pos:nxt])
                pos = nxt
            # Mirai's VAD needs the utterance to end in silence; give the last one time.
            await asyncio.sleep(1.0)
            result["utterances_sent"] = len(utterances)
        if not session_end.is_set():
            await asyncio.sleep(cfg["settle"])
            await send({"event": "end"})
            try:
                await asyncio.wait_for(session_end.wait(), cfg["final_wait"])
            except TimeoutError:
                result["errors"].append(f"no session.end within {cfg['final_wait']:g} s")
    except Exception as exc:
        result["errors"].append(f"send: {type(exc).__name__}: {exc}")
    finally:
        read_task.cancel()
        try:
            await ws.close()
        except Exception:
            pass
    if not manual:
        result["utterances_sent"] = result["finals"] + len(ends)  # VAD decides how many there were
    result["missing_finals"] = max(0, result["utterances_sent"] - result["finals"])
    return result


# ---------- one worker process ----------


async def loop_lag(samples: list, stop: asyncio.Event, interval=0.02):
    while not stop.is_set():
        t = time.monotonic()
        await asyncio.sleep(interval)
        samples.append((time.monotonic() - t - interval) * 1000)


async def resolve_endpoint(cfg):
    """(url, headers, label): the gateway, or the edge it names (or ``--edge-url``)."""
    headers = {"Authorization": f"Bearer {cfg['key']}"}
    url = cfg["url"]
    if not cfg["edge"]:
        return url, headers, "api"
    if cfg["edge_url"]:
        return cfg["edge_url"], headers, "edge"
    import httpx

    base = url.replace("wss://", "https://").replace("ws://", "http://").split("/v1/", 1)[0]
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(f"{base}/v2/stt/stream/tokens", headers=headers, json={})
    if response.status_code >= 300:
        raise SystemExit(
            f"--edge: the token route answered HTTP {response.status_code}: {response.text[:200]}"
        )
    edge_url = response.json().get("edge_url")
    if not edge_url:
        raise SystemExit(
            "--edge: the API offers no STT edge (edge_url is empty); use --no-edge or --edge-url"
        )
    return edge_url, headers, "edge"


async def worker(cfg, proc: int):
    url, headers, label = await resolve_endpoint(cfg)
    out = Path(cfg["out"])
    lag: list[float] = []
    stop = asyncio.Event()
    lag_task = asyncio.create_task(loop_lag(lag, stop))
    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    t_base = time.monotonic() + 0.5
    tasks = []
    for i in range(cfg["calls"]):
        index = proc * cfg["calls"] + i
        tasks.append(asyncio.create_task(run_call(index, cfg, url, headers, t_base + i * cfg["stagger"])))
    results = await asyncio.gather(*tasks)
    stop.set()
    await lag_task
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    for r in results:
        r["endpoint"] = label
        (out / f"call{r['call']:02d}.json").write_text(json.dumps(r, indent=1))
    (out / f"proc{proc}.json").write_text(
        json.dumps(
            {
                "proc": proc,
                "endpoint": label,
                "url": url,
                "loop_lag_ms": stats(lag),
                "cpu_s": (cpu1.ru_utime + cpu1.ru_stime) - (cpu0.ru_utime + cpu0.ru_stime),
            }
        )
    )


# ---------- the parent ----------


def serve_standin():
    """Run tests/fake_mirai_stt.py on a local port in a thread; returns (url, stop)."""
    import threading

    sys.path.insert(0, str(REPO / "tests"))
    from fake_mirai_stt import FakeMiraiSTT

    ready = threading.Event()
    box: dict = {}

    def run():
        async def main():
            fake = FakeMiraiSTT(partial_every_s=0.25)
            async with fake.serve() as url:
                box["url"] = url
                box["stop"] = asyncio.Event()
                box["loop"] = asyncio.get_running_loop()
                ready.set()
                await box["stop"].wait()

        asyncio.run(main())

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    ready.wait(10)
    return box["url"], lambda: box["loop"].call_soon_threadsafe(box["stop"].set)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL, help="the streaming endpoint (default: Mirai's sandbox)")
    p.add_argument("--key-file", help="file holding the API key (else MIRAI_API_KEY)")
    p.add_argument("--procs", type=int, default=1, help="worker processes")
    p.add_argument("--calls", type=int, default=5, help="calls per process, all at once (see --stagger)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds of audio per call (0: whole file)")
    p.add_argument(
        "--edge",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="connect to the edge the API names",
    )
    p.add_argument("--edge-url", help="connect to this edge URL (implies --edge)")
    p.add_argument("--language", default="hi-IN")
    p.add_argument("--model", default="mira-stt")
    p.add_argument("--sample-rate", type=int, choices=(8000, 16000), default=8000)
    p.add_argument("--endpointing", choices=("manual", "vad"), default="manual")
    p.add_argument("--stagger", type=float, default=0.2, help="seconds between call starts in a process")
    p.add_argument(
        "--final-p95-ms", type=float, default=100.0, help="speech_end -> final p95 the run must stay under"
    )
    p.add_argument("--audio-dir", help="real audio: name.wav [+ name.json sidecar of utterances]")
    p.add_argument("--standin", action="store_true", help="run against tests/fake_mirai_stt.py, no key")
    p.add_argument("--connect-timeout", type=float, default=10.0)
    p.add_argument(
        "--final-wait", type=float, default=5.0, help="seconds to wait for session.end after the last turn"
    )
    p.add_argument("--settle", type=float, default=0.5, help="seconds after the last turn before `end`")
    p.add_argument("--seed", type=int, default=1, help="synthetic audio seed")
    p.add_argument("--out", default="results/stt-e2e", help="directory for result.json and callNN.json")
    p.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    p.add_argument("--config", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.edge_url:
        args.edge = True
    return args


def read_key(args) -> str:
    if args.key_file:
        return Path(args.key_file).read_text().strip()
    return os.environ.get("MIRAI_API_KEY") or os.environ.get("MIRA_API_KEY") or ""


def main():
    args = parse_args()
    if args.worker is not None:
        cfg = json.loads(Path(args.config).read_text())
        asyncio.run(worker(cfg, args.worker))
        return 0

    stop_standin = None
    url, key = args.url, read_key(args)
    if args.standin:
        url, stop_standin = serve_standin()
        key = key or "sk_standin"
        if args.edge:
            raise SystemExit("--standin has no edge")
    if not key:
        raise SystemExit("Need an API key: --key-file FILE or MIRAI_API_KEY (or --standin).")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("call*.json"):
        old.unlink()
    cfg = {
        "url": url,
        "key": key,
        "calls": args.calls,
        "duration": args.duration,
        "edge": args.edge,
        "edge_url": args.edge_url,
        "language": args.language,
        "model": args.model,
        "sample_rate": args.sample_rate,
        "endpointing": args.endpointing,
        "stagger": args.stagger,
        "audio_dir": args.audio_dir,
        "connect_timeout": args.connect_timeout,
        "final_wait": args.final_wait,
        "settle": args.settle,
        "seed": args.seed,
        "out": str(out),
    }
    config_path = out / "_config.json"
    config_path.write_text(json.dumps(cfg))
    config_path.chmod(0o600)  # it carries the key
    started = time.time()
    try:
        procs = [
            subprocess.Popen([sys.executable, __file__, "--worker", str(i), "--config", str(config_path)])
            for i in range(args.procs)
        ]
        codes = [p.wait() for p in procs]
    finally:
        config_path.unlink(missing_ok=True)
        if stop_standin:
            stop_standin()
    wall = time.time() - started

    calls = [json.loads(f.read_text()) for f in sorted(out.glob("call*.json"))]
    workers = [json.loads(f.read_text()) for f in sorted(out.glob("proc*.json"))]
    summary = summarise(args, url, calls, workers, codes, wall)
    (out / "result.json").write_text(json.dumps(summary, indent=1))
    report(summary, calls)
    return 0 if all(summary["flags"].values()) else 1


def summarise(args, url, calls, workers, codes, wall):
    finals = [v for c in calls for v in c["final_ms"]]
    connect = [c["connect_ms"] for c in calls if c["connect_ms"] is not None]
    begin = [c["begin_ms"] for c in calls if c["begin_ms"] is not None]
    errors = [f"call{c['call']:02d}: {e}" for c in calls for e in c["errors"]]
    errors += [f"worker {i} exited {code}" for i, code in enumerate(codes) if code]
    expected = args.procs * args.calls
    if len(calls) < expected:
        errors.append(f"only {len(calls)} of {expected} calls reported")
    missing = sum(c.get("missing_finals", 0) for c in calls)
    lag = [w["loop_lag_ms"]["p99"] for w in workers if w["loop_lag_ms"]["p99"] is not None]
    p95 = pct(finals, 95)
    try:
        import pipecat_mirai

        version = pipecat_mirai.__version__
    except ImportError:
        version = None
    return {
        "benchmark": "stt-e2e",
        "package_version": version,
        "url": url if not args.standin else "standin",
        "endpoint": workers[0]["endpoint"] if workers else None,
        "edge_url": workers[0]["url"] if workers and workers[0]["endpoint"] == "edge" else None,
        "procs": args.procs,
        "calls_per_proc": args.calls,
        "sockets": expected,
        "duration_s": args.duration,
        "endpointing": args.endpointing,
        "sample_rate": args.sample_rate,
        "language": args.language,
        "audio": args.audio_dir or "synthetic",
        "wall_s": round(wall, 1),
        "connect_to_begin_ms": stats(begin),
        "connect_ms": stats(connect),
        "speech_end_to_final_ms": stats(finals),
        "first_partial_ms": stats([v for c in calls for v in c["first_partial_ms"]]),
        "partial_gap_ms": stats([v for c in calls for v in c["partial_gap_ms"]]),
        "utterances": sum(c["utterances_sent"] for c in calls),
        "finals": sum(c["finals"] for c in calls),
        "empty_finals": sum(c["empty_finals"] for c in calls),
        "partials": sum(c["partials"] for c in calls),
        "missing_finals": missing,
        "errors": errors,
        "loop_lag_p99_ms_worst": max(lag) if lag else None,
        "cpu_s_per_proc": [round(w["cpu_s"], 2) for w in workers],
        "billed_s": sum((c["billed"] or {}).get("audio_seconds_billed") or 0 for c in calls),
        "flags": {
            "no_errors": not errors,
            "no_missing_finals": missing == 0 and bool(finals),
            "final_p95_under_ms": p95 is not None and p95 < args.final_p95_ms,
        },
        "final_p95_limit_ms": args.final_p95_ms,
    }


def report(s, calls):
    print(
        f"\nSTT e2e: {s['sockets']} sockets ({s['procs']} procs x {s['calls_per_proc']}), "
        f"{s['endpoint']} {s['edge_url'] or s['url']}, {s['endpointing']} endpointing, "
        f"{s['sample_rate']} Hz, {s['audio']} audio, {s['wall_s']} s"
    )
    print(
        f"{'call':>4} {'conn':>7} {'begin':>7} {'utts':>5} {'finals':>6} {'miss':>5} "
        f"{'p50':>7} {'p95':>7} {'max':>7} {'partials':>8}  errors"
    )
    for c in calls:
        f = c["final_ms"]
        print(
            f"{c['call']:>4} {fmt(c['connect_ms']):>7} {fmt(c['begin_ms']):>7} {c['utterances_sent']:>5} "
            f"{c['finals']:>6} {c.get('missing_finals', 0):>5} {fmt(pct(f, 50)):>7} {fmt(pct(f, 95)):>7} "
            f"{fmt(max(f) if f else None):>7} {c['partials']:>8}  {'; '.join(c['errors'])[:60]}"
        )
    for name, key in (
        ("connect -> session.begin", "connect_to_begin_ms"),
        ("speech_end -> final", "speech_end_to_final_ms"),
        ("speech_start -> first partial", "first_partial_ms"),
        ("gap between partials", "partial_gap_ms"),
    ):
        v = s[key]
        print(
            f"{name:<30} n={v['n']:<5} p50 {fmt(v['p50'], ' ms')}  p95 {fmt(v['p95'], ' ms')}  "
            f"p99 {fmt(v['p99'], ' ms')}  max {fmt(v['max'], ' ms')}"
        )
    print(
        f"utterances {s['utterances']}, finals {s['finals']} ({s['empty_finals']} empty), "
        f"missing {s['missing_finals']}, partials {s['partials']}, errors {len(s['errors'])}, "
        f"worst loop-lag p99 {fmt(s['loop_lag_p99_ms_worst'], ' ms')}, audio billed {s['billed_s']} s"
    )
    for e in s["errors"][:10]:
        print(f"  error: {e}")
    for flag, ok in s["flags"].items():
        extra = f" (< {s['final_p95_limit_ms']:g} ms)" if flag == "final_p95_under_ms" else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {flag}{extra}")


if __name__ == "__main__":
    sys.exit(main())
