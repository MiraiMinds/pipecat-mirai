#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Soak: hundreds of calls, one after another, in one process, against local stand-ins.

What a worker that runs for days does to the shared pools, compressed: every
call is a pipeline with the service built with defaults, some calls are
interrupted, and some gaps between calls are longer than the pools' idle
timers (shortened here: refresh after 0.5 s, warm for 1 s, sockets recycled
after 3 s), so connections are refreshed, expire, get replaced and recycled
again and again. Every --every calls it records open file descriptors, RSS,
asyncio tasks and what the pools hold, and at the end checks they stayed
flat. It runs itself in a child with ``python -X dev -W default`` and fails on
any "Task was destroyed", "never awaited" or ResourceWarning in its output.

    unset VIRTUAL_ENV
    uv run python benchmarks/soak/soak.py --calls 200 --tts both
"""

import argparse
import asyncio
import gc
import json
import os
import random
import re
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEXT = "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?"


def open_fds() -> int:
    for path in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(path))
        except OSError:
            continue
    return -1


def rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    try:
        return int(out.stdout.strip()) / 1024
    except ValueError:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def child(a):
    sys.path.insert(0, str(REPO / "tests"))
    from fake_mirai_ws import FakeMiraiWS
    from loguru import logger
    from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
    from pipecat.pipeline.worker import PipelineParams
    from pipecat.tests.utils import SleepFrame, run_test
    from test_tts import FakeMirai

    import pipecat_mirai
    from pipecat_mirai import pool

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    # Days of timers, in seconds.
    pool.HTTP_REFRESH_AFTER = 0.5
    pool.HTTP_WARM_WINDOW = 1.0
    pool.WS_KEEPALIVE_SECS = 0.5
    pool.PooledWebsocket.max_age_secs = property(lambda self: 3.0)
    pool.PEAK_WINDOW_SECS = 5.0
    pool.WARM_RATE = 20.0
    os.environ["MIRAI_WARM_CONNECTIONS"] = str(a.warm)
    os.environ["MIRAI_WARM_WEBSOCKETS"] = str(a.warm)

    async def main():
        rng = random.Random(1)
        http = FakeMirai(seconds=0.3, reads=(1600,), pace=0.01)
        ws = FakeMiraiWS(seconds=0.3, reads=(1600,), pace=0.01)
        rows = []
        served_http = served_ws = 0
        async with http.serve() as http_url, ws.serve() as ws_url:
            kinds = ["http", "ws"] if a.tts == "both" else [a.tts]
            for n in range(a.calls):
                kind = kinds[n % len(kinds)]
                if kind == "http":
                    tts = pipecat_mirai.MiraiHttpTTSService(api_key="k", base_url=http_url, voice="shruti")
                else:
                    tts = pipecat_mirai.MiraiWebsocketTTSService(api_key="k", url=ws_url, voice="shruti")
                frames = [TTSSpeakFrame(TEXT), SleepFrame(0.1), TTSSpeakFrame("हाँ जी।")]
                if n % 5 == 4:
                    frames += [SleepFrame(0.05), InterruptionFrame(), TTSSpeakFrame("ठीक है।")]
                await run_test(
                    tts,
                    frames_to_send=frames,
                    pipeline_params=PipelineParams(audio_out_sample_rate=8000),
                    start_timeout=10.0,
                )
                # Mostly back to back; now and then a pause longer than the idle timers.
                await asyncio.sleep(rng.choice([0.0, 0.0, 0.05, 0.2, 1.5]))
                # The stand-ins keep a record of everything; only this process is measured.
                served_http += len([r for r in http.requests if r["method"] == "POST"])
                http.requests.clear()
                ws.spoken.clear()
                closed = [c for c in ws.conns if c.ws.state.name == "CLOSED"]
                served_ws += len(closed)
                ws.conns = [c for c in ws.conns if c not in closed]
                if (n + 1) % a.every == 0 or n == 0:
                    gc.collect()
                    stats = pipecat_mirai.shared_connection_stats()
                    rows.append(
                        {
                            "calls": n + 1,
                            "fds": open_fds(),
                            "rss_mb": round(rss_mb(), 1),
                            "tasks": len(asyncio.all_tasks()),
                            "http_slots": sum(len(c.transport.slots) for c in pool._http.values()),
                            "http": stats["http"],
                            "ws": stats["websocket"],
                            "server_http_conns": http.connections,
                            "server_ws_conns": served_ws + len(ws.conns),
                            "http_requests": served_http,
                        }
                    )
                    r = rows[-1]
                    print(
                        f"calls {r['calls']:4d}  fds {r['fds']:4d}  rss {r['rss_mb']:6.1f} MB  "
                        f"tasks {r['tasks']:3d}  http slots {r['http_slots']:3d}  "
                        f"server conns http {r['server_http_conns']} ws {r['server_ws_conns']}",
                        flush=True,
                    )
            await pipecat_mirai.close_shared_connections()
        print("ROWS " + json.dumps(rows), flush=True)

    asyncio.run(main())


def main(a):
    cmd = [sys.executable, "-X", "dev", "-W", "default", __file__, "_child", *sys.argv[1:]]
    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out, err = proc.stdout, proc.stderr
    for line in out.splitlines():
        if not line.startswith("ROWS "):
            print(line)
    rows = json.loads(next(line for line in out.splitlines() if line.startswith("ROWS "))[5:])
    bad = [
        line
        for line in err.splitlines()
        if re.search(r"Task was destroyed|never awaited|ResourceWarning|Unclosed|unclosed", line)
    ]
    warm = rows[1] if len(rows) > 1 else rows[0]  # after the first stretch: pools at size
    last = rows[-1]
    checks = {
        "exit 0": proc.returncode == 0,
        "no task/resource warnings": not bad,
        f"fds flat (±8 from call {warm['calls']})": abs(last["fds"] - warm["fds"]) <= 8,
        "tasks flat (±4)": abs(last["tasks"] - warm["tasks"]) <= 4,
        "rss growth < 30 MB": last["rss_mb"] - warm["rss_mb"] < 30,
        "http slots bounded": last["http_slots"] <= 3 * max(a.warm, 1) + 8,
    }
    print(f"\n{a.calls} calls in {time.monotonic() - t0:.0f} s")
    for name, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    for line in bad[:10]:
        print("  stderr:", line)
    if proc.returncode:
        print(err[-3000:])
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calls", type=int, default=200)
    p.add_argument("--tts", choices=["http", "ws", "both"], default="both")
    p.add_argument("--warm", type=int, default=4, help="pool size (MIRAI_WARM_CONNECTIONS / _WEBSOCKETS)")
    p.add_argument("--every", type=int, default=25)
    if len(sys.argv) > 1 and sys.argv[1] == "_child":
        sys.argv.pop(1)
        child(p.parse_args())
    else:
        main(p.parse_args())
