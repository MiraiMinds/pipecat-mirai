"""Connections shared across pipelines and kept warm: the HTTP pool, the WebSocket pool, prewarm()."""

import asyncio
import json

import pytest
from fake_mirai_ws import FakeMiraiWS
from pipecat.frames.frames import InterruptionFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from test_tts import TEXT, FakeMirai, Recorder, audio_of, errors_in, speak, tone
from websockets.protocol import State

from pipecat_mirai import (
    MiraiHttpTTSService,
    MiraiWebsocketTTSService,
    close_shared_connections,
    prewarm,
    shared_connection_stats,
)
from pipecat_mirai import pool as pool_module
from pipecat_mirai.pool import shared_http_client, take_websocket

KEY = "sk_test"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture(autouse=True)
async def _no_shared_connections_left():
    yield
    await close_shared_connections()


def http_tts(url, **kwargs):
    return MiraiHttpTTSService(api_key=KEY, base_url=url, **kwargs)


def ws_tts(url, **kwargs):
    return MiraiWebsocketTTSService(api_key=KEY, url=url, **kwargs)


def gets(fake):
    return [r for r in fake.requests if r["method"] == "GET"]


def posts(fake):
    return [r for r in fake.requests if r["method"] == "POST"]


async def ws_call(tts, frames=None, rate=8000):
    return await run_test(
        tts,
        frames_to_send=frames or [TTSSpeakFrame(TEXT)],
        pipeline_params=PipelineParams(audio_out_sample_rate=rate),
        start_timeout=10.0,
    )


# --- HTTP: one client per base URL and event loop ------------------------------------------


async def test_services_in_one_loop_share_connections():
    fake = FakeMirai()
    async with fake.serve() as url:
        for _ in range(3):  # three calls, one after the other, each with its own service
            down, up = await speak(http_tts(url), 8000)
            assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    assert fake.connections == 1  # the second and third calls paid no handshake
    assert [r["conn"] for r in posts(fake)] == [1, 1, 1]


async def test_without_the_shared_pool_every_call_connects():
    fake = FakeMirai()
    async with fake.serve() as url:
        services = [http_tts(url, warm_connection=False, shared_pool=False) for _ in range(2)]
        for tts in services:
            await speak(tts, 8000)
        assert shared_connection_stats()["http"] == []
    assert fake.connections == 2  # the 0.3.0 behaviour: a client per service
    assert all(tts._http.is_closed for tts in services)  # each closed with its pipeline


async def test_a_caller_owned_client_is_used_and_left_open():
    fake = FakeMirai()
    client = fake.client()
    tts = MiraiHttpTTSService(api_key=KEY, http_client=client, warm_connection=False)
    await speak(tts, 8000)
    assert len(posts(fake)) == 1
    assert not client.is_closed
    assert shared_connection_stats()["http"] == []
    await client.aclose()


async def test_the_first_service_warms_the_pool_paced_and_later_calls_find_it_warm(monkeypatch):
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "4")
    fake = FakeMirai()
    async with fake.serve() as url:
        # One call starts; the pool opens connections in the background, two at
        # once and then two a second (WARM_RATE), so a burst of greetings
        # always has most of the API key's request budget.
        await speak(http_tts(url), 8000, before=[SleepFrame(1.3)])
        warmups = gets(fake)
        assert len(warmups) == 4 and {r["conn"] for r in warmups} == {1, 2, 3, 4}
        assert warmups[1]["t"] - warmups[0]["t"] < 0.1
        assert warmups[2]["t"] - warmups[0]["t"] > 0.35
        (stats,) = shared_connection_stats()["http"]
        assert stats["target"] == 4 and stats["warm"] == 4 and stats["services"] == 0
        # Four calls start together: four warm connections, no handshake.
        connections = fake.connections
        results = await asyncio.gather(*(speak(http_tts(url), 8000) for _ in range(4)))
        assert all(not errors_in(up) for _, up in results)
        assert fake.connections == connections
        assert sorted(r["conn"] for r in posts(fake)[1:]) == [1, 2, 3, 4]


async def test_a_burst_bigger_than_the_pool_grows_it_for_next_time(monkeypatch):
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "2")
    monkeypatch.setattr(pool_module, "WARM_RATE", 50.0)
    fake = FakeMirai()
    async with fake.serve() as url:
        await asyncio.gather(*(speak(http_tts(url), 8000) for _ in range(5)))
        await asyncio.sleep(0.5)
        (stats,) = shared_connection_stats()["http"]
        assert stats["target"] == 5 + pool_module.AUTO_HEADROOM  # the peak, and headroom
        assert stats["warm"] == stats["target"]


async def test_prewarm_opens_connections_at_once_and_pipelines_that_start_together_use_them():
    fake = FakeMirai()
    async with fake.serve() as url:
        result = await prewarm(api_key=KEY, base_url=url + "/", connections=3, websocket=0)
        assert (result.http_connections, result.websockets, result.errors) == (3, 0, [])
        assert fake.connections == 3
        assert max(r["t"] for r in gets(fake)) - min(r["t"] for r in gets(fake)) < 0.1  # not paced
        assert all(r["headers"]["authorization"] == f"Bearer {KEY}" for r in gets(fake))
        (stats,) = shared_connection_stats()["http"]
        assert stats == {"base_url": url, "warm": 3, "busy": 0, "target": 3, "services": 0, "warmups": 3}

        services = [http_tts(url) for _ in range(3)]
        results = await asyncio.gather(*(speak(tts, 8000) for tts in services))
        for down, up in results:
            assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    assert fake.connections == 3  # every greeting found an open connection
    assert len(gets(fake)) == 3  # and no warm-up request tied one up meanwhile
    assert sorted(r["conn"] for r in posts(fake)) == [1, 2, 3]


async def test_each_connection_is_refreshed_before_mirai_would_close_it(monkeypatch):
    monkeypatch.setattr(pool_module, "HTTP_REFRESH_AFTER", 0.3)
    monkeypatch.setattr(pool_module, "HTTP_WARM_WINDOW", 0.6)
    assert pool_module.HTTP_REFRESH_AFTER < pool_module.HTTP_WARM_WINDOW < 70 < 75  # Mirai: 75 s
    fake = FakeMirai()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, base_url=url, connections=3, websocket=0)
        await asyncio.sleep(1.1)
        per_connection = [sum(r["conn"] == c for r in gets(fake)) for c in (1, 2, 3)]
        await close_shared_connections()
        refreshed = len(gets(fake))
        await asyncio.sleep(0.5)
    assert fake.connections == 3  # refreshed, never reopened
    assert all(n >= 3 for n in per_connection)  # every connection, not just the first in line
    assert len(gets(fake)) == refreshed  # and nothing after the pool closed


async def test_calls_spread_over_the_warm_connections():
    fake = FakeMirai()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, base_url=url, connections=3, websocket=0)
        for _ in range(3):  # one after another: each takes the connection idle longest
            await speak(http_tts(url), 8000)
    assert [r["conn"] for r in posts(fake)] == [1, 2, 3]


async def test_a_lost_connection_is_replaced():
    fake = FakeMirai()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, base_url=url, connections=2, websocket=0)
        shared = shared_http_client(url)
        slot = shared.transport.slots[0]
        await slot.transport.aclose()  # say the network dropped it
        slot.opened = False
        shared.nudge()
        await asyncio.sleep(0.3)
        assert shared.connection_counts() == (2, 0)
    assert fake.connections == 3


async def test_an_interrupted_stream_s_connection_is_replaced_by_the_pool(monkeypatch):
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "2")
    fake = FakeMirai(seconds={"लंबा": 20.0, "छोटा": 0.5}, reads=(3200,), pace=0.05)
    rec = Recorder()
    async with fake.serve() as url:
        await run_test(
            Pipeline([http_tts(url), rec]),
            frames_to_send=[
                SleepFrame(0.6),
                TTSSpeakFrame("लंबा"),
                SleepFrame(0.6),
                InterruptionFrame(),
                SleepFrame(0.3),
                TTSSpeakFrame("छोटा"),
                SleepFrame(0.5),
            ],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
        long, short = posts(fake)
        assert long["cut_at"] is not None  # the server saw the interruption
        assert short["conn"] != long["conn"]  # the next sentence used another warm connection
        (stats,) = shared_connection_stats()["http"]
        assert stats["warm"] == 2  # and the pool replaced the one that was cut


async def test_warm_ups_pause_when_mirai_says_429(monkeypatch):
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "4")
    fake = FakeMirai()
    respond = fake._respond

    def busy_models(path, body):
        if path.endswith("/models"):
            return 429, {"content-type": "application/json", "retry-after": "5"}, b"{}"
        return respond(path, body)

    fake._respond = busy_models
    async with fake.serve() as url:
        await speak(http_tts(url), 8000, before=[SleepFrame(1.2)])
    assert len(gets(fake)) == 2  # the first two were refused; nothing more for Retry-After


async def test_shared_client_outlives_its_last_service_briefly(monkeypatch):
    monkeypatch.setattr(pool_module, "HTTP_LINGER_SECS", 0.5)
    fake = FakeMirai()
    async with fake.serve() as url:
        await speak(http_tts(url), 8000)
        (stats,) = shared_connection_stats()["http"]
        assert stats["services"] == 0 and stats["warm"] == 1
        await asyncio.sleep(0.2)
        await speak(http_tts(url), 8000)  # the next call, soon after
        assert fake.connections == 1
        await asyncio.sleep(0.8)
        assert shared_connection_stats()["http"] == []  # closed once nothing used it for a while


async def test_a_service_reopens_a_shared_client_closed_under_it():
    fake = FakeMirai()
    async with fake.serve() as url:
        tts = http_tts(url)

        async def close_mid_call():
            await asyncio.sleep(0.3)
            await close_shared_connections()

        closer = asyncio.create_task(close_mid_call())
        down, up = await speak(tts, 8000, "पहला", "दूसरा", before=[SleepFrame(0.6)])
        await closer
    assert not errors_in(up)
    assert len(posts(fake)) == 2


async def test_shared_clients_are_per_event_loop():
    fake = FakeMirai()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, base_url=url, connections=1, websocket=0)
        here = shared_http_client(url)

        def in_another_loop():
            async def main():
                # What a MiraiHttpTTSService in this loop does (Pipecat can't run
                # a pipeline outside the main thread): hold the loop's shared
                # client, send a request, let go.
                there = shared_http_client(url)
                assert there is not here and there.loop is not here.loop
                owner = object()
                there.hold(owner, warm=False, headers=AUTH)
                response = await there.client.get(url + "/models")
                assert response.status_code == 200
                there.release(owner)
                return there

            return asyncio.run(main())

        there = await asyncio.to_thread(in_another_loop)
        assert there.closed  # closed with its event loop
        assert not here.closed and here.connection_counts() == (1, 0)
    assert fake.connections == 2  # the other loop never used this loop's connection
    assert [r["conn"] for r in gets(fake)] == [1, 2]


async def test_prewarm_checks_its_arguments(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    for kwargs in (
        {},
        {"api_key": KEY, "connections": 65},
        {"api_key": KEY, "connections": -1},
        {"api_key": KEY, "websocket": -1},
        {"api_key": KEY, "websocket_url": "https://x/stream"},
    ):
        with pytest.raises(ValueError):
            await prewarm(**kwargs)


async def test_prewarm_reports_failures_without_raising():
    fake = FakeMirai()
    async with fake.serve() as url:
        pass  # nothing listens there any more
    result = await prewarm(api_key=KEY, base_url=url, connections=2, websocket=0, timeout=1.0)
    assert result.http_connections == 0
    assert result.errors and "ConnectError" in result.errors[0]


# --- WebSocket pool -----------------------------------------------------------------------


def session_updates(conn):
    """The full session.update a service sends (not the pool's empty keepalives)."""
    return [m for m in conn.received if m.get("type") == "session.update" and "voice" in m]


async def test_a_pipeline_takes_a_waiting_socket_and_the_pool_refills():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        result = await prewarm(api_key=KEY, connections=0, websocket=2, websocket_url=url)
        assert (result.http_connections, result.websockets, result.errors) == (0, 2, [])
        assert len(fake.conns) == 2 and not fake.messages()  # both waiting at session.ready
        assert all(c.headers["authorization"] == f"Bearer {KEY}" for c in fake.conns)

        tts = ws_tts(url)
        down, up = await ws_call(tts)
        assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
        used = [c for c in fake.conns if session_updates(c)]
        assert [c.number for c in used] == [2]  # a waiting socket; no new handshake
        assert session_updates(used[0])[0]["sample_rate"] == 8000
        assert tts.session_id == "ttsws_2"
        await asyncio.sleep(0.2)
        assert used[0].ws.state is State.CLOSED  # closed with its pipeline, never reused
        (stats,) = shared_connection_stats()["websocket"]
        assert {k: stats[k] for k in ("ready", "target", "opened", "handed_out")} == {
            "ready": 2,
            "target": 2,
            "opened": 3,
            "handed_out": 1,
        }
    assert len(fake.conns) == 3


async def test_the_first_websocket_service_sets_the_pool_going(monkeypatch):
    monkeypatch.setenv("MIRAI_WARM_WEBSOCKETS", "3")
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await ws_call(ws_tts(url), [TTSSpeakFrame(TEXT), SleepFrame(1.3)])  # cold: connects itself
        assert len(fake.conns) == 4  # its own socket, then three waiting ones, paced
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 3 and stats["target"] == 3
        # Three calls start together: each takes a waiting socket.
        results = await asyncio.gather(*(ws_call(ws_tts(url)) for _ in range(3)))
        assert all(not errors_in(up) for _, up in results)
        used = sorted(c.number for c in fake.conns if session_updates(c))
        assert used == [1, 2, 3, 4]  # nothing beyond the four already open spoke


async def test_pipelines_starting_together_never_share_a_socket():
    fake = FakeMiraiWS(seconds=1.0)
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=3, websocket_url=url)
        services = [ws_tts(url, voice=v) for v in ("neha", "shruti", "ashu")]
        results = await asyncio.gather(*(ws_call(tts) for tts in services))
        for down, up in results:
            assert audio_of(down, 8000) == tone(1.0, 8000) and not errors_in(up)
    used = [c for c in fake.conns if session_updates(c)]
    assert sorted(c.number for c in used) == [1, 2, 3]  # the three waiting sockets
    assert all(len(session_updates(c)) == 1 for c in used)  # one pipeline each
    assert sorted(session_updates(c)[0]["voice"] for c in used) == ["ashu", "neha", "shruti"]
    assert [len([m for m in c.received if m["type"] == "text"]) for c in used] == [1, 1, 1]


async def test_only_the_same_url_and_key_take_a_waiting_socket():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key="sk_other", connections=0, websocket=1, websocket_url=url)
        await ws_call(ws_tts(url))
        assert [len(session_updates(c)) for c in fake.conns] == [0, 1]  # connected on its own
        assert fake.conns[1].headers["authorization"] == f"Bearer {KEY}"
        assert await take_websocket(url + "/", {"Authorization": "Bearer sk_other"}) is None


async def test_shared_pool_false_always_connects():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=1, websocket_url=url)
        await ws_call(ws_tts(url, shared_pool=False))
        assert [len(session_updates(c)) for c in fake.conns] == [0, 1]
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 1 and stats["handed_out"] == 0


async def test_waiting_sockets_get_keepalives(monkeypatch):
    monkeypatch.setattr(pool_module, "WS_KEEPALIVE_SECS", 0.2)
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=2, websocket_url=url)
        await asyncio.sleep(0.75)
        for conn in fake.conns:
            # Empty session.updates (they change nothing) every 0.2 s.
            assert 3 <= len(conn.received) <= 4
            assert all(m == {"type": "session.update"} for m in conn.received)
    assert len(fake.conns) == 2


async def test_the_pool_replaces_a_socket_the_server_closes():
    fake = FakeMiraiWS(close_after=0.3)  # Mirai ends the first session after 0.3 s
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=1, websocket_url=url)
        await asyncio.sleep(0.8)
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 1 and stats["opened"] == 2
        down, up = await ws_call(ws_tts(url))
        assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
        assert len(session_updates(fake.conns[1])) == 1  # the replacement served the call


async def test_old_waiting_sockets_are_replaced(monkeypatch):
    monkeypatch.setattr(pool_module.PooledWebsocket, "max_age_secs", property(lambda self: 0.3))
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=1, websocket_url=url)
        await asyncio.sleep(0.8)
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 1 and stats["opened"] >= 3
        assert fake.conns[0].ws.state is State.CLOSED


async def test_prewarm_zero_closes_the_waiting_sockets():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=2, websocket_url=url)
        await prewarm(api_key=KEY, connections=0, websocket=0, websocket_url=url)
        await asyncio.sleep(0.2)
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 0 and stats["target"] == 0
        assert all(c.ws.state is State.CLOSED for c in fake.conns)
        assert await take_websocket(url, AUTH) is None


async def test_waiting_sockets_are_per_event_loop():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=1, websocket_url=url)

        def in_another_loop():
            async def main():
                return await take_websocket(url, AUTH)

            return asyncio.run(main())

        assert await asyncio.to_thread(in_another_loop) is None
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 1


async def test_prewarm_derives_the_websocket_url_from_the_base_url():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        base = url.replace("ws://", "http://").replace("/audio/speech/stream", "")
        await prewarm(api_key=KEY, base_url=base, connections=0, websocket=1)
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["url"] == url and stats["ready"] == 1


async def test_a_taken_socket_is_the_takers_to_read():
    # The pool stopped reading it: the next message is the new owner's.
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=1, websocket_url=url)
        pooled = await take_websocket(url, AUTH)
        await pooled.websocket.send(json.dumps({"type": "session.update"}))
        assert json.loads(await pooled.websocket.recv())["type"] == "session.updated"
        await pooled.websocket.close()


# --- process lifecycle ----------------------------------------------------------------------

SHUTDOWN_SCRIPT = """
import asyncio, os, sys
sys.path.insert(0, {tests!r})
os.environ["MIRAI_WARM_CONNECTIONS"] = "3"
os.environ["MIRAI_WARM_WEBSOCKETS"] = "3"
from fake_mirai_ws import FakeMiraiWS
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from test_tts import FakeMirai
from pipecat_mirai import MiraiHttpTTSService, MiraiWebsocketTTSService

async def main():
    http, ws = FakeMirai(), FakeMiraiWS()
    async with http.serve() as http_url, ws.serve() as ws_url:
        for tts in (
            MiraiHttpTTSService(api_key="k", base_url=http_url),
            MiraiWebsocketTTSService(api_key="k", url=ws_url),
        ):
            await run_test(tts, frames_to_send=[TTSSpeakFrame("नमस्ते"), SleepFrame(0.6)],
                           pipeline_params=PipelineParams(audio_out_sample_rate=8000))
    # The pools are still warm here; the event loop ends without closing them.

asyncio.run(main())
print("clean exit")
"""


def test_a_process_exits_cleanly_with_its_pools_still_open(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = tmp_path / "worker.py"
    script.write_text(SHUTDOWN_SCRIPT.format(tests=str(Path(__file__).parent)))
    proc = subprocess.run(
        [sys.executable, "-X", "dev", "-W", "default", str(script)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "clean exit" in proc.stdout
    for bad in ("Task was destroyed", "never awaited", "ResourceWarning", "Unclosed", "unclosed"):
        assert bad not in proc.stderr, proc.stderr[-2000:]
