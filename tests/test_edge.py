"""Streaming from Mirai's edge: a token from the gateway, the socket on the edge, the gateway as fallback."""

import asyncio
import socket
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fake_mirai_gateway import FakeGateway
from fake_mirai_stt import FakeMiraiSTT
from fake_mirai_ws import FakeMiraiWS
from loguru import logger
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame, TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from test_stt import call as stt_call
from test_stt import errors_in as stt_errors
from test_stt import hz_of, texts, utterance
from test_tts import TEXT, audio_of, errors_in, tone

from pipecat_mirai import (
    MiraiSTTService,
    MiraiWebsocketTTSService,
    close_shared_connections,
    prewarm,
    shared_connection_stats,
)
from pipecat_mirai import edge as edge_module
from pipecat_mirai import pool as pool_module

KEY = "sk_test"
SENTENCES = ["आपका order कल deliver होगा।", "क्या मैं कुछ और बता सकती हूँ?", "धन्यवाद, आपका दिन शुभ हो।"]


@pytest.fixture(autouse=True)
async def _edge_on(monkeypatch):
    monkeypatch.delenv("MIRAI_TTS_EDGE", raising=False)  # the default: "auto"
    monkeypatch.delenv("MIRAI_STT_EDGE", raising=False)
    yield
    await close_shared_connections()


@pytest.fixture
def logs():
    """Every message logged during the test, at any level: (level, module, message)."""
    seen = []
    sink = logger.add(lambda m: seen.append((m.record["level"].name, m.record["name"], m.record["message"])))
    yield seen
    logger.remove(sink)


@pytest.fixture
def edge_warnings(logs):
    """The edge's warnings so far."""
    return lambda: [m for level, name, m in logs if level == "WARNING" and name == "pipecat_mirai.edge"]


def closed_port_url() -> str:
    """A ws:// URL nothing listens on: connections to it are refused."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"ws://127.0.0.1:{port}/v1/audio/speech/stream"


@dataclass
class Stack:
    url: str  # the gateway's streaming endpoint (what the service is given)
    gateway: FakeGateway
    delhi: FakeMiraiWS  # the gateway's streaming server
    edge: FakeMiraiWS
    edge_url: str


@asynccontextmanager
async def stack(*, offer=True, edge_url=None, gateway=None, delhi=None, edge=None):
    """The gateway (token route + stream) and an edge that takes the gateway's tokens.

    ``offer``: the token answer names the edge. ``edge_url`` names another one instead.
    """
    delhi_ws = FakeMiraiWS(**(delhi or {}))
    gw = FakeGateway(delhi_ws, key=KEY, **(gateway or {}))
    edge_ws = FakeMiraiWS(
        **{"auth": lambda headers: headers.get("Authorization") == f"Bearer {KEY}", **(edge or {})}
    )
    async with edge_ws.serve() as real_edge_url, gw.serve() as url:
        if offer:
            gw.edge_url = edge_url or real_edge_url
        yield Stack(url, gw, delhi_ws, edge_ws, real_edge_url)


def ws_tts(url, **kwargs):
    return MiraiWebsocketTTSService(api_key=KEY, url=url, **kwargs)


async def ws_call(tts, frames=None, rate=8000):
    return await run_test(
        tts,
        frames_to_send=frames or [TTSSpeakFrame(TEXT)],
        pipeline_params=PipelineParams(audio_out_sample_rate=rate),
        start_timeout=10.0,
    )


def bearer(seen) -> str:
    """The Authorization header of a connection, handshake or token request."""
    headers = seen["headers"] if isinstance(seen, dict) else seen.headers
    return headers.get("authorization", "")


def assert_key_never_at_edge(s: Stack):
    """The key reaches the edge in the Authorization header only: never in the URL."""
    assert s.edge.handshakes, "nothing reached the edge"
    for handshake in s.edge.handshakes:
        assert KEY not in handshake["path"]
        assert bearer(handshake) == f"Bearer {KEY}"


def spoke(fake: FakeMiraiWS) -> list[int]:
    """The connections a pipeline used (sent its full session.update on)."""
    return [
        c.number for c in fake.conns if any("voice" in m for m in c.received if m["type"] == "session.update")
    ]


def forget_backoff():
    """As if the backoff had run out."""
    for state in edge_module._health.values():
        state.retry_at = 0.0


# --- the edge path --------------------------------------------------------------------------


async def test_speaks_from_the_edge_with_the_api_key(logs):
    async with stack() as s:
        tts = ws_tts(s.url)
        down, up = await ws_call(tts)
    assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    # This gateway isn't one of Mirai's known ones: it was asked once which edge it offers...
    (mint,) = s.gateway.mints
    assert mint["path"].split("?")[0] == "/v2/tts/stream/tokens" and bearer(mint) == f"Bearer {KEY}"
    # ...and the socket opened there with the API key; the gateway's socket was never opened.
    (conn,) = s.edge.conns
    assert bearer(conn) == f"Bearer {KEY}"
    assert s.delhi.handshakes == []
    assert_key_never_at_edge(s)
    assert conn.received[0] == {
        "type": "session.update",
        "voice": "neha",
        "model": "mira-tts",
        "response_format": "pcm",
        "audio_transport": "binary",
        "sample_rate": 8000,
    }
    assert tts.connected_url == s.edge_url and tts.session_id == "ttsws_1"
    assert not any(KEY in m for _, _, m in logs)  # the key is never logged, at any level


async def test_the_gateway_is_asked_once_which_edge_and_every_socket_goes_there():
    async with stack() as s:
        for _ in range(3):
            _, up = await ws_call(ws_tts(s.url))
            assert not errors_in(up)
    assert len(s.gateway.mints) == 1  # the answer is remembered
    assert [bearer(c) for c in s.edge.conns] == [f"Bearer {KEY}"] * 3 and s.delhi.handshakes == []


async def test_calls_starting_together_all_reach_the_edge():
    async with stack() as s:
        results = await asyncio.gather(*(ws_call(ws_tts(s.url)) for _ in range(4)))
    assert all(not errors_in(up) for _, up in results)
    assert len(s.edge.conns) == 4 and s.delhi.handshakes == []
    assert len(s.gateway.mints) == 1  # sockets starting together share the one question


async def test_a_forced_edge_url_is_used_whatever_the_gateway_offers():
    async with stack(offer=False) as s:
        tts = ws_tts(s.url, edge=s.edge_url)
        _, up = await ws_call(tts)
    assert not errors_in(up)
    assert s.gateway.mints == [] and len(s.edge.conns) == 1 and s.delhi.handshakes == []
    assert tts.connected_url == s.edge_url
    assert_key_never_at_edge(s)


# --- no edge: the gateway, as before ---------------------------------------------------------


@pytest.mark.parametrize(
    "gateway",
    [
        {"edge_url": None},  # the gateway offers no edge now
        {"edge_field": False},  # today's gateway: a token, and no edge_url at all
        {"token_status": 404},  # an older gateway without the token route
    ],
    ids=["edge-url-null", "no-edge-field", "no-token-route"],
)
async def test_without_an_edge_the_socket_opens_on_the_gateway(gateway, edge_warnings):
    async with stack(offer=False, gateway=gateway) as s:
        tts = ws_tts(s.url)
        down, up = await ws_call(tts)
        assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
        assert s.edge.handshakes == []
        (conn,) = s.delhi.conns
        assert bearer(conn) == f"Bearer {KEY}"  # the gateway gets the API key, as before
        assert tts.connected_url == s.url
        # The next call doesn't ask again for a while.
        await ws_call(ws_tts(s.url))
    assert len(s.gateway.mints) == 1 and len(s.delhi.conns) == 2
    assert edge_warnings() == []  # nothing is wrong


async def test_edge_false_never_asks_for_a_token():
    async with stack() as s:
        _, up = await ws_call(ws_tts(s.url, edge=False))
    assert not errors_in(up)
    assert s.gateway.mints == [] and s.edge.handshakes == [] and len(s.delhi.conns) == 1


async def test_mirai_tts_edge_off_turns_the_edge_off(monkeypatch):
    monkeypatch.setenv("MIRAI_TTS_EDGE", "off")
    async with stack() as s:
        await ws_call(ws_tts(s.url))
    assert s.gateway.mints == [] and s.edge.handshakes == [] and len(s.delhi.conns) == 1


@pytest.mark.parametrize("bad", ["https://edge.example/v1/audio/speech/stream", "yes", ""])
async def test_invalid_edge_options_are_refused(bad):
    with pytest.raises(ValueError):
        ws_tts("wss://example/v1/audio/speech/stream", edge=bad)
    with pytest.raises(ValueError):
        await prewarm(api_key=KEY, connections=0, websocket=1, edge=bad)


# --- a broken edge: the gateway, and the edge left alone for a while --------------------------

FAILURES = {
    "edge-refuses": ({}, {"refuse_status": 503}, "HTTP 503"),
    "edge-unreachable": ({"edge_url": "closed"}, {}, "ConnectionRefusedError"),
    "no-session-ready": ({}, {"silent": True}, "not ready within"),
    "token-route-fails": (
        {"gateway": {"token_status": 503}},
        {},
        "couldn't say which edge to use (HTTP 503)",
    ),
    "token-route-hangs": (
        {"gateway": {"token_delay": 5.0}},
        {},
        "couldn't say which edge to use (ReadTimeout",
    ),
}


@pytest.mark.parametrize("failure", FAILURES, ids=list(FAILURES))
async def test_a_broken_edge_falls_back_to_the_gateway_and_is_left_alone(
    failure, monkeypatch, edge_warnings, logs
):
    stack_kwargs, edge_kwargs, reason = FAILURES[failure]
    monkeypatch.setattr(edge_module, "EDGE_READY_TIMEOUT", 0.5)
    monkeypatch.setattr(edge_module, "TOKEN_TIMEOUT", 0.5)
    if stack_kwargs.get("edge_url") == "closed":
        stack_kwargs = {"edge_url": closed_port_url()}
    async with stack(edge=edge_kwargs, **stack_kwargs) as s:
        tts = ws_tts(s.url)
        started = asyncio.get_running_loop().time()
        down, up = await ws_call(tts)
        took = asyncio.get_running_loop().time() - started
        assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
        assert tts.connected_url == s.url and len(s.delhi.conns) == 1
        assert bearer(s.delhi.conns[0]) == f"Bearer {KEY}"
        assert took < 3.0
        (warning,) = edge_warnings()
        assert "TTS edge is unavailable" in warning and reason in warning and "60 s" in warning
        # Inside the backoff the next call goes straight to the gateway: no token, no edge.
        mints, attempts = len(s.gateway.mints), len(s.edge.handshakes)
        await ws_call(ws_tts(s.url))
        assert len(s.gateway.mints) == mints and len(s.edge.handshakes) == attempts
        assert len(s.delhi.conns) == 2
    assert not any(KEY in m for _, _, m in logs)  # the key is never logged, at any level
    if s.edge.handshakes:
        assert_key_never_at_edge(s)


async def test_the_edge_is_tried_again_after_the_backoff_which_doubles(edge_warnings):
    async with stack(edge={"refuse_status": 503}) as s:
        await ws_call(ws_tts(s.url))
        (state,) = edge_module._health.values()
        assert state.failures == 1
        first_window = state.retry_at - asyncio.get_running_loop().time()
        assert 55 < first_window <= 60

        forget_backoff()
        await ws_call(ws_tts(s.url))  # fails again: twice as long
        assert state.failures == 2 and 115 < state.retry_at - asyncio.get_running_loop().time() <= 120
        assert len(edge_warnings()) == 1  # once per process

        s.edge.refuse_status = None  # the edge is back
        forget_backoff()
        tts = ws_tts(s.url)
        _, up = await ws_call(tts)
    assert not errors_in(up) and tts.connected_url == s.edge_url
    assert state.ok and state.failures == 0
    assert len(s.edge.handshakes) == 3 and len(s.edge.conns) == 1 and len(s.delhi.conns) == 2


# --- the pool ------------------------------------------------------------------------------


async def test_waiting_sockets_open_on_the_edge_and_calls_take_them():
    async with stack() as s:
        result = await prewarm(api_key=KEY, connections=0, websocket=2, websocket_url=s.url)
        assert (result.websockets, result.errors) == (2, [])
        assert len(s.edge.conns) == 2 and s.delhi.handshakes == []
        assert all(bearer(c) == f"Bearer {KEY}" for c in s.edge.conns)
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 2 and stats["edge"] == 2

        tts = ws_tts(s.url)
        down, up = await ws_call(tts)
        assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
        assert spoke(s.edge) == [2]  # a waiting edge socket: no handshake for the call
        assert tts.connected_url == s.edge_url
        await asyncio.sleep(0.3)
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["ready"] == 2 and stats["edge"] == 2  # the replacement is on the edge too
    assert len(s.gateway.mints) == 1  # asked once which edge; every socket opened there with the key
    assert_key_never_at_edge(s)


async def test_a_first_burst_tries_at_once_and_a_gateway_without_an_edge_is_then_remembered():
    async with stack(offer=False) as s:
        result = await prewarm(api_key=KEY, connections=0, websocket=4, websocket_url=s.url)
        assert result.websockets == 4
        # A fresh process doesn't queue a burst behind one probe: each socket asks at
        # once (a token request is cheap), and all of them land on the gateway.
        assert 1 <= len(s.gateway.mints) <= 4
        assert len(s.delhi.conns) == 4 and s.edge.handshakes == []
        # Once the answer is known, later sockets don't ask again within the backoff.
        before = len(s.gateway.mints)
        result = await prewarm(api_key=KEY, connections=0, websocket=2, websocket_url=s.url)
        assert len(s.gateway.mints) == before


async def test_waiting_gateway_sockets_move_to_the_edge_when_it_is_back():
    async with stack(edge={"refuse_status": 503}) as s:
        result = await prewarm(api_key=KEY, connections=0, websocket=2, websocket_url=s.url)
        assert result.websockets == 2 and len(s.delhi.conns) == 2
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["edge"] == 0

        s.edge.refuse_status = None
        forget_backoff()
        (pool,) = pool_module._ws.values()
        pool._wake.set()
        for _ in range(40):
            await asyncio.sleep(0.1)
            (stats,) = shared_connection_stats()["websocket"]
            if stats["edge"] == 2:
                break
        assert stats["ready"] == 2 and stats["edge"] == 2  # one at a time, the gateway's replaced
        await asyncio.sleep(0.2)
        assert all(c.ws.state.name == "CLOSED" for c in s.delhi.conns)
        _, up = await ws_call(ws_tts(s.url))
        assert not errors_in(up) and spoke(s.edge) and not spoke(s.delhi)


async def test_prewarm_with_edge_false_keeps_its_sockets_for_services_with_edge_false():
    async with stack() as s:
        await prewarm(api_key=KEY, connections=0, websocket=1, websocket_url=s.url, edge=False)
        assert len(s.delhi.conns) == 1 and s.gateway.mints == []
        await ws_call(ws_tts(s.url))  # "auto": not that pool's socket
        await ws_call(ws_tts(s.url, edge=False))  # takes it
    assert spoke(s.delhi) == [1] and len(s.edge.conns) == 1


# --- mid-call drops ------------------------------------------------------------------------


def llm_turn(*texts):
    return [LLMFullResponseStartFrame(), *(TextFrame(t) for t in texts), LLMFullResponseEndFrame()]


async def test_an_edge_drop_mid_call_reconnects_to_the_edge_and_resends_the_rest():
    edge = {"seconds": {s: 1.0 for s in SENTENCES}, "reads": (800,), "pace": 0.02, "drop_after_bytes": 4800}
    async with stack(edge=edge) as s:
        _, up = await ws_call(ws_tts(s.url), [*llm_turn(*(x + " " for x in SENTENCES)), SleepFrame(2.0)])
    assert not errors_in(up)
    assert len(s.edge.conns) == 2 and s.delhi.handshakes == []
    assert [bearer(c) for c in s.edge.conns] == [f"Bearer {KEY}"] * 2
    # The sentence the caller had started hearing isn't repeated; the rest is.
    assert [x for _, _, x in s.edge.spoken] == SENTENCES[1:]
    assert_key_never_at_edge(s)


async def test_an_edge_that_dies_mid_call_hands_the_call_to_the_gateway(edge_warnings):
    edge = {
        "seconds": {s: 1.0 for s in SENTENCES},
        "reads": (800,),
        "pace": 0.02,
        "drop_after_bytes": 4800,
        "refuse_attempts": {2, 3, 4},
    }
    async with stack(edge=edge) as s:
        tts = ws_tts(s.url)
        _, up = await ws_call(tts, [*llm_turn(*(x + " " for x in SENTENCES)), SleepFrame(2.0)])
    assert not errors_in(up)
    assert len(s.edge.conns) == 1 and len(s.delhi.conns) == 1
    assert [x for _, _, x in s.delhi.spoken] == SENTENCES[1:]
    assert tts.connected_url == s.url
    assert len(edge_warnings()) == 1 and "HTTP 503" in edge_warnings()[0]


# --- process lifecycle ----------------------------------------------------------------------

SHUTDOWN_SCRIPT = """
import asyncio, os, sys, threading
sys.path.insert(0, {tests!r})
os.environ.pop("MIRAI_TTS_EDGE", None)
os.environ["MIRAI_WARM_CONNECTIONS"] = "0"
os.environ["MIRAI_WARM_WEBSOCKETS"] = "3"
from fake_mirai_gateway import FakeGateway
from fake_mirai_stt import FakeMiraiSTT
from fake_mirai_ws import FakeMiraiWS
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from pipecat_mirai import MiraiWebsocketTTSService, shared_connection_stats

# Mirai (gateway and edge) runs in a thread of its own, up until after the worker's loop has ended.
up, stop, mirai = threading.Event(), threading.Event(), {{}}

def serve():
    async def run():
        delhi = FakeMiraiWS()
        gateway = FakeGateway(delhi, key="k")
        edge = FakeMiraiWS(auth=lambda headers: headers.get("Authorization") == "Bearer k")
        async with edge.serve() as edge_url, gateway.serve() as url:
            gateway.edge_url = edge_url
            mirai.update(url=url, delhi=delhi)
            up.set()
            while not stop.is_set():
                await asyncio.sleep(0.05)
    asyncio.run(run())

server = threading.Thread(target=serve)
server.start()
assert up.wait(10)

async def main():
    tts = MiraiWebsocketTTSService(api_key="k", url=mirai["url"])
    await run_test(tts, frames_to_send=[TTSSpeakFrame("नमस्ते"), SleepFrame(0.2)],
                   pipeline_params=PipelineParams(audio_out_sample_rate=8000))
    for _ in range(100):
        (stats,) = shared_connection_stats()["websocket"]
        if stats["edge"] == 3:
            break
        await asyncio.sleep(0.05)
    assert stats["edge"] == 3 and not mirai["delhi"].handshakes, stats
    # The pools are warm, their sockets open on the edge; the event loop ends without closing them.

asyncio.run(main())
stop.set()
server.join(10)
print("clean exit")
"""


def test_a_process_exits_cleanly_with_edge_sockets_still_open(tmp_path):
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


# --- Mirai's own gateway --------------------------------------------------------------------


@pytest.fixture
def known_edge(monkeypatch):
    """Make the fake gateway one of Mirai's own: its host has a known edge."""

    def known(s: Stack):
        monkeypatch.setitem(edge_module.KNOWN_EDGES, ("127.0.0.1", "tts"), s.edge_url)

    return known


async def test_mirais_own_gateway_goes_straight_to_its_edge_without_waiting(known_edge):
    # The gateway takes 2 s to say which edge it offers; the call doesn't wait for it.
    async with stack(gateway={"token_delay": 2.0}) as s:
        known_edge(s)
        tts = ws_tts(s.url)
        started = time.monotonic()
        down, up = await ws_call(tts)
        assert time.monotonic() - started < 1.5
        assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
        assert tts.connected_url == s.edge_url and s.delhi.handshakes == []
        assert [bearer(c) for c in s.edge.conns] == [f"Bearer {KEY}"]
        # ...but it was asked, in the background, once.
        await asyncio.sleep(2.2)
        assert len(s.gateway.mints) == 1
    assert_key_never_at_edge(s)


async def test_when_mirais_gateway_stops_offering_the_edge_sockets_go_back_to_it(known_edge):
    async with stack(offer=False) as s:
        known_edge(s)
        _, up = await ws_call(ws_tts(s.url))  # straight to the known edge, while the gateway is asked
        assert not errors_in(up) and len(s.edge.conns) == 1
        await asyncio.sleep(0.2)
        tts = ws_tts(s.url)
        _, up = await ws_call(tts)  # the gateway said no edge: this one stays on the gateway
        assert not errors_in(up) and tts.connected_url == s.url
        assert len(s.edge.conns) == 1 and len(s.delhi.handshakes) >= 1


async def test_a_refused_key_goes_to_the_gateway_without_putting_the_edge_off(known_edge, edge_warnings):
    async with stack(edge={"auth": lambda headers: False}) as s:
        known_edge(s)
        tts = ws_tts(s.url)
        _, up = await ws_call(tts)
        assert tts.connected_url == s.url and len(s.delhi.handshakes) == 1
        assert edge_warnings() == []  # a key problem is not an edge problem
        assert edge_module._health[(edge_module.token_url(s.url), "auto")].retry_at == 0.0


# --- crowding: many workers, one workspace -----------------------------------------------


async def test_a_pool_refused_by_the_edge_backs_off_instead_of_filling_the_gateway(known_edge):
    # The edge is at the workspace's open-socket limit (429). Waiting sockets nobody
    # asked for must not move to the gateway and spend its rate limit; a call still
    # speaks, through the gateway.
    async with stack(edge={"refuse_status": 429}) as s:
        known_edge(s)
        result = await prewarm(api_key=KEY, connections=0, websocket=3, websocket_url=s.url, timeout=1.0)
        assert result.websockets == 0 and result.errors
        assert s.delhi.handshakes == []  # nothing spilled onto the gateway
        tts = ws_tts(s.url, shared_pool=False)
        _, up = await ws_call(tts)
        assert not errors_in(up) and tts.connected_url == s.url
        assert len(s.delhi.handshakes) == 1


# --- the second service: streaming STT --------------------------------------------------------


@dataclass
class STTStack:
    url: str
    gateway: FakeGateway
    delhi: FakeMiraiSTT
    edge: FakeMiraiSTT
    edge_url: str


@asynccontextmanager
async def stt_stack(*, offer=True, gateway=None, delhi=None, edge=None):
    """The STT gateway (token route + stream) and an edge that takes the API key."""
    delhi_stt = FakeMiraiSTT(**(delhi or {}))
    gw = FakeGateway(delhi_stt, key=KEY, service="stt", **(gateway or {}))
    edge_stt = FakeMiraiSTT(
        **{"auth": lambda headers: headers.get("Authorization") == f"Bearer {KEY}", **(edge or {})}
    )
    async with edge_stt.serve() as real_edge_url, gw.serve() as url:
        if offer:
            gw.edge_url = real_edge_url
        yield STTStack(url, gw, delhi_stt, edge_stt, real_edge_url)


def known_stt_edge(monkeypatch, s: STTStack):
    monkeypatch.setitem(edge_module.KNOWN_EDGES, ("127.0.0.1", "stt"), s.edge_url)


def stt_service(url, **kwargs):
    return MiraiSTTService(api_key=KEY, url=url, endpointing="manual", **kwargs)


def test_each_service_has_its_own_token_route_and_known_edge():
    assert edge_module.token_url("wss://h/v1/audio/speech/stream") == "https://h/v2/tts/stream/tokens"
    assert (
        edge_module.token_url("wss://h/v1/audio/transcriptions/stream?x=1", "stt")
        == "https://h/v2/stt/stream/tokens"
    )
    assert edge_module.token_url("ws://127.0.0.1:1/v1/audio/transcriptions/stream", "stt").startswith(
        "http://"
    )
    sandbox = "sandbox.voice.miraiminds.co"
    assert edge_module.KNOWN_EDGES[(sandbox, "tts")].startswith("wss://tts-edge.")
    assert edge_module.KNOWN_EDGES[(sandbox, "stt")] == (
        "wss://stt-edge.voice.miraiminds.co/v1/audio/transcriptions/stream"
    )


def test_each_service_has_its_own_off_switch(monkeypatch):
    monkeypatch.setenv("MIRAI_STT_EDGE", "off")
    assert edge_module.edge_mode("auto", "stt") is None
    assert edge_module.edge_mode("auto", "tts") == "auto"
    monkeypatch.delenv("MIRAI_STT_EDGE")
    monkeypatch.setenv("MIRAI_TTS_EDGE", "0")
    assert edge_module.edge_mode("auto", "stt") == "auto" and edge_module.edge_mode("auto") is None
    assert edge_module.edge_mode("wss://e/v1/audio/transcriptions/stream", "stt").startswith("wss://e")


async def test_an_offered_stt_edge_takes_the_socket_with_the_key_and_the_settings():
    async with stt_stack() as s:
        service = stt_service(s.url, language="gu-IN")
        down, up = await stt_call(service, utterance(440))
    assert not stt_errors(up) and [hz_of(t) for t in texts(down)] == [440]
    (mint,) = s.gateway.mints
    assert mint["path"].split("?")[0] == "/v2/stt/stream/tokens" and bearer(mint) == f"Bearer {KEY}"
    (handshake,) = s.edge.handshakes
    assert KEY not in handshake["path"] and bearer(handshake) == f"Bearer {KEY}"
    # The settings that fix the stream travel to the edge in the query, as on the gateway.
    assert handshake["query"] == {
        "model": "mira-stt",
        "encoding": "linear16",
        "sample_rate": "8000",
        "endpointing": "manual",
        "language_code": "gu-IN",
    }
    assert s.delhi.handshakes == [] and service.connected_url == s.edge_url


async def test_mirais_own_stt_gateway_goes_straight_to_its_edge(monkeypatch):
    async with stt_stack(gateway={"token_delay": 2.0}) as s:
        known_stt_edge(monkeypatch, s)
        service = stt_service(s.url)
        started = time.monotonic()
        down, up = await stt_call(service, utterance(440))
        assert time.monotonic() - started < 1.5
        assert not stt_errors(up) and service.connected_url == s.edge_url and s.delhi.handshakes == []
        await asyncio.sleep(2.2)
        assert len(s.gateway.mints) == 1  # asked, in the background, once


async def test_a_dead_stt_edge_leaves_the_call_to_the_gateway_and_tts_unaffected(monkeypatch, edge_warnings):
    async with stt_stack() as s:
        s.gateway.edge_url = closed_port_url().replace("/speech/", "/transcriptions/")
        service = stt_service(s.url)
        down, up = await stt_call(service, utterance(440))
        assert not stt_errors(up) and [hz_of(t) for t in texts(down)] == [440]
        assert service.connected_url == s.url and len(s.delhi.handshakes) == 1
        (warning,) = edge_warnings()
        assert "STT edge is unavailable" in warning
        # The failure is the STT edge's: the TTS edge of the same host keeps its own good name.
        states = {key: state.ok for key, state in edge_module._health.items()}
        assert any(key[0].endswith("/v2/stt/stream/tokens") and not ok for key, ok in states.items())
        assert not any(key[0].endswith("/v2/tts/stream/tokens") and not ok for key, ok in states.items())


async def test_a_gateway_without_the_stt_token_route_is_used_directly():
    async with stt_stack(gateway={"token_status": 404}) as s:
        service = stt_service(s.url)
        _, up = await stt_call(service, utterance(440))
        assert not stt_errors(up) and service.connected_url == s.url and s.edge.handshakes == []


async def test_a_refused_key_at_the_stt_edge_goes_to_the_gateway(monkeypatch, edge_warnings):
    async with stt_stack(edge={"auth": lambda headers: False}) as s:
        known_stt_edge(monkeypatch, s)
        service = stt_service(s.url)
        _, up = await stt_call(service, utterance(440))
        assert service.connected_url == s.url and len(s.delhi.handshakes) == 1 and edge_warnings() == []


async def test_the_stt_edge_off_switch_keeps_sockets_on_the_gateway(monkeypatch):
    monkeypatch.setenv("MIRAI_STT_EDGE", "off")
    async with stt_stack() as s:
        service = stt_service(s.url)
        _, up = await stt_call(service, utterance(440))
        assert not stt_errors(up) and service.connected_url == s.url
        assert s.gateway.mints == [] and s.edge.handshakes == []


async def test_waiting_stt_sockets_open_on_the_edge(monkeypatch):
    async with stt_stack() as s:
        known_stt_edge(monkeypatch, s)
        result = await prewarm(api_key=KEY, connections=0, websocket=0, stt_websockets=2, stt_url=s.url)
        assert (result.stt_websockets, result.errors) == (2, [])
        assert len(s.edge.conns) == 2 and s.delhi.handshakes == []
        service = stt_service(s.url)
        down, up = await stt_call(service, utterance(440))
        assert not stt_errors(up) and service.connected_url == s.edge_url
        assert [hz_of(t) for t in texts(down)] == [440] and any(c.utterances for c in s.edge.conns[:2])
