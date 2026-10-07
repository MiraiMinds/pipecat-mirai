"""Streaming from Mirai's edge: a token from the gateway, the socket on the edge, the gateway as fallback."""

import asyncio
import socket
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fake_mirai_gateway import FakeGateway
from fake_mirai_ws import FakeMiraiWS
from loguru import logger
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, TextFrame, TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from test_tts import TEXT, audio_of, errors_in, tone

from pipecat_mirai import MiraiWebsocketTTSService, close_shared_connections, prewarm, shared_connection_stats
from pipecat_mirai import edge as edge_module
from pipecat_mirai import pool as pool_module

KEY = "sk_test"
SENTENCES = ["आपका order कल deliver होगा।", "क्या मैं कुछ और बता सकती हूँ?", "धन्यवाद, आपका दिन शुभ हो।"]


@pytest.fixture(autouse=True)
async def _edge_on(monkeypatch):
    monkeypatch.delenv("MIRAI_TTS_EDGE", raising=False)  # the default: "auto"
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
    edge_ws = FakeMiraiWS(auth=gw.spend, **(edge or {}))
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
    assert s.edge.handshakes, "nothing reached the edge"
    for handshake in s.edge.handshakes:
        assert KEY not in handshake["path"]
        assert not any(KEY in str(value) for value in handshake["headers"].values())


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


async def test_speaks_from_the_edge_with_a_fresh_token(logs):
    async with stack() as s:
        tts = ws_tts(s.url)
        down, up = await ws_call(tts)
    assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    # One token, asked for with the API key...
    (mint,) = s.gateway.mints
    assert mint["path"] == "/v2/tts/stream/tokens" and bearer(mint) == f"Bearer {KEY}"
    # ...and spent opening the socket on the edge; the gateway's socket was never opened.
    (conn,) = s.edge.conns
    assert bearer(conn) == f"Bearer {mint['token']}" and s.gateway.issued == {mint["token"]: True}
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
    assert not any(mint["token"] in m for _, _, m in logs)  # the token is never logged, at any level


async def test_each_socket_gets_its_own_token_over_one_kept_alive_connection():
    async with stack() as s:
        for _ in range(3):
            _, up = await ws_call(ws_tts(s.url))
            assert not errors_in(up)
    tokens = s.gateway.tokens
    assert len(tokens) == 3 and len(set(tokens)) == 3
    assert [bearer(c) for c in s.edge.conns] == [f"Bearer {t}" for t in tokens]  # each used once, in turn
    assert all(s.gateway.issued.values())
    assert len({m["conn"] for m in s.gateway.mints}) == 1  # the shared HTTP client kept its connection


async def test_calls_starting_together_all_reach_the_edge():
    async with stack() as s:
        results = await asyncio.gather(*(ws_call(ws_tts(s.url)) for _ in range(4)))
    assert all(not errors_in(up) for _, up in results)
    assert len(s.edge.conns) == 4 and s.delhi.handshakes == []
    assert len(set(s.gateway.tokens)) == 4


async def test_a_forced_edge_url_is_used_whatever_the_gateway_offers():
    async with stack(offer=False) as s:
        tts = ws_tts(s.url, edge=s.edge_url)
        _, up = await ws_call(tts)
    assert not errors_in(up)
    assert len(s.gateway.mints) == 1 and len(s.edge.conns) == 1 and s.delhi.handshakes == []
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
    "token-route-fails": ({"gateway": {"token_status": 503}}, {}, "token request failed (HTTP 503)"),
    "token-route-hangs": ({"gateway": {"token_delay": 5.0}}, {}, "token request failed (ReadTimeout"),
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
    for token in s.gateway.tokens:
        assert not any(token in m for _, _, m in logs)  # the token is never logged, at any level
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
        assert len(set(s.gateway.tokens)) == 2  # one fresh token per socket
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
    assert len(s.gateway.tokens) == 3 and all(s.gateway.issued.values())
    assert_key_never_at_edge(s)


async def test_a_burst_of_sockets_asks_a_gateway_without_an_edge_only_once():
    async with stack(offer=False) as s:
        result = await prewarm(api_key=KEY, connections=0, websocket=4, websocket_url=s.url)
        assert result.websockets == 4
    assert len(s.gateway.mints) == 1  # one socket found out; the others waited for its answer
    assert len(s.delhi.conns) == 4 and s.edge.handshakes == []


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
    tokens = s.gateway.tokens
    assert len(tokens) == 2 and [bearer(c) for c in s.edge.conns] == [f"Bearer {t}" for t in tokens]
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
import asyncio, os, sys
sys.path.insert(0, {tests!r})
os.environ.pop("MIRAI_TTS_EDGE", None)
os.environ["MIRAI_WARM_CONNECTIONS"] = "0"
os.environ["MIRAI_WARM_WEBSOCKETS"] = "3"
from fake_mirai_gateway import FakeGateway
from fake_mirai_ws import FakeMiraiWS
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import SleepFrame, run_test
from pipecat_mirai import MiraiWebsocketTTSService, shared_connection_stats

async def main():
    delhi = FakeMiraiWS()
    gateway = FakeGateway(delhi, key="k")
    edge = FakeMiraiWS(auth=gateway.spend)
    async with edge.serve() as edge_url, gateway.serve() as url:
        gateway.edge_url = edge_url
        tts = MiraiWebsocketTTSService(api_key="k", url=url)
        await run_test(tts, frames_to_send=[TTSSpeakFrame("नमस्ते"), SleepFrame(1.2)],
                       pipeline_params=PipelineParams(audio_out_sample_rate=8000))
        (stats,) = shared_connection_stats()["websocket"]
        assert stats["edge"] >= 1 and not delhi.handshakes, (stats, delhi.handshakes)
    # The pools are still warm here; the event loop ends without closing them.

asyncio.run(main())
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
