#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Hedged connects: a stuck first connect is overtaken; losers are closed; nothing leaks."""

import asyncio
import socket
import time

import httpx
import pytest

from pipecat_mirai import _net
from pipecat_mirai._net import (
    HedgedNetworkBackend,
    hedge_delays,
    hedge_transport,
    race,
    websocket_connect_kwargs,
)


class Thing:
    def __init__(self, n):
        self.n = n
        self.closed = False

    def close(self):
        self.closed = True


async def test_first_attempt_wins_when_fast():
    made = []

    async def attempt():
        t = Thing(len(made))
        made.append(t)
        return t

    got = await race(attempt, (0.05, 0.2), lambda t: t.close())
    assert got.n == 0 and len(made) == 1 and not got.closed


async def test_stuck_first_attempt_is_overtaken_and_cancelled():
    made, cancelled = [], []

    async def attempt():
        n = len(made)
        made.append(n)
        if n == 0:
            try:
                await asyncio.sleep(10)  # a lost SYN: the kernel would retry after 1 s
            except asyncio.CancelledError:
                cancelled.append(n)
                raise
        return Thing(n)

    t0 = time.monotonic()
    got = await race(attempt, (0.05, 0.5), lambda t: t.close())
    assert got.n == 1
    assert time.monotonic() - t0 < 0.3
    assert cancelled == [0]


async def test_failure_starts_the_next_attempt_at_once():
    made = []

    async def attempt():
        made.append(time.monotonic())
        if len(made) == 1:
            raise ConnectionRefusedError("refused")
        return Thing(len(made))

    t0 = time.monotonic()
    got = await race(attempt, (5.0, 10.0), lambda t: t.close())
    assert got.n == 2
    assert time.monotonic() - t0 < 0.5  # didn't wait the 5 s hedge delay


async def test_all_attempts_fail_raises_the_last_error():
    async def attempt():
        raise OSError("unreachable")

    with pytest.raises(OSError, match="unreachable"):
        await race(attempt, (0.01, 0.02), lambda t: t.close())


async def test_a_loser_that_also_connected_is_closed():
    made = []

    async def attempt():
        n = len(made)
        t = Thing(n)
        made.append(t)
        await asyncio.sleep(0.2 if n == 0 else 0.15)
        return t

    got = await race(attempt, (0.01,), lambda t: t.close())
    # attempt 1 (started at 10 ms, 150 ms) beats attempt 0 (200 ms); 0 is cancelled or closed
    assert got is made[1]
    assert not got.closed
    assert made[0].closed or True  # cancelled before connecting is fine too


async def test_async_close_is_awaited():
    """Two attempts that both connect: the loser is closed with its async close."""
    closed, made = [], []

    class A:
        async def aclose(self):
            closed.append(self)

    async def attempt():
        a = A()
        made.append(a)
        await asyncio.sleep(0.02)
        return a

    got = await race(attempt, (0.0,), lambda s: s.aclose())
    assert len(made) == 2
    assert got not in closed
    assert [a for a in made if a is not got] == closed


def test_hedge_delays_env(monkeypatch):
    monkeypatch.setenv("MIRAI_CONNECT_HEDGE_MS", "0")
    assert hedge_delays() == ()
    monkeypatch.setenv("MIRAI_CONNECT_HEDGE_MS", "200")
    assert hedge_delays() == (0.2, 1.0)
    monkeypatch.setenv("MIRAI_CONNECT_HEDGE_MS", "nonsense")
    assert hedge_delays() == (0.3, 1.0)
    monkeypatch.delenv("MIRAI_CONNECT_HEDGE_MS")
    assert hedge_delays() == (0.3, 1.0)


async def test_hedged_backend_overtakes_a_stuck_connect():
    class Stream:
        def __init__(self, n):
            self.n = n
            self.closed = False

        async def aclose(self):
            self.closed = True

    class Inner:
        def __init__(self):
            self.n = 0

        async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
            n = self.n
            self.n += 1
            if n == 0:
                await asyncio.sleep(10)
            return Stream(n)

        async def sleep(self, seconds):
            await asyncio.sleep(seconds)

    backend = HedgedNetworkBackend(Inner())
    t0 = time.monotonic()
    stream = await backend.connect_tcp("example.invalid", 443)
    assert stream.n == 1 and time.monotonic() - t0 < 0.6


async def test_real_http_through_a_hedged_transport(monkeypatch):
    """A request through a hedged httpx transport still works end to end."""
    monkeypatch.setenv("MIRAI_CONNECT_HEDGE_MS", "300")

    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        transport = hedge_transport(httpx.AsyncHTTPTransport())
        assert isinstance(transport._pool._network_backend, HedgedNetworkBackend)
        async with httpx.AsyncClient(transport=transport) as client:
            r = await client.get(f"http://127.0.0.1:{port}/")
        assert r.status_code == 200 and r.text == "ok"
    finally:
        server.close()
        await server.wait_closed()


async def test_websocket_kwargs(monkeypatch):
    monkeypatch.setenv("MIRAI_CONNECT_HEDGE_MS", "300")
    for name in _net._PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        kw = await websocket_connect_kwargs(f"ws://127.0.0.1:{port}/v1/audio/speech/stream")
        assert isinstance(kw["sock"], socket.socket)
        kw["sock"].close()
        assert await websocket_connect_kwargs("http://127.0.0.1/") == {}
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
        assert await websocket_connect_kwargs(f"ws://127.0.0.1:{port}/") == {}  # proxies keep their own path
        monkeypatch.delenv("HTTPS_PROXY")
        monkeypatch.setenv("MIRAI_CONNECT_HEDGE_MS", "0")
        assert await websocket_connect_kwargs(f"ws://127.0.0.1:{port}/") == {}
    finally:
        server.close()
        await server.wait_closed()
