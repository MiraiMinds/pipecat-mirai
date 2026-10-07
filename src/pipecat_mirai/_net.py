#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Hedged TCP connects.

A lost SYN costs a full retransmit timeout: about 1 s, then 3 s more. When a
load test starts many calls at once, a burst of new connections makes those
losses much more likely on some network paths. Measured from a European
server against Mirai's API, a burst of 20 simultaneous connects saw 18 of 60
take over 900 ms, against 1 of 3 for a single connect.

So a connect that hasn't completed after a short delay starts a second
attempt, and a third later; the first to complete is used and the others are
closed. Healthy paths connect well inside the delay, so this costs nothing
there. ``MIRAI_CONNECT_HEDGE_MS`` sets the delay (default 300); ``0`` turns
hedging off.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import socket
import time
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar
from urllib.parse import urlparse

import httpcore
from loguru import logger

T = TypeVar("T")

DEFAULT_HEDGE_MS = 300.0
_PROXY_ENV = ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy")


def hedge_delays() -> tuple[float, ...]:
    """Start times, after the first attempt, of the extra attempts (seconds)."""
    raw = os.environ.get("MIRAI_CONNECT_HEDGE_MS", "")
    try:
        ms = float(raw) if raw.strip() else DEFAULT_HEDGE_MS
    except ValueError:
        ms = DEFAULT_HEDGE_MS
    if ms <= 0:
        return ()
    first = ms / 1000.0
    return (first, max(1.0, 3 * first))


async def _close(value: Any, close: Callable[[Any], Any]) -> None:
    try:
        result = close(value)
        if inspect.isawaitable(result):
            await result
    except Exception:  # noqa: BLE001 - a loser that won't close is not our problem
        pass


async def race(
    attempt: Callable[[], Awaitable[T]],
    delays: tuple[float, ...],
    close: Callable[[T], Any],
) -> T:
    """Run ``attempt`` now and again at each of ``delays`` (or at once after a failure).

    Returns the first successful result; every other result is closed with
    ``close`` and every unfinished attempt is cancelled. Raises the last error
    when all attempts fail.
    """
    started = time.monotonic()
    schedule = [0.0, *delays]
    tasks: list[asyncio.Task] = []
    errors: list[BaseException] = []
    winner: Any = None
    won = False

    def launch():
        tasks.append(asyncio.ensure_future(attempt()))

    launch()
    try:
        while True:
            running = [t for t in tasks if not t.done()]
            more = len(tasks) < len(schedule)
            if not running:
                if more:
                    launch()  # the previous attempt failed: don't wait for the timer
                    continue
                raise errors[-1] if errors else OSError("connect failed")
            timeout = None
            if more:
                timeout = max(0.0, started + schedule[len(tasks)] - time.monotonic())
            done, _ = await asyncio.wait(running, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
            if not done:
                launch()
                continue
            for t in done:
                if t.cancelled():
                    continue
                exc = t.exception()
                if exc is None:
                    winner, won = t.result(), True
                    if len(tasks) > 1:
                        logger.debug(f"Mirai: connect won by attempt {tasks.index(t) + 1} of {len(tasks)}")
                    return winner
                errors.append(exc)
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        for t in tasks:
            try:
                result = await t
            except BaseException:  # noqa: BLE001 - cancelled or failed attempts
                continue
            if not (won and result is winner):
                await _close(result, close)


class HedgedNetworkBackend(httpcore.AsyncNetworkBackend):
    """httpcore network backend whose TCP connects are hedged."""

    def __init__(self, inner: httpcore.AsyncNetworkBackend | None = None):
        if inner is None:
            from httpcore._backends.auto import AutoBackend

            inner = AutoBackend()
        self._inner = inner

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        delays = hedge_delays()

        def attempt():
            return self._inner.connect_tcp(
                host, port, timeout=timeout, local_address=local_address, socket_options=socket_options
            )

        if not delays:
            return await attempt()
        return await race(attempt, delays, lambda stream: stream.aclose())

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        return await self._inner.connect_unix_socket(path, timeout=timeout, socket_options=socket_options)

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def hedge_transport(transport: Any) -> Any:
    """Give an ``httpx.AsyncHTTPTransport`` hedged connects (no-op if its internals differ)."""
    pool = getattr(transport, "_pool", None)
    if pool is not None and hasattr(pool, "_network_backend") and hedge_delays():
        pool._network_backend = HedgedNetworkBackend()
    return transport


def _proxied() -> bool:
    return any(os.environ.get(name) for name in _PROXY_ENV)


async def hedged_socket(host: str, port: int) -> socket.socket:
    """A connected, non-blocking TCP socket to ``host:port``, connected with hedging."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    family, type_, proto, _, address = infos[0]

    async def attempt():
        sock = socket.socket(family, type_, proto)
        sock.setblocking(False)
        try:
            await loop.sock_connect(sock, address)
        except BaseException:
            sock.close()
            raise
        return sock

    return await race(attempt, hedge_delays(), lambda sock: sock.close())


async def websocket_connect_kwargs(url: str) -> dict[str, Any]:
    """Extra ``websockets`` connect kwargs: a hedged, pre-connected socket when it applies.

    Empty (connect as usual) when hedging is off, a proxy is configured, or the
    URL isn't ws/wss.
    """
    if not hedge_delays() or _proxied():
        return {}
    u = urlparse(url)
    if u.scheme not in ("ws", "wss") or not u.hostname:
        return {}
    port = u.port or (443 if u.scheme == "wss" else 80)
    return {"sock": await hedged_socket(u.hostname, port)}
