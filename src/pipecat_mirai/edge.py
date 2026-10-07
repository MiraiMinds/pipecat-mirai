#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Streaming TTS from Mirai's edge, next to the speech GPUs.

Mirai serves the streaming TTS socket from an edge host beside its GPUs as
well as through its API gateway; the edge reaches the first audio byte in
about half the time. Before a socket opens, :class:`EdgeRoute` asks the
gateway for a single-use token (``POST /v2/tts/stream/tokens`` with the API
key). When the answer names an edge (``edge_url``), the socket opens there
with the token in its ``Authorization`` header, and speaks the same protocol.
The API key never goes to the edge, and a token is used for one socket only.

Anything that goes wrong on the way leaves the socket to the gateway, exactly
as before: no edge offered, a gateway without the token route (404/405), a
failed token request, an edge that refuses the connection or doesn't send
``session.ready`` within ``EDGE_READY_TIMEOUT``. A failure also keeps sockets
off the edge for ``EDGE_BACKOFF_SECS`` (doubling, up to
``EDGE_MAX_BACKOFF_SECS``), so a dead edge doesn't cost every call a timeout;
it is logged as a warning once per process. While the edge isn't known to
work, one socket at a time tries it and the others wait for its answer, so a
burst of sockets doesn't send a burst of token requests to a gateway that has
no edge to offer.

``MIRAI_TTS_EDGE=off`` turns the automatic edge off for the whole process.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from ._net import websocket_connect_kwargs
from .pool import shared_http_client

TOKEN_PATH = "/v2/tts/stream/tokens"
# The token request and, separately, the edge's connect + handshake + session.ready.
TOKEN_TIMEOUT = 3.0
EDGE_READY_TIMEOUT = 3.0
# After a failure, sockets stay on the gateway this long, doubling up to the maximum.
EDGE_BACKOFF_SECS = 60.0
EDGE_MAX_BACKOFF_SECS = 960.0
# The gateway offered no edge (or has no token route): ask again after this long.
NO_EDGE_RECHECK_SECS = 60.0
# The token request was rate limited (429) and gave no Retry-After.
BUSY_RECHECK_SECS = 2.0

_OFF = {"0", "off", "false", "no"}

# (token URL, mode) -> health; shared by every event loop in the process.
_lock = threading.Lock()
_health: dict[tuple[str, str], _Health] = {}
# (token URL, mode, id(loop)) -> the attempt other sockets are waiting on.
_probes: dict[tuple[str, str, int], asyncio.Future] = {}
_warned = False


@dataclass
class _Health:
    ok: bool = False  # the latest attempt reached the edge
    retry_at: float = 0.0  # monotonic; no attempt before this
    failures: int = 0  # consecutive failures, for the backoff


class _NoEdge(Exception):
    """The gateway has no edge to offer now."""


class _Busy(Exception):
    """The token request was rate limited."""

    def __init__(self, retry_after: float):
        super().__init__("rate limited")
        self.retry_after = retry_after


class _Failed(Exception):
    """The edge (or the token request) didn't work."""


@dataclass
class EdgeSocket:
    """A socket open on the edge that has sent ``session.ready``."""

    websocket: Any
    ready: dict
    url: str  # the edge's URL (never with the token)


Connect = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[Any]]


def edge_mode(edge: bool | str | None) -> str | None:
    """The ``edge`` option as ``None`` (off), ``"auto"`` or a ``ws(s)://`` URL to force."""
    if edge is None or edge is False:
        return None
    if edge is True or edge == "auto":
        if os.environ.get("MIRAI_TTS_EDGE", "").strip().lower() in _OFF:
            return None
        return "auto"
    if isinstance(edge, str) and edge.startswith(("ws://", "wss://")):
        return edge
    raise ValueError(f"edge must be 'auto', False or a ws:// or wss:// URL; got {edge!r}")


def token_url(stream_url: str) -> str:
    """``wss://host/v1/audio/speech/stream`` -> ``https://host/v2/tts/stream/tokens``."""
    u = urlparse(stream_url)
    scheme = {"wss": "https", "ws": "http"}.get(u.scheme, u.scheme)
    return f"{scheme}://{u.netloc}{TOKEN_PATH}"


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def _describe(exc: BaseException) -> str:
    """What went wrong, in a few words; never anything from the request (the token)."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return f"HTTP {status}"
    if isinstance(exc, _Failed):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "timed out"
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _reset():
    """Forget what is known about every edge (tests)."""
    global _warned
    with _lock:
        _health.clear()
        _probes.clear()
        _warned = False


class EdgeRoute:
    """How sockets for one streaming URL and API key reach the edge, when they can.

    ``url`` is the gateway's streaming endpoint, which also says where tokens
    come from (the same host). ``headers`` carry the API key, and go to the
    gateway only. ``mode`` is ``"auto"`` (the edge the gateway names) or an
    edge URL to use instead. ``http_base`` picks the shared HTTP client
    (``https://host/v1``), so token requests reuse its warm connections.
    """

    def __init__(self, url: str, headers: dict[str, str], mode: str, http_base: str):
        self.gateway_url = url
        self.mode = mode
        self.token_url = token_url(url)
        self._headers = dict(headers)
        self._http_base = http_base
        self._key = (self.token_url, mode)

    def _state(self) -> _Health:
        with _lock:
            return self._state_locked()

    def next_try_at(self) -> float:
        """When (monotonic) a socket may try the edge next; in the past if it may now."""
        state = self._state()
        loop_key = (*self._key, id(asyncio.get_running_loop()))
        if loop_key in _probes:
            return time.monotonic() + 0.5  # a try is under way
        return state.retry_at

    async def open(self, connect: Connect) -> EdgeSocket | None:
        """A socket on the edge, past ``session.ready``; ``None``: use the gateway.

        ``connect(url, headers, extra)`` opens a WebSocket (``extra``: connect
        kwargs such as a pre-connected socket).
        """
        loop = asyncio.get_running_loop()
        loop_key = (*self._key, id(loop))
        while True:
            state = self._state()
            if time.monotonic() < state.retry_at:
                return None
            if state.ok or state.failures == 0:
                # Known good, or never tried in this process: every socket tries
                # at once. A burst of calls at start-up must not queue behind one
                # probe (measured: 20 sockets spread over 3.3 s). Only after the
                # edge has failed does one socket probe while the rest wait.
                return await self._attempt(connect)
            with _lock:
                probe = _probes.get(loop_key)
                if probe is None:
                    probe = _probes[loop_key] = loop.create_future()
                    mine = True
                else:
                    mine = False
            if mine:
                try:
                    return await self._attempt(connect)
                finally:
                    with _lock:
                        if _probes.get(loop_key) is probe:
                            del _probes[loop_key]
                    if not probe.done():
                        probe.set_result(None)
            # Another socket is finding out whether the edge works: wait for its answer.
            await asyncio.wait({probe})

    async def _attempt(self, connect: Connect) -> EdgeSocket | None:
        try:
            token, edge_url = await self._mint()
        except _NoEdge as exc:
            self._no_edge(str(exc))
            return None
        except _Busy as exc:
            self._busy(exc.retry_after)
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed(f"the token request failed ({_describe(exc)})")
            return None
        url = edge_url if self.mode == "auto" else self.mode
        if not url:
            self._no_edge("the gateway offered no edge")
            return None
        try:
            websocket, ready = await self._connect(url, token, connect)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._failed(f"{_host(url)} was not ready within {EDGE_READY_TIMEOUT:g} s")
            return None
        except Exception as exc:
            self._failed(f"{_host(url)}: {_describe(exc)}")
            return None
        self._succeeded(url)
        return EdgeSocket(websocket, ready, url)

    async def _mint(self) -> tuple[str, str | None]:
        """A fresh single-use token from the gateway, and the edge it names."""
        shared = shared_http_client(self._http_base)
        owner = object()
        shared.hold(owner, warm=False, headers=self._headers)
        try:
            response = await shared.client.post(self.token_url, headers=self._headers, timeout=TOKEN_TIMEOUT)
        finally:
            shared.release(owner)
        status = response.status_code
        if status in (404, 405):
            raise _NoEdge(f"the gateway has no token route (HTTP {status})")
        if status == 429:
            try:
                retry_after = max(0.0, float(response.headers.get("retry-after")))
            except (TypeError, ValueError):
                retry_after = BUSY_RECHECK_SECS
            raise _Busy(retry_after)
        if not 200 <= status < 300:
            raise _Failed(f"HTTP {status}")
        try:
            body = response.json()
        except ValueError:
            raise _Failed("the answer was not JSON") from None
        token = body.get("token") if isinstance(body, dict) else None
        if not isinstance(token, str) or not token:
            raise _Failed("the answer had no token")
        edge_url = body.get("edge_url")
        if not (isinstance(edge_url, str) and edge_url.startswith(("ws://", "wss://"))):
            edge_url = None
        return token, edge_url

    async def _connect(self, url: str, token: str, connect: Connect) -> tuple[Any, dict]:
        """Open ``url`` with ``token`` and wait for ``session.ready``, within the time limit."""
        websocket = None
        headers = {"Authorization": f"Bearer {token}"}

        async def attempt() -> dict:
            nonlocal websocket
            extra = await websocket_connect_kwargs(url)
            try:
                websocket = await connect(url, headers, extra)
            except BaseException:
                sock = extra.get("sock")
                if sock is not None:
                    sock.close()
                raise
            raw = await websocket.recv()
            try:
                event = json.loads(raw) if isinstance(raw, str) else None
            except ValueError:
                event = None
            if not isinstance(event, dict) or event.get("type") != "session.ready":
                kind = event.get("type") if isinstance(event, dict) else "a message that is not JSON"
                code = event.get("code") if isinstance(event, dict) else None
                raise _Failed(f"sent {kind}{f' ({code})' if code else ''} instead of session.ready")
            return event

        try:
            ready = await asyncio.wait_for(attempt(), EDGE_READY_TIMEOUT)
        except BaseException:
            if websocket is not None:
                await _close_quietly(websocket)
            raise
        return websocket, ready

    # ---- what the attempts found ----

    def _succeeded(self, url: str):
        with _lock:
            state = self._state_locked()
            recovered = state.failures > 0
            state.ok, state.failures, state.retry_at = True, 0, 0.0
        if recovered:
            logger.info(f"Mirai: the TTS edge {_host(url)} is reachable again; streaming from it")
        logger.debug(f"Mirai: TTS socket opened on the edge {_host(url)}")

    def _failed(self, reason: str):
        global _warned
        now = time.monotonic()
        with _lock:
            state = self._state_locked()
            state.ok = False
            if state.retry_at > now:
                # Another socket's failure has already started a backoff.
                backoff = None
            else:
                state.failures += 1
                backoff = min(EDGE_MAX_BACKOFF_SECS, EDGE_BACKOFF_SECS * 2 ** (state.failures - 1))
                state.retry_at = now + backoff
            warn, _warned = not _warned, True
        if backoff is None:
            logger.debug(f"Mirai: TTS edge unavailable: {reason}")
            return
        message = (
            f"Mirai: the TTS edge is unavailable: {reason}. Streaming through {_host(self.gateway_url)} "
            f"instead, as before; trying the edge again in {backoff:g} s."
        )
        (logger.warning if warn else logger.debug)(message)

    def _no_edge(self, reason: str):
        with _lock:
            state = self._state_locked()
            state.ok, state.failures = False, 0
            state.retry_at = time.monotonic() + NO_EDGE_RECHECK_SECS
        logger.debug(f"Mirai: no TTS edge ({reason}); streaming through {_host(self.gateway_url)}")

    def _busy(self, retry_after: float):
        with _lock:
            state = self._state_locked()
            state.retry_at = max(state.retry_at, time.monotonic() + retry_after)
        logger.debug("Mirai: the TTS edge token request was rate limited; streaming through the gateway")

    def _state_locked(self) -> _Health:
        state = _health.get(self._key)
        if state is None:
            state = _health[self._key] = _Health()
        return state


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception as exc:
        logger.debug(f"error closing a Mirai socket: {exc!r}")
