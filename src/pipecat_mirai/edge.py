#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Streaming TTS and STT from Mirai's edge, next to the GPUs.

Mirai serves its streaming sockets (TTS ``/v1/audio/speech/stream``, STT
``/v1/audio/transcriptions/stream``) from an edge host beside its GPUs as well
as through its API gateway; the edge cuts the time to the first byte. A socket
for Mirai's own gateway opens straight on its edge with the API key in the
``Authorization`` header, exactly as on the gateway: the edge checks the key
with the gateway itself, so there is no extra round trip before the first
byte. The gateway is still asked which edge it offers (``edge_url`` from
``POST /v2/tts/stream/tokens`` or ``POST /v2/stt/stream/tokens``), in the
background, and its answer is remembered for ``EDGE_URL_TTL`` seconds: when
the gateway stops offering the edge, sockets go back to the gateway. For any
other gateway the answer is waited for before the first socket (sockets
starting together share the one request).

Anything that goes wrong on the way leaves the socket to the gateway, exactly
as before: no edge offered, a gateway without the token route (404/405), an
edge that refuses the connection or doesn't send its first event
(``session.ready`` for TTS, ``session.begin`` for STT) within
``EDGE_READY_TIMEOUT``. A failure also keeps sockets off that service's edge
for ``EDGE_BACKOFF_SECS`` (doubling, up to ``EDGE_MAX_BACKOFF_SECS``), so a
dead edge doesn't cost every call a timeout; it is logged as a warning once
per process. After a failure, one socket at a time tries the edge again and
the others wait for its answer. TTS and STT keep separate health.

``MIRAI_TTS_EDGE=off`` (``MIRAI_STT_EDGE=off``) turns the automatic edge off
for that service in the whole process.
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
from .pool import PROTOCOLS, shared_http_client

# Each streaming service's token route, on the gateway that serves its socket.
TOKEN_PATHS = {"tts": "/v2/tts/stream/tokens", "stt": "/v2/stt/stream/tokens"}
TOKEN_PATH = TOKEN_PATHS["tts"]
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
# Services whose edge failure has been logged as a warning (once per process each).
_warned: set[str] = set()

# Mirai's own gateways and the edges next to their GPUs, per service. A socket
# to one of these gateways goes straight to its edge with the API key, while
# the gateway is asked in the background whether it still offers that edge.
KNOWN_EDGES = {
    ("sandbox.voice.miraiminds.co", "tts"): "wss://tts-edge.voice.miraiminds.co/v1/audio/speech/stream",
    (
        "sandbox.voice.miraiminds.co",
        "stt",
    ): "wss://stt-edge.voice.miraiminds.co/v1/audio/transcriptions/stream",
}
EDGE_URL_TTL = 600.0  # seconds a gateway's answer about its edge is reused
# token URL -> (edge URL or None, monotonic time to ask again), learned from a gateway.
_edge_urls: dict[str, tuple[str | None, float]] = {}
# (token URL, id(loop)) -> the question to the gateway that sockets share.
_asking: dict[tuple[str, int], asyncio.Task] = {}


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


class _Refused(Exception):
    """The gateway or the edge refused this key (401/402/403/429): not the edge's fault.

    The socket opens on the gateway, which gives the reason in its own words;
    other sockets keep using the edge.
    """


_KEY_REFUSALS = {401, 402, 403, 429}


@dataclass
class EdgeSocket:
    """A socket open on the edge that has sent its first event (``session.ready`` / ``session.begin``)."""

    websocket: Any
    ready: dict
    url: str  # the edge's URL (never with the key, and without the query)


Connect = Callable[[str, dict[str, str], dict[str, Any]], Awaitable[Any]]


def edge_mode(edge: bool | str | None, service: str = "tts") -> str | None:
    """The ``edge`` option as ``None`` (off), ``"auto"`` or a ``ws(s)://`` URL to force.

    ``MIRAI_TTS_EDGE=off`` (``MIRAI_STT_EDGE=off`` for ``service="stt"``) turns ``"auto"`` off.
    """
    if edge is None or edge is False:
        return None
    if edge is True or edge == "auto":
        if os.environ.get(f"MIRAI_{service.upper()}_EDGE", "").strip().lower() in _OFF:
            return None
        return "auto"
    if isinstance(edge, str) and edge.startswith(("ws://", "wss://")):
        return edge
    raise ValueError(f"edge must be 'auto', False or a ws:// or wss:// URL; got {edge!r}")


def token_url(stream_url: str, service: str = "tts") -> str:
    """``wss://host/v1/audio/speech/stream`` -> ``https://host/v2/tts/stream/tokens``.

    ``service="stt"`` gives ``https://host/v2/stt/stream/tokens``.
    """
    u = urlparse(stream_url)
    scheme = {"wss": "https", "ws": "http"}.get(u.scheme, u.scheme)
    return f"{scheme}://{u.netloc}{TOKEN_PATHS[service]}"


def _with_query(url: str, query: str) -> str:
    """``url`` with ``query`` appended (an STT socket's settings travel in its query)."""
    if not query:
        return url
    return f"{url}{'&' if '?' in url else '?'}{query}"


def _host(url: str) -> str:
    return urlparse(url).netloc or url


def _status(exc: BaseException) -> int | None:
    """The HTTP status of a refused request or WebSocket handshake, if that is what failed."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _describe(exc: BaseException) -> str:
    """What went wrong, in a few words; never anything from the request (the key)."""
    status = _status(exc)
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
    with _lock:
        _health.clear()
        _probes.clear()
        _edge_urls.clear()
        _asking.clear()
        _warned.clear()


class EdgeRoute:
    """How sockets for one streaming URL and API key reach the edge, when they can.

    ``url`` is the gateway's streaming endpoint, which also says where tokens
    come from (the same host); its query, if any (an STT socket's settings),
    goes to the edge too. ``headers`` carry the API key. ``mode`` is
    ``"auto"`` (the edge the gateway names) or an edge URL to use instead.
    ``http_base`` picks the shared HTTP client (``https://host/v1``), so
    token requests reuse its warm connections. ``service`` is ``"tts"`` or
    ``"stt"``: its token route, the first event its socket sends, and its own
    health and backoff.
    """

    def __init__(self, url: str, headers: dict[str, str], mode: str, http_base: str, service: str = "tts"):
        if service not in TOKEN_PATHS:
            raise ValueError(f"service must be one of {', '.join(TOKEN_PATHS)}; got {service!r}")
        self.gateway_url = url
        self.mode = mode
        self.service = service
        self.protocol = PROTOCOLS[service]
        self.token_url = token_url(url, service)
        self.query = urlparse(url).query
        self._headers = dict(headers)
        self._http_base = http_base
        self._key = (self.token_url, mode)
        self._label = service.upper()

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

    async def open(self, connect: Connect, *, refusals: str = "gateway") -> EdgeSocket | None:
        """A socket on the edge, past ``session.ready``; ``None``: use the gateway.

        ``connect(url, headers, extra)`` opens a WebSocket (``extra``: connect
        kwargs such as a pre-connected socket). ``refusals``: what happens when
        the edge refuses this key (401/402/403/429): ``"gateway"`` returns
        ``None`` so a call opens on the gateway, which gives the reason in its own
        words; ``"raise"`` raises the refusal, for a pool's warm-up, which must back
        off rather than spend the gateway's rate limit on a socket nobody asked for.
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
                return await self._attempt(connect, refusals)
            with _lock:
                probe = _probes.get(loop_key)
                if probe is None:
                    probe = _probes[loop_key] = loop.create_future()
                    mine = True
                else:
                    mine = False
            if mine:
                try:
                    return await self._attempt(connect, refusals)
                finally:
                    with _lock:
                        if _probes.get(loop_key) is probe:
                            del _probes[loop_key]
                    if not probe.done():
                        probe.set_result(None)
            # Another socket is finding out whether the edge works: wait for its answer.
            await asyncio.wait({probe})

    async def _attempt(self, connect: Connect, refusals: str = "gateway") -> EdgeSocket | None:
        try:
            url = await self._edge_url()
        except _NoEdge as exc:
            self._no_edge(str(exc))
            return None
        except _Busy as exc:
            self._busy(exc.retry_after)
            return None
        except _Refused:
            return None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failed(f"the gateway couldn't say which edge to use ({_describe(exc)})")
            return None
        if not url:
            self._no_edge("the gateway offered no edge")
            return None
        try:
            websocket, ready = await self._connect(url, connect)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            self._failed(f"{_host(url)} was not ready within {EDGE_READY_TIMEOUT:g} s")
            return None
        except Exception as exc:
            if _status(exc) in _KEY_REFUSALS:
                if refusals == "raise":
                    raise
                # The key, the wallet or a limit: the gateway says which, and the edge stays in use.
                logger.debug(
                    f"Mirai: the {self._label} edge refused this key ({_describe(exc)}); using the gateway"
                )
                return None
            self._failed(f"{_host(url)}: {_describe(exc)}")
            return None
        self._succeeded(url)
        return EdgeSocket(websocket, ready, url)

    async def _edge_url(self) -> str | None:
        """Where this socket goes: a forced URL, the edge the gateway offers, or Mirai's known edge.

        The gateway's answer is reused for ``EDGE_URL_TTL`` seconds. For Mirai's
        own gateways the socket doesn't wait for it: it goes to the known edge
        (or the last answer) while the gateway is asked in the background.
        """
        if self.mode != "auto":
            return self.mode
        with _lock:
            learned = _edge_urls.get(self.token_url)
        if learned is not None and learned[1] > time.monotonic():
            return learned[0]
        known = KNOWN_EDGES.get((urlparse(self.gateway_url).hostname or "", self.service))
        if known is None:
            return await asyncio.shield(self._ask())
        self._ask().add_done_callback(_consume)
        return learned[0] if learned is not None else known

    def _ask(self) -> asyncio.Task:
        """The question to the gateway about its edge; sockets asking together share it."""
        loop = asyncio.get_running_loop()
        key = (self.token_url, id(loop))
        with _lock:
            task = _asking.get(key)
            if task is None or task.done() or task.get_loop() is not loop:
                task = _asking[key] = loop.create_task(self._ask_gateway())
        return task

    async def _ask_gateway(self) -> str | None:
        """Ask the gateway which edge it offers, and remember the answer.

        No answer (the gateway is down, slow or refused this key) changes
        nothing that was known, and is asked again after ``NO_EDGE_RECHECK_SECS``.
        """
        try:
            edge_url = await self._request_edge_url()
        except _NoEdge:
            edge_url = None
        except BaseException:
            with _lock:
                learned = _edge_urls.get(self.token_url)
                if learned is not None:
                    _edge_urls[self.token_url] = (learned[0], time.monotonic() + NO_EDGE_RECHECK_SECS)
            raise
        with _lock:
            _edge_urls[self.token_url] = (edge_url, time.monotonic() + EDGE_URL_TTL)
        return edge_url

    async def _request_edge_url(self) -> str | None:
        """One request to the gateway's token route, for the ``edge_url`` in its answer."""
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
        if status in _KEY_REFUSALS:
            raise _Refused(f"HTTP {status}")
        if not 200 <= status < 300:
            raise _Failed(f"HTTP {status}")
        try:
            body = response.json()
        except ValueError:
            raise _Failed("the answer was not JSON") from None
        if not isinstance(body, dict):
            raise _Failed("the answer was not an object")
        edge_url = body.get("edge_url")
        if isinstance(edge_url, str) and edge_url.startswith(("ws://", "wss://")):
            return edge_url
        return None

    async def _connect(self, url: str, connect: Connect) -> tuple[Any, dict]:
        """Open ``url`` with the API key and wait for its first event, within the time limit.

        The edge checks the key with the gateway itself, so the client needs no
        token request of its own: one connection, and the first byte can go.
        """
        websocket = None
        headers = dict(self._headers)
        protocol = self.protocol
        full_url = _with_query(url, self.query)

        async def attempt() -> dict:
            nonlocal websocket
            extra = await websocket_connect_kwargs(full_url)
            try:
                websocket = await connect(full_url, headers, extra)
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
            if not isinstance(event, dict) or protocol.kind_of(event) != protocol.ready:
                kind = protocol.kind_of(event) if isinstance(event, dict) else "a message that is not JSON"
                code = _code(event) if isinstance(event, dict) else None
                raise _Failed(f"sent {kind}{f' ({code})' if code else ''} instead of {protocol.ready}")
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
            logger.info(f"Mirai: the {self._label} edge {_host(url)} is reachable again; streaming from it")
        logger.debug(f"Mirai: {self._label} socket opened on the edge {_host(url)}")

    def _failed(self, reason: str):
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
            warn = self.service not in _warned
            _warned.add(self.service)
        if backoff is None:
            logger.debug(f"Mirai: {self._label} edge unavailable: {reason}")
            return
        message = (
            f"Mirai: the {self._label} edge is unavailable: {reason}. Streaming through "
            f"{_host(self.gateway_url)} instead, as before; trying the edge again in {backoff:g} s."
        )
        (logger.warning if warn else logger.debug)(message)

    def _no_edge(self, reason: str):
        with _lock:
            state = self._state_locked()
            state.ok, state.failures = False, 0
            state.retry_at = time.monotonic() + NO_EDGE_RECHECK_SECS
        logger.debug(f"Mirai: no {self._label} edge ({reason}); streaming through {_host(self.gateway_url)}")

    def _busy(self, retry_after: float):
        with _lock:
            state = self._state_locked()
            state.retry_at = max(state.retry_at, time.monotonic() + retry_after)
        logger.debug(
            f"Mirai: the {self._label} edge token request was rate limited; streaming through the gateway"
        )

    def _state_locked(self) -> _Health:
        state = _health.get(self._key)
        if state is None:
            state = _health[self._key] = _Health()
        return state


def _consume(task: asyncio.Task):
    """Retrieve a background question's outcome; nobody is waiting on it."""
    if not task.cancelled() and task.exception() is not None:
        logger.debug(f"Mirai: couldn't ask the gateway about its edge ({_describe(task.exception())})")


def _code(event: dict) -> Any:
    """An error event's code, flat (``code``) or nested (``error.code``)."""
    nested = event.get("error")
    if isinstance(nested, dict):
        return nested.get("code")
    return event.get("code")


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception as exc:
        logger.debug(f"error closing a Mirai socket: {exc!r}")
