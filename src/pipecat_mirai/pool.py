#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Connections to Mirai shared by every pipeline in a process, kept warm on their own.

A new connection to Mirai costs a TCP and TLS handshake before the first byte
of audio: a few hundred milliseconds from India, and a second or more when a
SYN is lost and retransmitted. A pipeline pays it once and reuses the
connection after that. But calls that start together (a load test, a burst
of real calls) each pay it on their first sentence, the greeting, unless
connections are already open when they start. This module keeps them open:

- HTTP: every :class:`~pipecat_mirai.MiraiTTSService` in an event loop shares
  one client per base URL. Its connections are tracked one by one, so the
  pool knows exactly which are open and idle, refreshes each before Mirai's
  75 s idle timeout, and opens more when fewer are warm than calls may need.
- WebSocket: authenticated sockets wait at ``session.ready``;
  :class:`~pipecat_mirai.MiraiWebsocketTTSService` takes one when its
  pipeline starts instead of connecting, and the pool opens a replacement.

Both start on their own when the first service of their kind starts, and keep
``max(AUTO, recent peak of concurrent calls + HEADROOM)`` warm from then on.
:func:`~pipecat_mirai.prewarm` starts them before any call does.

Every warm-up request counts against the API key's request rate limit (10
requests a second with bursts of 20 by default), as calls do. So warm-ups are
paced, at most ``WARM_RATE`` a second per API key and event loop, and pause
when Mirai answers 429. A burst of greetings always has most of the budget.

Everything belongs to one event loop: nothing opened in one loop is used from
another (asyncio connections can't be), and two loops get separate pools.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from ._net import hedge_transport, websocket_connect_kwargs

# ---- HTTP ----
# Mirai's gateway (nginx) closes an HTTP connection after 75 s idle. The client
# drops one at 70 s, so a request is never sent on a connection being closed.
HTTP_KEEPALIVE_EXPIRY = 70.0
# A connection idle this long is refreshed (one GET /v1/models)...
HTTP_REFRESH_AFTER = 55.0
# ...and one idle this long no longer counts as warm.
HTTP_WARM_WINDOW = 65.0
# How long a shared client outlives its last service when nothing keeps it warm.
HTTP_LINGER_SECS = HTTP_KEEPALIVE_EXPIRY
HTTP_TIMEOUT = httpx.Timeout(30.0, connect=10.0, read=15.0)
WARM_REQUEST_TIMEOUT = 10.0

# ---- WebSocket ----
# A waiting socket gets an empty session.update after this long without a
# message (or half Mirai's idle timeout, if shorter); Mirai closes a socket
# after 120 s without one.
WS_KEEPALIVE_SECS = 30.0
WS_OPEN_TIMEOUT = 10.0
WS_CLOSE_TIMEOUT = 2.0
# A waiting socket is replaced once this much of Mirai's maximum session
# length is left, so the call that takes it has room to run.
WS_SESSION_HEADROOM_SECS = 1200.0
WS_DEFAULT_MAX_AGE_SECS = 2400.0
WS_MAX_BACKOFF_SECS = 60.0

# ---- automatic sizing ----
# Kept warm from the first service on (MIRAI_WARM_CONNECTIONS /
# MIRAI_WARM_WEBSOCKETS override; 0 turns the automatic warm-up off).
AUTO_CONNECTIONS = 10
AUTO_WEBSOCKETS = 10
# More than the recent peak of concurrent calls, up to AUTO_MAX.
AUTO_HEADROOM = 2
AUTO_MAX = 32
# How long a peak is remembered.
PEAK_WINDOW_SECS = 600.0
# Warm-up requests (HTTP GETs and WebSocket handshakes) per second per API key
# and event loop, and the burst allowed after a quiet spell.
WARM_RATE = 2.0
WARM_BURST = 2.0
# Pause after Mirai answers a warm-up with 429 and no Retry-After.
WARM_PAUSE_SECS = 5.0

_lock = threading.Lock()
_http: dict[tuple[str, int], SharedHTTPClient] = {}
_ws: dict[tuple[str, str, int], WebsocketPool] = {}
_budgets: dict[tuple[str, int], _WarmBudget] = {}


def _purge():
    """Forget pools whose event loop has closed; their connections went with it."""
    for registry in (_http, _ws, _budgets):
        for key, value in list(registry.items()):
            if value.loop.is_closed():
                registry.pop(key, None)


def _credential(headers: dict[str, str] | None) -> str:
    """Identify an API key without keeping it in a dictionary key."""
    return hashlib.sha256(((headers or {}).get("Authorization") or "").encode()).hexdigest()


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"pipecat-mirai: ignoring {name}={raw!r}; expected a whole number")
        return default


def _retry_after(response: Any, default: float) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after")))
    except (TypeError, ValueError, AttributeError):
        return default


def _grown(base: int, peak: int) -> int:
    """``base``, or more once more calls than that have run at once: the peak and some headroom."""
    return base if peak <= base else max(base, min(AUTO_MAX, peak + AUTO_HEADROOM))


class _Peak:
    """The highest number of concurrent holders seen in the last PEAK_WINDOW_SECS."""

    def __init__(self):
        self.value = 0
        self.at = 0.0

    def note(self, current: int):
        now = time.monotonic()
        if current >= self.value or now - self.at > PEAK_WINDOW_SECS:
            self.value, self.at = current, now

    def get(self, current: int) -> int:
        self.note(current)
        return self.value


class _WarmBudget:
    """Paces warm-up requests for one API key in one event loop (a token bucket)."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.tokens = WARM_BURST
        self.at = time.monotonic()
        self.paused_until = 0.0

    def take(self, *, force: bool = False) -> bool:
        """Spend a token; ``force`` spends one even if none is left (never while paused)."""
        now = time.monotonic()
        if now < self.paused_until:
            return False
        self.tokens = min(WARM_BURST, self.tokens + (now - self.at) * WARM_RATE)
        self.at = now
        if self.tokens < 1 and not force:
            return False
        self.tokens -= 1
        return True

    def next_at(self) -> float:
        """When a token will next be available (monotonic)."""
        need = max(0.0, 1 - self.tokens) / WARM_RATE
        return max(self.paused_until, self.at + need)

    def pause(self, secs: float):
        self.paused_until = max(self.paused_until, time.monotonic() + secs)


def _budget(headers: dict[str, str] | None, loop: asyncio.AbstractEventLoop) -> _WarmBudget:
    key = (_credential(headers), id(loop))
    with _lock:
        budget = _budgets.get(key)
        if budget is None or budget.loop is not loop:
            budget = _budgets[key] = _WarmBudget(loop)
    return budget


def _ssl_context() -> ssl.SSLContext:
    """One TLS context for every connection (building one reads the CA bundle: ~30 ms)."""
    create = getattr(httpx, "create_ssl_context", None)
    if create is not None:
        return create()
    import certifi  # httpx < 0.28

    return ssl.create_default_context(cafile=certifi.where())


# ------------------------------------------------------------------------------------ HTTP


class _Slot:
    """One HTTP connection: a transport that never opens a second one."""

    __slots__ = ("transport", "busy", "opened", "last_used", "pinging")

    def __init__(self, transport: httpx.AsyncHTTPTransport):
        self.transport = transport
        self.busy = False  # a request is using it
        self.opened = False  # its last request completed: the connection is open
        self.last_used = 0.0
        self.pinging = False  # the busy request is a warm-up

    def warm(self, now: float) -> bool:
        return self.opened and not self.busy and now - self.last_used < HTTP_WARM_WINDOW


class _ReleasingStream(httpx.AsyncByteStream):
    """A response body that hands its connection back when it is closed."""

    def __init__(self, inner, release):
        self._inner = inner
        self._release = release
        self._complete = False
        self._closed = False

    async def __aiter__(self):
        async for chunk in self._inner:
            yield chunk
        self._complete = True

    async def aclose(self):
        if self._closed:
            return
        self._closed = True
        try:
            await self._inner.aclose()
        finally:
            # A body closed before its end takes the connection with it.
            self._release(self._complete)


class _SlotTransport(httpx.AsyncBaseTransport):
    """Sends each request on a connection of its own choosing.

    httpx's own pool hands a request the first idle connection in its list,
    so traffic keeps reusing the same few and the rest quietly expire, and
    nothing outside it can say which connection a request will use. Here each
    connection is a single-connection transport (a slot): a request gets the
    idle warm slot that has waited longest, which spreads calls across every
    open connection, and a warm-up can be aimed at exactly the connection that
    needs it.
    """

    def __init__(self):
        self.slots: list[_Slot] = []
        self._ssl: ssl.SSLContext | None = None

    def new_slot(self) -> _Slot:
        if self._ssl is None:
            self._ssl = _ssl_context()
        transport = hedge_transport(
            httpx.AsyncHTTPTransport(
                verify=self._ssl,
                limits=httpx.Limits(
                    max_connections=1, max_keepalive_connections=1, keepalive_expiry=HTTP_KEEPALIVE_EXPIRY
                ),
            )
        )
        slot = _Slot(transport)
        self.slots.append(slot)
        return slot

    def pick(self) -> _Slot:
        """The slot for a request: the longest-idle warm one, else any idle one, else a new one."""
        now = time.monotonic()
        idle = [s for s in self.slots if not s.busy]
        warm = [s for s in idle if s.warm(now)]
        if warm:
            return min(warm, key=lambda s: s.last_used)
        if idle:
            return idle[0]  # it will connect
        return self.new_slot()

    def counts(self) -> tuple[int, int]:
        """(warm idle, busy) connections."""
        now = time.monotonic()
        return sum(s.warm(now) for s in self.slots), sum(s.busy for s in self.slots)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        slot = self.pick()
        slot.busy = True
        return await self.send_on(slot, request)

    async def send_on(self, slot: _Slot, request: httpx.Request) -> httpx.Response:
        """Send ``request`` on ``slot``, which the caller has marked busy."""
        try:
            response = await slot.transport.handle_async_request(request)
        except BaseException:
            slot.busy = slot.pinging = slot.opened = False
            slot.last_used = time.monotonic()
            raise

        def release(complete: bool):
            slot.busy = slot.pinging = False
            slot.opened = complete
            slot.last_used = time.monotonic()

        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            stream=_ReleasingStream(response.stream, release),
            extensions=response.extensions,
        )

    async def prune(self, keep: int):
        """Close idle slots beyond ``keep``, the coldest first."""
        now = time.monotonic()
        extra = len(self.slots) - keep
        if extra <= 0:
            return
        idle = sorted((s for s in self.slots if not s.busy), key=lambda s: (s.warm(now), s.last_used))
        for slot in idle[:extra]:
            self.slots.remove(slot)
            with contextlib.suppress(Exception):
                await slot.transport.aclose()

    async def aclose(self):
        slots, self.slots = self.slots, []
        for slot in slots:
            with contextlib.suppress(Exception):
                await slot.transport.aclose()


class SharedHTTPClient:
    """One ``httpx.AsyncClient`` for a base URL, shared within one event loop.

    Services hold it while their pipeline runs. From the first holder that
    wants warm connections (or :func:`prewarm`) on, it keeps ``target()``
    connections open: it opens the missing ones and refreshes idle ones
    before Mirai's idle timeout, paced by the API key's warm-up budget. Left
    with no holders and nothing to keep warm, it closes after
    ``HTTP_LINGER_SECS``.
    """

    def __init__(self, base_url: str, loop: asyncio.AbstractEventLoop):
        self.base_url = base_url
        self.models_url = base_url + "/models"
        self.loop = loop
        self.transport = _SlotTransport()
        self.client = httpx.AsyncClient(transport=self.transport, timeout=HTTP_TIMEOUT)
        self.headers: dict[str, str] | None = None  # credentials for warm-up requests
        self.closed = False
        # prewarm()'s number (None: the automatic one), and whether a service asked for warmth.
        self.floor: int | None = None
        self.auto = False
        self.peak = _Peak()
        self.warmups = 0  # warm-up requests sent, over the client's life
        self.last_error: str | None = None
        self._holders: dict[int, bool] = {}  # id(service) -> wants warm connections
        self._keeper: asyncio.Task | None = None
        self._linger: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._changed = asyncio.Event()
        self._tasks: set[asyncio.Task] = set()
        # prewarm() opens its connections at once rather than paced, until then.
        self._unpaced_until = 0.0

    # ---- state ----

    @property
    def holders(self) -> int:
        """Services using the client now."""
        return len(self._holders)

    def connection_counts(self) -> tuple[int, int]:
        """(warm idle, busy) connections."""
        return self.transport.counts()

    def target(self) -> int:
        """Connections to keep open now."""
        if self.floor is None and not self.auto:
            return 0
        if self.floor is not None:
            base = self.floor
        else:
            base = _env_int("MIRAI_WARM_CONNECTIONS", AUTO_CONNECTIONS)
            if base == 0:
                return 0  # the automatic warm-up is turned off
        return _grown(base, self.peak.get(self.holders)) if self.auto else base

    # ---- holders ----

    def hold(self, owner: object, *, warm: bool, headers: dict[str, str]):
        """``owner`` uses the client until :meth:`release`; ``warm`` turns on the automatic warm-up."""
        self._holders[id(owner)] = warm
        self.headers = headers
        if warm:
            self.auto = True
        self.peak.note(self.holders)
        self._update()

    def release(self, owner: object):
        if self._holders.pop(id(owner), None) is not None:
            self._update()

    def set_floor(self, connections: int, headers: dict[str, str]):
        """prewarm(): keep at least ``connections`` open from now on, opening them at once."""
        self.floor = connections
        self.headers = headers
        self._unpaced_until = time.monotonic() + WARM_REQUEST_TIMEOUT
        self._update()

    def nudge(self):
        """Look at the connections again now (one was just closed, say)."""
        self._wake.set()

    def _update(self):
        """Start the keeper or the closing timer to match the holders."""
        if self.closed:
            return
        target = self.target()
        if target or self.transport.slots:
            if self._keeper is None or self._keeper.done():
                self._keeper = self.loop.create_task(self._keep(), name="mirai-http-pool")
            self._wake.set()
        if self._holders or target:
            if self._linger is not None:
                linger, self._linger = self._linger, None
                linger.cancel()
        elif self._linger is None:
            self._linger = self.loop.create_task(self._close_later(), name="mirai-http-linger")

    # ---- warming ----

    async def wait_warm(self, n: int, timeout: float) -> int:
        """Wait until ``n`` connections are warm, or ``timeout``; returns how many are."""
        deadline = time.monotonic() + timeout
        while not self.closed:
            if self.connection_counts()[0] >= n:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), remaining)
        return self.connection_counts()[0]

    async def _warm(self, slot: _Slot):
        """One authenticated ``GET /v1/models`` on ``slot`` (marked busy by the caller)."""
        self.warmups += 1
        budget = _budget(self.headers, self.loop)
        try:
            request = self.client.build_request(
                "GET", self.models_url, headers=self.headers or {}, timeout=WARM_REQUEST_TIMEOUT
            )
            response = await self.transport.send_on(slot, request)
            try:
                await response.aread()
            finally:
                await response.aclose()
        except Exception as exc:
            self.last_error = f"GET {self.models_url}: {exc!r}"
            logger.debug(f"Mirai: warm-up request failed: {self.last_error}")
            budget.pause(WARM_PAUSE_SECS)
            return
        finally:
            self._changed.set()
            self._wake.set()
        status = response.status_code
        if status == 429:
            budget.pause(_retry_after(response, WARM_PAUSE_SECS))
        elif status in (401, 403):
            self.last_error = f"Mirai rejected the API key (HTTP {status})"
            logger.warning(f"Mirai: {self.last_error}; check MIRAI_API_KEY. Not warming connections.")
            budget.pause(300.0)
        else:
            self.last_error = None

    async def _keep(self):
        """Open missing connections, refresh idle ones, close surplus ones."""
        me = asyncio.current_task()
        try:
            while not self.closed:
                target = self.target()
                now = time.monotonic()
                budget = _budget(self.headers, self.loop)
                force = now < self._unpaced_until
                busy = sum(s.busy for s in self.transport.slots)
                # The connections worth keeping: the most recently used open ones.
                open_idle = sorted(
                    (
                        s
                        for s in self.transport.slots
                        if not s.busy and s.opened and now - s.last_used < HTTP_KEEPALIVE_EXPIRY
                    ),
                    key=lambda s: s.last_used,
                    reverse=True,
                )
                next_due = now + (1.0 if target else 30.0)
                # Refresh first: keeping a connection is cheaper than opening one.
                for slot in sorted(open_idle[: max(0, target - busy)], key=lambda s: s.last_used):
                    if now - slot.last_used < HTTP_REFRESH_AFTER:
                        next_due = min(next_due, slot.last_used + HTTP_REFRESH_AFTER)
                        continue
                    if not budget.take(force=force):
                        next_due = min(next_due, budget.next_at())
                        break
                    slot.busy = slot.pinging = True
                    self._spawn(self._warm(slot))
                warm, busy = self.connection_counts()
                missing = target - warm - busy
                while missing > 0 and self.headers:
                    if not budget.take(force=force):
                        next_due = min(next_due, budget.next_at())
                        break
                    now = time.monotonic()
                    cold = [s for s in self.transport.slots if not s.busy and not s.warm(now)]
                    slot = cold[0] if cold else self.transport.new_slot()
                    slot.busy = slot.pinging = True
                    self._spawn(self._warm(slot))
                    missing -= 1
                await self.transport.prune(keep=max(target, busy) + AUTO_HEADROOM)
                if not target and not self.transport.slots:
                    self._keeper = None
                    return
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), max(0.05, next_due - time.monotonic()))
        except asyncio.CancelledError:
            if self._keeper is me:  # not cancelled by us: the event loop is shutting down
                await self.aclose()
            raise

    def _spawn(self, coro):
        task = self.loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _close_later(self):
        me = asyncio.current_task()
        try:
            await asyncio.sleep(HTTP_LINGER_SECS)
        except asyncio.CancelledError:
            if self._linger is me:  # not cancelled by us: the event loop is shutting down
                await self.aclose()
            raise
        if self._linger is me:
            self._linger = None
            await self.aclose()

    async def aclose(self):
        """Close the client and forget it. A service still holding it gets a new one."""
        if self.closed:
            return
        self.closed = True
        with _lock:
            for key, value in list(_http.items()):
                if value is self:
                    del _http[key]
        me = asyncio.current_task()
        tasks = [t for t in (self._keeper, self._linger, *self._tasks) if t is not None and t is not me]
        self._keeper = self._linger = None
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.client.aclose()
        self._changed.set()


def shared_http_client(base_url: str, *, create: bool = True) -> SharedHTTPClient | None:
    """The running event loop's shared client for ``base_url`` (``.../v1``, no trailing slash)."""
    loop = asyncio.get_running_loop()
    key = (base_url, id(loop))
    with _lock:
        _purge()
        shared = _http.get(key)
        if shared is None or shared.closed or shared.loop is not loop:
            if not create:
                return None
            shared = _http[key] = SharedHTTPClient(base_url, loop)
    return shared


# ------------------------------------------------------------------------------------ WebSocket


@dataclass(eq=False)
class PooledWebsocket:
    """An open, authenticated socket that has had ``session.ready``."""

    websocket: Any
    session_id: str | None = None
    idle_timeout_secs: float | None = None
    max_session_secs: float | None = None
    opened_at: float = field(default_factory=time.monotonic)
    last_sent: float = field(default_factory=time.monotonic)
    reader: asyncio.Task | None = None

    @property
    def is_open(self) -> bool:
        return self.websocket.state is State.OPEN

    def note(self, event: dict):
        """Take the session id and limits from ``session.ready`` / ``session.updated``."""
        self.session_id = event.get("session_id") or self.session_id
        limits = event.get("limits") or {}
        idle = limits.get("idle_timeout_secs")
        if isinstance(idle, int | float) and idle > 0:
            self.idle_timeout_secs = float(idle)
        longest = limits.get("max_session_secs")
        if isinstance(longest, int | float) and longest > 0:
            self.max_session_secs = float(longest)

    @property
    def keepalive_secs(self) -> float:
        if self.idle_timeout_secs:
            return min(WS_KEEPALIVE_SECS, self.idle_timeout_secs / 2)
        return WS_KEEPALIVE_SECS

    @property
    def max_age_secs(self) -> float:
        if self.max_session_secs:
            return max(self.max_session_secs / 2, self.max_session_secs - WS_SESSION_HEADROOM_SECS)
        return WS_DEFAULT_MAX_AGE_SECS


class WebsocketPool:
    """Sockets to one URL with one API key, waiting for pipelines in one event loop.

    ``target()`` sockets are kept open, each waiting at ``session.ready`` with
    an empty ``session.update`` before Mirai's idle timeout. :meth:`take`
    hands one to a single pipeline for good and wakes the pool to open a
    replacement; a socket never comes back.
    """

    def __init__(self, url: str, headers: dict[str, str], loop: asyncio.AbstractEventLoop, max_size: int):
        self.url = url
        self.headers = dict(headers)
        self.loop = loop
        self.max_size = max_size
        self.floor: int | None = None
        self.auto = False
        self.peak = _Peak()
        self.ready: list[PooledWebsocket] = []
        self.opening = 0
        self.closed = False
        self.opened = 0  # sockets opened, over the pool's life
        self.handed_out = 0
        self.last_error: str | None = None
        self._holders: set[int] = set()
        self._failures = 0
        self._retry_at = 0.0
        self._wake = asyncio.Event()
        self._changed = asyncio.Event()
        self._maintainer: asyncio.Task | None = None
        self._openers: set[asyncio.Task] = set()
        self._closers: set[asyncio.Task] = set()
        self._unpaced_until = 0.0

    @property
    def holders(self) -> int:
        return len(self._holders)

    def target(self) -> int:
        if self.floor is None and not self.auto:
            return 0
        if self.floor is not None:
            base = self.floor
        else:
            base = _env_int("MIRAI_WARM_WEBSOCKETS", AUTO_WEBSOCKETS)
            if base == 0:
                return 0
        return _grown(base, self.peak.get(self.holders)) if self.auto else base

    def hold(self, owner: object):
        """A pipeline for this URL and key started: keep sockets ready from now on."""
        self._holders.add(id(owner))
        self.auto = True
        self.peak.note(self.holders)
        self._kick()

    def release(self, owner: object):
        self._holders.discard(id(owner))

    def set_floor(self, sockets: int):
        """prewarm(): keep at least ``sockets`` waiting from now on, opening them at once."""
        self.floor = sockets
        self._unpaced_until = time.monotonic() + WS_OPEN_TIMEOUT
        self._kick()

    def _kick(self):
        if self.closed:
            return
        if self.target() and (self._maintainer is None or self._maintainer.done()):
            self._maintainer = self.loop.create_task(self._maintain(), name="mirai-ws-pool")
        self._wake.set()

    async def take(self) -> PooledWebsocket | None:
        """Hand a waiting socket to the caller, for good; ``None`` if none is waiting."""
        while self.ready and not self.closed:
            pooled = self.ready.pop()  # the newest: the most of its session left
            self._wake.set()  # open a replacement
            try:
                reader, pooled.reader = pooled.reader, None
                if reader is not None and not reader.done():
                    reader.cancel()
                    # Cancelling a websockets recv() loses nothing: whatever the
                    # reader hadn't read is the new owner's to read.
                    await asyncio.wait([reader])
            except BaseException:
                self._spawn(self._closers, _close_quietly(pooled.websocket))
                raise
            if pooled.is_open:
                self.handed_out += 1
                return pooled
            await _close_quietly(pooled.websocket)
        return None

    async def wait_ready(self, n: int, timeout: float) -> int:
        """Wait until ``n`` sockets are waiting, or opening them has failed, or ``timeout``."""
        deadline = time.monotonic() + timeout
        while len(self.ready) < n and not self.closed:
            if self._failures and not self.opening:
                break  # the last attempts failed; the pool keeps trying in the background
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            self._changed.clear()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._changed.wait(), remaining)
        return len(self.ready)

    def _spawn(self, tasks: set[asyncio.Task], coro) -> asyncio.Task:
        task = self.loop.create_task(coro)
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return task

    def _drop(self, pooled: PooledWebsocket):
        if pooled in self.ready:
            self.ready.remove(pooled)
        reader, pooled.reader = pooled.reader, None
        if reader is not None and reader is not asyncio.current_task():
            reader.cancel()
        self._spawn(self._closers, _close_quietly(pooled.websocket))
        self._wake.set()
        self._changed.set()

    async def _read(self, pooled: PooledWebsocket):
        """Read a waiting socket: note session events, notice it closing."""
        try:
            async for message in pooled.websocket:
                if isinstance(message, bytes):
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    continue
                if not isinstance(event, dict):
                    continue
                kind = event.get("type")
                if kind in ("session.ready", "session.updated"):
                    pooled.note(event)
                elif kind in ("session.closed", "error"):
                    logger.debug(f"Mirai: waiting socket {pooled.session_id}: {kind} {event}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug(f"Mirai: waiting socket {pooled.session_id} failed: {exc!r}")
        logger.debug(f"Mirai: waiting socket {pooled.session_id} closed; replacing it")
        self._drop(pooled)

    async def _open_one(self):
        """Open a socket and wait for session.ready (``opening`` was counted by the caller)."""
        websocket = None
        try:
            extra = await asyncio.wait_for(websocket_connect_kwargs(self.url), WS_OPEN_TIMEOUT)
            websocket = await websocket_connect(
                self.url,
                additional_headers=self.headers,
                max_size=self.max_size,
                open_timeout=WS_OPEN_TIMEOUT,
                close_timeout=WS_CLOSE_TIMEOUT,
                **extra,
            )
            raw = await asyncio.wait_for(websocket.recv(), WS_OPEN_TIMEOUT)
            event = json.loads(raw) if isinstance(raw, str) else {}
            if not isinstance(event, dict) or event.get("type") != "session.ready":
                detail = event.get("message") if isinstance(event, dict) else None
                raise ConnectionError(f"expected session.ready, got {event.get('type')!r}: {detail}")
            pooled = PooledWebsocket(websocket)
            pooled.note(event)
            if self.closed or len(self.ready) >= self.target():
                await _close_quietly(websocket)
                return
            pooled.reader = self.loop.create_task(self._read(pooled), name="mirai-ws-waiting")
            self.ready.append(pooled)
            self.opened += 1
            self._failures = 0
            self.last_error = None
        except asyncio.CancelledError:
            if websocket is not None:
                await _close_quietly(websocket)
            raise
        except Exception as exc:
            self._failures += 1
            response = getattr(exc, "response", None)
            status = getattr(response, "status_code", None)
            backoff = min(WS_MAX_BACKOFF_SECS, 2.0 ** (self._failures - 1))
            if status == 429:
                backoff = max(backoff, _retry_after(response, WARM_PAUSE_SECS))
                _budget(self.headers, self.loop).pause(backoff)
            elif status in (401, 402, 403, 404, 426):
                backoff = WS_MAX_BACKOFF_SECS  # retrying soon won't change the answer
            self._retry_at = time.monotonic() + backoff
            self.last_error = f"HTTP {status}" if status else repr(exc)
            log = logger.warning if status in (401, 402, 403) or self._failures == 1 else logger.debug
            log(f"Mirai: could not open a waiting socket to {self.url}: {self.last_error}")
            if websocket is not None:
                await _close_quietly(websocket)
        finally:
            self.opening -= 1
            self._wake.set()
            self._changed.set()

    async def _maintain(self):
        me = asyncio.current_task()
        try:
            while not self.closed:
                target = self.target()
                now = time.monotonic()
                for pooled in list(self.ready):
                    if not pooled.is_open or now - pooled.opened_at >= pooled.max_age_secs:
                        self._drop(pooled)
                while len(self.ready) > target:
                    self._drop(self.ready[0])  # the oldest
                next_due = now + 3600.0
                for pooled in list(self.ready):
                    interval = pooled.keepalive_secs
                    if now - pooled.last_sent >= interval and pooled in self.ready:
                        try:
                            # Changes nothing; any message resets Mirai's idle clock.
                            await pooled.websocket.send(json.dumps({"type": "session.update"}))
                            pooled.last_sent = time.monotonic()
                        except Exception as exc:
                            logger.debug(f"Mirai: keepalive on a waiting socket failed: {exc!r}")
                            if pooled in self.ready:
                                self._drop(pooled)
                            continue
                    next_due = min(
                        next_due, pooled.last_sent + interval, pooled.opened_at + pooled.max_age_secs
                    )
                missing = target - len(self.ready) - self.opening
                if missing > 0:
                    budget = _budget(self.headers, self.loop)
                    if time.monotonic() < self._retry_at:
                        next_due = min(next_due, self._retry_at)
                    else:
                        force = time.monotonic() < self._unpaced_until
                        while missing > 0:
                            if not budget.take(force=force):
                                next_due = min(next_due, budget.next_at())
                                break
                            self.opening += 1
                            missing -= 1
                            self._spawn(self._openers, self._open_one())
                if not target and not self.ready and not self.opening:
                    self._maintainer = None
                    return
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), max(0.05, next_due - time.monotonic()))
        except asyncio.CancelledError:
            if self._maintainer is me:  # not cancelled by us: the event loop is shutting down
                await self.aclose()
            raise

    async def aclose(self):
        """Close every waiting socket and stop the pool."""
        if self.closed:
            return
        self.closed = True
        with _lock:
            for key, value in list(_ws.items()):
                if value is self:
                    del _ws[key]
        me = asyncio.current_task()
        tasks = [
            t
            for t in (self._maintainer, *self._openers, *(p.reader for p in self.ready))
            if t and t is not me
        ]
        self._maintainer = None
        for task in tasks:
            task.cancel()
        ready, self.ready = self.ready, []
        await asyncio.gather(
            *tasks, *(_close_quietly(p.websocket) for p in ready), *self._closers, return_exceptions=True
        )
        self._changed.set()


def websocket_pool(
    url: str, headers: dict[str, str], max_size: int, *, create: bool = True
) -> WebsocketPool | None:
    """The running event loop's pool for ``url`` and these credentials."""
    loop = asyncio.get_running_loop()
    key = (url, _credential(headers), id(loop))
    with _lock:
        _purge()
        pool = _ws.get(key)
        if pool is None or pool.closed or pool.loop is not loop:
            if not create:
                return None
            pool = _ws[key] = WebsocketPool(url, headers, loop, max_size)
    return pool


async def take_websocket(url: str, headers: dict[str, str]) -> PooledWebsocket | None:
    """A waiting socket for ``url`` with these credentials, if the running loop's pool has one."""
    pool = websocket_pool(url, headers, 0, create=False)
    if pool is None:
        return None
    return await pool.take()


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception as exc:
        logger.debug(f"error closing a Mirai socket: {exc!r}")


# ------------------------------------------------------------------------------------ public


def shared_connection_stats() -> dict:
    """What the running event loop's shared connections hold, for logs and load tests.

    Returns ``{"http": [...], "websocket": [...]}``: per shared HTTP client its
    ``base_url``, ``warm`` (open, idle) and ``busy`` connections, the
    ``target`` it keeps warm, the ``services`` using it and the ``warmups``
    sent so far; per WebSocket pool its ``url``, sockets ``ready`` and
    ``opening``, ``target``, ``services``, and the sockets ``opened`` and
    ``handed_out`` so far.
    """
    loop = asyncio.get_running_loop()
    with _lock:
        https = [s for s in _http.values() if s.loop is loop and not s.closed]
        pools = [p for p in _ws.values() if p.loop is loop and not p.closed]
    http = []
    for s in https:
        warm, busy = s.connection_counts()
        http.append(
            {
                "base_url": s.base_url,
                "warm": warm,
                "busy": busy,
                "target": s.target(),
                "services": s.holders,
                "warmups": s.warmups,
            }
        )
    websocket = [
        {
            "url": p.url,
            "ready": len(p.ready),
            "opening": p.opening,
            "target": p.target(),
            "services": p.holders,
            "opened": p.opened,
            "handed_out": p.handed_out,
        }
        for p in pools
    ]
    return {"http": http, "websocket": websocket}


async def close_shared_connections():
    """Close the running event loop's shared HTTP clients and WebSocket pools.

    Optional: they close themselves when the event loop ends. Call it on a
    graceful shutdown, or between tests. A service still running afterwards
    opens a new shared client on its next request.
    """
    loop = asyncio.get_running_loop()
    with _lock:
        pools = [p for p in (*_http.values(), *_ws.values()) if p.loop is loop]
    for pool in pools:
        await pool.aclose()
