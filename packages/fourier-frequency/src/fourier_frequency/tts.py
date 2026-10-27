#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text to speech for any Python voice agent.

Two ways to Mirai's speech, one engine each:

- :class:`TTSStream`: one WebSocket for a whole call (the one to use). Text
  goes to Mirai as the agent produces it, Mirai cuts it into sentences,
  synthesises the next sentence while the current one streams, and sends the
  audio back on the same socket. The socket comes from Mirai's edge next to
  the speech GPUs when it can, usually ready from the process's pool, and is
  reopened if it drops, with the reply it cut short resent once. If the
  streaming endpoint can't be reached at all, the call speaks over HTTP.
- :class:`HttpTTS`: one streaming ``POST /v1/audio/speech`` per sentence, over
  kept-alive connections the process shares and keeps warm.

Audio is 16-bit mono PCM at ``sample_rate`` (or G.711 with ``encoding``), in
40 ms pieces after a short first-audio buffer, resampled locally only when
Mirai sends another rate. What happens is told to a :class:`TTSListener`; the
default one hands each reply's audio to its :class:`TTSTurn`, which is all a
plain asyncio orchestrator needs::

    async with TTSStream(voice="shruti", sample_rate=8000, encoding="mulaw") as tts:
        turn = tts.turn()
        await turn.send("नमस्ते! ")
        await turn.end()
        async for audio in turn:
            await phone.send(audio)

The WebSocket protocol, in short. Client: ``session.update`` (voice, model,
response_format, sample_rate, audio_transport), ``text`` {context_id, text,
flush?}, ``flush``, ``cancel``, ``close``. Server: ``session.ready`` /
``session.updated``, ``audio.start`` (the sentence, its rate and encoding),
binary audio frames, ``audio.done`` (chars billed), ``context.done`` (a flush
is fully spoken), ``context.cancelled`` (the cancel ack; nothing for that
context follows it), ``error`` and ``session.closed``. One sentence plays at a
time, so every binary frame belongs to the latest ``audio.start``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
import numpy as np
import soxr
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK, InvalidHandshake
from websockets.protocol import State

from fourier_frequency import errors
from fourier_frequency._net import websocket_connect_kwargs
from fourier_frequency.audio import lin2alaw, lin2ulaw
from fourier_frequency.edge import EdgeRoute, edge_mode
from fourier_frequency.errors import FrequencyError
from fourier_frequency.pool import (
    SharedHTTPClient,
    WebsocketPool,
    shared_http_client,
    take_websocket,
    websocket_pool,
)
from fourier_frequency.pool import _retry_after as _handshake_retry_after

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://sandbox.voice.miraiminds.co/v1"
DEFAULT_WEBSOCKET_URL = "wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream"
# The rate Mirai sends when the request doesn't name one, and the rate assumed
# when a response carries no X-Sample-Rate header.
SOURCE_SAMPLE_RATE = 48000
# PCM rates the server can produce on request (the ``sample_rate`` request field).
SERVER_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
VOICES = ("ashu", "neha", "shruti", "sameer")
DEFAULT_VOICE = "neha"
DEFAULT_MODEL = "mira-tts"
# What the audio handed out is encoded as: 16-bit PCM, or G.711 for phone media streams.
ENCODINGS = ("pcm_s16le", "mulaw", "alaw")
# Audio is handed out in pieces of this length, whatever size the network
# reads are. 40 ms is what phone media streams (and Pipecat's output
# transports) send per write.
FRAME_SECS = 0.04
# A sentence Mirai refuses with 429 (busy, or the key's request rate) is sent
# again after its Retry-After, for at most this long in all; past that a caller
# would sit through the silence, so it is reported.
RETRY_BUDGET_SECS = 3.0
# A 502/503/504, or a connection that fails before any response, is retried
# once after this pause.
FAST_RETRY_DELAY_SECS = 0.1
# A socket refused with 429 (the key's request rate, or the workspace's open-socket
# limit) is retried for this long, waiting Retry-After (at most CONNECT_RETRY_WAIT
# at a time), before the call reports it: the refusal is usually a moment's
# crowding, such as many workers starting at once.
CONNECT_RETRY_SECS = 5.0
CONNECT_RETRY_WAIT = 2.0
# A capacity error is retried once when Mirai asks for a wait no longer than
# this. Longer than that, a caller would sit through the silence; it is
# reported instead.
RETRY_AFTER_LIMIT_SECS = 5.0
# Ceiling on one server message. Audio frames are a few KB; this is headroom.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
# A dropped socket is reopened up to RECONNECT_ATTEMPTS times, waiting between
# failed attempts from RECONNECT_MIN_WAIT doubling to RECONNECT_MAX_WAIT (the
# schedule Pipecat's websocket services use). A socket that dies within
# QUICK_FAILURE_SECS of opening, QUICK_FAILURES times in a row, is given up on:
# the server accepts the handshake and then refuses the session.
RECONNECT_ATTEMPTS = 3
RECONNECT_MIN_WAIT = 4.0
RECONNECT_MAX_WAIT = 10.0
QUICK_FAILURE_SECS = 5.0
QUICK_FAILURES = 3
# How long a close waits for the server to acknowledge it.
CLOSE_TIMEOUT = 2.0

# Mirai's error codes, as error categories. An unknown code is SERVER.
_ERROR_CATEGORIES = {
    "unauthorized": errors.AUTHENTICATION,
    "forbidden": errors.AUTHORIZATION,
    "invalid_request": errors.INVALID_REQUEST,
    "model_not_found": errors.INVALID_REQUEST,
    "insufficient_balance": errors.QUOTA,
    "at_capacity": errors.RATE_LIMIT,
    "rate_limited": errors.RATE_LIMIT,
    "queue_full": errors.RATE_LIMIT,
    "too_many_contexts": errors.RATE_LIMIT,
    "session_closing": errors.CONNECTIVITY,
}
_HANDSHAKE_CATEGORIES = {
    401: errors.AUTHENTICATION,
    402: errors.QUOTA,
    403: errors.AUTHORIZATION,
    429: errors.RATE_LIMIT,
}
# What to do about each, in one line.
_ADVICE = {
    "unauthorized": "Check MIRAI_API_KEY.",
    "forbidden": "The API key is revoked or not allowed this request.",
    "insufficient_balance": "The workspace is out of credit. Top it up in the Mirai console.",
    "at_capacity": "Ask Mirai to raise this workspace's concurrency limit.",
    "rate_limited": "Ask Mirai to raise the API key's request rate limit.",
    "queue_full": "Ask Mirai to raise this workspace's concurrency limit.",
    401: "Check MIRAI_API_KEY.",
    402: "The workspace is out of credit. Top it up in the Mirai console.",
    403: "The API key is revoked or not allowed this request.",
    429: "Ask Mirai to raise the API key's request rate limit.",
}
# Where the HTTP fallback cuts streamed tokens into sentences, as Mirai does.
_SENTENCE_END = re.compile(r"[.?!।॥…][\"')\]]*\s|\n")
# Endpoints already reported as unavailable (the fallback is announced once per process).
_fallback_announced: set[str] = set()


def api_key_from_env(api_key: str | None, who: str) -> str:
    """``api_key``, or ``MIRAI_API_KEY`` / ``MIRA_API_KEY`` from the environment."""
    key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
    if not key:
        raise ValueError(f"Set MIRAI_API_KEY or pass api_key to {who}.")
    return key


def check_server_rate(server_sample_rate):
    """Refuse a ``server_sample_rate`` Mirai can't be asked for."""
    if not (
        server_sample_rate is None
        or server_sample_rate == "auto"
        or (type(server_sample_rate) is int and server_sample_rate in SERVER_SAMPLE_RATES)
    ):
        raise ValueError(
            f"server_sample_rate must be 'auto', None or one of "
            f"{', '.join(map(str, SERVER_SAMPLE_RATES))}; got {server_sample_rate!r}"
        )


def _check_encoding(encoding: str):
    if encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {', '.join(ENCODINGS)}; got {encoding!r}")


def _encode(pcm: bytes, encoding: str) -> bytes:
    if encoding == "mulaw":
        return lin2ulaw(np.frombuffer(pcm, dtype="<i2")).tobytes()
    if encoding == "alaw":
        return lin2alaw(np.frombuffer(pcm, dtype="<i2")).tobytes()
    return pcm


class TTSListener:
    """What a TTS engine tells its host. Override what you need; the rest do nothing.

    ``turn_id`` is the host's id for one reply (Pipecat's audio context id).
    :class:`TTSStream` calls every method; :class:`HttpTTS` only the
    synthesis, first-audio and text-billed ones (it yields audio instead).
    """

    async def on_audio(self, turn_id: str, audio: bytes) -> None:
        """A piece of the turn's audio (``FRAME_SECS`` long, the last one shorter)."""

    async def on_turn_done(self, turn_id: str) -> None:
        """Mirai has spoken all of the ended turn (or what's left of it can't be spoken)."""

    def turn_active(self, turn_id: str) -> bool:
        """Whether the host still wants the turn's audio. ``False`` stops it."""
        return True

    def keep_turn_open(self, turn_id: str) -> None:
        """The turn is waiting (a retry in a second or two); don't time it out."""

    async def on_synthesis_started(self, turn_id: str) -> None:
        """An HTTP request for a sentence is about to go out (time to first byte starts)."""

    async def on_first_audio(self, turn_id: str) -> None:
        """The first audio byte of a turn (WebSocket) or of a sentence (HTTP) arrived."""

    async def on_synthesis_ended(self, turn_id: str) -> None:
        """An HTTP sentence is over, spoken or not."""

    async def on_no_audio(self, turn_id: str) -> None:
        """The turn was cancelled before any of its audio arrived."""

    async def on_usage(self, chars: int) -> None:
        """Mirai billed ``chars`` characters for a sentence (WebSocket)."""

    async def on_text_billed(self, text: str) -> None:
        """Mirai took ``text`` to speak over HTTP (billed by its length)."""

    async def on_error(self, error: FrequencyError, turn_id: str | None = None) -> None:
        """Something failed. The stream carries on unless ``error.permanent``."""

    async def on_connected(self) -> None:
        """A socket is open and set up."""

    async def on_disconnected(self) -> None:
        """The socket was closed."""

    async def on_connection_error(self, message: str) -> None:
        """A socket could not be opened, or reopened."""


# ---------- HTTP ----------


class HttpTTS:
    """Mirai TTS over HTTP: one streaming ``POST /v1/audio/speech`` per sentence.

    Most agents should use :class:`TTSStream`, which streams over a WebSocket
    and falls back to this by itself. Each sentence is one streaming request
    over a kept-alive connection. The connections are shared by every
    ``HttpTTS`` for the same ``base_url`` in the process and kept warm on
    their own (see :mod:`fourier_frequency.pool`), so calls that start
    together, or a call after a quiet spell, don't wait for a TCP and TLS
    handshake. Mirai is asked for audio at ``sample_rate`` (an 8 kHz phone
    call downloads 128 kbit/s instead of 48 kHz's 768 kbit/s).

    Example::

        tts = HttpTTS(voice="shruti", sample_rate=8000)
        async for audio in tts.speak("नमस्ते!"):
            ...
        await tts.aclose()

    A busy Mirai (429) is asked again after its Retry-After for up to
    ``RETRY_BUDGET_SECS``; a 502/503/504 or a connection that fails before any
    response is retried once. Anything else raises :class:`FrequencyError`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        sample_rate: int = 24000,
        server_sample_rate: int | Literal["auto"] | None = "auto",
        encoding: str = "pcm_s16le",
        prebuffer_secs: float = 0.15,
        warm_connection: bool = True,
        keep_warm_secs: float | None = 30.0,
        http_client: httpx.AsyncClient | None = None,
        shared_pool: bool = True,
        listener: TTSListener | None = None,
        name: str | None = None,
    ):
        """Set up the engine; nothing connects until :meth:`start` or the first sentence.

        Args:
            api_key: Mirai API key. Defaults to the ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``) environment variable.
            base_url: API base URL including ``/v1``.
            voice: One of :data:`VOICES`.
            model: Mirai model name.
            sample_rate: The rate of the audio handed out.
            server_sample_rate: The PCM rate to ask Mirai for. ``"auto"`` (the
                default) asks for ``sample_rate`` when Mirai serves it and
                leaves the field out otherwise. An ``int`` from
                :data:`SERVER_SAMPLE_RATES` asks for that rate. ``None`` leaves
                the field out, so Mirai sends 48 kHz. Audio is resampled
                locally whenever the rate Mirai reports (``X-Sample-Rate``)
                differs from ``sample_rate``.
            encoding: ``"pcm_s16le"`` (the default), or ``"mulaw"`` /
                ``"alaw"`` for phone media streams.
            prebuffer_secs: Audio to collect before the first piece of each
                sentence is handed out, so a short first chunk followed by a
                pause doesn't start playback and then stall.
            warm_connection: Keep connections to Mirai open ahead of need (the
                shared pool's warm-up, or with a client of its own a
                ``GET /models`` at start and after an interruption).
            keep_warm_secs: With a client of its own: repeat that request when
                the connection has been idle this long (Mirai closes idle
                connections after 75 s). ``None`` turns it off.
            http_client: A caller-owned ``httpx.AsyncClient``; never closed here.
            shared_pool: Use the event loop's shared, self-warming client for
                ``base_url`` (the default). ``False``: a client of its own,
                closed by :meth:`aclose`.
            listener: Told when a sentence starts, its first audio and what was billed.
            name: How log lines name this engine.
        """
        key = api_key_from_env(api_key, name or "HttpTTS")
        check_server_rate(server_sample_rate)
        _check_encoding(encoding)
        if not prebuffer_secs >= 0:
            raise ValueError(f"prebuffer_secs must be >= 0; got {prebuffer_secs!r}")
        if keep_warm_secs is not None and not keep_warm_secs > 0:
            raise ValueError(f"keep_warm_secs must be > 0 or None; got {keep_warm_secs!r}")
        self.name = name or "HttpTTS"
        self.voice = voice
        self.model = model
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.listener = listener or TTSListener()
        base = base_url.rstrip("/")
        self._base_url = base
        self._speech_url = base + "/audio/speech"
        self._models_url = base + "/models"
        self._headers = {"Authorization": f"Bearer {key}"}
        self._server_rate_option = server_sample_rate
        self._prebuffer_secs = float(prebuffer_secs)
        self._warm_connection = warm_connection
        self._keep_warm_secs = keep_warm_secs
        self._warm_task: asyncio.Task | None = None
        self._keep_warm_task: asyncio.Task | None = None
        self._owns_http = http_client is None and not shared_pool
        self._use_shared = http_client is None and shared_pool
        # The event loop's shared client, held from start() (or the first sentence) to aclose().
        self._shared: SharedHTTPClient | None = None
        self._http: httpx.AsyncClient | None = http_client
        if self._owns_http:
            # Mirai's gateway closes a connection after 75 s idle. Drop it a little
            # earlier here, so a request is never sent on one the server is closing;
            # keep_warm_secs keeps it from ever getting that idle while a call runs.
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0, read=15.0),
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=70),
            )
        self._server_rate_refused = False
        self._closing = False
        # Responses still streaming, so an interruption or shutdown can close them.
        self._streams: set[httpx.Response] = set()
        # Bumped on every interruption; a sentence started before it hands out nothing more.
        self._interruptions = 0
        self._last_activity = time.monotonic()
        self.last_error: str | None = None
        # The rate of the PCM Mirai sent for the latest sentence (X-Sample-Rate).
        self.last_server_sample_rate: int | None = None
        # Mirai's id (X-Request-Id) of the latest sentence, for support requests.
        self.last_request_id: str | None = None

    def __str__(self) -> str:
        return self.name

    async def __aenter__(self) -> HttpTTS:
        self.start()
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()

    def requested_rate(self) -> int | None:
        """The ``sample_rate`` to put in the request, or ``None`` to leave it out."""
        if self._server_rate_refused:
            return None
        if self._server_rate_option == "auto":
            return self.sample_rate if self.sample_rate in SERVER_SAMPLE_RATES else None
        return self._server_rate_option

    # ---- connections ----

    def start(self):
        """Take the shared client and start warming connections (optional; the first sentence does it)."""
        self._closing = False
        self._hold_shared()
        self._start_warm_up()
        if (
            self._shared is None
            and self._warm_connection
            and self._keep_warm_secs
            and self._keep_warm_task is None
        ):
            self._keep_warm_task = asyncio.create_task(self._keep_warm(), name=f"{self}.keep_warm")

    def _hold_shared(self):
        """Take (or retake, if it was closed under us) the event loop's shared client."""
        if not self._use_shared or self._closing:
            return
        if self._shared is not None:
            if not self._shared.closed:
                return
            self._shared.release(self)
        self._shared = shared_http_client(self._base_url)
        self._shared.hold(self, warm=self._warm_connection, headers=self._headers)
        self._http = self._shared.client

    def client(self) -> httpx.AsyncClient:
        """The client sentences go out on."""
        self._hold_shared()
        if self._http is None:  # pragma: no cover - a shared client is always taken above
            raise RuntimeError(f"{self} has no HTTP client")
        return self._http

    def _after_cut(self):
        """A stream was closed before its end: its connection is gone."""
        if self._shared is not None:
            self._shared.nudge()  # the pool notices the closed connection and replaces it
        else:
            self._start_warm_up()  # those connections are gone; open the next one now

    def _start_warm_up(self):
        """Open a connection to Mirai in the background, unless one is opening."""
        if self._shared is not None or not self._warm_connection or self._closing:
            return  # the shared pool warms itself
        if self._http is None or self._http.is_closed:
            return
        if self._warm_task is not None and not self._warm_task.done():
            return
        try:
            self._warm_task = asyncio.create_task(self._warm_up(), name=f"{self}.warm_up")
        except Exception as exc:  # e.g. no running loop; warming is only an optimisation
            logger.debug(f"{self}: not pre-opening the connection to Mirai: {exc}")

    async def _warm_up(self):
        try:
            response = await self._http.get(self._models_url, headers=self._headers, timeout=10.0)
        except Exception as exc:  # never fatal: the next request connects on its own
            logger.debug(f"{self}: could not pre-open the connection to Mirai: {exc!r}")
            return
        finally:
            self._last_activity = time.monotonic()
        if response.status_code in (401, 403):
            logger.warning(
                f"{self}: Mirai rejected the API key (HTTP {response.status_code}); check MIRAI_API_KEY"
            )
        else:
            logger.debug(f"{self}: connection to Mirai open (GET /models: HTTP {response.status_code})")

    async def _keep_warm(self):
        """While the engine runs, keep the idle connection from timing out."""
        interval = self._keep_warm_secs
        while True:
            await asyncio.sleep(max(1.0, self._last_activity + interval - time.monotonic()))
            if self._streams or time.monotonic() - self._last_activity < interval:
                continue  # in use, or used since we went to sleep
            if self._warm_task is not None and not self._warm_task.done():
                continue
            await self._warm_up()

    # ---- interruptions and shutdown ----

    def begin_interrupt(self) -> list[httpx.Response]:
        """The listener stopped wanting what's playing: nothing started before now hands out more.

        Returns the responses still streaming, for :meth:`finish_interrupt`
        to close once whoever was reading them has stopped (closing one under
        a reader that is still waiting on it would make that reader fail).
        """
        self._interruptions += 1
        return list(self._streams)

    async def finish_interrupt(self, cut_short: list[httpx.Response], *, after_cut: bool = True):
        """Close the responses an interruption cut short.

        A sentence parked between two pieces of audio only closes once its
        generator is finalised, which can wait on garbage collection, and until
        then the server keeps generating into a slot nobody will hear.
        """
        for response in cut_short:
            await response.aclose()
        if cut_short and after_cut:
            self._after_cut()

    async def interrupt(self):
        """Stop every sentence now (barge-in), and close their streams."""
        await self.finish_interrupt(self.begin_interrupt())

    async def close_streams(self):
        """Close every response still streaming."""
        for response in list(self._streams):
            await response.aclose()

    async def aclose(self):
        """Close the streams and the warm-up tasks; let go of (or close) the client."""
        self._closing = True
        await self.close_streams()
        tasks = (self._warm_task, self._keep_warm_task)
        self._warm_task = self._keep_warm_task = None
        for task in tasks:
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        if self._shared is not None:
            # Other calls may be using it; it keeps itself warm for the next ones.
            shared, self._shared = self._shared, None
            shared.release(self)
        elif self._owns_http and self._http is not None and not self._http.is_closed:
            await self._http.aclose()

    # ---- speaking ----

    async def _send_speech(self, text: str, rate: int | None) -> httpx.Response:
        body = {
            "model": self.model,
            "voice": self.voice,
            "input": text,
            "response_format": "pcm",
        }
        if rate is not None:
            body["sample_rate"] = rate
        client = self.client()
        request = client.build_request("POST", self._speech_url, headers=self._headers, json=body)
        return await client.send(request, stream=True)

    async def _post_speech(self, text: str) -> httpx.Response:
        """Send the sentence and return the open, streaming response.

        A server that doesn't take ``sample_rate`` may answer 400. The request
        is then sent once more without it. If that one gets past validation,
        the field is left out for the rest of the session and the server's
        48 kHz audio is resampled locally. If it is refused too, the request
        itself is at fault: that error is reported and nothing is remembered.
        """
        rate = self.requested_rate()
        response = await self._send_speech(text, rate)
        if rate is None or response.status_code != 400:
            return response
        try:
            await response.aread()  # read to the end so the connection is reused
        finally:
            await response.aclose()
        retry = await self._send_speech(text, None)
        if retry.status_code != 400:
            self._server_rate_refused = True
            detail = _error_detail(response) or "no detail"
            logger.warning(
                f"{self}: Mirai refused sample_rate={rate} (HTTP 400: {detail}); "
                f"using its default rate for this session and resampling locally"
            )
        return retry

    async def _open_speech(self, text: str) -> httpx.Response:
        """The sentence's open 2xx response, after any retries; raises :class:`FrequencyError`."""
        waited = 0.0
        retried = False
        while True:
            try:
                response = await self._post_speech(text)
            except httpx.TransportError as exc:
                # Nothing came back: a refused or reset connection (a kept-alive
                # one the server had just closed, say). Once more, on a new one.
                if retried or isinstance(exc, httpx.ReadTimeout):
                    raise FrequencyError(
                        f"Mirai TTS: could not reach {self._speech_url} ({exc!r}"
                        f"{', also on a retry' if retried else ''}). Check the network and base_url.",
                        category=errors.CONNECTIVITY,
                    ) from exc
                retried = True
                logger.info(f"{self}: no response from Mirai ({exc!r}); retrying once")
                await asyncio.sleep(FAST_RETRY_DELAY_SECS)
                continue
            self.last_request_id = response.headers.get("x-request-id") or self.last_request_id
            if not response.is_error:
                return response
            try:
                await response.aread()
            finally:
                await response.aclose()
            status = response.status_code
            code, detail = _error_info(response)
            request_id = response.headers.get("x-request-id")
            if status == 429:
                delay = _retry_after(response, 0.5)
                if waited + delay <= RETRY_BUDGET_SECS:
                    waited += delay
                    logger.info(
                        f"{self}: Mirai is busy (HTTP 429 {code}{_rid(request_id)}); "
                        f"sending the sentence again in {delay:g} s"
                    )
                    await asyncio.sleep(delay)
                    continue
            elif status in (502, 503, 504) and not retried:
                retried = True
                logger.info(f"{self}: Mirai answered HTTP {status}{_rid(request_id)}; retrying once")
                await asyncio.sleep(FAST_RETRY_DELAY_SECS)
                continue
            raise FrequencyError(
                _explain(status, code, detail, request_id, waited),
                category=_http_category(status),
                code=code or None,
                status=status,
                request_id=request_id,
                permanent=status in (401, 402, 403),
            )

    async def speak(self, text: str, turn_id: str = "") -> AsyncIterator[bytes]:
        """Synthesize ``text``; yield its audio in ``FRAME_SECS`` pieces as it streams in.

        Raises :class:`FrequencyError` when Mirai refuses the sentence or the
        stream breaks off (after yielding what arrived). Stops quietly when
        :meth:`interrupt` (or :meth:`begin_interrupt`) is called meanwhile.
        """
        if not text.strip():
            return
        self.last_error = None
        epoch = self._interruptions
        rate = self.sample_rate
        frame_bytes = max(1, round(rate * FRAME_SECS)) * 2
        prebuffer_bytes = round(rate * self._prebuffer_secs) * 2
        pending = bytearray()  # resampled audio not handed out yet: < one frame once playing
        playing = prebuffer_bytes == 0
        carry = b""
        received = 0
        cut: Exception | None = None
        listener = self.listener
        try:
            await listener.on_synthesis_started(turn_id)
            self._last_activity = time.monotonic()
            response = await self._open_speech(text)
            self._streams.add(response)
            try:
                source_rate = _source_rate(response)
                resampler = (
                    soxr.ResampleStream(source_rate, rate, 1, dtype="int16", quality="HQ")
                    if source_rate != rate
                    else None
                )
                if source_rate != self.last_server_sample_rate:
                    resampling = f", resampling to {rate} Hz" if resampler else ""
                    logger.debug(f"{self}: Mirai sent {source_rate} Hz PCM{resampling}")
                self.last_server_sample_rate = source_rate
                await listener.on_text_billed(text)
                try:
                    async for chunk in response.aiter_bytes():
                        # Network reads can end mid-sample; carry the odd byte forward.
                        data = carry + chunk
                        end = len(data) - len(data) % 2
                        carry, data = data[end:], data[:end]
                        if not data:
                            continue
                        if not received:
                            await listener.on_first_audio(turn_id)  # TTFB = first audio byte received
                        received += len(data)
                        if resampler:
                            data = resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
                        pending += data
                        if not playing:
                            if len(pending) < prebuffer_bytes:
                                continue
                            playing = True
                        # Hand out whole frames; a burst of several seconds becomes many
                        # 40 ms frames, and only the remainder stays here.
                        whole = len(pending) - len(pending) % frame_bytes
                        for offset in range(0, whole, frame_bytes):
                            yield _encode(bytes(pending[offset : offset + frame_bytes]), self.encoding)
                            if self._interruptions != epoch or self._closing:
                                return  # cut short while parked here: hand out nothing more
                        del pending[:whole]
                except httpx.TransportError as exc:
                    if self._interruptions != epoch or self._closing:
                        return  # closed by the interruption itself
                    cut = exc  # the rest of the sentence is lost; play what arrived
                if not received:
                    if cut is not None:
                        raise cut
                    raise ValueError("no audio returned")
                if carry and cut is None:
                    raise ValueError("audio ended with an incomplete PCM sample")
                if resampler:
                    pending += resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True).tobytes()
                for offset in range(0, len(pending), frame_bytes):
                    yield _encode(bytes(pending[offset : offset + frame_bytes]), self.encoding)
                    if self._interruptions != epoch or self._closing:
                        return
                if cut is not None:
                    seconds = received / 2 / source_rate
                    raise FrequencyError(
                        f"Mirai TTS: the stream broke off after {seconds:.1f} s of audio ({cut!r})"
                        f"{_rid(response.headers.get('x-request-id'))}; the rest of the sentence was "
                        f"not spoken. Usually a network drop between this server and Mirai.",
                        category=errors.CONNECTIVITY,
                        request_id=response.headers.get("x-request-id"),
                    )
            finally:
                self._streams.discard(response)
                self._last_activity = time.monotonic()
                await response.aclose()
        except FrequencyError as exc:
            self.last_error = str(exc)
            raise
        except (httpx.HTTPError, ValueError) as exc:
            if self._interruptions != epoch or self._closing:
                return  # a stream an interruption closed under us
            self.last_error = f"Mirai TTS: {exc}{_rid(self.last_request_id)}"
            raise FrequencyError(self.last_error, category=errors.SERVER) from exc
        except (asyncio.CancelledError, GeneratorExit):
            # Interrupted. Closing an unfinished response drops its connection,
            # so open the next one now, while the caller is still talking.
            self._after_cut()
            raise
        finally:
            # An interruption closes this generator: the HTTP stream is closed
            # with it and nothing already-interrupted is retried or flushed.
            await listener.on_synthesis_ended(turn_id)


# ---------- WebSocket ----------


@dataclass
class _Turn:
    """One reply (the host's turn id), and the Mirai context that speaks it.

    The Mirai context id is the host's, until the context has to be
    restarted (a retry, or a new socket); then it gets a fresh id so nothing
    from the old one can be mistaken for the new.
    """

    id: str
    server_id: str
    # Text sent under server_id, and how much of it Mirai has finished speaking.
    sent: str = ""
    cursor: int = 0
    # Flushes sent under server_id that have no context.done yet.
    marks: int = 0
    # The host has ended the turn: no more text is coming.
    final: bool = False
    # A retry (capacity or reconnect) has been used; there is only one.
    retried: bool = False
    # Text waiting to be sent to server_id (during a retry); None otherwise.
    held: str | None = None
    retry_task: asyncio.Task | None = None
    got_audio: bool = False

    def record_sent(self, text: str, separate: bool):
        # Sentences sent one by one are cut by Mirai one by one; keep a space
        # between them here so a resend of several isn't run together.
        if separate and self.sent and text and not self.sent[-1].isspace() and not text[0].isspace():
            self.sent += " "
        self.sent += text

    def spoken(self, sentence: str):
        """Mirai finished ``sentence``: one of its cuts of ``sent``, trimmed."""
        sentence = sentence.strip()
        at = self.sent.find(sentence, self.cursor) if sentence else -1
        if at >= 0:
            self.cursor = at + len(sentence)

    def unspoken(self, skip: str = "") -> str:
        """Text sent but not finished; past ``skip`` too, if it comes next."""
        cursor = self.cursor
        skip = skip.strip()
        if skip and (at := self.sent.find(skip, cursor)) >= 0:
            cursor = at + len(skip)
        return self.sent[cursor:].lstrip()


@dataclass
class _Sentence:
    """The sentence whose audio is streaming (the latest ``audio.start``)."""

    request_id: str | None
    server_id: str | None
    text: str
    turn: _Turn | None  # None: its audio is dropped (cancelled, or unplayable)
    resampler: Any = None
    carry: bytes = b""
    pending: bytearray = field(default_factory=bytearray)
    playing: bool = False
    pushed: int = 0


class TTSStream:
    """Mirai text to speech over one WebSocket for a whole call.

    The socket opens on :meth:`connect` and carries every reply. Sockets are
    opened ahead of need by a pool shared by every stream in the process (see
    :mod:`fourier_frequency.pool`), so a call usually takes one that is
    already open and authenticated and starts without a handshake. Each socket
    opens on Mirai's edge, next to the speech GPUs, when Mirai offers one
    (about half the time to first byte), and on ``url`` otherwise (see
    :mod:`fourier_frequency.edge`).

    Text goes to Mirai as the agent produces it: whole sentences
    (``text_mode="sentence"``, each sent with a flush so it is spoken at once)
    or LLM tokens (``"token"``; Mirai cuts sentences and the turn's end
    flushes the rest). Mirai synthesises the next sentence while the current
    one plays, so there is no gap and no handshake between sentences.

    Cancelling a turn cancels it on the server at once and drops whatever of
    it is still arriving. A dropped connection is reopened, and a reply it cut
    short is resent once, from the first sentence the caller hasn't started
    hearing. A sentence Mirai refuses for capacity is resent once after its
    Retry-After. If the streaming endpoint can't be reached at all (refused,
    not found, a proxy that doesn't pass WebSockets), the call speaks over
    Mirai's HTTP endpoint instead (``http_fallback``).

    Use it through turns (:meth:`turn`), or hand it a :class:`TTSListener`
    and drive it by turn id (:meth:`send_text`, :meth:`end_turn`,
    :meth:`cancel_turn`), as the Pipecat service does.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        base_url: str | None = None,
        voice: str = DEFAULT_VOICE,
        model: str = DEFAULT_MODEL,
        sample_rate: int = 24000,
        server_sample_rate: int | Literal["auto"] | None = "auto",
        encoding: str = "pcm_s16le",
        text_mode: Literal["sentence", "token"] = "sentence",
        prebuffer_secs: float = 0.15,
        keepalive_secs: float | None = 30.0,
        shared_pool: bool = True,
        http_fallback: bool = True,
        edge: bool | str = "auto",
        reconnect: bool = True,
        http_client: httpx.AsyncClient | None = None,
        listener: TTSListener | None = None,
        name: str | None = None,
    ):
        """Set up the stream; nothing connects until :meth:`connect`.

        Args:
            api_key: Mirai API key. Defaults to the ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``) environment variable. Sent as an
                ``Authorization: Bearer`` header.
            url: The streaming endpoint, ``wss://<host>/v1/audio/speech/stream``.
                Defaults to Mirai's (or to ``base_url``'s, when that is given).
            base_url: The API base URL including ``/v1`` (``https://<host>/v1``);
                the streaming endpoint is derived from it. ``url`` wins.
            voice: One of :data:`VOICES`.
            model: Mirai model name.
            sample_rate: The rate of the audio handed out (8000 for a phone call).
            server_sample_rate: The PCM rate to ask Mirai for. ``"auto"`` (the
                default) asks for ``sample_rate`` when Mirai serves it and
                leaves the field out otherwise; an ``int`` asks for that rate;
                ``None`` leaves it out, so Mirai sends 48 kHz. Audio is
                resampled locally only when the rate Mirai sends differs.
            encoding: ``"pcm_s16le"`` (the default), or ``"mulaw"`` /
                ``"alaw"`` (8 kHz phone media streams).
            text_mode: ``"sentence"``: each :meth:`TTSTurn.send` is one or
                more whole sentences, spoken at once. ``"token"``: send LLM
                tokens as they come; Mirai cuts sentences and the turn's end
                speaks the rest.
            prebuffer_secs: Audio to collect before the first piece of each
                sentence is handed out, so a short first chunk followed by a
                pause doesn't start playback and then stall.
            keepalive_secs: When nothing has been sent for this long, send an
                empty ``session.update`` so Mirai doesn't close the socket as
                idle during a long pause in the call. ``None`` turns it off.
            shared_pool: Take a socket the process's pool opened ahead of
                need when one is waiting (the default); the first stream to
                connect sets the pool going. ``False`` always connects.
            http_fallback: If the streaming endpoint can't be reached (a
                handshake answered 404, 426 or 5xx, refused, or timed out),
                speak this call over Mirai's HTTP endpoint on the same host
                instead of failing, and log it once. Authentication, credit
                and rate-limit refusals are reported, not worked around.
            edge: ``"auto"`` (the default): open each socket on Mirai's edge
                when Mirai offers one, with the API key in the header; if the
                edge can't be reached or isn't ready within 3 s, the socket
                opens on ``url`` and the edge is left alone for a while.
                ``False`` always uses ``url``; a ``wss://`` URL uses that edge.
                ``MIRAI_TTS_EDGE=off`` turns ``"auto"`` off.
            reconnect: Reopen a socket that drops (up to
                ``RECONNECT_ATTEMPTS`` tries). ``False`` reports the drop and
                reconnects only when the next text is sent.
            http_client: An ``httpx.AsyncClient`` for the HTTP fallback, if
                you manage your own; by default the process's shared one.
            listener: Told about audio, turns, usage, errors and the
                connection. By default, the stream's turns (:meth:`turn`).
            name: How log lines name this stream.
        """
        key = api_key_from_env(api_key, name or "TTSStream")
        check_server_rate(server_sample_rate)
        _check_encoding(encoding)
        if text_mode not in ("sentence", "token"):
            raise ValueError(f"text_mode must be 'sentence' or 'token'; got {text_mode!r}")
        if not prebuffer_secs >= 0:
            raise ValueError(f"prebuffer_secs must be >= 0; got {prebuffer_secs!r}")
        if url is None:
            url = stream_url(base_url) if base_url else DEFAULT_WEBSOCKET_URL
        if keepalive_secs is not None and not keepalive_secs > 0:
            raise ValueError(f"keepalive_secs must be > 0 or None; got {keepalive_secs!r}")
        if not url.startswith(("ws://", "wss://")):
            raise ValueError(f"url must be a ws:// or wss:// URL; got {url!r}")
        mode = edge_mode(edge)

        self.name = name or "TTSStream"
        self.voice = voice
        self.model = model
        self.sample_rate = sample_rate
        self.encoding = encoding
        # Text is sent as tokens (flushed at the turn's end), not as sentences.
        self.tokens = text_mode == "token"
        self._turn_objects: dict[str, TTSTurn] = {}
        self.listener = listener or _TurnListener(self)
        self._url = url
        self._headers = {"Authorization": f"Bearer {key}"}
        self._server_rate_option = server_sample_rate
        self._prebuffer_secs = float(prebuffer_secs)
        self._keepalive_secs = keepalive_secs
        self._shared_pool = shared_pool
        self._http_fallback = http_fallback
        self._reconnect_on_error = reconnect
        self._pool: WebsocketPool | None = None
        # The HTTP fallback: the same host's speech endpoint, and why it is in use (None: it isn't).
        self._http_base = http_base(url)
        self._http = HttpTTS(
            api_key=key,
            base_url=self._http_base,
            voice=voice,
            model=model,
            sample_rate=sample_rate,
            server_sample_rate=server_sample_rate,
            prebuffer_secs=prebuffer_secs,
            # A caller's own client is used as it is; the shared one is held warm.
            warm_connection=http_client is None,
            keep_warm_secs=None,
            http_client=http_client,
            shared_pool=True,
            listener=self.listener,
            name=self.name,
        )
        self._speech_url = self._http_base + "/audio/speech"
        self._edge = EdgeRoute(url, self._headers, mode, self._http_base) if mode else None
        self._fallback: str | None = None
        self._fallback_text: dict[str, str] = {}
        self._frame_bytes = 2
        self._prebuffer_bytes = 0

        self._websocket: Any = None
        self._receive_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._last_sent = time.monotonic()
        self._idle_timeout_secs: float | None = None
        # Reconnecting: whether the host is closing the stream, a reconnect is
        # under way, the stream gave up for good, and quick failures in a row.
        self._disconnecting = False
        self._reconnect_in_progress = False
        self._usable = True
        self._quick_failures = 0
        self._last_connect_time = 0.0

        # Host turn id -> turn, and Mirai context id -> turn.
        self._turns: dict[str, _Turn] = {}
        self._by_server: dict[str, _Turn] = {}
        # Mirai context ids used so far (never reused), and those cancelled
        # whose context.cancelled hasn't arrived.
        self._used_ids: set[str] = set()
        self._cancelled: set[str] = set()
        self._sentence: _Sentence | None = None

        self.last_error: str | None = None
        self._connect_error: FrequencyError | None = None
        # The rate of the PCM Mirai sent for the latest sentence (audio.start).
        self.last_server_sample_rate: int | None = None
        # The socket's session id (``ttsws_...``), from session.ready.
        self.session_id: str | None = None
        # Where the latest socket was connected: the edge's URL, or ``url``.
        self.connected_url: str | None = None

    def __str__(self) -> str:
        return self.name

    async def __aenter__(self) -> TTSStream:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info):
        await self.close()

    @property
    def url(self) -> str:
        """The streaming endpoint (the gateway's)."""
        return self._url

    @property
    def keepalive_secs(self) -> float | None:
        return self._keepalive_secs

    @property
    def http(self) -> HttpTTS:
        """The HTTP engine the call falls back to."""
        return self._http

    @property
    def fallback(self) -> str | None:
        """Why the call speaks over HTTP (``"HTTP 404"``, ...), or ``None`` while on the socket."""
        return self._fallback

    def requested_rate(self) -> int | None:
        """The PCM rate asked of Mirai, or ``None`` for its default."""
        if self._server_rate_option == "auto":
            return self.sample_rate if self.sample_rate in SERVER_SAMPLE_RATES else None
        return self._server_rate_option

    # ---------- lifecycle ----------

    async def connect(self):
        """Open the socket (from the pool when one is waiting) and set up the session.

        Never raises for Mirai's sake: a refusal is told to the listener
        (``on_error``, ``on_connection_error``) and the next text tries again;
        an endpoint that isn't there moves the call to HTTP.
        """
        rate = self.sample_rate
        self._frame_bytes = max(1, round(rate * FRAME_SECS)) * 2
        self._prebuffer_bytes = round(rate * self._prebuffer_secs) * 2
        self._http.sample_rate = rate
        self._http.encoding = self.encoding
        self._http._closing = False
        if self.voice not in VOICES:
            logger.warning(f"{self}: voice {self.voice!r} is not one of {', '.join(VOICES)}")
        requested = self.requested_rate()
        asked = f"{requested} Hz" if requested else f"its default {SOURCE_SAMPLE_RATE} Hz"
        logger.debug(f"{self}: asking Mirai for {asked} PCM; output {rate} Hz")
        if self._shared_pool and self._pool is None:
            self._pool = websocket_pool(self._url, self._headers, MAX_MESSAGE_BYTES, edge=self._edge)
            self._pool.hold(self)  # the first one sets the pool going
        await self._connect()

    async def close(self):
        """Close the socket, stop every turn and let go of shared connections."""
        self._disconnecting = True
        self._http._closing = True
        for name in ("_keepalive_task", "_receive_task"):
            task = getattr(self, name)
            setattr(self, name, None)
            if task is not None and task is not asyncio.current_task() and not task.done():
                await _cancel(task)
        await self._http.close_streams()
        await self._disconnect_websocket()
        await self._release_shared()

    async def _release_shared(self):
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.release(self)
        await self._http.aclose()

    async def _connect(self):
        self._disconnecting = False
        self._quick_failures = 0
        if self._fallback:
            return
        try:
            await self._connect_websocket()
        except Exception as exc:
            if self._can_fall_back(exc):
                self._fall_back(exc)
                return
            await self._report_connect_failure(exc)
        if self._websocket is not None:
            task = self._receive_task
            if (
                task is not None
                and task is not asyncio.current_task()
                and not task.done()
                and self._reconnect_in_progress
                and self._is_open()
            ):
                # The receive task is sleeping out a reconnect backoff, but the
                # socket is up again: read it now, not in a few seconds.
                self._reconnect_in_progress = False
                await _cancel(task)
                self._receive_task = None
            if self._receive_task is None or self._receive_task.done():
                self._receive_task = asyncio.create_task(self._receive_task_handler(), name=f"{self}.receive")
            if self._keepalive_secs and (self._keepalive_task is None or self._keepalive_task.done()):
                self._keepalive_task = asyncio.create_task(
                    self._keepalive_task_handler(), name=f"{self}.keepalive"
                )

    # ---------- HTTP fallback ----------

    def _can_fall_back(self, exc: Exception, *, transient: bool = True) -> bool:
        """Whether ``exc`` says the streaming endpoint itself is unavailable.

        Definitive: the endpoint isn't there or doesn't speak WebSocket (404,
        405, 426, a broken handshake). Transient, counted only with
        ``transient``: a 5xx, a refused connection, a timeout.
        """
        if not self._http_fallback:
            return False
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status is not None:
            return status in (404, 405, 426) or (transient and status >= 500)
        if isinstance(exc, InvalidHandshake):
            return True
        return transient and isinstance(exc, OSError | TimeoutError)

    def _fall_back(self, exc: Exception):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        self._fallback = f"HTTP {status}" if status else repr(exc)
        message = (
            f"{self}: Mirai's streaming endpoint {self._url} is unavailable ({self._fallback}); "
            f"speaking over {self._speech_url} instead. Calls work as before, with a TCP and TLS "
            f"handshake now and then; set http_fallback=False to fail instead."
        )
        if self._url in _fallback_announced:
            logger.debug(message)
        else:
            _fallback_announced.add(self._url)
            logger.warning(message)

    async def _speak_over_http(self, text: str, turn_id: str):
        """Speak ``text`` over HTTP into the turn, as the socket would."""
        listener = self.listener
        try:
            async with contextlib.aclosing(self._http.speak(text, turn_id)) as audio:
                async for piece in audio:
                    if not listener.turn_active(turn_id):
                        return  # interrupted
                    await listener.on_audio(turn_id, piece)
        except FrequencyError as exc:
            self.last_error = str(exc)
            await listener.on_error(exc, turn_id)

    async def _fallback_tts(self, text: str, turn_id: str):
        if not self.tokens:
            await self._speak_over_http(text, turn_id)
            return
        # Tokens: cut sentences here, as Mirai's stream would; end_turn speaks the rest.
        pending = self._fallback_text.get(turn_id, "") + text
        while match := _SENTENCE_END.search(pending):
            sentence, pending = pending[: match.end()], pending[match.end() :]
            if sentence.strip():
                await self._speak_over_http(sentence, turn_id)
        self._fallback_text[turn_id] = pending

    # ---------- the socket ----------

    async def _connect_websocket(self):
        """Open the socket and set up the session. Raises if it can't."""
        async with self._connect_lock:
            if self._is_open():
                return
            if self._websocket is not None:
                # The socket died and this noticed before the receive task did:
                # settle what was on it, so it is resent below.
                dead, self._websocket = self._websocket, None
                await self._abandon_turns(intentional=False)
                await _close_quietly(dead)
            pooled = (
                await take_websocket(self._url, self._headers, edge=self._edge) if self._shared_pool else None
            )
            on_edge = None
            if pooled is None and self._edge is not None:
                on_edge = await self._edge.open(self._connect_to)
            if pooled is not None:
                # Already open and authenticated; nothing else has used it.
                # The session.update below sets this call's voice and rate.
                logger.debug(f"{self}: using waiting socket {pooled.session_id} to {pooled.url or self._url}")
                self._websocket = pooled.websocket
                self.session_id = pooled.session_id
                self.connected_url = pooled.url or self._url
                if pooled.idle_timeout_secs:
                    self._idle_timeout_secs = pooled.idle_timeout_secs
            elif on_edge is not None:
                logger.debug(f"{self}: connected to the edge {on_edge.url}")
                self._websocket = on_edge.websocket
                self.connected_url = on_edge.url
                await self._on_event(on_edge.ready)  # its session id and limits
            else:
                logger.debug(f"{self}: connecting to {self._url}")
                self._websocket = await self._connect_gateway()
                self.connected_url = self._url
            self._cancelled.clear()
            self._sentence = None
            await self._send(self._session_update())
            await self.listener.on_connected()
        # Replies a dropped socket cut short go out on the new one.
        for turn in list(self._turns.values()):
            if turn.held is not None and turn.retry_task is None:
                await self._send_held(turn)

    async def _connect_gateway(self):
        """Open ``url``, waiting out a 429 for up to ``CONNECT_RETRY_SECS``."""
        deadline = time.monotonic() + CONNECT_RETRY_SECS
        while True:
            extra = await asyncio.wait_for(websocket_connect_kwargs(self._url), 10.0)
            try:
                return await self._connect_to(self._url, self._headers, extra)
            except Exception as exc:
                sock = extra.get("sock")
                if sock is not None:
                    sock.close()
                response = getattr(exc, "response", None)
                if getattr(response, "status_code", None) != 429:
                    raise
                wait = min(CONNECT_RETRY_WAIT, _handshake_retry_after(response, 1.0))
                if time.monotonic() + wait > deadline:
                    raise
                logger.debug(f"{self}: Mirai answered 429 to the socket; trying again in {wait:g} s")
                await asyncio.sleep(wait)

    async def _connect_to(self, url: str, headers: dict[str, str], extra: dict[str, Any]):
        return await websocket_connect(
            url,
            additional_headers=headers,
            max_size=MAX_MESSAGE_BYTES,
            close_timeout=CLOSE_TIMEOUT,
            **extra,
        )

    def _session_update(self) -> dict:
        msg = {
            "type": "session.update",
            "voice": self.voice,
            "model": self.model,
            "response_format": "pcm",
            "audio_transport": "binary",
        }
        rate = self.requested_rate()
        if rate is not None:
            msg["sample_rate"] = rate
        return msg

    async def _disconnect_websocket(self):
        websocket, self._websocket = self._websocket, None
        await self._abandon_turns(intentional=self._disconnecting)
        if websocket is None:
            return
        # A plain close: Mirai drops (and doesn't bill) anything unsent.
        await _close_quietly(websocket)
        logger.debug(f"{self}: disconnected from Mirai")
        await self.listener.on_disconnected()

    async def _report_connect_failure(self, exc: Exception):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            message = f"Mirai TTS refused the WebSocket: HTTP {status}"
            detail = _ws_error_detail(getattr(response, "body", None))
            if detail:
                message += f": {detail}"
            request_id = getattr(response, "headers", {}).get("x-request-id")
            if request_id:
                message += f" [request {request_id}]"
            if status in _ADVICE:
                message += f". {_ADVICE[status]}"
            category = _HANDSHAKE_CATEGORIES.get(status, errors.CONNECTIVITY)
        else:
            request_id = None
            message = f"Mirai TTS could not connect to {self._url}: {exc}"
            category = errors.CONNECTIVITY
        self.last_error = message
        error = FrequencyError(message, category=category, status=status, request_id=request_id)
        self._connect_error = error
        error.__cause__ = exc
        await self.listener.on_error(error)
        await self.listener.on_connection_error(message)

    def _is_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

    def is_open(self) -> bool:
        """Whether the socket is open (``False`` while speaking over HTTP)."""
        return self._is_open()

    def _get_websocket(self):
        if self._websocket is None:
            raise ConnectionError("not connected to Mirai")
        return self._websocket

    async def _send(self, msg: dict):
        websocket = self._websocket
        if websocket is None or websocket.state is not State.OPEN:
            raise ConnectionError("not connected to Mirai")
        await websocket.send(json.dumps(msg, ensure_ascii=False))
        self._last_sent = time.monotonic()

    async def _keepalive_task_handler(self):
        """Keep an idle socket from being closed by Mirai's idle timeout."""
        while True:
            interval = self._keepalive_secs
            if self._idle_timeout_secs:
                interval = min(interval, self._idle_timeout_secs / 2)
            await asyncio.sleep(max(0.05, self._last_sent + interval - time.monotonic()))
            if time.monotonic() - self._last_sent < interval or not self._is_open():
                continue
            try:
                # Changes nothing; any message resets the server's idle clock.
                await self._send({"type": "session.update"})
            except Exception as exc:
                logger.debug(f"{self}: keepalive failed: {exc!r}")

    # ---------- reconnecting ----------

    async def _receive_task_handler(self):
        """Read the socket; reopen it when it drops, as many times as allowed."""
        while True:
            self._last_connect_time = time.monotonic()
            error: Exception | None = None
            try:
                await self._receive_messages()
                message = f"{self} connection closed by server"
            except ConnectionClosedOK as exc:
                logger.debug(f"{self} connection closed normally: {exc}")
                break
            except ConnectionClosedError as exc:
                message, error = f"{self} connection closed, but with an error: {exc}", exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                message, error = f"{self} error receiving messages: {exc}", exc
            if not await self._maybe_try_reconnect(message, error):
                break

    async def _maybe_try_reconnect(self, message: str, error: Exception | None) -> bool:
        """Whether to go on reading after a drop: reopen the socket, unless it's no use."""
        if self._disconnecting:
            if error:
                logger.warning(f"{self} error during disconnect: {error}")
            else:
                logger.debug(f"{self} receive loop ended during disconnect")
            return False
        # A server that accepts the handshake and then closes the session at
        # once (a refused key, say) won't be fixed by waiting longer: stop.
        lasted = time.monotonic() - self._last_connect_time
        if lasted < QUICK_FAILURE_SECS:
            self._quick_failures += 1
            logger.warning(
                f"{self} connection lasted only {lasted:.1f}s "
                f"({self._quick_failures}/{QUICK_FAILURES} consecutive quick failures)"
            )
            if self._quick_failures >= QUICK_FAILURES:
                text = f"{self} connection failed {QUICK_FAILURES} times immediately after connecting"
                logger.error(text)
                await self._report_error(text, None, permanent=True)
                return False
        else:
            self._quick_failures = 0
        logger.warning(message)
        if not self._reconnect_on_error:
            await self._report_error(message, error)
            return False
        return await self._try_reconnect()

    async def _try_reconnect(self) -> bool:
        if self._reconnect_in_progress:
            logger.warning(f"{self} reconnect attempt aborted: already in progress")
            return False
        if not self._usable:
            logger.error(f"{self} not reconnecting: the stream gave up")
            return False
        self._reconnect_in_progress = True
        last: Exception | None = None
        try:
            for attempt in range(1, RECONNECT_ATTEMPTS + 1):
                try:
                    logger.warning(f"{self} reconnecting, attempt {attempt}")
                    if await self._reconnect_websocket(attempt):
                        logger.info(f"{self} reconnected successfully on attempt {attempt}")
                        self._last_connect_time = time.monotonic()
                        return True
                except Exception as exc:
                    last = exc
                    logger.error(f"{self} reconnection attempt {attempt} failed: {exc}")
                    await self._report_error(f"{self} reconnection attempt {attempt} failed: {exc}", exc)
                if not self._usable:
                    return False
                await asyncio.sleep(_backoff(attempt))
            text = f"{self} failed to reconnect after {RECONNECT_ATTEMPTS} attempts"
            if last:
                text += f": {last}"
            logger.error(text)
            await self._report_error(text, last, permanent=True)
            return False
        finally:
            self._reconnect_in_progress = False

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        if self._fallback or self._is_open():
            return True  # already reopened (by send_text) while this waited to retry
        try:
            logger.warning(f"{self} reconnecting (attempt: {attempt_number})")
            await self._disconnect_websocket()
            await self._connect_websocket()
            if not await self._verify_connection():
                raise ConnectionError(f"{self} websocket reconnection failed verification")
            return True
        except Exception as exc:
            # Mid-call: a 5xx or refused connection may be a restart, so
            # the reconnect gets a second try before the call moves to HTTP.
            if not self._can_fall_back(exc, transient=attempt_number >= 2):
                raise
            self._fall_back(exc)  # the endpoint is gone: carry on over HTTP
            return True

    async def _verify_connection(self) -> bool:
        try:
            if not self._websocket or self._websocket.state is State.CLOSED:
                return False
            await self._websocket.ping()
            return True
        except Exception as exc:
            logger.error(f"{self} connection verification failed: {exc}")
            return False

    async def _report_error(self, message: str, exc: Exception | None, *, permanent: bool = False):
        if permanent:
            self._usable = False
        error = FrequencyError(message, category=errors.CONNECTIVITY, permanent=permanent)
        error.__cause__ = exc
        await self.listener.on_connection_error(message)
        await self.listener.on_error(error)

    # ---------- sending text ----------

    def turn(self, turn_id: str | None = None) -> TTSTurn:
        """A new reply, spoken on this stream (with the default listener)."""
        if not isinstance(self.listener, _TurnListener):
            raise RuntimeError("turn() needs the default listener; drive the stream by turn id instead")
        turn = TTSTurn(self, turn_id or uuid.uuid4().hex)
        self._turn_objects[turn.id] = turn
        return turn

    async def send_text(self, turn_id: str, text: str):
        """Send ``text`` for ``turn_id``; its audio goes to the listener as it arrives.

        Raises :class:`FrequencyError` when there is no socket to send on (the
        listener has been told why). Over the HTTP fallback, a sentence is
        spoken before this returns.
        """
        if not self._fallback and not self._is_open():
            await self._connect()
            if not self._fallback and not self._is_open():
                raise self._not_connected("Mirai TTS: not connected")
        if self._fallback:
            await self._fallback_tts(text, turn_id)
            return
        turn = self._turns.get(turn_id)
        if turn is None:
            self.last_error = None
            turn = _Turn(id=turn_id, server_id=self._new_server_id(turn_id))
            self._turns[turn_id] = turn
            self._by_server[turn.server_id] = turn
        if turn.held is not None:  # a retry is pending: it sends this too
            turn.held += text
            return
        flush = not self.tokens
        msg = {"type": "text", "context_id": turn.server_id, "text": text}
        if flush:
            msg["flush"] = True
            turn.marks += 1
        turn.record_sent(text, separate=flush)
        try:
            await self._send(msg)
        except Exception as exc:
            # The socket died under us. Reconnecting settles this turn (the
            # text just recorded is resent once, with the rest of the reply).
            logger.warning(f"{self}: could not send text ({exc!r}); reconnecting")
            await self._connect()
            if not self._is_open():
                raise self._not_connected(f"Mirai TTS: could not send text: {exc}") from exc

    def _not_connected(self, otherwise: str) -> FrequencyError:
        """Why there is no socket: the connect failure just reported, if that is the latest error."""
        known = self._connect_error
        if known is not None and self.last_error == known.message:
            return FrequencyError(
                known.message, category=known.category, status=known.status, request_id=known.request_id
            )
        return FrequencyError(self.last_error or otherwise, category=errors.CONNECTIVITY)

    async def end_turn(self, turn_id: str):
        """No more text is coming for ``turn_id``: speak what's left, then end it."""
        listener = self.listener
        if self._fallback and turn_id not in self._turns:
            rest = self._fallback_text.pop(turn_id, "")
            if rest.strip() and listener.turn_active(turn_id):
                await self._speak_over_http(rest, turn_id)
            if listener.turn_active(turn_id):
                await listener.on_turn_done(turn_id)
            return
        turn = self._turns.get(turn_id)
        if turn is None:
            # Nothing reached Mirai for it (a send failed): end it now rather
            # than leave the host waiting.
            if listener.turn_active(turn_id):
                await listener.on_turn_done(turn_id)
            return
        turn.final = True
        if turn.held is not None:
            return  # the retry flushes
        if self.tokens:
            turn.marks += 1
            try:
                await self._send({"type": "flush", "context_id": turn.server_id})
            except Exception as exc:
                logger.warning(f"{self}: could not flush context {turn.id} ({exc!r}); reconnecting")
                await self._connect()  # resends the rest of the turn, flushed
                return
        await self._maybe_complete(turn)

    async def update(self, *, voice: str | None = None, model: str | None = None):
        """Change the voice or model; it applies from the next sentence."""
        if voice is not None:
            self.voice = voice
        if model is not None:
            self.model = model
        self._http.voice, self._http.model = self.voice, self.model
        if self.voice not in VOICES:
            logger.warning(f"{self}: voice {self.voice!r} is not one of {', '.join(VOICES)}")
        if self._is_open():
            try:
                await self._send(self._session_update())
            except Exception as exc:
                logger.warning(f"{self}: could not update the session: {exc!r}")

    def _new_server_id(self, turn_id: str) -> str:
        server_id, n = turn_id, 0
        while server_id in self._used_ids:
            n += 1
            server_id = f"{turn_id}#{n}"
        self._used_ids.add(server_id)
        return server_id

    def _restart(self, turn: _Turn):
        """Move ``turn`` to a fresh Mirai context; the old one is forgotten."""
        self._by_server.pop(turn.server_id, None)
        turn.server_id = self._new_server_id(turn.id)
        turn.sent, turn.cursor, turn.marks = "", 0, 0
        self._by_server[turn.server_id] = turn

    async def _send_held(self, turn: _Turn):
        """Send the text a retry held back, flushed if the turn is over."""
        text, turn.held = turn.held or "", None
        if not self._is_current(turn):
            await self._forget(turn)
            return
        flush = turn.final or not self.tokens
        if text.strip():
            msg = {"type": "text", "context_id": turn.server_id, "text": text}
            if flush:
                msg["flush"] = True
                turn.marks += 1
            turn.record_sent(text, separate=False)
            try:
                await self._send(msg)
            except Exception as exc:
                logger.warning(f"{self}: could not resend context {turn.id}: {exc!r}")
                if flush:
                    turn.marks -= 1
                return
        await self._maybe_complete(turn)

    def _is_current(self, turn: _Turn) -> bool:
        return self._turns.get(turn.id) is turn and self.listener.turn_active(turn.id)

    async def _maybe_complete(self, turn: _Turn):
        """End the turn once Mirai has spoken all of it."""
        if not turn.final or turn.marks > 0 or turn.held is not None:
            return
        await self._forget(turn)
        if self.listener.turn_active(turn.id):
            await self.listener.on_turn_done(turn.id)

    async def _forget(self, turn: _Turn):
        if self._turns.get(turn.id) is turn:
            del self._turns[turn.id]
        if self._by_server.get(turn.server_id) is turn:
            del self._by_server[turn.server_id]
        task, turn.retry_task = turn.retry_task, None
        if task is not None and task is not asyncio.current_task() and not task.done():
            await _cancel(task)

    # ---------- interruptions ----------

    def begin_interrupt(self) -> list[httpx.Response]:
        """The caller barged in: what the HTTP fallback is speaking stops here.

        Returns the HTTP responses to close in :meth:`finish_interrupt`, once
        the host has stopped reading them.
        """
        self._fallback_text.clear()
        return self._http.begin_interrupt()

    async def finish_interrupt(self, cut_short: list[httpx.Response]):
        """Close what :meth:`begin_interrupt` cut short and cancel every turn on the server."""
        await self._http.finish_interrupt(cut_short, after_cut=False)
        # Turns the host no longer tracks (one that timed out between
        # sentences, say) may still have text at Mirai: cancel those too.
        for turn in list(self._turns.values()):
            await self._cancel_turn(turn)

    async def interrupt(self):
        """Barge-in: stop every turn now, here and on the server."""
        for turn in list(self._turn_objects.values()):
            turn._finish(cancelled=True)
        await self.finish_interrupt(self.begin_interrupt())

    async def cancel_turn(self, turn_id: str):
        """Stop ``turn_id``: cancel it on the server and drop what is still arriving."""
        turn = self._turns.get(turn_id)
        if turn is not None:
            await self._cancel_turn(turn)

    async def _cancel_turn(self, turn: _Turn):
        server_id = turn.server_id
        sent_something = bool(turn.sent) or turn.marks > 0
        await self._forget(turn)
        if self._sentence is not None and self._sentence.turn is turn:
            self._sentence.turn = None  # the rest of its audio is dropped
        if not turn.got_audio:
            await self.listener.on_no_audio(turn.id)
        if not sent_something:
            return
        # Everything for this context is dropped until Mirai acknowledges.
        self._cancelled.add(server_id)
        try:
            await self._send({"type": "cancel", "context_id": server_id})
        except Exception as exc:
            self._cancelled.discard(server_id)  # that socket is gone, and its audio with it
            logger.debug(f"{self}: could not cancel context {server_id}: {exc!r}")

    # ---------- receiving ----------

    async def _receive_messages(self):
        if self._fallback:
            await asyncio.Event().wait()  # nothing to read: the call is on HTTP now
        async for message in self._get_websocket():
            if isinstance(message, bytes):
                await self._on_audio(message)
                continue
            try:
                event = json.loads(message)
            except ValueError:
                logger.warning(f"{self}: ignoring a message that is not JSON: {message[:200]!r}")
                continue
            if isinstance(event, dict):
                await self._on_event(event)

    async def _on_event(self, event: dict):
        kind = event.get("type")
        context_id = event.get("context_id")
        if kind == "audio.start":
            await self._on_audio_start(event)
        elif kind == "audio.chunk":
            await self._on_audio(base64.b64decode(event.get("data") or ""))
        elif kind == "audio.done":
            await self._on_audio_done(event)
        elif kind == "context.done":
            turn = None if context_id in self._cancelled else self._by_server.get(context_id)
            if turn is not None:
                turn.marks = max(0, turn.marks - 1)
                await self._maybe_complete(turn)
        elif kind == "context.cancelled":
            self._cancelled.discard(context_id)
        elif kind == "error":
            await self._on_error(event)
        elif kind in ("session.ready", "session.updated"):
            self.session_id = event.get("session_id") or self.session_id
            idle = (event.get("limits") or {}).get("idle_timeout_secs")
            if isinstance(idle, int | float) and idle > 0:
                self._idle_timeout_secs = float(idle)
            logger.debug(
                f"{self}: {kind} {self.session_id}: voice {event.get('voice')}, "
                f"{event.get('response_format')} {event.get('sample_rate')} Hz"
            )
        elif kind == "session.closed":
            logger.debug(f"{self}: Mirai is closing the session ({event.get('reason')})")
        else:
            logger.debug(f"{self}: ignoring Mirai event {kind!r}")

    async def _on_audio_start(self, event: dict):
        server_id = event.get("context_id")
        turn = None if server_id in self._cancelled else self._by_server.get(server_id)
        sentence = _Sentence(
            request_id=event.get("request_id"), server_id=server_id, text=event.get("text") or "", turn=turn
        )
        self._sentence = sentence
        if turn is None:
            return
        encoding = str(event.get("encoding") or "pcm_s16le").lower()
        try:
            rate = int(event.get("sample_rate") or SOURCE_SAMPLE_RATE)
        except (TypeError, ValueError):
            rate = 0
        if encoding != "pcm_s16le" or rate <= 0:
            sentence.turn = None
            await self._report(
                f"Mirai TTS sent {encoding} audio at {event.get('sample_rate')!r} Hz; expected pcm_s16le",
                errors.SERVER,
                turn.id,
            )
            return
        if rate != self.sample_rate:
            sentence.resampler = soxr.ResampleStream(rate, self.sample_rate, 1, dtype="int16", quality="HQ")
        if rate != self.last_server_sample_rate:
            resampling = f", resampling to {self.sample_rate} Hz" if sentence.resampler else ""
            logger.debug(f"{self}: Mirai sent {rate} Hz PCM{resampling}")
        self.last_server_sample_rate = rate

    async def _on_audio(self, data: bytes):
        sentence = self._sentence
        if sentence is None or sentence.turn is None:
            return  # cancelled, or not ours: never handed out
        # Mirai sends whole samples, but be safe if a frame ever ends mid-sample.
        data = sentence.carry + data
        end = len(data) - len(data) % 2
        sentence.carry, data = data[end:], data[:end]
        if not data:
            return
        turn = sentence.turn
        if not turn.got_audio:
            turn.got_audio = True
            await self.listener.on_first_audio(turn.id)  # TTFB = first audio byte received
        if sentence.resampler is not None:
            data = sentence.resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
        sentence.pending += data
        if not sentence.playing:
            if len(sentence.pending) < self._prebuffer_bytes:
                return
            sentence.playing = True
        await self._push_audio(sentence, last=False)

    async def _push_audio(self, sentence: _Sentence, *, last: bool):
        """Hand out whole 40 ms frames (and, at the end, the remainder)."""
        size = self._frame_bytes
        pending = sentence.pending
        end = len(pending) if last else len(pending) - len(pending) % size
        for offset in range(0, end, size):
            turn = sentence.turn
            if turn is None:
                return
            audio = bytes(pending[offset : offset + size])
            await self.listener.on_audio(turn.id, _encode(audio, self.encoding))
            sentence.pushed += len(audio)
        del pending[:end]

    async def _on_audio_done(self, event: dict):
        sentence = self._sentence
        request_id = event.get("request_id")
        if sentence is not None and (request_id is None or sentence.request_id == request_id):
            self._sentence = None
            turn = sentence.turn
            if turn is not None:
                if sentence.carry:
                    logger.warning(f"{self}: Mirai's audio ended with half a sample; dropped it")
                if sentence.resampler is not None:
                    tail = sentence.resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True)
                    sentence.pending += tail.tobytes()
                await self._push_audio(sentence, last=True)
                if sentence.server_id == turn.server_id:
                    turn.spoken(sentence.text)
        # Billed whether or not it was heard: report it either way.
        chars = event.get("chars_billed")
        if isinstance(chars, int) and chars > 0:
            await self.listener.on_usage(chars)

    async def _on_error(self, event: dict):
        code = str(event.get("code") or "error")
        message = str(event.get("message") or "")
        server_id = event.get("context_id")
        request_id = event.get("request_id")
        sentence = self._sentence
        if sentence is not None and request_id and sentence.request_id == request_id:
            self._sentence = None  # that sentence is over; what's buffered of it is dropped
        if server_id and server_id in self._cancelled:
            return  # about a context already cancelled
        turn = self._by_server.get(server_id) if server_id else None
        retry_after = event.get("retry_after_secs") or 0
        if (
            turn is not None
            and isinstance(retry_after, int | float)
            and 0 < retry_after <= RETRY_AFTER_LIMIT_SECS
            and not turn.retried
            and self._is_current(turn)
        ):
            await self._retry_later(turn, float(retry_after), code, message)
            return
        where = f" (context {turn.id})" if turn is not None else ""
        ids = ", ".join(
            f"{name} {value}"
            for name, value in (("session", self.session_id), ("request", request_id))
            if value
        )
        advice = f". {_ADVICE[code]}" if code in _ADVICE else ""
        await self._report(
            f"Mirai TTS error {code}: {message}{where}{f' [{ids}]' if ids else ''}{advice}",
            _ERROR_CATEGORIES.get(code, errors.SERVER),
            turn.id if turn is not None else None,
            code=code,
            request_id=request_id,
        )
        # The socket stays open and Mirai goes on with the next sentence; the
        # context's flush is still answered with context.done.

    async def _retry_later(self, turn: _Turn, delay: float, code: str, message: str):
        """Cancel what's left of ``turn`` and send it again after ``delay``.

        Mirai goes on to the next sentence after a refused one; cancelling
        keeps it from playing the rest of the reply with a sentence missing.
        """
        turn.retried = True
        old = turn.server_id
        turn.held = turn.unspoken()
        self._restart(turn)
        logger.warning(f"{self}: Mirai {code} ({message}); resending context {turn.id} in {delay:g} s")
        self._cancelled.add(old)
        try:
            await self._send({"type": "cancel", "context_id": old})
        except Exception:
            self._cancelled.discard(old)
        turn.retry_task = asyncio.create_task(self._retry_after(turn, delay), name=f"{self}.retry")

    async def _retry_after(self, turn: _Turn, delay: float):
        deadline = time.monotonic() + delay
        while (left := deadline - time.monotonic()) > 0:
            self.listener.keep_turn_open(turn.id)
            await asyncio.sleep(min(left, 1.0))
        turn.retry_task = None
        if self._turns.get(turn.id) is not turn:
            return
        if self._is_open():
            await self._send_held(turn)
        else:
            await self._connect()  # sends it once the socket is up

    async def _abandon_turns(self, *, intentional: bool):
        """The socket is gone: settle every turn that was using it.

        On a drop, a turn that still has unspoken text is resent once on the
        next socket, from the first sentence the caller hasn't started
        hearing. A second loss, or a turn no longer current, is reported.
        """
        sentence, self._sentence = self._sentence, None
        self._cancelled.clear()
        for turn in list(self._turns.values()):
            if intentional:
                await self._forget(turn)
                continue
            if turn.held is not None:
                continue  # already waiting to be resent
            heard = (
                sentence.text if sentence is not None and sentence.turn is turn and sentence.pushed else ""
            )
            rest = turn.unspoken(skip=heard)
            outstanding = bool(rest.strip()) or turn.marks > 0
            self._restart(turn)
            if not outstanding:
                await self._maybe_complete(turn)
            elif not turn.retried and self._is_current(turn):
                turn.retried = True
                turn.held = rest
                logger.warning(f"{self}: connection lost; resending the rest of context {turn.id}")
            else:
                await self._report(
                    f"Mirai TTS connection lost; the rest of context {turn.id} was not spoken",
                    errors.CONNECTIVITY,
                    turn.id,
                )
                await self._maybe_complete(turn)

    async def _report(
        self,
        message: str,
        category: str,
        turn_id: str | None = None,
        *,
        code: str | None = None,
        request_id: str | None = None,
    ):
        self.last_error = message
        error = FrequencyError(message, category=category, code=code, request_id=request_id)
        await self.listener.on_error(error, turn_id)


# ---------- turns, for hosts without their own listener ----------


class TTSTurn:
    """One reply on a :class:`TTSStream`: send its text, then read its audio.

    Sending and reading usually run side by side (the LLM's tokens in, the
    audio out to the phone); the audio starts long before the text ends.
    Iterating yields ``bytes`` until Mirai has spoken the whole turn (after
    :meth:`end`) or the turn is cancelled. Errors that cost the turn a
    sentence are in :attr:`errors`; the turn carries on with the rest.
    """

    def __init__(self, stream: TTSStream, turn_id: str):
        self.id = turn_id
        self._stream = stream
        self._audio: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.errors: list[FrequencyError] = []
        self.done = False
        self.cancelled = False

    async def send(self, text: str):
        """Send text: whole sentences (``text_mode="sentence"``) or tokens."""
        if self.done:
            raise RuntimeError(f"turn {self.id} is over")
        await self._stream.send_text(self.id, text)

    async def end(self):
        """No more text: Mirai speaks the rest, then the audio ends."""
        if not self.done:
            await self._stream.end_turn(self.id)

    async def cancel(self):
        """Stop now (barge-in): cancelled on the server, no more audio."""
        if self.cancelled:
            return
        self._finish(cancelled=True)
        await self._stream.cancel_turn(self.id)

    def _finish(self, *, cancelled: bool = False):
        if self.done:
            return
        self.done = True
        self.cancelled = cancelled
        self._audio.put_nowait(None)
        self._stream._turn_objects.pop(self.id, None)

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self.cancelled and self._audio.empty():
            raise StopAsyncIteration
        audio = await self._audio.get()
        if audio is None or self.cancelled:
            self._audio.put_nowait(None)  # a second reader stops too
            raise StopAsyncIteration
        return audio


class _TurnListener(TTSListener):
    """The default listener: each turn's audio goes to its :class:`TTSTurn`."""

    def __init__(self, stream: TTSStream):
        self._stream = stream

    def _turn(self, turn_id: str) -> TTSTurn | None:
        return self._stream._turn_objects.get(turn_id)

    async def on_audio(self, turn_id: str, audio: bytes) -> None:
        turn = self._turn(turn_id)
        if turn is not None and not turn.done:
            turn._audio.put_nowait(audio)

    async def on_turn_done(self, turn_id: str) -> None:
        turn = self._turn(turn_id)
        if turn is not None:
            turn._finish()

    def turn_active(self, turn_id: str) -> bool:
        turn = self._turn(turn_id)
        return turn is not None and not turn.done

    async def on_error(self, error: FrequencyError, turn_id: str | None = None) -> None:
        turn = self._turn(turn_id) if turn_id else None
        if turn is not None:
            turn.errors.append(error)
        level = logging.ERROR if error.permanent else logging.WARNING
        logger.log(level, f"{self._stream}: {error}")
        if error.permanent:
            # The stream gave up: no turn will hear more, so none waits for it.
            for waiting in list(self._stream._turn_objects.values()):
                waiting.errors.append(error)
                waiting._finish()


# ---------- helpers ----------


def _backoff(attempt: int) -> float:
    """Seconds to wait after failed reconnect ``attempt`` (1-based)."""
    return max(RECONNECT_MIN_WAIT, min(2 ** (attempt - 1), RECONNECT_MAX_WAIT))


async def _cancel(task: asyncio.Task):
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception as exc:
        logger.debug(f"error closing a Mirai socket: {exc!r}")


def http_base(url: str) -> str:
    """``wss://host/v1/audio/speech/stream`` -> ``https://host/v1``."""
    for ws, http in (("wss://", "https://"), ("ws://", "http://")):
        if url.startswith(ws):
            url = http + url[len(ws) :]
    return re.sub(r"/audio/speech/stream/?$", "", url.rstrip("/"))


def stream_url(base: str) -> str:
    """``https://host/v1`` -> ``wss://host/v1/audio/speech/stream``."""
    base = base.rstrip("/")
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            return ws + base[len(http) :] + "/audio/speech/stream"
    return base + "/audio/speech/stream"


def _ws_error_detail(body: Any) -> str | None:
    """The ``error.message`` of a JSON error body (a refused handshake's), if there is one."""
    try:
        detail = json.loads(body).get("error", {}).get("message")
    except (TypeError, ValueError, AttributeError):
        return None
    return detail[:500] if isinstance(detail, str) else None


def _source_rate(response: httpx.Response) -> int:
    """Check that ``response`` carries 16-bit PCM and return its sample rate."""
    media_type = response.headers.get("content-type", "").split(";")[0].strip()
    if media_type != "audio/pcm":
        raise ValueError(f"expected audio/pcm, received {media_type or 'no content type'}")
    encoding = response.headers.get("x-audio-encoding", "pcm_s16le").strip().lower()
    if encoding != "pcm_s16le":
        raise ValueError(f"expected pcm_s16le audio, received {encoding or 'no encoding'}")
    header = response.headers.get("x-sample-rate")
    if header is None:
        return SOURCE_SAMPLE_RATE  # servers before per-request rates always sent 48 kHz
    try:
        rate = int(header)
    except ValueError:
        rate = 0
    if rate <= 0:
        raise ValueError(f"invalid X-Sample-Rate {header!r}")
    return rate


def _error_info(response: httpx.Response) -> tuple[str, str | None]:
    """The ``error.code`` and ``error.message`` of a JSON error body."""
    try:
        error = response.json().get("error", {})
    except (ValueError, httpx.HTTPError, AttributeError):
        return "", None
    if not isinstance(error, dict):
        return "", None
    code = error.get("code") if isinstance(error.get("code"), str) else ""
    message = error.get("message") if isinstance(error.get("message"), str) else None
    return code, message[:500] if message else None


def _error_detail(response: httpx.Response) -> str | None:
    """The ``error.message`` of a JSON error body, if there is one."""
    return _error_info(response)[1]


def _retry_after(response: httpx.Response, default: float) -> float:
    try:
        return max(0.0, float(response.headers.get("retry-after")))
    except (TypeError, ValueError):
        return default


def _rid(request_id: str | None) -> str:
    return f" [request {request_id}]" if request_id else ""


def _http_category(status: int) -> str:
    if status == 401:
        return errors.AUTHENTICATION
    if status == 402:
        return errors.QUOTA
    if status == 403:
        return errors.AUTHORIZATION
    if status == 429:
        return errors.RATE_LIMIT
    if status >= 500:
        return errors.SERVER
    return errors.INVALID_REQUEST


def _explain(status: int, code: str, detail: str | None, request_id: str | None, waited: float) -> str:
    """One line: what Mirai said, and what to do about it."""
    said = f"HTTP {status}" + (f": {detail}" if detail else (f" {code}" if code else ""))
    retried = f", still after {waited:g} s of retries" if waited else ""
    if status == 401:
        advice = "Mirai rejected the API key. Check MIRAI_API_KEY."
    elif status == 402:
        advice = "The workspace is out of credit. Top it up in the Mirai console."
    elif status == 403:
        advice = "The API key is revoked or not allowed this request."
    elif status == 429 and code == "rate_limited":
        advice = f"The API key's request rate limit was reached{retried}. Ask Mirai to raise it."
    elif status == 429:
        advice = f"Mirai was at capacity{retried}. Ask Mirai to raise this workspace's concurrency limit."
    elif status >= 500:
        advice = (
            "Mirai had a server error, also on a retry. If it persists, contact Mirai with the request id."
        )
    else:
        advice = "Mirai refused the request."
    return f"Mirai TTS returned {said}{_rid(request_id)}. {advice}"
