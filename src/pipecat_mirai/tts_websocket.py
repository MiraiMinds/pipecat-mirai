#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech over one WebSocket per pipeline.

``MiraiWebsocketTTSService`` keeps a single socket to Mirai's
``/v1/audio/speech/stream`` open for the whole call. Text is sent as Pipecat
produces it, Mirai cuts it into sentences, synthesises the next sentence while
the current one streams, and sends the audio back on the same socket. Nothing
pays a TCP and TLS handshake after the first one.

The protocol, in short. Client: ``session.update`` (voice, model,
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
import json
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import soxr
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    MetricsFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.metrics.metrics import TTSUsageMetricsData
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import WebsocketTTSService
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts
from websockets.protocol import State

from pipecat_mirai.tts import (
    FRAME_SECS,
    SERVER_SAMPLE_RATES,
    SOURCE_SAMPLE_RATE,
    VOICES,
    MiraiTTSSettings,
)

DEFAULT_WEBSOCKET_URL = "wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream"
# A capacity error is retried once when Mirai asks for a wait no longer than
# this. Longer than that, a caller would sit through the silence; it is
# reported instead.
RETRY_AFTER_LIMIT_SECS = 5.0
# Ceiling on one server message. Audio frames are a few KB; this is headroom.
MAX_MESSAGE_BYTES = 16 * 1024 * 1024

# Mirai's error codes, as Pipecat error categories. An unknown code is SERVER.
_ERROR_CATEGORIES = {
    "unauthorized": ErrorCategory.AUTHENTICATION,
    "forbidden": ErrorCategory.AUTHORIZATION,
    "invalid_request": ErrorCategory.INVALID_REQUEST,
    "model_not_found": ErrorCategory.INVALID_REQUEST,
    "insufficient_balance": ErrorCategory.QUOTA,
    "at_capacity": ErrorCategory.RATE_LIMIT,
    "rate_limited": ErrorCategory.RATE_LIMIT,
    "queue_full": ErrorCategory.RATE_LIMIT,
    "too_many_contexts": ErrorCategory.RATE_LIMIT,
    "session_closing": ErrorCategory.CONNECTIVITY,
}
_HANDSHAKE_CATEGORIES = {
    401: ErrorCategory.AUTHENTICATION,
    402: ErrorCategory.QUOTA,
    403: ErrorCategory.AUTHORIZATION,
    429: ErrorCategory.RATE_LIMIT,
}


@dataclass
class _Turn:
    """One Pipecat audio context, and the Mirai context that speaks it.

    The Mirai context id is the Pipecat one, until the context has to be
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
    # Pipecat has flushed the turn: no more text is coming.
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


class MiraiWebsocketTTSService(WebsocketTTSService):
    """Stream Mirai TTS over one WebSocket for the whole pipeline.

    The socket opens when the pipeline starts and carries every utterance.
    Text goes to Mirai as Pipecat produces it: each sentence as soon as it is
    aggregated (the default), or each LLM token with
    ``text_aggregation_mode=TextAggregationMode.TOKEN``. Mirai cuts sentences
    itself and starts synthesising the next one while the current one plays,
    so there is no gap and no handshake between sentences.

    Audio is asked for at the pipeline's output rate and pushed in 40 ms
    frames after a short first-audio buffer, like :class:`MiraiTTSService`.
    An interruption cancels the reply on the server at once and drops
    whatever of it is still arriving. A dropped connection is reopened, and
    a reply it cut short is resent once.

    Example::

        tts = MiraiWebsocketTTSService(
            api_key=os.getenv("MIRAI_API_KEY"),
            settings=MiraiWebsocketTTSService.Settings(voice="shruti"),
        )

    Event handlers (from Pipecat): ``on_connected``, ``on_disconnected``,
    ``on_connection_error``, ``on_tts_request``.
    """

    Settings = MiraiTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = DEFAULT_WEBSOCKET_URL,
        voice: str | None = None,
        model: str | None = None,
        sample_rate: int | None = None,
        server_sample_rate: int | Literal["auto"] | None = "auto",
        prebuffer_secs: float = 0.15,
        keepalive_secs: float | None = 30.0,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Mirai WebSocket TTS service.

        Args:
            api_key: Mirai API key. Defaults to the ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``) environment variable. Sent as an
                ``Authorization: Bearer`` header.
            url: The streaming endpoint, ``wss://<host>/v1/audio/speech/stream``.
            voice: Shortcut for ``settings.voice``. Defaults to ``"neha"``.
            model: Shortcut for ``settings.model``. Defaults to ``"mira-tts"``.
            sample_rate: Output sample rate. Defaults to the pipeline's
                ``audio_out_sample_rate``.
            server_sample_rate: The PCM rate to ask Mirai for. ``"auto"`` (the
                default) asks for the output rate when Mirai serves it (8000,
                16000, 24000 or 48000 Hz) and leaves the field out otherwise.
                An ``int`` from that list asks for that rate. ``None`` leaves
                it out, so Mirai sends 48 kHz. Each sentence's ``audio.start``
                says what rate it is in, and audio is resampled locally only
                when that differs from the output rate.
            prebuffer_secs: Audio to collect before the first frame of each
                sentence is pushed, so a short first chunk followed by a
                pause doesn't start playback and then stall. ``0`` pushes a
                frame as soon as 40 ms is in hand. TTFB is still measured at
                the first byte received.
            keepalive_secs: When nothing has been sent for this long, send an
                empty ``session.update`` so Mirai doesn't close the socket as
                idle during a long pause in the call (it closes after 120 s
                without a message). ``None`` turns it off.
            settings: Runtime-updatable settings; values here win over the
                ``voice``/``model`` shortcuts.
            **kwargs: Passed through to :class:`WebsocketTTSService`, e.g.
                ``text_aggregation_mode`` or ``reconnect_on_error``.
        """
        key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
        if not key:
            raise ValueError("Set MIRAI_API_KEY or pass api_key to MiraiWebsocketTTSService.")
        if not (
            server_sample_rate is None
            or server_sample_rate == "auto"
            or (type(server_sample_rate) is int and server_sample_rate in SERVER_SAMPLE_RATES)
        ):
            raise ValueError(
                f"server_sample_rate must be 'auto', None or one of "
                f"{', '.join(map(str, SERVER_SAMPLE_RATES))}; got {server_sample_rate!r}"
            )
        if not prebuffer_secs >= 0:
            raise ValueError(f"prebuffer_secs must be >= 0; got {prebuffer_secs!r}")
        if keepalive_secs is not None and not keepalive_secs > 0:
            raise ValueError(f"keepalive_secs must be > 0 or None; got {keepalive_secs!r}")
        if not url.startswith(("ws://", "wss://")):
            raise ValueError(f"url must be a ws:// or wss:// URL; got {url!r}")

        default_settings = self.Settings(model="mira-tts", voice="neha", language=None)
        if voice is not None:
            default_settings.voice = voice
        if model is not None:
            default_settings.model = model
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(
            sample_rate=sample_rate,
            push_start_frame=True,
            push_stop_frames=True,
            settings=default_settings,
            **kwargs,
        )
        self._url = url
        self._headers = {"Authorization": f"Bearer {key}"}
        self._server_rate_option = server_sample_rate
        self._prebuffer_secs = float(prebuffer_secs)
        self._keepalive_secs = keepalive_secs
        self._frame_bytes = 2
        self._prebuffer_bytes = 0

        self._receive_task: asyncio.Task | None = None
        self._keepalive_task: asyncio.Task | None = None
        self._connect_lock = asyncio.Lock()
        self._last_sent = time.monotonic()
        self._idle_timeout_secs: float | None = None

        # Pipecat context id -> turn, and Mirai context id -> turn.
        self._turns: dict[str, _Turn] = {}
        self._by_server: dict[str, _Turn] = {}
        # Mirai context ids used so far (never reused), and those cancelled
        # whose context.cancelled hasn't arrived.
        self._used_ids: set[str] = set()
        self._cancelled: set[str] = set()
        self._sentence: _Sentence | None = None

        self.last_error: str | None = None
        # The rate of the PCM Mirai sent for the latest sentence (audio.start).
        self.last_server_sample_rate: int | None = None
        # The socket's session id (``ttsws_...``), from session.ready.
        self.session_id: str | None = None

    def can_generate_metrics(self) -> bool:
        """Mirai TTS reports TTFB and usage metrics."""
        return True

    # Mirai bills each sentence and says how much in audio.done; that is the
    # usage reported. Pipecat's own count (the length of the text it sent) is
    # skipped: in sentence mode through this method, and in token mode, where
    # Pipecat reports the turn's accumulated `_streamed_text` directly, by
    # keeping that accumulator empty.
    async def start_tts_usage_metrics(self, text: str):
        """Usage is reported from Mirai's ``audio.done`` instead."""

    @property
    def _streamed_text(self) -> str:
        return ""

    @_streamed_text.setter
    def _streamed_text(self, value: str):
        pass

    # ---------- lifecycle ----------

    async def start(self, frame: StartFrame):
        """Start the service and open the socket."""
        await super().start(frame)
        rate = self.sample_rate
        self._frame_bytes = max(1, round(rate * FRAME_SECS)) * 2
        self._prebuffer_bytes = round(rate * self._prebuffer_secs) * 2
        if self._settings.voice not in VOICES:
            logger.warning(f"{self}: voice {self._settings.voice!r} is not one of {', '.join(VOICES)}")
        requested = self._requested_server_rate()
        asked = f"{requested} Hz" if requested else f"its default {SOURCE_SAMPLE_RATE} Hz"
        logger.debug(f"{self}: asking Mirai for {asked} PCM; output {rate} Hz")
        await self._connect()

    def _requested_server_rate(self) -> int | None:
        if self._server_rate_option == "auto":
            return self.sample_rate if self.sample_rate in SERVER_SAMPLE_RATES else None
        return self._server_rate_option

    def _session_update(self) -> dict:
        msg = {
            "type": "session.update",
            "voice": self._settings.voice,
            "model": self._settings.model,
            "response_format": "pcm",
            "audio_transport": "binary",
        }
        rate = self._requested_server_rate()
        if rate is not None:
            msg["sample_rate"] = rate
        return msg

    async def _connect(self):
        await super()._connect()
        try:
            await self._connect_websocket()
        except Exception as exc:
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
                await self.cancel_task(task)
                self._receive_task = None
            if self._receive_task is None or self._receive_task.done():
                self._receive_task = self.create_task(
                    self._receive_task_handler(self._report_error), name="receive"
                )
            if self._keepalive_secs and (self._keepalive_task is None or self._keepalive_task.done()):
                self._keepalive_task = self.create_task(self._keepalive_task_handler(), name="keepalive")

    async def _disconnect(self):
        await super()._disconnect()
        for name in ("_keepalive_task", "_receive_task"):
            task = getattr(self, name)
            setattr(self, name, None)
            if task is not None and task is not asyncio.current_task() and not task.done():
                await self.cancel_task(task)
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        """Open the socket and set up the session. Raises if it can't."""
        async with self._connect_lock:
            if self._is_open():
                return
            logger.debug(f"{self}: connecting to {self._url}")
            self._websocket = await self._websocket_connect(
                self._url, additional_headers=self._headers, max_size=MAX_MESSAGE_BYTES
            )
            self._cancelled.clear()
            self._sentence = None
            await self._send(self._session_update())
            await self._call_event_handler("on_connected")
        # Replies a dropped socket cut short go out on the new one.
        for turn in list(self._turns.values()):
            if turn.held is not None and turn.retry_task is None:
                await self._send_held(turn)

    async def _reconnect_websocket(self, attempt_number: int) -> bool:
        if self._is_open():
            return True  # already reopened (by run_tts) while this waited to retry
        return await super()._reconnect_websocket(attempt_number)

    async def _disconnect_websocket(self):
        websocket, self._websocket = self._websocket, None
        await self._abandon_turns(intentional=self._disconnecting)
        if websocket is None:
            return
        try:
            # A plain close: Mirai drops (and doesn't bill) anything unsent.
            await websocket.close()
        except Exception as exc:
            logger.debug(f"{self}: error closing the socket: {exc!r}")
        logger.debug(f"{self}: disconnected from Mirai")
        await self._call_event_handler("on_disconnected")

    async def _report_connect_failure(self, exc: Exception):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
        if status is not None:
            message = f"Mirai TTS refused the WebSocket: HTTP {status}"
            detail = _error_detail(getattr(response, "body", None))
            if detail:
                message += f": {detail}"
            category = _HANDSHAKE_CATEGORIES.get(status, ErrorCategory.CONNECTIVITY)
        else:
            message = f"Mirai TTS could not connect to {self._url}: {exc}"
            category = ErrorCategory.CONNECTIVITY
        self.last_error = message
        await self.push_error(error_msg=message, exception=exc, category=category)
        await self._call_event_handler("on_connection_error", message)

    def _is_open(self) -> bool:
        return self._websocket is not None and self._websocket.state is State.OPEN

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

    # ---------- sending text ----------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send ``text`` for ``context_id``; its audio arrives on the receive task."""
        if not self._is_open():
            await self._connect()
            if not self._is_open():
                yield ErrorFrame(error=self.last_error or "Mirai TTS: not connected")
                return
        turn = self._turns.get(context_id)
        if turn is None:
            self.last_error = None
            turn = _Turn(id=context_id, server_id=self._new_server_id(context_id))
            self._turns[context_id] = turn
            self._by_server[turn.server_id] = turn
        if turn.held is not None:  # a retry is pending: it sends this too
            turn.held += text
            yield None
            return
        flush = not self._is_streaming_tokens
        msg = {"type": "text", "context_id": turn.server_id, "text": text}
        if flush:
            msg["flush"] = True
            turn.marks += 1
        turn.record_sent(text, separate=flush)
        try:
            await self._send(msg)
        except Exception as exc:
            if flush:
                turn.marks -= 1
            self.last_error = f"Mirai TTS: could not send text: {exc}"
            yield ErrorFrame(error=self.last_error, exception=exc, category=ErrorCategory.CONNECTIVITY)
            return
        yield None

    async def flush_audio(self, context_id: str | None = None):
        """Pipecat has sent the whole turn: ask Mirai to speak what's left."""
        context_id = context_id or self.get_active_audio_context_id()
        if not context_id:
            return
        turn = self._turns.get(context_id)
        if turn is None:
            # Nothing reached Mirai for it (a send failed): end it now rather
            # than leave Pipecat waiting out its stop timeout.
            if self.audio_context_available(context_id):
                await self.remove_audio_context(context_id)
            return
        turn.final = True
        if turn.held is not None:
            return  # the retry flushes
        if self._is_streaming_tokens:
            turn.marks += 1
            try:
                await self._send({"type": "flush", "context_id": turn.server_id})
            except Exception as exc:
                turn.marks -= 1
                logger.warning(f"{self}: could not flush context {turn.id}: {exc!r}")
                return  # the reconnect resends it, flushed
        await self._maybe_complete(turn)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta; a new voice or model applies from the next sentence."""
        changed = await super()._update_settings(delta)
        if changed.keys() & {"voice", "model"}:
            if self._settings.voice not in VOICES:
                logger.warning(f"{self}: voice {self._settings.voice!r} is not one of {', '.join(VOICES)}")
            if self._is_open():
                try:
                    await self._send(self._session_update())
                except Exception as exc:
                    logger.warning(f"{self}: could not update the session: {exc!r}")
        return changed

    def _new_server_id(self, context_id: str) -> str:
        server_id, n = context_id, 0
        while server_id in self._used_ids:
            n += 1
            server_id = f"{context_id}#{n}"
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
        flush = turn.final or not self._is_streaming_tokens
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
        return self._turns.get(turn.id) is turn and self.audio_context_available(turn.id)

    async def _maybe_complete(self, turn: _Turn):
        """End the Pipecat audio context once Mirai has spoken all of the turn."""
        if not turn.final or turn.marks > 0 or turn.held is not None:
            return
        await self._forget(turn)
        if self.audio_context_available(turn.id):
            await self.remove_audio_context(turn.id)

    async def _forget(self, turn: _Turn):
        if self._turns.get(turn.id) is turn:
            del self._turns[turn.id]
        if self._by_server.get(turn.server_id) is turn:
            del self._by_server[turn.server_id]
        task, turn.retry_task = turn.retry_task, None
        if task is not None and task is not asyncio.current_task() and not task.done():
            await self.cancel_task(task)

    # ---------- interruptions ----------

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        await super()._handle_interruption(frame, direction)
        # Contexts Pipecat no longer tracks (one that timed out between
        # sentences, say) may still have text at Mirai: cancel those too.
        for turn in list(self._turns.values()):
            await self._cancel_turn(turn)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the interrupted context on the server."""
        turn = self._turns.get(context_id)
        if turn is not None:
            await self._cancel_turn(turn)
        await super().on_audio_context_interrupted(context_id)

    async def _cancel_turn(self, turn: _Turn):
        server_id = turn.server_id
        sent_something = bool(turn.sent) or turn.marks > 0
        await self._forget(turn)
        if self._sentence is not None and self._sentence.turn is turn:
            self._sentence.turn = None  # the rest of its audio is dropped
        if not turn.got_audio:
            await self.cancel_ttfb_metrics()
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
                ErrorCategory.SERVER,
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
            return  # cancelled, or not ours: never pushed
        # Mirai sends whole samples, but be safe if a frame ever ends mid-sample.
        data = sentence.carry + data
        end = len(data) - len(data) % 2
        sentence.carry, data = data[end:], data[:end]
        if not data:
            return
        turn = sentence.turn
        if not turn.got_audio:
            turn.got_audio = True
            await self.stop_ttfb_metrics()  # TTFB = first audio byte received
        if sentence.resampler is not None:
            data = sentence.resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
        sentence.pending += data
        if not sentence.playing:
            if len(sentence.pending) < self._prebuffer_bytes:
                return
            sentence.playing = True
        await self._push_audio(sentence, last=False)

    async def _push_audio(self, sentence: _Sentence, *, last: bool):
        """Push whole 40 ms frames (and, at the end, the remainder)."""
        size = self._frame_bytes
        pending = sentence.pending
        end = len(pending) if last else len(pending) - len(pending) % size
        for offset in range(0, end, size):
            turn = sentence.turn
            if turn is None:
                return
            audio = bytes(pending[offset : offset + size])
            await self.append_to_audio_context(
                turn.id, TTSAudioRawFrame(audio, self.sample_rate, 1, context_id=turn.id)
            )
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
            await self._report_usage(chars)

    async def _report_usage(self, chars: int):
        if not (self.can_generate_metrics() and self.usage_metrics_enabled):
            return
        logger.debug(f"{self.name} usage characters: {chars}")
        data = TTSUsageMetricsData(processor=self.name, model=self._settings.model, value=chars)
        await self.push_frame(MetricsFrame(data=[data]))

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
        await self._report(
            f"Mirai TTS error {code}: {message}{where}", _ERROR_CATEGORIES.get(code, ErrorCategory.SERVER)
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
        turn.retry_task = self.create_task(self._retry_after(turn, delay), name="retry")

    async def _retry_after(self, turn: _Turn, delay: float):
        deadline = time.monotonic() + delay
        while (left := deadline - time.monotonic()) > 0:
            self._keep_context_open(turn.id)
            await asyncio.sleep(min(left, 1.0))
        turn.retry_task = None
        if self._turns.get(turn.id) is not turn:
            return
        if self._is_open():
            await self._send_held(turn)
        else:
            await self._connect()  # sends it once the socket is up

    def _keep_context_open(self, context_id: str):
        # Pipecat ends an audio context that gets no frame for its stop
        # timeout (3 s); a retry may wait about that long.
        refresh = getattr(self, "_refresh_audio_context", None)
        if refresh is not None and self.audio_context_available(context_id):
            refresh(context_id)

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
                    ErrorCategory.CONNECTIVITY,
                )
                await self._maybe_complete(turn)

    async def _report(self, message: str, category: ErrorCategory):
        self.last_error = message
        await self.push_error(error_msg=message, category=category)


def _error_detail(body: Any) -> str | None:
    """The ``error.message`` of a JSON error body, if there is one."""
    try:
        detail = json.loads(body).get("error", {}).get("message")
    except (TypeError, ValueError, AttributeError):
        return None
    return detail[:500] if isinstance(detail, str) else None
