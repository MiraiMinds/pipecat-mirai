#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai speech-to-text over one WebSocket per pipeline.

``MiraiSTTService`` streams the caller's audio to Mirai's
``/v1/audio/transcriptions/stream`` and turns what comes back into Pipecat
transcription frames. The protocol is Mirai's STT gateway's, which is
Sarvam-v3 compatible. Client: ``audio_input`` {audio: base64}, ``speech_start``
/ ``speech_end`` / ``flush`` (manual endpointing), ``config.update``, ``ping``,
``end``. Server: ``session.begin`` (request id, session id, limits),
``vad.speech_start`` / ``vad.speech_end`` (server endpointing),
``transcript.partial`` (a full snapshot of the utterance so far),
``transcript.final`` (exactly one per utterance, in order), ``config.updated``,
``pong``, ``session.draining``, ``session.end`` and ``error`` {code, message,
is_fatal}. Settings that fix the stream (model, encoding, rate, endpointing)
travel in the connection's query string; language and tuning can change on an
open socket with ``config.update``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse

import numpy as np
import soxr
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    STTMetadataFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import STTUsage
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import STTService
from pipecat.transcriptions.language import Language
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.tracing.service_decorators import traced_stt
from pipecat.utils.types import NOT_GIVEN, NotGiven, is_given
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from pipecat_mirai._net import websocket_connect_kwargs
from pipecat_mirai.edge import EdgeRoute, edge_mode
from pipecat_mirai.pool import STT_PROTOCOL, WebsocketPool, _retry_after, take_websocket, websocket_pool

DEFAULT_STT_WEBSOCKET_URL = "wss://sandbox.voice.miraiminds.co/v1/audio/transcriptions/stream"
DEFAULT_LANGUAGE = "hi-IN"
DEFAULT_MODEL = "mira-stt"
# The languages Mirai's streaming STT accepts (``language_code``).
STT_LANGUAGES = ("hi-IN", "en-IN", "gu-IN", "mr-IN", "pa-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN")
# The rates and encodings the socket takes.
WIRE_SAMPLE_RATES = (8000, 16000)
ENCODINGS = ("linear16", "mulaw", "alaw")
# Audio goes out in pieces this long: base64 in JSON has real per-message
# overhead, and the gateway reads it in about this size.
CHUNK_SECS = 0.06
# Manual endpointing: audio held between turns and sent at speech_start. The
# pipeline's VAD fires only after start_secs of speech (and a queue hop), so
# without it the utterance the server sees begins mid-word.
PREROLL_SECS = 0.5
# Audio the server hasn't answered yet (a final), kept to replay on a new
# socket after a drop. Past this, the oldest is dropped.
REPLAY_SECS = 30.0
REPLAY_CHUNK_SECS = 0.5
# An idle socket gets a ping and this much silence every KEEPALIVE_SECS (or
# half the session's idle timeout, if shorter). Manual endpointing sends
# nothing between turns, and a caller who listens for a minute must not lose
# their transcriber to the idle timeout.
KEEPALIVE_SECS = 20.0
KEEPALIVE_AUDIO_SECS = 0.04
# A socket refused with 429 is retried for this long, waiting Retry-After (at
# most CONNECT_RETRY_WAIT at a time), before it is reported.
CONNECT_RETRY_SECS = 5.0
CONNECT_RETRY_WAIT = 2.0
OPEN_TIMEOUT = 10.0
CLOSE_TIMEOUT = 2.0
# A pooled socket opened with another language gets config.update; this long for config.updated.
CONFIG_UPDATE_TIMEOUT = 2.0
# After the service gives up on a connection, the next attempt waits at least this long.
RETRY_AFTER_FAILURE_SECS = 5.0
# A graceful stop waits this long for the last finals and session.end.
STOP_WAIT_SECS = 2.0
# A session is replaced (at a moment between utterances) once this much of its
# maximum length is left, or at half of it if that comes later. A replacement
# that can't be opened is tried again after REPLACE_RETRY_SECS.
ROTATE_HEADROOM_SECS = 60.0
REPLACE_RETRY_SECS = 30.0
# Pipecat's turn strategies wait up to this long after the VAD stops for a
# final (a finalized transcript ends the wait at once).
DEFAULT_TTFS_P99 = 0.5
MAX_MESSAGE_BYTES = 1 << 20

# Settings that config.update can change on an open socket, by query name.
_RUNTIME_FIELDS = ("stream_type", "mode", "threshold", "silence_duration_ms", "min_speech_duration_ms")
# Server-side VAD tuning: sent only when the server does the endpointing.
_VAD_FIELDS = frozenset({"threshold", "silence_duration_ms", "min_speech_duration_ms", "prefix_padding_ms"})
# What the pool's waiting sockets are opened with; a pipeline that wants other
# runtime settings sends config.update when it takes one.
_POOL_RUNTIME = {"language_code": DEFAULT_LANGUAGE}

_ERROR_CATEGORIES = {
    "unauthorized": ErrorCategory.AUTHENTICATION,
    "invalid_token": ErrorCategory.AUTHENTICATION,
    "forbidden": ErrorCategory.AUTHORIZATION,
    "invalid_config": ErrorCategory.INVALID_REQUEST,
    "invalid_request": ErrorCategory.INVALID_REQUEST,
    "insufficient_balance": ErrorCategory.QUOTA,
    "at_capacity": ErrorCategory.RATE_LIMIT,
    "rate_limited": ErrorCategory.RATE_LIMIT,
    "idle_timeout": ErrorCategory.CONNECTIVITY,
    "unavailable": ErrorCategory.CONNECTIVITY,
}
# Codes after which another connection would be refused the same way.
_PERMANENT_CODES = frozenset(
    {
        "unauthorized",
        "invalid_token",
        "forbidden",
        "invalid_config",
        "invalid_request",
        "insufficient_balance",
    }
)
_HANDSHAKE_CATEGORIES = {
    400: ErrorCategory.INVALID_REQUEST,
    401: ErrorCategory.AUTHENTICATION,
    402: ErrorCategory.QUOTA,
    403: ErrorCategory.AUTHORIZATION,
    404: ErrorCategory.INVALID_REQUEST,
    426: ErrorCategory.INVALID_REQUEST,
    429: ErrorCategory.RATE_LIMIT,
}
_PERMANENT_STATUSES = frozenset({400, 401, 402, 403, 404, 426})
_ADVICE = {
    "unauthorized": "Check MIRAI_API_KEY.",
    401: "Check MIRAI_API_KEY.",
    "forbidden": "The API key is revoked or not allowed this request.",
    403: "The API key is revoked or not allowed this request.",
    "insufficient_balance": "The workspace is out of credit. Top it up in the Mirai console.",
    402: "The workspace is out of credit. Top it up in the Mirai console.",
    "at_capacity": "Ask Mirai to raise this workspace's concurrent STT session limit.",
    429: "Ask Mirai to raise this workspace's concurrent STT session limit.",
    "invalid_config": "Check the language, model, sample_rate, encoding and endpointing settings.",
    400: "Check the language, model, sample_rate, encoding and endpointing settings.",
}


@dataclass
class MiraiSTTSettings(STTSettings):
    """Runtime-updatable settings for :class:`MiraiSTTService`.

    Parameters:
        model: Mirai model name. Defaults to ``"mira-stt"``. A change opens a
            new socket at the next pause between utterances.
        language: ``hi-IN`` (default), ``en-IN``, ``gu-IN``, ``mr-IN``,
            ``pa-IN``, ``bn-IN``, ``ta-IN``, ``te-IN``, ``kn-IN`` or ``ml-IN``
            (a ``Language`` works too). Changes apply on the open socket.
        stream_type: ``"fast"`` or ``"balanced"``; ``None`` leaves it to Mirai.
        mode: ``"transcribe"`` or ``"codemix"`` (English words kept in Latin
            script); ``None`` leaves it to Mirai.
        threshold: Server VAD sensitivity, 0 to 1 (server endpointing only).
        silence_duration_ms: Silence that ends an utterance (server endpointing only).
        min_speech_duration_ms: Shortest speech that opens an utterance (server endpointing only).
        prefix_padding_ms: Audio kept before speech (server endpointing only;
            fixed for a socket, so a change opens a new one).
    """

    stream_type: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    mode: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    silence_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_speech_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prefix_padding_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class _ServerRefused(Exception):
    """The server answered the upgrade, then sent an error instead of session.begin."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.message = message


@dataclass(eq=False)
class _Utterance:
    """One utterance under manual endpointing: speech_start ... speech_end, then its final."""

    number: int
    audio: bytearray = field(default_factory=bytearray)  # wire bytes sent for it (the last REPLAY_SECS)
    language: str = ""  # the language it was spoken under
    ended: bool = False
    end_sent_wall: float = 0.0  # when its speech_end was last written to a socket


@dataclass(eq=False)
class _Out:
    """A message waiting to be sent."""

    text: str
    audio_bytes: int = 0  # wire bytes of audio in it, for usage
    ends: _Utterance | None = None  # it is this utterance's speech_end


@dataclass(eq=False)
class _Socket:
    """An open socket past session.begin."""

    websocket: Any
    url: str  # where it is connected, without the query
    edge: bool
    begin: dict
    runtime: dict  # the runtime settings it was opened or configured with
    opened_at: float = field(default_factory=time.monotonic)  # when its session began
    reader: asyncio.Task | None = None
    fatal: bool = False  # the server sent a fatal error on it

    @property
    def session_id(self) -> str | None:
        return self.begin.get("session_id")

    @property
    def request_id(self) -> str | None:
        return self.begin.get("request_id")

    def _limit(self, name: str) -> float | None:
        value = (self.begin.get("limits") or {}).get(name)
        return float(value) if isinstance(value, int | float) and value > 0 else None

    @property
    def idle_timeout_secs(self) -> float | None:
        return self._limit("idle_timeout_secs")

    @property
    def rotate_at(self) -> float | None:
        """When (monotonic) this session should be replaced at the next pause, if it has a maximum length."""
        longest = self._limit("max_session_secs")
        if longest is None:
            return None
        return self.opened_at + max(longest / 2, longest - ROTATE_HEADROOM_SECS)


class MiraiSTTService(STTService):
    """Mirai speech-to-text for Pipecat: one WebSocket for the whole pipeline.

    The caller's audio streams to Mirai's ``/v1/audio/transcriptions/stream``
    in 60 ms pieces; partial transcripts come back as
    ``InterimTranscriptionFrame`` and each utterance's final transcript as one
    ``TranscriptionFrame`` (``finalized=True``), in order.

    Who decides where an utterance ends (``endpointing``):

    - ``"manual"``: the pipeline's VAD. ``VADUserStartedSpeakingFrame`` becomes
      ``speech_start`` (with the last 0.5 s of audio, so the first word isn't
      cut) and ``VADUserStoppedSpeakingFrame`` becomes ``speech_end``; the
      final follows within tens of milliseconds. Between turns nothing is
      sent but a keepalive.
    - ``"vad"``: Mirai's VAD. Audio streams continuously and the service
      proposes turn boundaries (``ProposedUserStarted/StoppedSpeakingFrame``)
      from Mirai's ``vad.speech_start`` / ``vad.speech_end``, and asks the
      user aggregator to follow them.
    - ``"auto"`` (default): ``"manual"`` when the pipeline has a VAD (a
      ``vad_analyzer`` on the user aggregator, a ``VADProcessor``), else
      ``"vad"``; decided when the pipeline starts.

    Audio at 8 kHz (phone calls) goes as it is; any other rate is resampled
    to 16 kHz (``sample_rate``). The socket opens on Mirai's edge, next to the
    GPUs, when Mirai offers one, else on ``url`` (see
    :mod:`pipecat_mirai.edge`), and usually comes ready from the process's
    pool (see :mod:`pipecat_mirai.pool`). It opens in the background, so the
    pipeline never waits for it: audio is held until it is up. If it drops,
    the service dials once more and replays the audio Mirai hadn't answered
    (up to 30 s); if that fails too it reports an ``ErrorFrame``.

    Metrics: TTFB is ``speech_end`` sent (``vad.speech_end`` received under
    server endpointing) to the final received, reported with processing time
    for the same span; usage is the audio seconds sent.

    Example::

        stt = MiraiSTTService(api_key=os.getenv("MIRAI_API_KEY"), language="hi-IN")

    Event handlers (from Pipecat): ``on_connected``, ``on_disconnected``,
    ``on_connection_error``.
    """

    Settings = MiraiSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        language: str | Language = DEFAULT_LANGUAGE,
        model: str = DEFAULT_MODEL,
        sample_rate: int | Literal["auto"] = "auto",
        encoding: Literal["auto", "linear16", "mulaw", "alaw"] = "auto",
        endpointing: Literal["auto", "manual", "vad"] = "auto",
        edge: bool | str = "auto",
        shared_pool: bool = True,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Mirai STT service.

        Args:
            api_key: Mirai API key. Defaults to ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``). Sent as an ``Authorization: Bearer`` header.
            url: The streaming endpoint,
                ``wss://<host>/v1/audio/transcriptions/stream``. Defaults to
                Mirai's sandbox.
            language: Shortcut for ``settings.language``. Defaults to ``"hi-IN"``.
            model: Shortcut for ``settings.model``. Defaults to ``"mira-stt"``.
            sample_rate: The rate audio is sent at. ``"auto"`` (the default)
                sends 8 kHz audio as it is and resamples anything else to
                16 kHz; ``8000`` or ``16000`` resamples to that rate when the
                pipeline's input rate differs.
            encoding: ``"auto"`` (linear16), ``"linear16"``, or ``"mulaw"`` /
                ``"alaw"`` (8 kHz only; half the bytes).
            endpointing: ``"auto"`` (default), ``"manual"`` or ``"vad"``: see the
                class docstring.
            edge: ``"auto"`` (default): open the socket on Mirai's edge when it
                is offered, else on ``url``. ``False`` always uses ``url``; a
                ``wss://`` URL uses that edge instead. ``MIRAI_STT_EDGE=off``
                turns ``"auto"`` off.
            shared_pool: Take a socket the process's pool opened ahead of need
                (the default; the first service starts the pool). ``False``
                always connects and starts no pool.
            settings: Runtime-updatable settings; values here win over the
                ``language``/``model`` shortcuts.
            **kwargs: Passed through to :class:`STTService`, e.g.
                ``audio_passthrough`` or ``ttfs_p99_latency``.
        """
        key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
        if not key:
            raise ValueError("Set MIRAI_API_KEY or pass api_key to MiraiSTTService.")
        url = url or DEFAULT_STT_WEBSOCKET_URL
        if not url.startswith(("ws://", "wss://")):
            raise ValueError(f"url must be a ws:// or wss:// URL; got {url!r}")
        if urlparse(url).query:
            raise ValueError("url must not carry a query; pass settings as arguments instead")
        if sample_rate != "auto" and sample_rate not in WIRE_SAMPLE_RATES:
            raise ValueError(f"sample_rate must be 'auto', 8000 or 16000; got {sample_rate!r}")
        if encoding != "auto" and encoding not in ENCODINGS:
            raise ValueError(f"encoding must be 'auto', 'linear16', 'mulaw' or 'alaw'; got {encoding!r}")
        if encoding in ("mulaw", "alaw") and sample_rate == 16000:
            raise ValueError(f"{encoding} is sent at 8000 Hz; use encoding='linear16' for 16000")
        if endpointing not in ("auto", "manual", "vad"):
            raise ValueError(f"endpointing must be 'auto', 'manual' or 'vad'; got {endpointing!r}")
        mode = edge_mode(edge, "stt")

        default_settings = self.Settings(
            model=model,
            language=language,
            stream_type=None,
            mode=None,
            threshold=None,
            silence_duration_ms=None,
            min_speech_duration_ms=None,
            prefix_padding_ms=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        kwargs.setdefault("ttfs_p99_latency", DEFAULT_TTFS_P99)
        super().__init__(settings=default_settings, **kwargs)

        self._key = key
        self._base_url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._rate_option = sample_rate
        self._encoding = "linear16" if encoding == "auto" else encoding
        self._endpointing_option = endpointing
        self._edge_mode = mode
        self._shared_pool = shared_pool
        self._http_base = _http_base(self._base_url)

        # Decided at start.
        self._endpointing = endpointing if endpointing != "auto" else "vad"
        self._wire_rate = 16000
        self._bytes_per_sample = 2 if self._encoding == "linear16" else 1
        self._resampler: Any = None
        self._carry = b""
        self._url = self._base_url
        self._pool_url = self._base_url
        self._edge: EdgeRoute | None = None
        self._pool_edge: EdgeRoute | None = None
        self._pool: WebsocketPool | None = None

        # The connection.
        self._sock: _Socket | None = None
        self._spare: _Socket | None = None  # a new session waiting for a pause to take over
        self._dialing: asyncio.Task | None = None
        self._urgent = False  # the current session is draining: replace it now
        self._replace_at = 0.0  # no replacement session is dialled before this (monotonic)
        self._sender: asyncio.Task | None = None
        self._outbox: deque[_Out] = deque()
        self._wake = asyncio.Event()
        self._ended = asyncio.Event()  # session.end arrived (or the socket is gone)
        self._last_sent = time.monotonic()
        self._connected_once = False
        self._permanent = False  # a refusal another connection would get too
        self._retry_at = 0.0
        self._closing = False  # a graceful stop is under way
        self._stopping = False
        self._started = False

        # What is being transcribed.
        self._ring = bytearray()  # manual: the pre-roll between turns
        self._coalesce = bytearray()  # audio not yet cut into a chunk
        self._current: _Utterance | None = None  # manual: the open utterance
        self._pending: deque[_Utterance] = deque()  # manual: speech_end sent, final not in
        self._utterances = 0
        self._unanswered = bytearray()  # vad: audio since the last final, for a replay
        self._vad_open = False
        self._vad_ends: deque[float] = deque()  # vad: when each vad.speech_end arrived (wall clock)
        self._usage_pending = 0.0

        self.last_error: str | None = None
        #: The socket's session id (``sttws_...``) and request id, from session.begin.
        self.session_id: str | None = None
        self.request_id: str | None = None
        #: Where the socket is connected: the edge's URL or ``url`` (no query).
        self.connected_url: str | None = None
        #: speech_end sent (vad.speech_end received) to the latest final, in milliseconds.
        self.last_final_latency_ms: float | None = None
        #: Audio seconds sent over the service's life (replays included: they are billed).
        self.audio_seconds_sent = 0.0

    # ---------- what Pipecat asks ----------

    def can_generate_metrics(self) -> bool:
        """Mirai STT reports TTFB, processing and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str:
        """``Language.HI`` / ``Language.HI_IN`` -> ``"hi-IN"``."""
        code = str(getattr(language, "value", language))
        base = code.split("-", 1)[0].lower()
        for supported in STT_LANGUAGES:
            if supported.split("-", 1)[0] == base:
                return supported
        return code

    @property
    def endpointing(self) -> str:
        """``"manual"`` or ``"vad"``: who ends utterances (decided when the pipeline starts)."""
        return self._endpointing

    @property
    def wire_sample_rate(self) -> int:
        """The rate audio is sent at (decided when the pipeline starts)."""
        return self._wire_rate

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Under server endpointing, ask the user aggregator to follow Mirai's turn boundaries."""
        frame = super().service_metadata_frame()
        if self._endpointing == "vad":
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    # ---------- lifecycle ----------

    async def start(self, frame: StartFrame):
        """Start the service; the socket opens in the background."""
        await super().start(frame)
        in_rate = self.sample_rate
        if self._encoding != "linear16":
            self._wire_rate = 8000
        elif self._rate_option == "auto":
            self._wire_rate = in_rate if in_rate in WIRE_SAMPLE_RATES else 16000
        else:
            self._wire_rate = int(self._rate_option)
        self._resampler = (
            soxr.ResampleStream(in_rate, self._wire_rate, 1, dtype="int16", quality="HQ")
            if in_rate != self._wire_rate
            else None
        )
        if self._endpointing_option == "auto":
            self._endpointing = "manual" if self._pipeline_has_vad() else "vad"
        if self._settings.language not in STT_LANGUAGES:
            logger.warning(
                f"{self}: language {self._settings.language!r} is not one of {', '.join(STT_LANGUAGES)}"
            )
        self._build_urls()
        resampling = f", resampled from {in_rate} Hz" if self._resampler else ""
        logger.debug(
            f"{self}: {self._endpointing} endpointing, {self._encoding} at {self._wire_rate} Hz{resampling}"
        )
        self._started = True
        if self._shared_pool and self._pool is None:
            self._pool = websocket_pool(
                self._pool_url, self._headers, MAX_MESSAGE_BYTES, edge=self._pool_edge, protocol=STT_PROTOCOL
            )
            self._pool.hold(self)  # the first one sets the pool going
        self._sender = self.create_task(self._send_loop(), name="mirai-stt-send")
        self._ensure_connected()

    async def stop(self, frame: EndFrame):
        """Finish: close the open utterance, wait briefly for the last finals, then close."""
        await self._shutdown(graceful=True)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Close at once."""
        await self._shutdown(graceful=False)
        await super().cancel(frame)

    async def cleanup(self):
        """Release everything at pipeline teardown."""
        try:
            await super().cleanup()
        finally:
            await self._shutdown(graceful=False)

    async def _shutdown(self, *, graceful: bool):
        if self._stopping:
            return
        if graceful and self._started and not self._permanent:
            self._closing = True
            if self._current is not None:
                await self._speech_stopped()
            elif self._endpointing != "manual":
                self._cut(final=True)  # the last part of a chunk
            deadline = time.monotonic() + STOP_WAIT_SECS
            ended_on: _Socket | None = None
            while (left := deadline - time.monotonic()) > 0:
                sock = self._sock
                if sock is None:
                    # Still dialling, or the socket dropped with audio unanswered: wait for the redial.
                    if self._permanent or not self._has_unanswered():
                        break
                    self._ensure_connected()
                    await asyncio.sleep(0.02)
                    continue
                if ended_on is not sock:
                    self._ended.clear()
                    self._queue(_Out(json.dumps({"event": "end"})))
                    ended_on = sock
                try:
                    await asyncio.wait_for(self._ended.wait(), min(left, 0.05))
                except TimeoutError:
                    continue
                if self._sock is sock:
                    break  # session.end
            else:
                logger.debug(f"{self}: no session.end within {STOP_WAIT_SECS:g} s")
        self._stopping = True
        current = asyncio.current_task()
        for name in ("_dialing", "_sender"):
            task = getattr(self, name)
            setattr(self, name, None)
            if task is not None and task is not current and not task.done():
                await self.cancel_task(task)
        for sock in (self._sock, self._spare):
            if sock is not None:
                await self._close(sock)
        had_socket = self._sock is not None
        self._sock = self._spare = None
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.release(self)
        if had_socket:
            await self._call_event_handler("on_disconnected")

    # ---------- audio in ----------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Queue the caller's audio; transcripts arrive on the receive task."""
        if self._started and not self._stopping and not self._permanent:
            data = self._to_wire(audio)
            if data:
                if self._endpointing == "manual" and self._current is None:
                    self._ring += data
                    excess = len(self._ring) - self._bytes(PREROLL_SECS)
                    if excess > 0:
                        del self._ring[:excess]
                else:
                    self._coalesce += data
                    self._cut(final=False)
        yield None

    def _record_stt_audio_usage(self, audio: bytes | bytearray):
        # Usage is the audio actually sent (after resampling, without what is
        # held between turns), counted as it goes out: see _send_loop.
        pass

    async def emit_stt_usage_metrics(self):
        """Report the audio seconds sent since the last report."""
        seconds, self._usage_pending = self._usage_pending, 0.0
        if seconds > 0:
            await self.start_stt_usage_metrics(STTUsage(audio_seconds=seconds))

    def _to_wire(self, audio: bytes) -> bytes:
        """Pipeline PCM to the socket's rate and encoding (whole samples only)."""
        data = self._carry + audio
        end = len(data) - len(data) % 2
        self._carry, data = data[end:], data[:end]
        if not data:
            return b""
        if self._resampler is not None:
            data = self._resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
        if self._encoding == "mulaw":
            data = lin2ulaw(np.frombuffer(data, dtype="<i2")).tobytes()
        elif self._encoding == "alaw":
            data = lin2alaw(np.frombuffer(data, dtype="<i2")).tobytes()
        return data

    def _bytes(self, secs: float) -> int:
        return int(self._wire_rate * secs) * self._bytes_per_sample

    def _seconds(self, nbytes: int) -> float:
        return nbytes / (self._wire_rate * self._bytes_per_sample) if self._wire_rate else 0.0

    def _cut(self, *, final: bool):
        """Send whole chunks of the coalesced audio (and, when ``final``, the rest)."""
        size = self._bytes(CHUNK_SECS)
        while len(self._coalesce) >= size or (final and self._coalesce):
            piece = bytes(self._coalesce[:size])
            del self._coalesce[:size]
            self._send_audio(piece)

    def _send_audio(self, piece: bytes):
        if self._endpointing == "manual":
            if self._current is None:
                return
            self._current.audio += piece
        else:
            self._unanswered += piece
        self._trim_unanswered()
        self._queue(_audio_out(piece))
        self._ensure_connected()

    def _trim_unanswered(self):
        """Keep at most REPLAY_SECS of unanswered audio, dropping the oldest."""
        limit = self._bytes(REPLAY_SECS)
        if self._endpointing != "manual":
            excess = len(self._unanswered) - limit
            if excess > 0:
                del self._unanswered[:excess]
            return
        held = [*self._pending, *([self._current] if self._current is not None else [])]
        excess = sum(len(u.audio) for u in held) - limit
        for utt in held:
            if excess <= 0:
                break
            cut = min(excess, len(utt.audio))
            del utt.audio[:cut]
            excess -= cut

    def _has_unanswered(self) -> bool:
        if self._endpointing == "manual":
            return bool(self._pending) or self._current is not None
        return bool(self._unanswered) or self._vad_open or bool(self._vad_ends)

    def _idle(self) -> bool:
        """No utterance open and none waiting for its final: a session can be swapped now."""
        if self._endpointing == "manual":
            return self._current is None and not self._pending
        return not self._vad_open and not self._vad_ends

    # ---------- manual endpointing ----------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Turn the pipeline's VAD frames into speech_start / speech_end under manual endpointing."""
        await super().process_frame(frame, direction)
        if self._endpointing != "manual" or not self._started:
            return
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self._speech_started()
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self._speech_stopped()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        # TTFB is measured from speech_end sent (or vad.speech_end) to the
        # final, not from the VAD's estimate of when speech ended.
        self._user_speaking = False

    async def _speech_started(self):
        if self._current is not None or self._stopping or self._permanent:
            return
        self._utterances += 1
        self._current = _Utterance(self._utterances, language=self._settings.language)
        self._queue(_Out(json.dumps({"event": "speech_start"})))
        self._coalesce = self._ring
        self._ring = bytearray()
        self._cut(final=False)
        self._ensure_connected()

    async def _speech_stopped(self):
        utt = self._current
        if utt is None:
            return
        self._cut(final=True)
        utt.ended = True
        self._current = None
        self._pending.append(utt)
        self._queue(_Out(json.dumps({"event": "speech_end"}), ends=utt))

    # ---------- the connection ----------

    def _build_urls(self):
        """The socket's query (fixed settings, then runtime ones) and the pool's."""
        padding = self._settings.prefix_padding_ms
        fixed = _fixed_query(
            model=self._settings.model,
            encoding=self._encoding,
            sample_rate=self._wire_rate,
            endpointing=self._endpointing,
            prefix_padding_ms=padding if is_given(padding) else None,
        )
        self._url = f"{self._base_url}?{urlencode({**fixed, **self._runtime()})}"
        self._pool_url = f"{self._base_url}?{urlencode({**fixed, **_POOL_RUNTIME})}"
        if self._edge_mode:
            self._edge = EdgeRoute(self._url, self._headers, self._edge_mode, self._http_base, "stt")
            self._pool_edge = EdgeRoute(
                self._pool_url, self._headers, self._edge_mode, self._http_base, "stt"
            )

    def _runtime(self) -> dict:
        """The settings an open socket can change with config.update, by query name."""
        params: dict[str, Any] = {"language_code": self._settings.language}
        for name in _RUNTIME_FIELDS:
            value = getattr(self._settings, name)
            if not is_given(value) or value is None:
                continue
            if name in _VAD_FIELDS and self._endpointing != "vad":
                continue
            params[name] = value
        return params

    def _ensure_connected(self):
        """Dial in the background if there is no socket and no dial under way (and it is allowed)."""
        if self._sock is not None or not self._started or self._stopping or self._permanent:
            return
        if self._dialing is not None and not self._dialing.done():
            return
        if time.monotonic() < self._retry_at:
            return
        self._dialing = self.create_task(self._dial_task(), name="mirai-stt-dial")

    async def _dial_task(self):
        me = asyncio.current_task()
        try:
            sock = await self._dial()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._dialing is me:
                self._dialing = None
            if self._stopping:
                return
            if self._sock is None:
                await self._connect_failed(exc)
            else:
                # The current session carries on; if it is draining, its close dials again.
                logger.debug(f"{self}: could not open a replacement session: {_describe(exc)}")
                self._replace_at = time.monotonic() + REPLACE_RETRY_SECS
            return
        if self._dialing is me:
            self._dialing = None
        if self._stopping:
            await self._close(sock)
            return
        if self._sock is None:
            await self._attach(sock)
        elif self._urgent:
            await self._swap(sock)
        else:
            if self._spare is not None:
                await self._close(self._spare)
            self._spare = sock
            await self._maybe_swap()

    async def _dial(self) -> _Socket:
        """A socket past session.begin: a waiting one from the pool, else the edge, else ``url``."""
        if self._shared_pool:
            pooled = await take_websocket(
                self._pool_url, self._headers, edge=self._pool_edge, protocol=STT_PROTOCOL
            )
            if pooled is not None:
                sock = _Socket(
                    pooled.websocket,
                    _strip_query(pooled.url or self._pool_url),
                    pooled.edge,
                    pooled.ready,
                    runtime=dict(_POOL_RUNTIME),
                    opened_at=pooled.opened_at,
                )
                try:
                    await self._configure(sock)
                except asyncio.CancelledError:
                    await _close_quietly(sock.websocket)
                    raise
                except Exception as exc:
                    logger.debug(
                        f"{self}: a waiting socket didn't take the session's settings: {_describe(exc)}"
                    )
                    await self._close(sock)
                else:
                    logger.debug(f"{self}: using waiting socket {sock.session_id} on {sock.url}")
                    return sock
        runtime = self._runtime()
        if self._edge is not None:
            on_edge = await self._edge.open(self._connect_to)
            if on_edge is not None:
                logger.debug(f"{self}: connected to the edge {on_edge.url}")
                return _Socket(on_edge.websocket, on_edge.url, True, on_edge.ready, runtime=runtime)
        websocket, begin = await self._connect_gateway()
        return _Socket(websocket, self._base_url, False, begin, runtime=runtime)

    async def _configure(self, sock: _Socket):
        """Bring a pooled socket's runtime settings to this session's, waiting for config.updated."""
        wanted = self._runtime()
        change = {k: v for k, v in wanted.items() if sock.runtime.get(k) != v}
        if not change:
            return
        await sock.websocket.send(json.dumps({"event": "config.update", **change}))

        async def updated():
            while True:
                event = _parse(await sock.websocket.recv())
                kind = _kind(event)
                if kind == "config.updated":
                    return
                if kind == "error":
                    code, message, _ = _error_parts(event)
                    raise _ServerRefused(code, message)

        await asyncio.wait_for(updated(), CONFIG_UPDATE_TIMEOUT)
        sock.runtime = wanted

    async def _connect_to(self, url: str, headers: dict[str, str], extra: dict[str, Any]):
        return await websocket_connect(
            url,
            additional_headers=headers,
            max_size=MAX_MESSAGE_BYTES,
            open_timeout=OPEN_TIMEOUT,
            close_timeout=CLOSE_TIMEOUT,
            **extra,
        )

    async def _connect_gateway(self) -> tuple[Any, dict]:
        """Open ``url`` and wait for session.begin, waiting out a 429 for up to ``CONNECT_RETRY_SECS``."""
        deadline = time.monotonic() + CONNECT_RETRY_SECS
        while True:
            extra = await asyncio.wait_for(websocket_connect_kwargs(self._url), OPEN_TIMEOUT)
            try:
                websocket = await self._connect_to(self._url, self._headers, extra)
            except Exception as exc:
                sock = extra.get("sock")
                if sock is not None:
                    sock.close()
                response = getattr(exc, "response", None)
                if getattr(response, "status_code", None) != 429:
                    raise
                wait = min(CONNECT_RETRY_WAIT, _retry_after(response, 1.0))
                if time.monotonic() + wait > deadline:
                    raise
                logger.debug(f"{self}: Mirai answered 429 to the socket; trying again in {wait:g} s")
                await asyncio.sleep(wait)
                continue
            try:
                begin = await asyncio.wait_for(_first_event(websocket), OPEN_TIMEOUT)
            except BaseException:
                await _close_quietly(websocket)
                raise
            return websocket, begin

    async def _attach(self, sock: _Socket):
        """Make ``sock`` the session's socket and replay what is unanswered on it."""
        self._sock = sock
        self._urgent = False
        self._ended.clear()
        self.session_id = sock.session_id or self.session_id
        self.request_id = sock.request_id or self.request_id
        self.connected_url = sock.url
        sock.reader = self.create_task(self._read(sock), name="mirai-stt-receive")
        replay = self._replay()
        if sock.runtime != self._runtime():
            replay.appendleft(_Out(json.dumps({"event": "config.update", **self._runtime()})))
            sock.runtime = self._runtime()
        replayed = sum(item.audio_bytes for item in replay)
        if replayed and self._connected_once:
            logger.info(f"{self}: replaying {self._seconds(replayed):.1f} s of audio not yet transcribed")
        self._outbox = replay
        self._last_sent = time.monotonic()
        self._connected_once = True
        self._wake.set()
        logger.debug(f"{self}: session {sock.session_id} on {sock.url}")
        await self._call_event_handler("on_connected")

    def _replay(self) -> deque[_Out]:
        """Everything the server hasn't answered, as messages for a new socket."""
        out: deque[_Out] = deque()
        if self._endpointing == "manual":
            for utt in self._pending:
                out.append(_Out(json.dumps({"event": "speech_start"})))
                out.extend(self._audio_pieces(utt.audio))
                out.append(_Out(json.dumps({"event": "speech_end"}), ends=utt))
            if self._current is not None:
                out.append(_Out(json.dumps({"event": "speech_start"})))
                out.extend(self._audio_pieces(self._current.audio))
        else:
            out.extend(self._audio_pieces(self._unanswered))
        return out

    def _audio_pieces(self, audio: bytes | bytearray) -> list[_Out]:
        step = max(self._bytes_per_sample, self._bytes(REPLAY_CHUNK_SECS))
        return [_audio_out(bytes(audio[i : i + step])) for i in range(0, len(audio), step)]

    def _queue(self, item: _Out):
        """Send ``item`` on the current socket. Without one it isn't queued: the state replays it."""
        if self._sock is None:
            return
        self._outbox.append(item)
        self._wake.set()

    async def _swap(self, sock: _Socket):
        """Move the session to ``sock`` now, replaying what the old one hadn't answered."""
        old, self._sock = self._sock, None
        if old is not None:
            await self._close(old, end=True)
        await self._attach(sock)

    async def _maybe_swap(self):
        """Hand over to the waiting replacement session if nothing is in flight."""
        if self._spare is None or self._stopping or self._closing or not self._idle():
            return
        spare, self._spare = self._spare, None
        if spare.websocket.state is not State.OPEN:
            return
        logger.debug(f"{self}: moving to a new session {spare.session_id}")
        await self._swap(spare)

    def _rotation_due(self) -> bool:
        sock = self._sock
        at = sock.rotate_at if sock is not None else None
        return at is not None and time.monotonic() >= at

    def _start_replacement(self, *, urgent: bool = False):
        """Open the next session in the background; it takes over at a pause (now, if ``urgent``)."""
        if urgent:
            self._urgent = True
        if self._stopping or self._permanent or (self._dialing is not None and not self._dialing.done()):
            return
        now = time.monotonic()
        if not urgent and now < self._replace_at:
            return
        self._replace_at = now + OPEN_TIMEOUT + CONNECT_RETRY_SECS  # this attempt's own time
        self._dialing = self.create_task(self._dial_task(), name="mirai-stt-dial")

    async def _close(self, sock: _Socket, *, end: bool = False):
        reader, sock.reader = sock.reader, None
        if reader is not None and reader is not asyncio.current_task() and not reader.done():
            await self.cancel_task(reader)
        if end and sock.websocket.state is State.OPEN:
            try:
                await sock.websocket.send(json.dumps({"event": "end"}))
            except Exception:
                pass
        await _close_quietly(sock.websocket)

    async def _lost(self, sock: _Socket, reason: str):
        """The socket closed under us: dial once more and replay, or report it."""
        if sock is not self._sock:
            return
        self._sock = None
        self._outbox.clear()
        self._ended.set()
        await self._close(sock)
        await self._call_event_handler("on_disconnected")
        if self._stopping or (self._closing and not self._has_unanswered()):
            return
        if self._permanent or (sock.fatal and not self._has_unanswered()):
            self._clear_unanswered()
            return
        if self._spare is not None and self._spare.websocket.state is State.OPEN:
            spare, self._spare = self._spare, None
            await self._attach(spare)
            return
        held = self._seconds(self._unanswered_bytes())
        logger.warning(
            f"{self}: Mirai STT connection lost ({reason}); reconnecting"
            + (f" and replaying {held:.1f} s of audio" if held else "")
        )
        if self._dialing is None or self._dialing.done():
            self._retry_at = 0.0
            self._ensure_connected()

    def _unanswered_bytes(self) -> int:
        if self._endpointing == "manual":
            current = len(self._current.audio) if self._current is not None else 0
            return sum(len(u.audio) for u in self._pending) + current
        return len(self._unanswered)

    def _clear_unanswered(self):
        self._pending.clear()
        self._current = None
        self._coalesce.clear()
        self._unanswered.clear()
        self._vad_open = False
        self._vad_ends.clear()

    async def _connect_failed(self, exc: Exception):
        """No socket could be had: report it and stop holding audio for it."""
        status = _status(exc)
        code = exc.code if isinstance(exc, _ServerRefused) else None
        lost = self._seconds(self._unanswered_bytes())
        if status is not None:
            message = f"Mirai STT refused the WebSocket: HTTP {status}"
            response = getattr(exc, "response", None)
            detail = _error_detail(getattr(response, "body", None))
            if detail:
                message += f": {detail}"
            if status in _ADVICE:
                message += f". {_ADVICE[status]}"
            category = _HANDSHAKE_CATEGORIES.get(status, ErrorCategory.CONNECTIVITY)
            self._permanent = status in _PERMANENT_STATUSES
        elif code is not None:
            message = f"Mirai STT refused the session: {exc}"
            if code in _ADVICE:
                message += f". {_ADVICE[code]}"
            category = _ERROR_CATEGORIES.get(code, ErrorCategory.SERVER)
            self._permanent = code in _PERMANENT_CODES
        else:
            verb = "reconnect to" if self._connected_once else "connect to"
            message = f"Mirai STT could not {verb} {self._base_url}: {_describe(exc)}"
            category = ErrorCategory.CONNECTIVITY
        if lost:
            message += f" ({lost:.1f} s of audio was not transcribed)"
        self._clear_unanswered()
        self._retry_at = time.monotonic() + RETRY_AFTER_FAILURE_SECS
        self.last_error = message
        await self.push_error(error_msg=message, exception=exc, category=category)
        await self._call_event_handler("on_connection_error", message)

    # ---------- sending ----------

    def _ping_interval(self) -> float:
        interval = KEEPALIVE_SECS
        idle = self._sock.idle_timeout_secs if self._sock is not None else None
        if idle:
            interval = min(interval, idle / 2)
        return interval

    async def _send_loop(self):
        """Send queued messages in order on the current socket; keep it alive when quiet."""
        while True:
            sock = self._sock
            if sock is None or not self._outbox:
                self._wake.clear()
                timeout = None
                if sock is not None:
                    due = self._last_sent + self._ping_interval()
                    rotate_at = sock.rotate_at
                    if rotate_at is not None and self._spare is None:
                        due = min(due, max(rotate_at, self._replace_at))
                    timeout = max(0.05, due - time.monotonic())
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout)
                except TimeoutError:
                    await self._quiet(sock)
                continue
            item = self._outbox[0]
            try:
                await sock.websocket.send(item.text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._lost(sock, f"send failed: {_describe(exc)}")
                continue
            if sock is not self._sock:
                continue  # replaced while sending; the new socket's replay has it
            if self._outbox and self._outbox[0] is item:
                self._outbox.popleft()
            self._last_sent = time.monotonic()
            if item.audio_bytes:
                self._count(item.audio_bytes)
            if item.ends is not None:
                item.ends.end_sent_wall = time.time()

    def _count(self, nbytes: int):
        seconds = self._seconds(nbytes)
        self._usage_pending += seconds
        self.audio_seconds_sent += seconds

    async def _quiet(self, sock: _Socket | None):
        """Nothing sent for a while: keep the socket open, and replace an old session at a pause."""
        if sock is None or sock is not self._sock:
            return
        if time.monotonic() - self._last_sent >= self._ping_interval() and (
            self._endpointing != "manual" or self._current is None
        ):
            silence = bytes(self._bytes(KEEPALIVE_AUDIO_SECS))
            if self._encoding == "mulaw":
                silence = b"\xff" * len(silence)
            elif self._encoding == "alaw":
                silence = b"\xd5" * len(silence)
            try:
                await sock.websocket.send(json.dumps({"event": "ping"}))
                # Outside an utterance the server discards it; the idle timeout counts audio.
                await sock.websocket.send(_audio_out(silence).text)
            except Exception as exc:
                await self._lost(sock, f"keepalive failed: {_describe(exc)}")
                return
            self._last_sent = time.monotonic()
            self._count(len(silence))
        if self._rotation_due() and self._spare is None:
            self._start_replacement()
        await self._maybe_swap()

    # ---------- receiving ----------

    async def _read(self, sock: _Socket):
        reason = "closed by the server"
        try:
            async for message in sock.websocket:
                if sock is not self._sock:
                    return
                if not isinstance(message, str):
                    continue
                try:
                    event = json.loads(message)
                except ValueError:
                    logger.warning(f"{self}: ignoring a message that is not JSON: {message[:200]!r}")
                    continue
                if isinstance(event, dict):
                    await self._on_event(event, sock)
            code = getattr(sock.websocket, "close_code", None)
            if code is not None:
                why = getattr(sock.websocket, "close_reason", "") or ""
                reason = f"closed by the server: {code} {why}".strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            reason = _describe(exc)
        if sock is self._sock and not self._stopping:
            await self._lost(sock, reason)

    async def _on_event(self, event: dict, sock: _Socket):
        kind = _kind(event)
        if kind == "transcript.partial":
            await self._on_partial(event)
        elif kind == "transcript.final":
            await self._on_final(event)
        elif kind == "vad.speech_start":
            if self._endpointing == "vad" and not self._vad_open:
                self._vad_open = True
                await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif kind == "vad.speech_end":
            if self._endpointing == "vad" and self._vad_open:
                self._vad_open = False
                self._vad_ends.append(time.time())
                await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
        elif kind == "session.begin":
            sock.begin = event
        elif kind in ("pong", "config.updated"):
            pass
        elif kind == "session.draining":
            reason = event.get("reason")
            logger.debug(f"{self}: Mirai is ending session {sock.session_id} ({reason}); moving to a new one")
            if self._spare is not None and self._spare.websocket.state is State.OPEN:
                spare, self._spare = self._spare, None
                await self._swap(spare)
            else:
                self._start_replacement(urgent=True)
        elif kind == "session.end":
            logger.debug(
                f"{self}: session {sock.session_id} ended: {event.get('audio_seconds_billed')} s billed, "
                f"{event.get('total_utterances')} utterances"
            )
            self._ended.set()
        elif kind == "error":
            await self._on_server_error(event, sock)
        else:
            logger.debug(f"{self}: ignoring Mirai event {kind!r}")

    async def _on_partial(self, event: dict):
        text = _text(event)
        if not text:
            return
        await self.push_frame(
            InterimTranscriptionFrame(
                text, self._user_id, time_now_iso8601(), self._frame_language(event), result=event
            )
        )

    async def _on_final(self, event: dict):
        started: float | None = None
        spoken_in: str | None = None
        if self._endpointing == "manual":
            utt = self._pending.popleft() if self._pending else None
            if utt is not None:
                spoken_in = utt.language
                if utt.end_sent_wall:
                    started = utt.end_sent_wall
        else:
            started = self._vad_ends.popleft() if self._vad_ends else None
            # What came before is answered; keep a little as pre-roll for a replay.
            excess = len(self._unanswered) - self._bytes(PREROLL_SECS)
            if excess > 0:
                del self._unanswered[:excess]
        now = time.time()
        result = dict(event)
        result.setdefault("session_id", self.session_id)
        result.setdefault("request_id", self.request_id)
        if started is not None:
            latency = max(0.0, now - started)
            self.last_final_latency_ms = latency * 1000
            result["latency_ms"] = round(latency * 1000, 1)
            await self.start_ttfb_metrics(start_time=started)
            await self.stop_ttfb_metrics(end_time=now)
            await self.start_processing_metrics(start_time=started)
            await self.stop_processing_metrics(end_time=now)
        # Usage before the frame, so tracing can attach it to the span the frame closes.
        await self.emit_stt_usage_metrics()
        text = _text(event)
        language = self._frame_language(event, spoken_in)
        # One frame per utterance even when empty: a finalized transcript tells
        # the turn strategies nothing more is coming (empty text adds nothing).
        await self.push_frame(
            TranscriptionFrame(
                text, self._user_id, time_now_iso8601(), language, result=result, finalized=True
            )
        )
        if text:
            await self._trace_transcription(text, True, language)
        await self._maybe_swap()

    async def _on_server_error(self, event: dict, sock: _Socket):
        code, message, fatal = _error_parts(event)
        ids = ", ".join(f"{name} {value}" for name, value in (("session", sock.session_id),) if value)
        text = f"Mirai STT error {code}: {message}{f' [{ids}]' if ids else ''}"
        if not fatal:
            logger.warning(f"{self}: {text}")
            return
        sock.fatal = True
        if code in _PERMANENT_CODES:
            self._permanent = True
        if code in _ADVICE:
            text += f". {_ADVICE[code]}"
        self.last_error = text
        await self.push_error(error_msg=text, category=_ERROR_CATEGORIES.get(code, ErrorCategory.SERVER))

    def _frame_language(self, event: dict, spoken_in: str | None = None) -> Language | None:
        code = event.get("language_code") or event.get("language") or spoken_in or self._settings.language
        try:
            return Language(code)
        except ValueError:
            return None

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language | None = None):
        """Record a transcript for tracing."""

    # ---------- settings ----------

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Language and tuning change on the open socket; model and prefix padding at the next pause."""
        changed = await super()._update_settings(delta)
        if not changed or not self._started:
            return changed
        before = self._sock.runtime if self._sock is not None else None
        self._build_urls()
        runtime = self._runtime()
        if self._sock is not None and runtime != before:
            update = {k: v for k, v in runtime.items() if (before or {}).get(k) != v}
            self._queue(_Out(json.dumps({"event": "config.update", **update})))
            self._sock.runtime = runtime
        if changed.keys() & {"model", "prefix_padding_ms"} and self._sock is not None:
            # Fixed for a socket: open a new session and move to it at the next pause.
            self._start_replacement()
        return changed

    # ---------- the pipeline ----------

    def _pipeline_has_vad(self) -> bool:
        """Whether a processor linked to this one runs a VAD (a VADProcessor, or any ``vad_analyzer``)."""
        try:
            from pipecat.processors.audio.vad_processor import VADProcessor
        except ImportError:  # pragma: no cover
            VADProcessor = ()  # noqa: N806
        seen: set[int] = set()
        stack: list[Any] = [self]
        while stack:
            processor = stack.pop()
            if processor is None or id(processor) in seen:
                continue
            seen.add(id(processor))
            if processor is not self:
                if VADProcessor and isinstance(processor, VADProcessor):
                    return True
                if getattr(getattr(processor, "_params", None), "vad_analyzer", None) is not None:
                    return True
            stack.append(getattr(processor, "_next", None))
            stack.append(getattr(processor, "_prev", None))
            try:
                children = processor.processors if hasattr(processor, "processors") else None
            except Exception:
                children = None
            stack.extend(children or [])
            # A pipeline's source and sink reach its neighbours through the pipeline itself.
            for name in ("_upstream_push_frame", "_downstream_push_frame"):
                stack.append(getattr(getattr(processor, name, None), "__self__", None))
        return False


# ---------- helpers ----------


def _fixed_query(
    *, model: str, encoding: str, sample_rate: int, endpointing: str, prefix_padding_ms: int | None = None
) -> dict[str, Any]:
    """The settings fixed for a socket's life, in its query."""
    fixed: dict[str, Any] = {
        "model": model,
        "encoding": encoding,
        "sample_rate": sample_rate,
        "endpointing": endpointing,
    }
    if endpointing == "vad" and prefix_padding_ms is not None:
        fixed["prefix_padding_ms"] = prefix_padding_ms
    return fixed


def pool_url(
    url: str = DEFAULT_STT_WEBSOCKET_URL,
    *,
    sample_rate: int = 8000,
    endpointing: str = "manual",
    model: str = DEFAULT_MODEL,
    encoding: str = "linear16",
) -> str:
    """The URL the process's waiting STT sockets for these settings open (what :func:`prewarm` fills)."""
    fixed = _fixed_query(model=model, encoding=encoding, sample_rate=sample_rate, endpointing=endpointing)
    return f"{url.rstrip('/')}?{urlencode({**fixed, **_POOL_RUNTIME})}"


def _audio_out(piece: bytes) -> _Out:
    return _Out(
        json.dumps({"event": "audio_input", "audio": base64.b64encode(piece).decode()}),
        audio_bytes=len(piece),
    )


def _parse(raw: Any) -> dict:
    try:
        event = json.loads(raw) if isinstance(raw, str) else None
    except ValueError:
        event = None
    return event if isinstance(event, dict) else {}


def _kind(event: dict) -> Any:
    return event.get("event") or event.get("type")


def _text(event: dict) -> str:
    text = event.get("text")
    if text is None and isinstance(event.get("data"), dict):
        text = event["data"].get("transcript")
    return str(text or "").strip()


def _error_parts(event: dict) -> tuple[str, str, bool]:
    """(code, message, is_fatal) of an error event, flat or nested under ``error``."""
    error = event.get("error") if isinstance(event.get("error"), dict) else event
    code = str(error.get("code") or "error")
    message = str(error.get("message") or "")
    fatal = bool(error.get("is_fatal", event.get("is_fatal", False)))
    return code, message, fatal


async def _first_event(websocket) -> dict:
    """The socket's first event, which must be session.begin."""
    event = _parse(await websocket.recv())
    kind = _kind(event)
    if kind == "session.begin":
        return event
    if kind == "error":
        code, message, _ = _error_parts(event)
        raise _ServerRefused(code, message)
    raise ConnectionError(f"expected session.begin, got {kind!r}")


def _status(exc: BaseException) -> int | None:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _describe(exc: BaseException) -> str:
    status = _status(exc)
    if status is not None:
        return f"HTTP {status}"
    if isinstance(exc, TimeoutError):
        return "timed out"
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _error_detail(body: Any) -> str | None:
    """The ``error.message`` of a JSON error body, if there is one."""
    try:
        detail = json.loads(body).get("error", {}).get("message")
    except (TypeError, ValueError, AttributeError):
        return None
    return detail[:500] if isinstance(detail, str) else None


def _strip_query(url: str) -> str:
    return url.split("?", 1)[0]


def _http_base(url: str) -> str:
    """``wss://host/v1/audio/transcriptions/stream`` -> ``https://host/v1``."""
    for ws, http in (("wss://", "https://"), ("ws://", "http://")):
        if url.startswith(ws):
            url = http + url[len(ws) :]
    url = url.rstrip("/")
    suffix = "/audio/transcriptions/stream"
    return url[: -len(suffix)] if url.endswith(suffix) else url


def stt_query(url: str) -> dict[str, str]:
    """The query parameters of an STT socket URL (for logs and tests)."""
    return dict(parse_qsl(urlparse(url).query))


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception as exc:
        logger.debug(f"error closing a Mirai socket: {exc!r}")


def lin2ulaw(pcm: np.ndarray) -> np.ndarray:
    """16-bit PCM to G.711 mu-law bytes."""
    x = pcm.astype(np.int32)
    sign = (x < 0).astype(np.int32) << 7
    magnitude = np.minimum(np.abs(x), 32635) + 0x84
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def lin2alaw(pcm: np.ndarray) -> np.ndarray:
    """16-bit PCM to G.711 A-law bytes."""
    x = pcm.astype(np.int32)
    sign = (x >= 0).astype(np.int32) << 7
    magnitude = np.where(x < 0, -x - 1, x) >> 3  # 13-bit magnitude, 0..4095
    segment = np.where(magnitude >= 32, np.floor(np.log2(np.maximum(magnitude, 1))).astype(np.int32) - 4, 0)
    shift = np.maximum(segment, 1)
    mantissa = (magnitude >> shift) & 0x0F
    return ((sign | (segment << 4) | mantissa) ^ 0x55).astype(np.uint8)
