#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai speech-to-text over one WebSocket per call: :class:`STTStream`.

A stream sends a caller's audio to Mirai's ``/v1/audio/transcriptions/stream``
(``Authorization: Bearer`` with the API key) and tells a :class:`STTListener`
what comes back; without one, the transcripts wait in :meth:`STTStream.events`::

    async with STTStream(language="hi-IN", input_sample_rate=8000, encoding="mulaw") as stt:
        stt.send(pcm16)  # never blocks; held until the socket is up, replayed after a drop
        async for event in stt.events():
            if isinstance(event, Transcript) and event.final:
                print(event.text)

The wire protocol, in short. Client: ``audio_input`` {audio: base64},
``speech_start`` / ``speech_end`` / ``flush`` (manual endpointing),
``config.update``, ``ping``, ``end``. Server: ``session.begin`` (request id,
session id, limits), ``vad.speech_start`` / ``vad.speech_end`` (server
endpointing), ``transcript.partial`` (a full snapshot of the utterance so far),
``transcript.final`` (exactly one per utterance, in order), ``config.updated``,
``pong``, ``session.draining``, ``session.end`` (audio seconds billed, cost in
paise) and ``error`` {code, message, is_fatal}. Settings that fix the stream
(model, encoding, rate, endpointing) travel in the connection's query string;
language (``language_code``, a code or ``auto``), ``secondary_languages`` and
tuning can change on an open socket with ``config.update``. In an ``auto`` or
secondary-language session each transcript names the language it was heard in.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import time
import traceback
from collections import deque
from collections.abc import AsyncIterator, Coroutine, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlparse

import numpy as np
import soxr
from websockets.asyncio.client import connect as websocket_connect
from websockets.protocol import State

from fourier_frequency import errors
from fourier_frequency._net import websocket_connect_kwargs
from fourier_frequency.audio import lin2alaw, lin2ulaw
from fourier_frequency.edge import EdgeRoute, edge_mode
from fourier_frequency.errors import FrequencyError
from fourier_frequency.pool import STT_PROTOCOL, WebsocketPool, _retry_after, take_websocket, websocket_pool
from fourier_frequency.tts import api_key_from_env

logger = logging.getLogger(__name__)

DEFAULT_STT_WEBSOCKET_URL = "wss://sandbox.voice.miraiminds.co/v1/audio/transcriptions/stream"
DEFAULT_LANGUAGE = "hi-IN"
DEFAULT_MODEL = "mira-stt"
#: The languages Mirai's speech-to-text transcribes (``language_code`` on the socket).
STT_LANGUAGES = ("hi-IN", "en-IN", "gu-IN", "mr-IN", "pa-IN", "bn-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN")
#: ``language`` that asks Mirai to work out each utterance's language among :data:`STT_LANGUAGES`.
AUTO_LANGUAGE = "auto"
# Mirai's names for it ("unknown" is Sarvam's), sent as AUTO_LANGUAGE.
_AUTO = frozenset({"auto", "unknown"})
# Other providers' names for language detection: refused, with a pointer to "auto".
_DETECT = frozenset({"detect", "multi", "multilingual"})
#: The most ``secondary_languages`` a session takes.
MAX_SECONDARY_LANGUAGES = 3
# The rates and encodings the socket takes.
WIRE_SAMPLE_RATES = (8000, 16000)
ENCODINGS = ("linear16", "mulaw", "alaw")
# Audio goes out in pieces this long: base64 in JSON has real per-message
# overhead, and the gateway reads it in about this size.
CHUNK_SECS = 0.06
# Manual endpointing: audio held between turns and sent at speech_start. The
# host's VAD fires only after some speech (and a queue hop), so without it the
# utterance the server sees begins mid-word.
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
# After the stream gives up on a connection, the next attempt waits at least this long.
RETRY_AFTER_FAILURE_SECS = 5.0
# A graceful stop waits this long for the last finals and session.end.
STOP_WAIT_SECS = 2.0
# A session is replaced (at a moment between utterances) once this much of its
# maximum length is left, or at half of it if that comes later. A replacement
# that can't be opened is tried again after REPLACE_RETRY_SECS.
ROTATE_HEADROOM_SECS = 60.0
REPLACE_RETRY_SECS = 30.0
MAX_MESSAGE_BYTES = 1 << 20
# A task the stream cancels gets this long to finish.
CANCEL_TIMEOUT = 1.0

# Settings that config.update can change on an open socket, by query name.
_RUNTIME_FIELDS = ("stream_type", "mode", "threshold", "silence_duration_ms", "min_speech_duration_ms")
# Server-side VAD tuning: sent only when the server does the endpointing.
_VAD_FIELDS = frozenset({"threshold", "silence_duration_ms", "min_speech_duration_ms", "prefix_padding_ms"})
# What the pool's waiting sockets are opened with; a stream that wants other
# runtime settings sends config.update when it takes one.
_POOL_RUNTIME = {"language_code": DEFAULT_LANGUAGE}

_ERROR_CATEGORIES = {
    "unauthorized": errors.AUTHENTICATION,
    "invalid_token": errors.AUTHENTICATION,
    "forbidden": errors.AUTHORIZATION,
    "invalid_config": errors.INVALID_REQUEST,
    "invalid_request": errors.INVALID_REQUEST,
    "insufficient_balance": errors.QUOTA,
    "at_capacity": errors.RATE_LIMIT,
    "rate_limited": errors.RATE_LIMIT,
    "idle_timeout": errors.CONNECTIVITY,
    "unavailable": errors.CONNECTIVITY,
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
    400: errors.INVALID_REQUEST,
    401: errors.AUTHENTICATION,
    402: errors.QUOTA,
    403: errors.AUTHORIZATION,
    404: errors.INVALID_REQUEST,
    426: errors.INVALID_REQUEST,
    429: errors.RATE_LIMIT,
}
_PERMANENT_STATUSES = frozenset({400, 401, 402, 403, 404, 426})
_CHECK_SETTINGS = (
    "Check the language, secondary_languages, model, sample_rate, encoding and endpointing settings."
)
_ADVICE = {
    "unauthorized": "Check MIRAI_API_KEY.",
    401: "Check MIRAI_API_KEY.",
    "forbidden": "The API key is revoked or not allowed this request.",
    403: "The API key is revoked or not allowed this request.",
    "insufficient_balance": "The workspace is out of credit. Top it up in the Mirai console.",
    402: "The workspace is out of credit. Top it up in the Mirai console.",
    "at_capacity": "Ask Mirai to raise this workspace's concurrent STT session limit.",
    429: "Ask Mirai to raise this workspace's concurrent STT session limit.",
    "invalid_config": _CHECK_SETTINGS,
    400: _CHECK_SETTINGS,
}
# The settings STTStream.update() takes; anything not passed stays as it is.
_KEEP: Any = object()


def stt_language(language: Any, *, allow_auto: bool = True) -> str:
    """The language code Mirai's speech-to-text takes for ``language``, or a ``ValueError``.

    One of :data:`STT_LANGUAGES` in any case (``"gu-in"``), the bare language
    (``"gu"``) or an enum whose value is one (a Pipecat ``Language``:
    ``Language.GU``, ``Language.GU_IN``) all give ``"gu-IN"``. ``"auto"`` (or
    ``"unknown"``) gives ``"auto"``: Mirai works out each utterance's language
    among the ten. Languages outside the list are refused, and so is
    ``"auto"`` when ``allow_auto`` is false.
    """
    code = str(getattr(language, "value", language) or "").strip()
    lowered = code.lower()
    for supported in STT_LANGUAGES:
        if lowered in (supported.lower(), supported.split("-", 1)[0]):
            return supported
    if lowered in _AUTO and allow_auto:
        return AUTO_LANGUAGE
    if lowered in _DETECT:
        raise ValueError(
            f"Mirai speech-to-text spells language detection 'auto' (got {code!r}); "
            f"or pass the language the caller speaks, one of {', '.join(STT_LANGUAGES)}"
        )
    auto = ", or 'auto'" if allow_auto else ""
    raise ValueError(
        f"language must be one of {', '.join(STT_LANGUAGES)} (or the bare code: 'hi', 'gu', ...)"
        f"{auto}; got {language!r}"
    )


def stt_secondary_languages(
    languages: Iterable[Any] | str | None, language: Any = DEFAULT_LANGUAGE
) -> tuple[str, ...]:
    """``secondary_languages`` as Mirai's codes, or a ``ValueError``.

    A list (or a comma-separated string) of up to
    :data:`MAX_SECONDARY_LANGUAGES` of :data:`STT_LANGUAGES`, each taken as
    :func:`stt_language` takes it. Duplicates and ``language`` itself are
    dropped. They need a fixed ``language``: with ``"auto"`` every language is
    already considered, so a non-empty list is refused, as the API refuses it.
    """
    if languages is None:
        return ()
    if isinstance(languages, str):  # "en-IN,gu-IN", or a Pipecat Language (a str enum)
        items: list[Any] = [part for part in languages.split(",") if part.strip()]
    else:
        items = list(languages)
    primary = stt_language(language)
    codes: list[str] = []
    for item in items:
        try:
            code = stt_language(item, allow_auto=False)
        except ValueError as exc:
            raise ValueError(f"secondary_languages: {exc}") from None
        if code != primary and code not in codes:
            codes.append(code)
    if codes and primary == AUTO_LANGUAGE:
        raise ValueError(
            "secondary_languages need a fixed language: with language='auto' Mirai already "
            "considers every language"
        )
    if len(codes) > MAX_SECONDARY_LANGUAGES:
        raise ValueError(
            f"secondary_languages takes at most {MAX_SECONDARY_LANGUAGES} languages; got {len(codes)}"
        )
    return tuple(codes)


def _check_options(url: str | None, sample_rate: Any, encoding: Any) -> str:
    """``url`` (Mirai's sandbox when ``None``), or a ``ValueError`` for a setting the socket doesn't take."""
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
    return url


# ---------- what a stream tells its host ----------


class STTListener:
    """What a stream tells its host. Override what you need; the rest do nothing.

    The stream awaits each call on its own tasks, in the order things happen,
    so a slow method holds up what comes after it. ``language`` is the code a
    transcript was heard in (the event's own, else the one asked for), or
    ``None`` in an ``"auto"`` session when Mirai named none.
    """

    async def on_connected(self) -> None:
        """A socket is up: the first, or a new one after a drop or a session change."""

    async def on_disconnected(self) -> None:
        """The socket closed (a new one follows when audio is waiting for it), or the stream did."""

    async def on_connection_error(self, message: str) -> None:
        """No socket could be had; ``message`` is what :meth:`on_error` was just told."""

    async def on_error(self, error: FrequencyError) -> None:
        """A refusal, a failed reconnect or a fatal error from Mirai.

        The audio it held is dropped. The stream tries again with the next
        audio, unless ``error.permanent``: then it has stopped for good.
        """

    async def on_speech_started(self) -> None:
        """Mirai's VAD heard speech begin (``endpointing="vad"`` only)."""

    async def on_speech_stopped(self) -> None:
        """Mirai's VAD heard speech end; the utterance's final follows."""

    async def on_partial(self, text: str, language: str | None, event: dict) -> None:
        """The utterance so far (a full snapshot each time, never empty); ``event`` is Mirai's."""

    async def on_final_latency(self, started: float, received: float) -> None:
        """An utterance's final arrived (just before :meth:`on_final`), ``time.time()`` both.

        ``started`` is when the utterance's end was known: ``speech_end`` sent
        (manual endpointing) or ``vad.speech_end`` received (Mirai's VAD).
        Told only when that is known.
        """

    async def on_usage(self, audio_seconds: float) -> None:
        """Audio seconds sent since the last report (replays included: they are billed).

        Told just before each final that has some; :meth:`STTStream.take_usage`
        hands over the rest.
        """

    async def on_final(self, text: str, language: str | None, event: dict) -> None:
        """An utterance's final transcript: exactly one per utterance, in order.

        ``text`` is empty when the utterance had no words (it still ends the
        utterance). ``event`` is Mirai's ``transcript.final`` with
        ``session_id``, ``request_id`` and, when known, ``latency_ms``.
        """

    async def on_closed(self) -> None:
        """The stream closed; nothing more comes."""


@dataclass
class Transcript:
    """A partial (``final=False``) or an utterance's final transcript."""

    text: str
    final: bool
    language: str | None = None  # the language heard (None: an auto session and Mirai named none)
    latency_ms: float | None = None  # a final: the utterance's end to the final, when known
    event: dict = field(default_factory=dict)  # Mirai's event (see STTListener.on_final)


@dataclass
class SpeechStarted:
    """Mirai's VAD heard speech begin (``endpointing="vad"``)."""


@dataclass
class SpeechStopped:
    """Mirai's VAD heard speech end; the utterance's final follows."""


@dataclass
class StreamError:
    """Something failed (see :meth:`STTListener.on_error`)."""

    error: FrequencyError


@dataclass
class StreamClosed:
    """The stream closed: the last event."""


STTEvent = Transcript | SpeechStarted | SpeechStopped | StreamError | StreamClosed


class _Events(STTListener):
    """The default listener: everything worth reading, queued for :meth:`STTStream.events`."""

    def __init__(self):
        self.queue: asyncio.Queue[STTEvent] = asyncio.Queue()
        self.closed = False

    async def on_error(self, error: FrequencyError) -> None:
        self.queue.put_nowait(StreamError(error))

    async def on_speech_started(self) -> None:
        self.queue.put_nowait(SpeechStarted())

    async def on_speech_stopped(self) -> None:
        self.queue.put_nowait(SpeechStopped())

    async def on_partial(self, text: str, language: str | None, event: dict) -> None:
        self.queue.put_nowait(Transcript(text, False, language, event=event))

    async def on_final(self, text: str, language: str | None, event: dict) -> None:
        self.queue.put_nowait(Transcript(text, True, language, event.get("latency_ms"), event))

    async def on_closed(self) -> None:
        self.closed = True
        self.queue.put_nowait(StreamClosed())


# ---------- the stream's state ----------


@dataclass
class _Settings:
    """What a stream asks Mirai for; ``None`` leaves it to Mirai."""

    model: str
    language: str
    secondary_languages: tuple[str, ...] = ()
    stream_type: str | None = None
    mode: str | None = None
    threshold: float | None = None
    silence_duration_ms: int | None = None
    min_speech_duration_ms: int | None = None
    prefix_padding_ms: int | None = None


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


# ---------- the stream ----------


class STTStream:
    """Mirai speech-to-text for one call: one WebSocket, kept up for the whole call.

    The caller's audio (16-bit mono PCM at ``input_sample_rate``) streams to
    Mirai in 60 ms pieces; partial transcripts come back as they form and each
    utterance's final transcript once, in order (:class:`STTListener`, or
    :meth:`events`).

    Who decides where an utterance ends (``endpointing``):

    - ``"vad"`` (default): Mirai's VAD. Audio streams continuously and Mirai
      says where speech began and ended (``SpeechStarted`` / ``SpeechStopped``).
    - ``"manual"``: your VAD. :meth:`speech_started` sends ``speech_start``
      with the last 0.5 s of audio, so the first word isn't cut, and
      :meth:`speech_stopped` sends ``speech_end``; the final follows within
      tens of milliseconds. Between turns nothing is sent but a keepalive.

    Audio at 8 kHz (phone calls) goes as it is; any other rate is resampled
    to 16 kHz (``sample_rate``). The socket opens on ``url``, Mirai's API,
    which sits next to the GPUs (were Mirai ever to offer an STT edge, it
    would open there: see :mod:`fourier_frequency.edge`), and usually comes
    ready from the process's pool (see :mod:`fourier_frequency.pool`). It
    opens in the background, so :meth:`send` never waits for it: audio is held
    until it is up. If it drops, the stream dials once more and replays the
    audio Mirai hadn't answered (up to 30 s); if that fails too it reports an
    error. A session near Mirai's maximum length is replaced at a pause between
    utterances, and one Mirai is draining at once.

    What Mirai billed is in ``audio_seconds_billed`` and ``cost_paise`` (from
    each ``session.end``); ``audio_seconds_sent`` is what went out.

    Example::

        async with STTStream(language="gu-IN", input_sample_rate=8000, encoding="mulaw") as stt:
            stt.send(pcm16)
            async for event in stt.events():
                ...
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str | None = None,
        language: Any = DEFAULT_LANGUAGE,
        secondary_languages: Iterable[Any] | str | None = None,
        model: str = DEFAULT_MODEL,
        input_sample_rate: int = 16000,
        sample_rate: int | Literal["auto"] = "auto",
        encoding: Literal["auto", "linear16", "mulaw", "alaw"] = "auto",
        endpointing: Literal["manual", "vad"] = "vad",
        edge: bool | str = "auto",
        shared_pool: bool = True,
        stream_type: str | None = None,
        mode: str | None = None,
        threshold: float | None = None,
        silence_duration_ms: int | None = None,
        min_speech_duration_ms: int | None = None,
        prefix_padding_ms: int | None = None,
        listener: STTListener | None = None,
        name: str | None = None,
    ):
        """Set up a stream; :meth:`connect` (or ``async with``) opens it.

        Args:
            api_key: Mirai API key. Defaults to ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``). Sent as an ``Authorization: Bearer`` header.
            url: The streaming endpoint,
                ``wss://<host>/v1/audio/transcriptions/stream``. Defaults to
                Mirai's sandbox.
            language: The language the caller speaks: ``"hi-IN"`` (the
                default; Hindi and Hinglish), ``"en-IN"``, ``"gu-IN"``,
                ``"mr-IN"``, ``"pa-IN"``, ``"bn-IN"``, ``"ta-IN"``, ``"te-IN"``,
                ``"kn-IN"`` or ``"ml-IN"``, in any case or as the bare code
                (``"gu"``); or ``"auto"`` when you don't know it (see
                :func:`stt_language`).
            secondary_languages: Up to three more of the ten that callers may
                switch to, with a fixed ``language`` (see
                :func:`stt_secondary_languages`).
            model: Mirai model name. Defaults to ``"mira-stt"``.
            input_sample_rate: The rate of the 16-bit PCM given to :meth:`send`.
            sample_rate: The rate audio is sent at. ``"auto"`` (the default)
                sends 8 kHz audio as it is and resamples anything else to
                16 kHz; ``8000`` or ``16000`` resamples to that rate when
                ``input_sample_rate`` differs.
            encoding: ``"auto"`` (linear16), ``"linear16"``, or ``"mulaw"`` /
                ``"alaw"`` (8 kHz only; half the bytes). Encoded here: the input
                is always 16-bit PCM.
            endpointing: ``"vad"`` (Mirai's VAD; the default) or ``"manual"``
                (yours: :meth:`speech_started` / :meth:`speech_stopped`).
            edge: ``"auto"`` (default): open the socket on ``url``, or on an
                STT edge if Mirai's API ever offers one. ``False`` always uses
                ``url``; a ``wss://`` URL uses that edge instead.
                ``MIRAI_STT_EDGE=off`` turns ``"auto"`` off.
            shared_pool: Take a socket the process's pool opened ahead of need
                (the default; the first stream starts the pool). ``False``
                always connects and starts no pool.
            stream_type: ``"fast"`` or ``"balanced"``; ``None`` leaves it to Mirai.
            mode: ``"transcribe"`` or ``"codemix"`` (English words kept in Latin
                script); ``None`` leaves it to Mirai.
            threshold: Mirai's VAD sensitivity, 0 to 1 (``endpointing="vad"`` only).
            silence_duration_ms: Silence that ends an utterance (``"vad"`` only).
            min_speech_duration_ms: Shortest speech that opens an utterance (``"vad"`` only).
            prefix_padding_ms: Audio kept before speech (``"vad"`` only; fixed
                for a socket, so a change opens a new one).
            listener: Told what happens (:class:`STTListener`). Without one, it
                waits in :meth:`events`.
            name: How log lines name this stream.

        Raises:
            ValueError: An option Mirai doesn't take, or no API key.
        """
        self.name = name or type(self).__name__
        key = api_key_from_env(api_key, self.name)
        url = _check_options(url, sample_rate, encoding)
        if endpointing not in ("manual", "vad"):
            raise ValueError(f"endpointing must be 'manual' or 'vad'; got {endpointing!r}")
        if type(input_sample_rate) is not int or input_sample_rate <= 0:
            raise ValueError(f"input_sample_rate must be a positive int; got {input_sample_rate!r}")
        route = edge_mode(edge, "stt")
        language = stt_language(language)
        self._settings = _Settings(
            model=model,
            language=language,
            secondary_languages=stt_secondary_languages(secondary_languages, language),
            stream_type=stream_type,
            mode=mode,
            threshold=threshold,
            silence_duration_ms=silence_duration_ms,
            min_speech_duration_ms=min_speech_duration_ms,
            prefix_padding_ms=prefix_padding_ms,
        )
        self._listener: STTListener = listener if listener is not None else _Events()

        self._base_url = url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._rate_option = sample_rate
        self._encoding = "linear16" if encoding == "auto" else encoding
        self._edge_mode = route
        self._shared_pool = shared_pool
        self._http_base = _http_base(self._base_url)
        self._input_rate = input_sample_rate

        # Decided at start.
        self._endpointing = endpointing
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
        #: Audio seconds sent over the stream's life (replays included: they are billed).
        self.audio_seconds_sent = 0.0
        #: What Mirai billed, summed over the sessions that ended with ``session.end``
        #: (``audio_seconds_billed``, ``cost_paise``).
        self.audio_seconds_billed = 0.0
        self.cost_paise = 0.0

    def __str__(self) -> str:
        return self.name

    async def __aenter__(self) -> STTStream:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close(graceful=exc_type is None)

    # ---------- what the host asks ----------

    @property
    def endpointing(self) -> str:
        """``"manual"`` or ``"vad"``: who ends utterances."""
        return self._endpointing

    @property
    def wire_sample_rate(self) -> int:
        """The rate audio is sent at (decided when the stream starts)."""
        return self._wire_rate

    @property
    def encoding(self) -> str:
        """``"linear16"``, ``"mulaw"`` or ``"alaw"``: how audio is sent."""
        return self._encoding

    @property
    def language(self) -> str:
        """The language asked for: one of :data:`STT_LANGUAGES`, or ``"auto"``."""
        return self._settings.language

    @property
    def secondary_languages(self) -> tuple[str, ...]:
        """The languages callers may switch to (``()``: ``language`` alone)."""
        return self._settings.secondary_languages

    def take_usage(self) -> float:
        """Audio seconds sent since the last report (:meth:`STTListener.on_usage`) or call."""
        seconds, self._usage_pending = self._usage_pending, 0.0
        return seconds

    async def events(self) -> AsyncIterator[STTEvent]:
        """What the stream heard, in order, until :class:`StreamClosed`.

        :class:`Transcript` (partials and finals), :class:`SpeechStarted` /
        :class:`SpeechStopped` (Mirai's VAD), :class:`StreamError` and, last,
        :class:`StreamClosed`. Events wait here until they are read. Only for a
        stream made without ``listener``: a listener is told them instead.
        """
        listener = self._listener
        if not isinstance(listener, _Events):
            raise RuntimeError(
                f"{self} tells its listener what happens; events() is for a stream without one"
            )
        while not (listener.closed and listener.queue.empty()):
            event = await listener.queue.get()
            yield event
            if isinstance(event, StreamClosed):
                return

    # ---------- lifecycle ----------

    async def connect(self, *, input_sample_rate: int | None = None, endpointing: str | None = None):
        """Start streaming; the socket opens in the background (``async with`` calls this).

        ``input_sample_rate`` and ``endpointing``, when given, replace the
        constructor's: for a host that learns them only when its call starts.
        """
        if self._stopping:
            raise RuntimeError(f"{self} is closed; open a new stream")
        if self._started:
            return
        if input_sample_rate is not None:
            self._input_rate = input_sample_rate
        if endpointing is not None:
            if endpointing not in ("manual", "vad"):
                raise ValueError(f"endpointing must be 'manual' or 'vad'; got {endpointing!r}")
            self._endpointing = endpointing
        in_rate = self._input_rate
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
        self._sender = self._spawn(self._send_loop(), "mirai-stt-send")
        self._ensure_connected()

    async def close(self, *, graceful: bool = True):
        """Close the stream.

        ``graceful`` (the default) ends the open utterance and waits briefly
        (``STOP_WAIT_SECS``) for the last finals and Mirai's ``session.end``
        (what it billed); ``graceful=False`` closes at once. Closing twice does
        nothing.
        """
        await self._shutdown(graceful=graceful)

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
                await _cancel(task)
        for sock in (self._sock, self._spare):
            if sock is not None:
                await self._close(sock)
        had_socket = self._sock is not None
        self._sock = self._spare = None
        if self._pool is not None:
            pool, self._pool = self._pool, None
            pool.release(self)
        if had_socket:
            await self._listener.on_disconnected()
        await self._listener.on_closed()

    def _spawn(self, coroutine: Coroutine, name: str) -> asyncio.Task:
        """A task of this stream's; an exception it ends with is logged, as Pipecat's tasks log theirs."""
        label = f"{self}::{name}"

        async def run():
            try:
                return await coroutine
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last = traceback.extract_tb(exc.__traceback__)[-1]
                logger.error(f"{label} unexpected exception ({last.filename}:{last.lineno}): {exc}")

        task = asyncio.get_running_loop().create_task(run(), name=label)

        def close_unstarted(_: asyncio.Task):
            # Cancelled before run() awaited it: close it, or Python warns it was never awaited.
            if inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED:
                coroutine.close()

        task.add_done_callback(close_unstarted)
        return task

    # ---------- audio in ----------

    def send(self, audio: bytes | bytearray):
        """Queue 16-bit PCM at ``input_sample_rate``. Never blocks.

        Under Mirai's VAD every byte goes; under manual endpointing only what
        is inside an utterance, plus its pre-roll. Audio is held while the
        socket opens, replayed on a new one after a drop, and dropped before
        :meth:`connect`, after :meth:`close` and once the stream has stopped
        for good (a permanent error).
        """
        if self._started and not self._stopping and not self._permanent:
            data = self._to_wire(audio if isinstance(audio, bytes) else bytes(audio))
            if data:
                if self._endpointing == "manual" and self._current is None:
                    self._ring += data
                    excess = len(self._ring) - self._bytes(PREROLL_SECS)
                    if excess > 0:
                        del self._ring[:excess]
                else:
                    self._coalesce += data
                    self._cut(final=False)

    def _to_wire(self, audio: bytes) -> bytes:
        """Input PCM to the socket's rate and encoding (whole samples only)."""
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

    async def speech_started(self):
        """Your VAD heard speech begin (``endpointing="manual"``; ignored otherwise).

        Sends ``speech_start`` with the last 0.5 s of audio before it.
        """
        if self._endpointing != "manual" or not self._started:
            return
        await self._speech_started()

    async def speech_stopped(self):
        """Your VAD heard speech end (``endpointing="manual"``): the utterance's final follows."""
        if self._endpointing != "manual" or not self._started:
            return
        await self._speech_stopped()

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
        fixed = _fixed_query(
            model=self._settings.model,
            encoding=self._encoding,
            sample_rate=self._wire_rate,
            endpointing=self._endpointing,
            prefix_padding_ms=self._settings.prefix_padding_ms,
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
        language = self._settings.language
        params: dict[str, Any] = {"language_code": language}
        if language != AUTO_LANGUAGE:
            secondary = [code for code in self._settings.secondary_languages or () if code != language]
            if secondary:
                params["secondary_languages"] = ",".join(secondary)
        for name in _RUNTIME_FIELDS:
            value = getattr(self._settings, name)
            if value is None:
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
        self._dialing = self._spawn(self._dial_task(), "mirai-stt-dial")

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
        change = _runtime_change(sock.runtime, wanted)
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
        sock.reader = self._spawn(self._read(sock), "mirai-stt-receive")
        replay = self._replay()
        runtime = self._runtime()
        if sock.runtime != runtime:
            update = {**runtime, **_runtime_change(sock.runtime, runtime)}
            replay.appendleft(_Out(json.dumps({"event": "config.update", **update})))
            sock.runtime = runtime
        replayed = sum(item.audio_bytes for item in replay)
        if replayed and self._connected_once:
            logger.info(f"{self}: replaying {self._seconds(replayed):.1f} s of audio not yet transcribed")
        self._outbox = replay
        self._last_sent = time.monotonic()
        self._connected_once = True
        self._wake.set()
        logger.debug(f"{self}: session {sock.session_id} on {sock.url}")
        await self._listener.on_connected()

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
        self._dialing = self._spawn(self._dial_task(), "mirai-stt-dial")

    async def _close(self, sock: _Socket, *, end: bool = False):
        reader, sock.reader = sock.reader, None
        if reader is not None and reader is not asyncio.current_task() and not reader.done():
            await _cancel(reader)
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
        await self._listener.on_disconnected()
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
            body = getattr(response, "body", None)
            detail = _error_detail(body)
            if detail:
                message += f": {detail}"
            if status in _ADVICE:
                message += f". {_ADVICE[status]}"
            category = _HANDSHAKE_CATEGORIES.get(status, errors.CONNECTIVITY)
            self._permanent = status in _PERMANENT_STATUSES
            code = _error_code(body)
        elif code is not None:
            message = f"Mirai STT refused the session: {exc}"
            if code in _ADVICE:
                message += f". {_ADVICE[code]}"
            category = _ERROR_CATEGORIES.get(code, errors.SERVER)
            self._permanent = code in _PERMANENT_CODES
        else:
            verb = "reconnect to" if self._connected_once else "connect to"
            message = f"Mirai STT could not {verb} {self._base_url}: {_describe(exc)}"
            category = errors.CONNECTIVITY
        if lost:
            message += f" ({lost:.1f} s of audio was not transcribed)"
        self._clear_unanswered()
        self._retry_at = time.monotonic() + RETRY_AFTER_FAILURE_SECS
        self.last_error = message
        error = FrequencyError(
            message, category=category, code=code, status=status, permanent=self._permanent
        )
        error.__cause__ = exc
        await self._listener.on_error(error)
        await self._listener.on_connection_error(message)

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
                await self._listener.on_speech_started()
        elif kind == "vad.speech_end":
            if self._endpointing == "vad" and self._vad_open:
                self._vad_open = False
                self._vad_ends.append(time.time())
                await self._listener.on_speech_stopped()
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
            billed, cost = event.get("audio_seconds_billed"), event.get("cost_paise")
            if isinstance(billed, int | float):
                self.audio_seconds_billed += billed
            if isinstance(cost, int | float):
                self.cost_paise += cost
            logger.debug(
                f"{self}: session {sock.session_id} ended: {billed} s billed, {cost} paise, "
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
        await self._listener.on_partial(text, self._language_heard(event), event)

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
            await self._listener.on_final_latency(started, now)
        # Usage before the final, so a host's tracing can attach it to the span the final closes.
        seconds = self.take_usage()
        if seconds > 0:
            await self._listener.on_usage(seconds)
        # One final per utterance even when empty: a final tells the host's turn
        # logic nothing more is coming (empty text adds nothing).
        await self._listener.on_final(_text(event), self._language_heard(event, spoken_in), result)
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
        await self._listener.on_error(
            FrequencyError(
                text,
                category=_ERROR_CATEGORIES.get(code, errors.SERVER),
                code=code,
                request_id=sock.request_id,
                permanent=code in _PERMANENT_CODES,
            )
        )

    def _language_heard(self, event: dict, spoken_in: str | None = None) -> str | None:
        """The language a transcript was heard in: the event's own, else the one asked for.

        ``None`` when that is ``"auto"`` and the event names none.
        """
        code = event.get("language") or event.get("language_code") or spoken_in or self._settings.language
        return None if code == AUTO_LANGUAGE else str(code)

    # ---------- settings ----------

    async def update(
        self,
        *,
        language: Any = _KEEP,
        secondary_languages: Any = _KEEP,
        model: Any = _KEEP,
        stream_type: Any = _KEEP,
        mode: Any = _KEEP,
        threshold: Any = _KEEP,
        silence_duration_ms: Any = _KEEP,
        min_speech_duration_ms: Any = _KEEP,
        prefix_padding_ms: Any = _KEEP,
    ) -> set[str]:
        """Change settings mid-call: only what is passed. Returns the names that changed.

        Language, secondary languages and tuning change on the open socket
        (``config.update``); model and prefix padding open a new session and
        move to it at the next pause between utterances. A change to
        ``"auto"`` clears ``secondary_languages`` unless they are passed too.
        A language Mirai doesn't transcribe (or secondary languages it won't
        take) raises ``ValueError`` and changes nothing.
        """
        given = {
            name: value
            for name, value in (
                ("language", language),
                ("secondary_languages", secondary_languages),
                ("model", model),
                ("stream_type", stream_type),
                ("mode", mode),
                ("threshold", threshold),
                ("silence_duration_ms", silence_duration_ms),
                ("min_speech_duration_ms", min_speech_duration_ms),
                ("prefix_padding_ms", prefix_padding_ms),
            )
            if value is not _KEEP
        }
        if "language" in given:
            given["language"] = stt_language(given["language"])
        primary = given.get("language", self._settings.language)
        if "secondary_languages" in given:
            given["secondary_languages"] = stt_secondary_languages(given["secondary_languages"], primary)
        elif primary == AUTO_LANGUAGE:
            given["secondary_languages"] = ()  # auto already considers every language
        changed: set[str] = set()
        for name, value in given.items():
            if getattr(self._settings, name) != value:
                setattr(self._settings, name, value)
                changed.add(name)
        if not changed or not self._started:
            return changed
        before = self._sock.runtime if self._sock is not None else None
        self._build_urls()
        runtime = self._runtime()
        if self._sock is not None and runtime != before:
            update = _runtime_change(before, runtime)
            self._queue(_Out(json.dumps({"event": "config.update", **update})))
            self._sock.runtime = runtime
        if changed & {"model", "prefix_padding_ms"} and self._sock is not None:
            # Fixed for a socket: open a new session and move to it at the next pause.
            self._start_replacement()
        return changed


# ---------- helpers ----------


def _runtime_change(before: dict | None, wanted: dict) -> dict[str, Any]:
    """The ``config.update`` fields that take a socket from ``before`` to ``wanted``.

    Secondary languages the socket has and ``wanted`` drops go as ``""``,
    which clears them (and must travel with a change to ``auto``).
    """
    before = before or {}
    change = {k: v for k, v in wanted.items() if before.get(k) != v}
    if before.get("secondary_languages") and "secondary_languages" not in wanted:
        change["secondary_languages"] = ""
    return change


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
    """The URL the process's waiting STT sockets for these settings open (what ``prewarm`` fills)."""
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


def _error_body(body: Any) -> dict:
    """The ``error`` object of a JSON error body (a refused handshake's), or ``{}``."""
    try:
        error = json.loads(body).get("error", {})
    except (TypeError, ValueError, AttributeError):
        return {}
    return error if isinstance(error, dict) else {}


def _error_detail(body: Any) -> str | None:
    """The ``error.message`` of a JSON error body, if there is one."""
    detail = _error_body(body).get("message")
    return detail[:500] if isinstance(detail, str) else None


def _error_code(body: Any) -> str | None:
    """The ``error.code`` of a JSON error body, if there is one."""
    code = _error_body(body).get("code")
    return code if isinstance(code, str) and code else None


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


async def _cancel(task: asyncio.Task, timeout: float = CANCEL_TIMEOUT):
    """Cancel ``task`` and wait (at most ``timeout``) for it to finish, as Pipecat's task manager does."""
    current = asyncio.current_task()
    if task is current:
        return
    cancels_requested = current.cancelling() if current else 0
    task.cancel()
    try:
        await asyncio.wait_for(task, timeout)
    except TimeoutError:
        logger.warning(f"{task.get_name()}: timed out waiting for task to cancel")
    except asyncio.CancelledError:
        # Either ``task`` finished cancelling, or the caller was cancelled while
        # it waited. Only the latter may propagate.
        if current is not None and current.cancelling() > cancels_requested:
            raise
    except Exception as exc:
        logger.error(f"{task.get_name()} unexpected exception while cancelling task: {exc}")


async def _close_quietly(websocket):
    try:
        await websocket.close()
    except Exception as exc:
        logger.debug(f"error closing a Mirai socket: {exc!r}")
