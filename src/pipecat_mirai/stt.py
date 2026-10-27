#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai speech-to-text over one WebSocket per pipeline.

``MiraiSTTService`` streams the caller's audio to Mirai's
``/v1/audio/transcriptions/stream`` (``Authorization: Bearer`` with the API
key) and turns what comes back into Pipecat transcription frames. The engine
is :class:`fourier_frequency.STTStream` (the socket, the pool, reconnects and
replays, session rotation, keepalives, resampling and G.711, and the wire
protocol, described in :mod:`fourier_frequency.stt`); this module is how it
meets Pipecat: VAD frames in, transcription frames, metrics and errors out.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

# The engine's names, importable from here as they always were.
from fourier_frequency.audio import lin2alaw, lin2ulaw  # noqa: F401
from fourier_frequency.edge import edge_mode
from fourier_frequency.errors import FrequencyError
from fourier_frequency.stt import (  # noqa: F401
    AUTO_LANGUAGE,
    CHUNK_SECS,
    CLOSE_TIMEOUT,
    CONFIG_UPDATE_TIMEOUT,
    CONNECT_RETRY_SECS,
    CONNECT_RETRY_WAIT,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    DEFAULT_STT_WEBSOCKET_URL,
    ENCODINGS,
    KEEPALIVE_AUDIO_SECS,
    KEEPALIVE_SECS,
    MAX_MESSAGE_BYTES,
    MAX_SECONDARY_LANGUAGES,
    OPEN_TIMEOUT,
    PREROLL_SECS,
    REPLACE_RETRY_SECS,
    REPLAY_CHUNK_SECS,
    REPLAY_SECS,
    RETRY_AFTER_FAILURE_SECS,
    ROTATE_HEADROOM_SECS,
    STOP_WAIT_SECS,
    STT_LANGUAGES,
    WIRE_SAMPLE_RATES,
    STTListener,
    STTStream,
    _check_options,
    _fixed_query,
    _http_base,
    _runtime_change,
    pool_url,
    stt_language,
    stt_query,
    stt_secondary_languages,
)
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

# Pipecat's turn strategies wait up to this long after the VAD stops for a
# final (a finalized transcript ends the wait at once).
DEFAULT_TTFS_P99 = 0.5
# The settings the stream takes from MiraiSTTSettings, besides language and model.
_TUNING = (
    "stream_type",
    "mode",
    "threshold",
    "silence_duration_ms",
    "min_speech_duration_ms",
    "prefix_padding_ms",
)


@dataclass
class MiraiSTTSettings(STTSettings):
    """Runtime-updatable settings for :class:`MiraiSTTService`.

    Parameters:
        model: Mirai model name. Defaults to ``"mira-stt"``. A change opens a
            new socket at the next pause between utterances.
        language: The language the caller speaks: ``hi-IN`` (default; Hindi
            and Hinglish), ``en-IN``, ``gu-IN``, ``mr-IN``, ``pa-IN``,
            ``bn-IN``, ``ta-IN``, ``te-IN``, ``kn-IN`` or ``ml-IN``. Any case,
            the bare code (``"gu"``) or a Pipecat ``Language`` works too.
            ``"auto"`` (or ``"unknown"``) has Mirai work out each utterance's
            language among the ten (see :func:`stt_language`). A change applies
            on the open socket (``config.update``); a change to ``"auto"``
            clears ``secondary_languages``.
        secondary_languages: Up to three more of the ten that callers may
            switch to, with a fixed ``language`` (see
            :func:`stt_secondary_languages`). ``language`` stays the default and
            an utterance moves to one of these only when Mirai is confident.
            ``()`` (the default) listens for ``language`` alone. A change applies
            on the open socket.
        stream_type: ``"fast"`` or ``"balanced"``; ``None`` leaves it to Mirai.
        mode: ``"transcribe"`` or ``"codemix"`` (English words kept in Latin
            script); ``None`` leaves it to Mirai.
        threshold: Server VAD sensitivity, 0 to 1 (server endpointing only).
        silence_duration_ms: Silence that ends an utterance (server endpointing only).
        min_speech_duration_ms: Shortest speech that opens an utterance (server endpointing only).
        prefix_padding_ms: Audio kept before speech (server endpointing only;
            fixed for a socket, so a change opens a new one).
    """

    secondary_languages: tuple[str, ...] | list[str] | str | None | NotGiven = field(
        default_factory=lambda: NOT_GIVEN
    )
    stream_type: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    mode: str | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    silence_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    min_speech_duration_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    prefix_padding_ms: int | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


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
    to 16 kHz (``sample_rate``). The socket opens on ``url``, Mirai's API,
    which sits next to the GPUs (were Mirai ever to offer an STT edge, it
    would open there: see :mod:`fourier_frequency.edge`), and usually comes ready
    from the process's pool (see :mod:`fourier_frequency.pool`). It opens in the
    background, so the pipeline never waits for it: audio is held until it is
    up. If it drops, the service dials once more and replays the audio Mirai
    hadn't answered (up to 30 s); if that fails too it reports an
    ``ErrorFrame``.

    Metrics: TTFB is ``speech_end`` sent (``vad.speech_end`` received under
    server endpointing) to the final received, reported with processing time
    for the same span; usage is the audio seconds sent. What Mirai billed is
    in ``audio_seconds_billed`` and ``cost_paise`` (from each ``session.end``).

    Languages: pass the one your callers speak (``language="gu-IN"``); add
    ``secondary_languages`` when they may switch between known languages; use
    ``language="auto"`` only when you don't know it. In an ``auto`` or
    secondary-language session each frame's ``language`` is the language Mirai
    heard that utterance in.

    Example::

        stt = MiraiSTTService(language="gu-IN")  # api_key from MIRAI_API_KEY
        stt = MiraiSTTService(language="hi-IN", secondary_languages=["en-IN"])
        stt = MiraiSTTService(language="auto")

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
        secondary_languages: Iterable[str | Language] | str | None = None,
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
            language: The language the caller speaks (shortcut for
                ``settings.language``): ``"hi-IN"`` (the default; Hindi and
                Hinglish), ``"en-IN"``, ``"gu-IN"``, ``"mr-IN"``, ``"pa-IN"``,
                ``"bn-IN"``, ``"ta-IN"``, ``"te-IN"``, ``"kn-IN"`` or
                ``"ml-IN"``, in any case, as the bare code (``"gu"``) or as a
                Pipecat ``Language``; or ``"auto"`` when you don't know it.
                Anything else raises ``ValueError``.
            secondary_languages: Shortcut for ``settings.secondary_languages``:
                up to three more of the ten codes that callers may switch to,
                as a list (``["en-IN"]``) or a comma-separated string. Only
                with a fixed ``language``: with ``"auto"``, or with a code
                outside the list or more than three, it raises ``ValueError``.
            model: Shortcut for ``settings.model``. Defaults to ``"mira-stt"``.
            sample_rate: The rate audio is sent at. ``"auto"`` (the default)
                sends 8 kHz audio as it is and resamples anything else to
                16 kHz; ``8000`` or ``16000`` resamples to that rate when the
                pipeline's input rate differs.
            encoding: ``"auto"`` (linear16), ``"linear16"``, or ``"mulaw"`` /
                ``"alaw"`` (8 kHz only; half the bytes).
            endpointing: ``"auto"`` (default), ``"manual"`` or ``"vad"``: see the
                class docstring.
            edge: ``"auto"`` (default): open the socket on ``url``, or on an
                STT edge if Mirai's API ever offers one (it offers none today,
                and nothing is tried or waited for). ``False`` always uses
                ``url``; a ``wss://`` URL uses that edge instead.
                ``MIRAI_STT_EDGE=off`` turns ``"auto"`` off.
            shared_pool: Take a socket the process's pool opened ahead of need
                (the default; the first service starts the pool). ``False``
                always connects and starts no pool.
            settings: Runtime-updatable settings; values here win over the
                ``language``/``secondary_languages``/``model`` shortcuts.
            **kwargs: Passed through to :class:`STTService`, e.g.
                ``audio_passthrough`` or ``ttfs_p99_latency``.
        """
        key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
        if not key:
            raise ValueError("Set MIRAI_API_KEY or pass api_key to MiraiSTTService.")
        url = _check_options(url, sample_rate, encoding)
        if endpointing not in ("auto", "manual", "vad"):
            raise ValueError(f"endpointing must be 'auto', 'manual' or 'vad'; got {endpointing!r}")
        edge_mode(edge, "stt")  # refused here, before the settings, as it always was

        default_settings = self.Settings(
            model=model,
            language=language,
            secondary_languages=secondary_languages,
            stream_type=None,
            mode=None,
            threshold=None,
            silence_duration_ms=None,
            min_speech_duration_ms=None,
            prefix_padding_ms=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)
        default_settings.language = stt_language(default_settings.language)
        default_settings.secondary_languages = stt_secondary_languages(
            default_settings.secondary_languages, default_settings.language
        )
        kwargs.setdefault("ttfs_p99_latency", DEFAULT_TTFS_P99)
        super().__init__(settings=default_settings, **kwargs)

        self._endpointing_option = endpointing
        self.stream = STTStream(
            api_key=key,
            url=url,
            sample_rate=sample_rate,
            encoding=encoding,
            # "auto" is decided when the pipeline starts; until then it reads as "vad".
            endpointing=endpointing if endpointing != "auto" else "vad",
            edge=edge,
            shared_pool=shared_pool,
            listener=_Frames(self),
            name=self.name,
            **self._stream_settings(),
        )

    # ---------- what Pipecat asks ----------

    def can_generate_metrics(self) -> bool:
        """Mirai STT reports TTFB, processing and usage metrics."""
        return True

    def language_to_service_language(self, language: Language) -> str | None:
        """``Language.GU`` / ``Language.GU_IN`` -> ``"gu-IN"``; ``None`` if Mirai doesn't transcribe it."""
        try:
            return stt_language(language)
        except ValueError:
            return None

    @property
    def endpointing(self) -> str:
        """``"manual"`` or ``"vad"``: who ends utterances (decided when the pipeline starts)."""
        return self.stream.endpointing

    @property
    def wire_sample_rate(self) -> int:
        """The rate audio is sent at (decided when the pipeline starts)."""
        return self.stream.wire_sample_rate

    @property
    def last_error(self) -> str | None:
        """The last error reported, if any."""
        return self.stream.last_error

    @property
    def session_id(self) -> str | None:
        """The socket's session id (``sttws_...``), from session.begin."""
        return self.stream.session_id

    @property
    def request_id(self) -> str | None:
        """The socket's request id, from session.begin."""
        return self.stream.request_id

    @property
    def connected_url(self) -> str | None:
        """Where the socket is connected: the edge's URL or ``url`` (no query)."""
        return self.stream.connected_url

    @property
    def last_final_latency_ms(self) -> float | None:
        """speech_end sent (vad.speech_end received) to the latest final, in milliseconds."""
        return self.stream.last_final_latency_ms

    @property
    def audio_seconds_sent(self) -> float:
        """Audio seconds sent over the service's life (replays included: they are billed)."""
        return self.stream.audio_seconds_sent

    @property
    def audio_seconds_billed(self) -> float:
        """What Mirai billed, summed over the sessions that ended with ``session.end``."""
        return self.stream.audio_seconds_billed

    @property
    def cost_paise(self) -> float:
        """What Mirai charged, summed over the sessions that ended with ``session.end``."""
        return self.stream.cost_paise

    def service_metadata_frame(self) -> STTMetadataFrame:
        """Under server endpointing, ask the user aggregator to follow Mirai's turn boundaries."""
        frame = super().service_metadata_frame()
        if self.stream.endpointing == "vad":
            frame.user_turn_strategies = ExternalUserTurnStrategies()
        return frame

    # ---------- lifecycle ----------

    async def start(self, frame: StartFrame):
        """Start the service; the socket opens in the background."""
        await super().start(frame)
        endpointing = None
        if self._endpointing_option == "auto":
            endpointing = "manual" if self._pipeline_has_vad() else "vad"
        await self.stream.connect(input_sample_rate=self.sample_rate, endpointing=endpointing)

    async def stop(self, frame: EndFrame):
        """Finish: close the open utterance, wait briefly for the last finals, then close."""
        await self.stream.close(graceful=True)
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Close at once."""
        await self.stream.close(graceful=False)
        await super().cancel(frame)

    async def cleanup(self):
        """Release everything at pipeline teardown."""
        try:
            await super().cleanup()
        finally:
            await self.stream.close(graceful=False)

    # ---------- audio in ----------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Queue the caller's audio; transcripts arrive on the receive task."""
        self.stream.send(audio)
        yield None

    def _record_stt_audio_usage(self, audio: bytes | bytearray):
        # Usage is the audio actually sent (after resampling, without what is
        # held between turns), counted by the stream as it goes out.
        pass

    async def emit_stt_usage_metrics(self):
        """Report the audio seconds sent since the last report."""
        seconds = self.stream.take_usage()
        if seconds > 0:
            await self.start_stt_usage_metrics(STTUsage(audio_seconds=seconds))

    # ---------- manual endpointing ----------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Turn the pipeline's VAD frames into speech_start / speech_end under manual endpointing."""
        await super().process_frame(frame, direction)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            await self.stream.speech_started()  # nothing unless manual endpointing has started
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            await self.stream.speech_stopped()

    async def _handle_vad_user_stopped_speaking(self, frame: VADUserStoppedSpeakingFrame):
        # TTFB is measured from speech_end sent (or vad.speech_end) to the
        # final, not from the VAD's estimate of when speech ended.
        self._user_speaking = False

    @traced_stt
    async def _trace_transcription(self, transcript: str, is_final: bool, language: Language | None = None):
        """Record a transcript for tracing."""

    # ---------- settings ----------

    def _stream_settings(self) -> dict[str, Any]:
        """``_settings`` as the stream takes them (unset: ``None``, Mirai's default)."""
        names = ("model", "language", "secondary_languages", *_TUNING)
        values = {name: getattr(self._settings, name) for name in names}
        return {name: value if is_given(value) else None for name, value in values.items()}

    async def _update_settings(self, delta: STTSettings) -> dict[str, Any]:
        """Language and tuning change on the open socket; model and prefix padding at the next pause.

        A language Mirai doesn't transcribe is not sent: the socket keeps its
        language and a warning says why. The same goes for secondary languages
        that aren't valid, or that come with ``"auto"``; a change to ``"auto"``
        clears the secondary languages.
        """
        if is_given(delta.language):
            try:
                delta.language = stt_language(delta.language)
            except ValueError as exc:
                logger.warning(f"{self}: {exc}; keeping {self._settings.language}")
                delta.language = NOT_GIVEN
        language = delta.language if is_given(delta.language) else self._settings.language
        secondary = getattr(delta, "secondary_languages", NOT_GIVEN)
        if is_given(secondary):
            try:
                delta.secondary_languages = stt_secondary_languages(secondary, language)
            except ValueError as exc:
                logger.warning(f"{self}: {exc}; keeping {list(self._settings.secondary_languages or ())}")
                delta.secondary_languages = NOT_GIVEN
        if language == AUTO_LANGUAGE and not is_given(getattr(delta, "secondary_languages", NOT_GIVEN)):
            self._settings.secondary_languages = ()  # auto already considers every language
        changed = await super()._update_settings(delta)
        if changed:
            # The stream sends what changed on the open socket (or keeps it for start).
            await self.stream.update(**self._stream_settings())
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


class _Frames(STTListener):
    """What the stream tells the service, as Pipecat frames, metrics, errors and event handlers."""

    def __init__(self, service: MiraiSTTService):
        self._service = service

    async def on_connected(self) -> None:
        await self._service._call_event_handler("on_connected")

    async def on_disconnected(self) -> None:
        await self._service._call_event_handler("on_disconnected")

    async def on_connection_error(self, message: str) -> None:
        await self._service._call_event_handler("on_connection_error", message)

    async def on_error(self, error: FrequencyError) -> None:
        # The category alone says whether Pipecat may keep using the service,
        # as before the engine moved: no force_treat_as_permanent.
        cause = error.__cause__
        await self._service.push_error(
            error_msg=str(error),
            exception=cause if isinstance(cause, Exception) else None,
            category=ErrorCategory(error.category),
        )

    async def on_speech_started(self) -> None:
        await self._service.broadcast_frame(ProposedUserStartedSpeakingFrame)

    async def on_speech_stopped(self) -> None:
        await self._service.broadcast_frame(ProposedUserStoppedSpeakingFrame)

    async def on_partial(self, text: str, language: str | None, event: dict) -> None:
        service = self._service
        await service.push_frame(
            InterimTranscriptionFrame(
                text, service._user_id, time_now_iso8601(), _frame_language(language), result=event
            )
        )

    async def on_final_latency(self, started: float, received: float) -> None:
        service = self._service
        await service.start_ttfb_metrics(start_time=started)
        await service.stop_ttfb_metrics(end_time=received)
        await service.start_processing_metrics(start_time=started)
        await service.stop_processing_metrics(end_time=received)

    async def on_usage(self, audio_seconds: float) -> None:
        # Before the frame, so tracing can attach it to the span the frame closes.
        await self._service.start_stt_usage_metrics(STTUsage(audio_seconds=audio_seconds))

    async def on_final(self, text: str, language: str | None, event: dict) -> None:
        service = self._service
        heard = _frame_language(language)
        # One frame per utterance even when empty: a finalized transcript tells
        # the turn strategies nothing more is coming (empty text adds nothing).
        await service.push_frame(
            TranscriptionFrame(
                text, service._user_id, time_now_iso8601(), heard, result=event, finalized=True
            )
        )
        if text:
            await service._trace_transcription(text, True, heard)


def _frame_language(code: str | None) -> Language | None:
    """A transcript's language as Pipecat's ``Language``; ``None`` if there is none or Pipecat has no such."""
    if not code:
        return None
    try:
        return Language(code)
    except ValueError:
        return None
