#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech over one WebSocket per pipeline.

``MiraiTTSService`` keeps a single socket to Mirai's
``/v1/audio/speech/stream`` open for the whole call. Text is sent as Pipecat
produces it, Mirai cuts it into sentences, synthesises the next sentence while
the current one streams, and sends the audio back on the same socket. Nothing
pays a TCP and TLS handshake after the first one.

The engine (the socket, the edge, the pool, reconnects, resends, the HTTP
fallback) is :class:`fourier_frequency.TTSStream`, which any Python voice
agent can use; this module maps Pipecat's audio contexts, frames, metrics and
settings onto it.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Literal

import httpx
from fourier_frequency.errors import FrequencyError
from fourier_frequency.tts import (
    CONNECT_RETRY_SECS,
    CONNECT_RETRY_WAIT,
    DEFAULT_WEBSOCKET_URL,
    MAX_MESSAGE_BYTES,
    RETRY_AFTER_LIMIT_SECS,
    TTSListener,
    TTSStream,
    api_key_from_env,
)
from fourier_frequency.tts import check_server_rate as _check_server_rate
from fourier_frequency.tts import http_base as _http_base
from fourier_frequency.tts import stream_url as _stream_url
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
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
from pipecat.services.tts_service import TTSService
from pipecat.services.websocket_service import WebsocketService
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.tracing.service_decorators import traced_tts

from pipecat_mirai.pacing import DEFAULT_LEAD_SECS, ensure_output_lead
from pipecat_mirai.tts import MiraiTTSSettings

__all__ = [
    "CONNECT_RETRY_SECS",
    "CONNECT_RETRY_WAIT",
    "DEFAULT_WEBSOCKET_URL",
    "MAX_MESSAGE_BYTES",
    "RETRY_AFTER_LIMIT_SECS",
    "MiraiTTSService",
    "MiraiWebsocketTTSService",
    "_http_base",
    "_stream_url",
]

_UNSET: Any = object()


def error_category(name: str) -> ErrorCategory:
    """The Pipecat category of a fourier-frequency error (they share the values)."""
    try:
        return ErrorCategory(name)
    except ValueError:
        return ErrorCategory.SERVER


class _Contexts(TTSListener):
    """The stream's turns are the service's Pipecat audio contexts."""

    def __init__(self, service: MiraiTTSService):
        self._service = service

    async def on_audio(self, turn_id: str, audio: bytes) -> None:
        service = self._service
        await service.append_to_audio_context(
            turn_id, TTSAudioRawFrame(audio, service.sample_rate, 1, context_id=turn_id)
        )

    async def on_turn_done(self, turn_id: str) -> None:
        await self._service.remove_audio_context(turn_id)

    def turn_active(self, turn_id: str) -> bool:
        return self._service.audio_context_available(turn_id)

    def keep_turn_open(self, turn_id: str) -> None:
        # Pipecat ends an audio context that gets no frame for its stop
        # timeout (3 s); a retry may wait about that long.
        service = self._service
        refresh = getattr(service, "_refresh_audio_context", None)
        if refresh is not None and service.audio_context_available(turn_id):
            refresh(turn_id)

    async def on_synthesis_started(self, turn_id: str) -> None:
        await self._service.start_ttfb_metrics()

    async def on_first_audio(self, turn_id: str) -> None:
        await self._service.stop_ttfb_metrics()

    async def on_synthesis_ended(self, turn_id: str) -> None:
        await self._service.stop_ttfb_metrics()

    async def on_no_audio(self, turn_id: str) -> None:
        await self._service.cancel_ttfb_metrics()

    async def on_usage(self, chars: int) -> None:
        service = self._service
        if not (service.can_generate_metrics() and service.usage_metrics_enabled):
            return
        logger.debug(f"{service.name} usage characters: {chars}")
        data = TTSUsageMetricsData(processor=service.name, model=service._settings.model, value=chars)
        await service.push_frame(MetricsFrame(data=[data]))

    async def on_text_billed(self, text: str) -> None:
        # Over the HTTP fallback, usage is the text, as Pipecat counts it.
        await TTSService.start_tts_usage_metrics(self._service, text)

    async def on_error(self, error: FrequencyError, turn_id: str | None = None) -> None:
        await self._service.push_error(
            error_msg=str(error),
            exception=error.__cause__,
            category=error_category(error.category),
            force_treat_as_permanent=error.permanent,
        )

    async def on_connected(self) -> None:
        await self._service._call_event_handler("on_connected")

    async def on_disconnected(self) -> None:
        await self._service._call_event_handler("on_disconnected")

    async def on_connection_error(self, message: str) -> None:
        await self._service._call_event_handler("on_connection_error", message)


class MiraiTTSService(TTSService):
    """Mirai text to speech for Pipecat: one WebSocket for the whole pipeline.

    This is the one service to use. It picks the fastest way to Mirai by
    itself: Mirai's edge next to the speech GPUs, then the API's own streaming
    endpoint, then (if WebSockets can't get through at all) HTTP. Code written
    for the HTTP ``MiraiTTSService`` of 0.4 and earlier keeps working:
    ``base_url``, ``http_client``, ``warm_connection`` and ``keep_warm_secs``
    are still accepted. ``MiraiWebsocketTTSService`` is the same class.

    The socket opens when the pipeline starts and carries every utterance.
    Sockets are opened ahead of need by a pool shared by every service in the
    process (see :mod:`fourier_frequency.pool`), so a pipeline usually takes one
    that is already open and authenticated and starts without a handshake.
    Each socket opens on Mirai's edge, next to the speech GPUs, when Mirai
    offers one (about half the time to first byte), and on ``url`` otherwise
    (see :mod:`fourier_frequency.edge`).
    Text goes to Mirai as Pipecat produces it: each sentence as soon as it is
    aggregated (the default), or each LLM token with
    ``text_aggregation_mode=TextAggregationMode.TOKEN``. Mirai cuts sentences
    itself and starts synthesising the next one while the current one plays,
    so there is no gap and no handshake between sentences.

    Audio is asked for at the pipeline's output rate and pushed in 40 ms
    frames after a short first-audio buffer.
    An interruption cancels the reply on the server at once and drops
    whatever of it is still arriving. A dropped connection is reopened, and
    a reply it cut short is resent once. If the streaming endpoint can't be
    reached at all (refused, not found, a proxy that doesn't pass WebSockets),
    the call speaks over Mirai's HTTP endpoint instead (``http_fallback``).
    On a phone call over a websocket transport the service also lets the
    transport send audio a little ahead of real time (``output_lead_secs``).

    The engine is :class:`fourier_frequency.TTSStream` (``self.stream``).

    Example::

        tts = MiraiTTSService(
            api_key=os.getenv("MIRAI_API_KEY"),
            settings=MiraiTTSService.Settings(voice="shruti"),
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
        url: str | None = None,
        base_url: str | None = None,
        voice: str | None = None,
        model: str | None = None,
        sample_rate: int | None = None,
        server_sample_rate: int | Literal["auto"] | None = "auto",
        prebuffer_secs: float = 0.15,
        keepalive_secs: float | None = 30.0,
        shared_pool: bool = True,
        output_lead_secs: float | None = DEFAULT_LEAD_SECS,
        http_fallback: bool = True,
        edge: bool | str = "auto",
        settings: Settings | None = None,
        http_client: httpx.AsyncClient | None = None,
        warm_connection: bool = True,
        keep_warm_secs: float | None | object = _UNSET,
        reconnect_on_error: bool = True,
        **kwargs,
    ):
        """Initialize the Mirai TTS service.

        Args:
            api_key: Mirai API key. Defaults to the ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``) environment variable. Sent as an
                ``Authorization: Bearer`` header.
            url: The streaming endpoint, ``wss://<host>/v1/audio/speech/stream``.
                Defaults to Mirai's (or to ``base_url``'s, when that is given).
            base_url: The API base URL including ``/v1``, as the HTTP service
                took it (``https://<host>/v1``); the streaming endpoint is
                derived from it. ``url`` wins if both are given.
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
            shared_pool: Take a socket the process's pool opened ahead of
                need for this URL and API key when one is waiting (the
                default), and connect only when none is; the first service to
                start sets the pool going (see :mod:`fourier_frequency.pool`). The
                socket is this pipeline's alone and is closed when it ends.
                ``False`` always connects and starts no pool.
            output_lead_secs: When the pipeline starts, let the first output
                transport after this service run up to this far ahead of real
                time, as :func:`pipecat_mirai.apply_output_lead` does (phone
                calls over Pipecat's websocket transports). ``None`` turns it
                off.
            http_fallback: If the streaming endpoint can't be reached (a
                handshake answered 404, 426 or 5xx, refused, or timed out),
                speak this call over Mirai's HTTP endpoint (``POST
                /v1/audio/speech`` on the same host) instead of failing, and
                log it once. Authentication, credit and rate-limit refusals
                are reported, not worked around. ``False`` reports them all.
            edge: ``"auto"`` (the default): open each socket on Mirai's edge,
                next to the speech GPUs, with the API key (in the header,
                never in a URL). If the edge can't be reached or isn't ready
                within 3 s, the socket opens on ``url`` as before and the edge
                is left alone for a while (60 s, doubling). ``False`` always
                uses ``url``. A ``wss://`` URL uses that edge instead.
                ``MIRAI_TTS_EDGE=off`` turns ``"auto"`` off.
            settings: Runtime-updatable settings; values here win over the
                ``voice``/``model`` shortcuts.
            http_client: An ``httpx.AsyncClient`` for the HTTP fallback, if
                you manage your own; by default the process's shared one.
            warm_connection: Accepted for code written for the HTTP service;
                the socket always opens while the pipeline starts.
            keep_warm_secs: The HTTP service's name for ``keepalive_secs``.
            reconnect_on_error: Reopen a socket that drops (three tries,
                4 to 10 s apart after a failed one). ``False`` reports the
                drop and reconnects only for the next utterance.
            **kwargs: Passed through to :class:`TTSService`, e.g.
                ``text_aggregation_mode``.
        """
        key = api_key_from_env(api_key, "MiraiTTSService")
        _check_server_rate(server_sample_rate)
        if output_lead_secs is not None and not output_lead_secs >= 0:
            raise ValueError(f"output_lead_secs must be >= 0 or None; got {output_lead_secs!r}")
        if not prebuffer_secs >= 0:
            raise ValueError(f"prebuffer_secs must be >= 0; got {prebuffer_secs!r}")
        if keep_warm_secs is not _UNSET:
            keepalive_secs = keep_warm_secs  # type: ignore[assignment]
        if url is None:
            url = _stream_url(base_url) if base_url else DEFAULT_WEBSOCKET_URL
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
        self._output_lead_secs = output_lead_secs
        self.stream = TTSStream(
            api_key=key,
            url=url,
            voice=default_settings.voice,
            model=default_settings.model,
            server_sample_rate=server_sample_rate,
            prebuffer_secs=prebuffer_secs,
            keepalive_secs=keepalive_secs,
            shared_pool=shared_pool,
            http_fallback=http_fallback,
            edge=edge,
            reconnect=reconnect_on_error,
            http_client=http_client,
            listener=_Contexts(self),
            name=self.name,
        )

    # The stream's state, where 0.7 and earlier kept it.
    @property
    def last_error(self) -> str | None:
        return self.stream.last_error

    @property
    def last_server_sample_rate(self) -> int | None:
        return self.stream.last_server_sample_rate

    @property
    def session_id(self) -> str | None:
        return self.stream.session_id

    @property
    def connected_url(self) -> str | None:
        return self.stream.connected_url

    @property
    def _url(self) -> str:
        return self.stream.url

    @property
    def _keepalive_secs(self) -> float | None:
        return self.stream.keepalive_secs

    @property
    def _fallback(self) -> str | None:
        return self.stream.fallback

    def _client(self) -> httpx.AsyncClient:
        return self.stream.http.client()

    def can_generate_metrics(self) -> bool:
        """Mirai TTS reports TTFB and usage metrics."""
        return True

    @property
    def supports_processing_metrics(self) -> bool:
        """No: ``run_tts`` sends the text and returns; audio arrives on the stream's own task."""
        return False

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
        stream = self.stream
        stream.sample_rate = self.sample_rate
        stream.voice, stream.model = self._settings.voice, self._settings.model
        stream.tokens = self._is_streaming_tokens
        ensure_output_lead(self, self._output_lead_secs)
        await stream.connect()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the socket."""
        await super().stop(frame)
        await self.stream.close()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the socket at once."""
        await super().cancel(frame)
        await self.stream.close()

    async def cleanup(self):
        """Release resources at pipeline teardown."""
        try:
            await super().cleanup()
        finally:
            await self.stream.close()

    # ---------- sending text ----------

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Send ``text`` for ``context_id``; its audio arrives on the stream's receive task."""
        self.stream.tokens = self._is_streaming_tokens
        try:
            await self.stream.send_text(context_id, text)
        except FrequencyError as exc:
            yield ErrorFrame(error=str(exc))
            return
        yield None

    async def flush_audio(self, context_id: str | None = None):
        """Pipecat has sent the whole turn: ask Mirai to speak what's left."""
        context_id = context_id or self.get_active_audio_context_id()
        if not context_id:
            return
        self.stream.tokens = self._is_streaming_tokens
        await self.stream.end_turn(context_id)

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:
        """Apply a settings delta; a new voice or model applies from the next sentence."""
        changed = await super()._update_settings(delta)
        if changed.keys() & {"voice", "model"}:
            await self.stream.update(voice=self._settings.voice, model=self._settings.model)
        return changed

    # ---------- interruptions ----------

    async def _handle_interruption(self, frame: InterruptionFrame, direction: FrameDirection):
        cut_short = self.stream.begin_interrupt()  # what the HTTP fallback is speaking stops here
        await super()._handle_interruption(frame, direction)
        await self.stream.finish_interrupt(cut_short)

    async def on_audio_context_interrupted(self, context_id: str):
        """Cancel the interrupted context on the server."""
        await self.stream.cancel_turn(context_id)
        await super().on_audio_context_interrupted(context_id)


# No longer a WebsocketTTSService (the socket is the stream's), but still a
# websocket-streaming service to code that asks, such as Pipecat's evals: its
# run_tts sends text and yields no audio.
WebsocketService.register(MiraiTTSService)

# The name this service had in 0.3 and 0.4; the same class.
MiraiWebsocketTTSService = MiraiTTSService
