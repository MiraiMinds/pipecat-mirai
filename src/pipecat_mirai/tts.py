#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech service for Pipecat, over HTTP.

Mirai streams mono 16-bit PCM from an OpenAI-compatible ``POST /v1/audio/speech``
endpoint. This service asks for the pipeline's output rate (8, 16, 24 or 48 kHz)
so the audio needs no conversion, and reads the rate the server actually sent
from the ``X-Sample-Rate`` response header. If that differs (a server that only
speaks 48 kHz, or a pipeline rate Mirai doesn't serve), the audio is converted
with a per-utterance streaming resampler.

The engine is :class:`fourier_frequency.HttpTTS`; this module turns it into
Pipecat frames, metrics and settings.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

import httpx
from fourier_frequency.errors import FrequencyError
from fourier_frequency.tts import (
    DEFAULT_BASE_URL,
    FAST_RETRY_DELAY_SECS,
    FRAME_SECS,
    RETRY_BUDGET_SECS,
    SERVER_SAMPLE_RATES,
    SOURCE_SAMPLE_RATE,
    VOICES,
    HttpTTS,
    TTSListener,
    api_key_from_env,
    is_custom_voice,
)
from fourier_frequency.tts import check_server_rate as _check_server_rate
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterruptionFrame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

from pipecat_mirai.pacing import DEFAULT_LEAD_SECS, ensure_output_lead

__all__ = [
    "DEFAULT_BASE_URL",
    "FAST_RETRY_DELAY_SECS",
    "FRAME_SECS",
    "RETRY_BUDGET_SECS",
    "SERVER_SAMPLE_RATES",
    "SOURCE_SAMPLE_RATE",
    "VOICES",
    "MiraiHttpTTSService",
    "MiraiTTSError",
    "MiraiTTSSettings",
]

# Raised (and carried in ErrorFrame.exception) when a sentence fails; 0.7 and earlier had their own class.
MiraiTTSError = FrequencyError


@dataclass
class MiraiTTSSettings(TTSSettings):
    """Runtime-updatable settings for :class:`MiraiTTSService`.

    Parameters:
        model: Mirai model name. Defaults to ``"mira-tts"``.
        voice: One of ``ashu``, ``neha``, ``shruti`` or ``sameer``.
        language: Unused; the model reads the script of the input text
            (Devanagari Hindi, Hinglish, Gujarati).
    """


class _Metrics(TTSListener):
    """The HTTP engine's events, as the service's Pipecat metrics."""

    def __init__(self, service: MiraiHttpTTSService):
        self._service = service

    async def on_synthesis_started(self, turn_id: str) -> None:
        await self._service.start_ttfb_metrics()

    async def on_first_audio(self, turn_id: str) -> None:
        await self._service.stop_ttfb_metrics()

    async def on_synthesis_ended(self, turn_id: str) -> None:
        await self._service.stop_ttfb_metrics()

    async def on_text_billed(self, text: str) -> None:
        await self._service.start_tts_usage_metrics(text)


class MiraiHttpTTSService(TTSService):
    """Mirai TTS over HTTP: one streaming request per sentence.

    Most bots should use :class:`~pipecat_mirai.MiraiTTSService`, which streams
    over a WebSocket from Mirai's edge and falls back to HTTP by itself. This
    service is for when you want HTTP only (it was ``MiraiTTSService`` before
    0.5.0).

    Each sentence is one streaming HTTP request over a kept-alive connection.
    The connections are shared by every ``MiraiHttpTTSService`` for the same
    ``base_url`` in the process and kept warm on their own (see
    :mod:`fourier_frequency.pool`), so calls that start together, or a call after
    a quiet spell, don't wait for a TCP and TLS handshake. Audio is pushed in
    40 ms frames as it arrives, after a short first-audio buffer
    (``prebuffer_secs``) so playback doesn't start and then stall.
    Interrupting the bot closes the request immediately and drops audio not
    yet pushed.

    Mirai is asked for audio at the pipeline's output rate: an 8 kHz phone
    pipeline downloads 128 kbit/s per call instead of 48 kHz's 768 kbit/s.
    On a phone call over a websocket transport (Twilio, Plivo, Exotel,
    Telnyx, ...) the service also lets the transport send audio a little ahead
    of real time (``output_lead_secs``), so a busy server doesn't break up the
    caller's audio.

    Example::

        tts = MiraiHttpTTSService(
            api_key=os.getenv("MIRAI_API_KEY"),
            settings=MiraiHttpTTSService.Settings(voice="shruti"),
        )
    """

    Settings = MiraiTTSSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        voice: str | None = None,
        model: str | None = None,
        sample_rate: int | None = None,
        server_sample_rate: int | Literal["auto"] | None = "auto",
        prebuffer_secs: float = 0.15,
        warm_connection: bool = True,
        keep_warm_secs: float | None = 30.0,
        http_client: httpx.AsyncClient | None = None,
        shared_pool: bool = True,
        output_lead_secs: float | None = DEFAULT_LEAD_SECS,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the Mirai TTS service.

        Args:
            api_key: Mirai API key. Defaults to the ``MIRAI_API_KEY`` (or
                ``MIRA_API_KEY``) environment variable.
            base_url: API base URL including ``/v1``.
            voice: Shortcut for ``settings.voice``. Defaults to ``"neha"``.
            model: Shortcut for ``settings.model``. Defaults to ``"mira-tts"``.
            sample_rate: Output sample rate. Defaults to the pipeline's
                ``audio_out_sample_rate``.
            server_sample_rate: The PCM rate to ask Mirai for. ``"auto"`` (the
                default) asks for the output rate when Mirai serves it (8000,
                16000, 22050, 24000, 44100 or 48000 Hz) and leaves the field out otherwise.
                An ``int`` from that list asks for that rate. ``None`` leaves
                the field out, so Mirai sends 48 kHz as it did before 0.3.
                Audio is resampled locally whenever the rate Mirai reports
                (``X-Sample-Rate``) differs from the output rate.
            prebuffer_secs: Audio to collect before the first frame of each
                utterance is pushed. Mirai's first chunk can be short and
                followed by a brief gap; starting playback on it alone would
                stall. ``0`` pushes audio as soon as a 40 ms frame is in hand.
                TTFB metrics still measure the first byte received.
            warm_connection: Keep connections to Mirai open ahead of need.
                With the shared pool (the default), the first service to start
                in a process starts the pool's warm-up (see
                :mod:`fourier_frequency.pool`). With its own client, the service
                opens its connection (an authenticated ``GET /models``) when
                the pipeline starts and again after an interruption drops it.
                Failures are logged and ignored. ``False``: this service asks
                for no background requests.
            keep_warm_secs: With its own client (``shared_pool=False`` or
                ``http_client``): while the pipeline runs, repeat that request
                when the connection has been idle this long (Mirai closes idle
                connections after 75 s). ``None`` turns it off. The shared pool
                refreshes each of its connections on its own.
            http_client: Optional caller-owned ``httpx.AsyncClient``. When
                given, ``shared_pool`` is ignored and the client is not closed.
            shared_pool: Use the event loop's shared, self-warming client for
                ``base_url`` (the default). ``False`` gives the service a
                client of its own that is closed when the pipeline ends, as in
                0.3.0.
            output_lead_secs: When the pipeline starts, let the first output
                transport after this service run up to this far ahead of real
                time, as :func:`pipecat_mirai.apply_output_lead` does. Only
                Pipecat's websocket transports (phone calls) pace audio the
                way this helps; others are left alone, and so is a transport
                that already has a lead. ``None`` turns it off.
            settings: Runtime-updatable settings; values here win over the
                ``voice``/``model`` shortcuts.
            **kwargs: Passed through to :class:`TTSService`.
        """
        key = api_key_from_env(api_key, "MiraiHttpTTSService")
        _check_server_rate(server_sample_rate)
        if not prebuffer_secs >= 0:
            raise ValueError(f"prebuffer_secs must be >= 0; got {prebuffer_secs!r}")
        if keep_warm_secs is not None and not keep_warm_secs > 0:
            raise ValueError(f"keep_warm_secs must be > 0 or None; got {keep_warm_secs!r}")
        if output_lead_secs is not None and not output_lead_secs >= 0:
            raise ValueError(f"output_lead_secs must be >= 0 or None; got {output_lead_secs!r}")

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
        self._engine = HttpTTS(
            api_key=key,
            base_url=base_url,
            voice=default_settings.voice,
            model=default_settings.model,
            server_sample_rate=server_sample_rate,
            prebuffer_secs=prebuffer_secs,
            warm_connection=warm_connection,
            keep_warm_secs=keep_warm_secs,
            http_client=http_client,
            shared_pool=shared_pool,
            listener=_Metrics(self),
            name=self.name,
        )

    # What 0.7 and earlier kept on the service.
    @property
    def last_error(self) -> str | None:
        return self._engine.last_error

    @property
    def last_server_sample_rate(self) -> int | None:
        return self._engine.last_server_sample_rate

    @property
    def last_request_id(self) -> str | None:
        return self._engine.last_request_id

    @property
    def _http(self) -> httpx.AsyncClient | None:
        return self._engine._http

    def can_generate_metrics(self) -> bool:
        """Mirai TTS reports TTFB and usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service; the output sample rate is known from here on."""
        await super().start(frame)
        self._engine.sample_rate = self.sample_rate
        self._sync_settings()
        if self._settings.voice not in VOICES and not is_custom_voice(self._settings.voice):
            logger.warning(f"{self}: voice {self._settings.voice!r} is not one of {', '.join(VOICES)}")
        requested = self._engine.requested_rate()
        asked = f"{requested} Hz" if requested else f"its default {SOURCE_SAMPLE_RATE} Hz"
        logger.debug(f"{self}: asking Mirai for {asked} PCM; output {self.sample_rate} Hz")
        ensure_output_lead(self, self._output_lead_secs)
        self._engine.start()

    def _sync_settings(self):
        self._engine.voice = self._settings.voice
        self._engine.model = self._settings.model

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame; on an interruption, close the streams it cut short."""
        if not isinstance(frame, InterruptionFrame):
            await super().process_frame(frame, direction)
            return
        # Pipecat cancels the running utterance here, and that normally closes
        # its HTTP stream too; the engine closes what is left once it has.
        cut_short = self._engine.begin_interrupt()
        await super().process_frame(frame, direction)
        await self._engine.finish_interrupt(cut_short)

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` and yield audio frames as they stream in."""
        self._sync_settings()
        try:
            async with contextlib.aclosing(self._engine.speak(text, context_id)) as audio:
                async for piece in audio:
                    yield TTSAudioRawFrame(piece, self.sample_rate, 1, context_id=context_id)
        except FrequencyError as exc:
            yield ErrorFrame(error=str(exc), exception=exc)

    async def _update_settings(self, delta: TTSSettings):
        changed = await super()._update_settings(delta)
        self._sync_settings()
        return changed

    # ---- lifecycle ----

    async def stop(self, frame: EndFrame):
        """Stop the service and close (or let go of) its HTTP client."""
        try:
            await super().stop(frame)
        finally:
            await self._engine.aclose()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close (or let go of) its HTTP client."""
        try:
            await super().cancel(frame)
        finally:
            await self._engine.aclose()

    async def cleanup(self):
        """Release resources at pipeline teardown."""
        try:
            await super().cleanup()
        finally:
            await self._engine.aclose()
