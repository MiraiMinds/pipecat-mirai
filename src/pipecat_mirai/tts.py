#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech service for Pipecat.

Mirai streams mono 16-bit PCM at 48 kHz from an OpenAI-compatible
``POST /v1/audio/speech`` endpoint. This service streams that audio into a
Pipecat pipeline and converts it to the pipeline's output sample rate (8 kHz for
phone calls, 16/24/48 kHz elsewhere) with a per-utterance streaming resampler.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx
import numpy as np
import soxr
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import TTSService
from pipecat.utils.tracing.service_decorators import traced_tts

DEFAULT_BASE_URL = "https://sandbox.voice.miraiminds.co/v1"
SOURCE_SAMPLE_RATE = 48000
VOICES = ("ashu", "neha", "shruti", "sameer")


@dataclass
class MiraiTTSSettings(TTSSettings):
    """Runtime-updatable settings for :class:`MiraiTTSService`.

    Parameters:
        model: Mirai model name. Defaults to ``"mira-tts"``.
        voice: One of ``ashu``, ``neha``, ``shruti`` or ``sameer``.
        language: Unused; the model reads the script of the input text
            (Devanagari Hindi, Hinglish, Gujarati).
    """


class MiraiTTSService(TTSService):
    """Stream Mirai TTS audio into a Pipecat pipeline.

    Each utterance is one streaming HTTP request. Audio frames are pushed as they
    arrive, so the first audio typically leaves the service ~100 ms after the
    request. Interrupting the bot closes the request immediately.

    Example::

        tts = MiraiTTSService(
            api_key=os.getenv("MIRAI_API_KEY"),
            settings=MiraiTTSService.Settings(voice="shruti"),
        )

    For phone calls over a websocket transport (Twilio, Plivo, Exotel, Telnyx,
    ...), also call :func:`pipecat_mirai.apply_output_lead` on the transport;
    see the README for why.
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
        http_client: httpx.AsyncClient | None = None,
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
                ``audio_out_sample_rate``. Mirai audio is resampled to it.
            http_client: Optional caller-owned ``httpx.AsyncClient``.
            settings: Runtime-updatable settings; values here win over the
                ``voice``/``model`` shortcuts.
            **kwargs: Passed through to :class:`TTSService`.
        """
        key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
        if not key:
            raise ValueError("Set MIRAI_API_KEY or pass api_key to MiraiTTSService.")

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
        self._speech_url = base_url.rstrip("/") + "/audio/speech"
        self._headers = {"Authorization": f"Bearer {key}"}
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=120),
        )
        self.last_error: str | None = None

    def can_generate_metrics(self) -> bool:
        """Mirai TTS reports TTFB and usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service; the output sample rate is known from here on."""
        await super().start(frame)
        if self._settings.voice not in VOICES:
            logger.warning(f"{self}: voice {self._settings.voice!r} is not one of {', '.join(VOICES)}")
        logger.debug(f"{self}: Mirai 48 kHz PCM -> {self.sample_rate} Hz")

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` and yield audio frames as they stream in."""
        if not text.strip():
            return
        self.last_error = None
        resampler = (
            soxr.ResampleStream(SOURCE_SAMPLE_RATE, self.sample_rate, 1, dtype="int16", quality="HQ")
            if self.sample_rate != SOURCE_SAMPLE_RATE
            else None
        )
        carry = b""
        received = False
        try:
            await self.start_ttfb_metrics()
            async with self._http.stream(
                "POST",
                self._speech_url,
                headers=self._headers,
                json={
                    "model": self._settings.model,
                    "voice": self._settings.voice,
                    "input": text,
                    "response_format": "pcm",
                },
            ) as response:
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()
                media_type = response.headers.get("content-type", "").split(";")[0].strip()
                if media_type != "audio/pcm":
                    raise ValueError(f"expected audio/pcm, received {media_type or 'no content type'}")
                rate = response.headers.get("x-sample-rate", str(SOURCE_SAMPLE_RATE))
                if rate != str(SOURCE_SAMPLE_RATE):
                    raise ValueError(f"unexpected PCM sample rate {rate}")
                await self.start_tts_usage_metrics(text)
                async for chunk in response.aiter_bytes():
                    # Network reads can end mid-sample; carry the odd byte forward.
                    data = carry + chunk
                    end = len(data) - len(data) % 2
                    carry, data = data[end:], data[:end]
                    if not data:
                        continue
                    if not received:
                        received = True
                        await self.stop_ttfb_metrics()
                    if resampler:
                        data = resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
                    if data:
                        yield TTSAudioRawFrame(data, self.sample_rate, 1, context_id=context_id)
                if carry:
                    raise ValueError("audio ended with an incomplete PCM sample")
                if not received:
                    raise ValueError("no audio returned")
                if resampler:
                    tail = resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True).tobytes()
                    if tail:
                        yield TTSAudioRawFrame(tail, self.sample_rate, 1, context_id=context_id)
        except httpx.HTTPStatusError as exc:
            message = f"Mirai TTS returned HTTP {exc.response.status_code}"
            try:
                detail = exc.response.json().get("error", {}).get("message")
                if isinstance(detail, str):
                    message += f": {detail[:500]}"
            except (ValueError, httpx.HTTPError, AttributeError):
                pass
            self.last_error = message
            yield ErrorFrame(error=message, exception=exc)
        except (httpx.HTTPError, ValueError) as exc:
            self.last_error = f"Mirai TTS: {exc}"
            yield ErrorFrame(error=self.last_error, exception=exc)
        finally:
            # An interruption closes this generator: the HTTP stream is closed
            # with it and nothing already-interrupted is retried or flushed.
            await self.stop_ttfb_metrics()

    async def stop(self, frame: EndFrame):
        """Stop the service and close the HTTP client it owns."""
        try:
            await super().stop(frame)
        finally:
            await self._close_http()

    async def cancel(self, frame: CancelFrame):
        """Cancel the service and close the HTTP client it owns."""
        try:
            await super().cancel(frame)
        finally:
            await self._close_http()

    async def cleanup(self):
        """Release resources at pipeline teardown."""
        try:
            await super().cleanup()
        finally:
            await self._close_http()

    async def _close_http(self):
        if self._owns_http and not self._http.is_closed:
            await self._http.aclose()
