#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech service for Pipecat.

Mirai streams mono 16-bit PCM from an OpenAI-compatible ``POST /v1/audio/speech``
endpoint. This service asks for the pipeline's output rate (8, 16, 24 or 48 kHz)
so the audio needs no conversion, and reads the rate the server actually sent
from the ``X-Sample-Rate`` response header. If that differs (a server that only
speaks 48 kHz, or a pipeline rate Mirai doesn't serve), the audio is converted
with a per-utterance streaming resampler.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Literal

import httpx
import numpy as np
import soxr
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

DEFAULT_BASE_URL = "https://sandbox.voice.miraiminds.co/v1"
# The rate Mirai sends when the request doesn't name one, and the rate assumed
# when a response carries no X-Sample-Rate header.
SOURCE_SAMPLE_RATE = 48000
# PCM rates the server can produce on request (the ``sample_rate`` request field).
SERVER_SAMPLE_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
VOICES = ("ashu", "neha", "shruti", "sameer")
# Audio is pushed downstream in frames of this length, whatever size the network
# reads are. 40 ms is what Pipecat's output transports send per write by default.
FRAME_SECS = 0.04


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

    Each utterance is one streaming HTTP request. Requests share one kept-alive
    connection, so only the first pays for the TCP and TLS handshake, and with
    ``warm_connection`` that happens while the pipeline starts. Audio is pushed
    in 40 ms frames as it arrives, after a short first-audio buffer
    (``prebuffer_secs``) so playback doesn't start and then stall. Interrupting
    the bot closes the request immediately and drops audio not yet pushed.

    Mirai is asked for audio at the pipeline's output rate: an 8 kHz phone
    pipeline downloads 128 kbit/s per call instead of 48 kHz's 768 kbit/s.

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
        server_sample_rate: int | Literal["auto"] | None = "auto",
        prebuffer_secs: float = 0.15,
        warm_connection: bool = True,
        keep_warm_secs: float | None = 30.0,
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
            warm_connection: Open the connection to Mirai (an authenticated
                ``GET /models``) in the background when the pipeline starts,
                and again after an interruption drops it, so the next sentence
                doesn't wait for a TCP and TLS handshake. Failures are logged
                and ignored. ``False`` turns off every background request.
            keep_warm_secs: While the pipeline runs, repeat that request when
                the connection has been idle this long, so a long pause in the
                call doesn't let it close (Mirai closes idle connections after
                75 s). ``None`` turns it off.
            http_client: Optional caller-owned ``httpx.AsyncClient``.
            settings: Runtime-updatable settings; values here win over the
                ``voice``/``model`` shortcuts.
            **kwargs: Passed through to :class:`TTSService`.
        """
        key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
        if not key:
            raise ValueError("Set MIRAI_API_KEY or pass api_key to MiraiTTSService.")
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
        if keep_warm_secs is not None and not keep_warm_secs > 0:
            raise ValueError(f"keep_warm_secs must be > 0 or None; got {keep_warm_secs!r}")

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
        base = base_url.rstrip("/")
        self._speech_url = base + "/audio/speech"
        self._models_url = base + "/models"
        self._headers = {"Authorization": f"Bearer {key}"}
        self._server_rate_option = server_sample_rate
        self._server_rate_refused = False
        self._prebuffer_secs = float(prebuffer_secs)
        self._warm_connection = warm_connection
        self._keep_warm_secs = keep_warm_secs
        self._warm_task: asyncio.Task | None = None
        self._keep_warm_task: asyncio.Task | None = None
        self._closing = False
        # Responses still streaming, so an interruption or shutdown can close them.
        self._streams: set[httpx.Response] = set()
        # Bumped on every interruption; an utterance started before it pushes nothing more.
        self._interruptions = 0
        self._last_activity = time.monotonic()
        self._owns_http = http_client is None
        # Mirai's gateway closes a connection after 75 s idle. Drop it a little
        # earlier here, so a request is never sent on one the server is closing;
        # keep_warm_secs keeps it from ever getting that idle while a call runs.
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),
            limits=httpx.Limits(max_connections=8, max_keepalive_connections=8, keepalive_expiry=70),
        )
        self.last_error: str | None = None
        # The rate of the PCM Mirai sent for the latest utterance (X-Sample-Rate).
        self.last_server_sample_rate: int | None = None

    def can_generate_metrics(self) -> bool:
        """Mirai TTS reports TTFB and usage metrics."""
        return True

    async def start(self, frame: StartFrame):
        """Start the service; the output sample rate is known from here on."""
        await super().start(frame)
        if self._settings.voice not in VOICES:
            logger.warning(f"{self}: voice {self._settings.voice!r} is not one of {', '.join(VOICES)}")
        requested = self._requested_server_rate()
        asked = f"{requested} Hz" if requested else f"its default {SOURCE_SAMPLE_RATE} Hz"
        logger.debug(f"{self}: asking Mirai for {asked} PCM; output {self.sample_rate} Hz")
        self._start_warm_up()
        if self._warm_connection and self._keep_warm_secs and self._keep_warm_task is None:
            self._keep_warm_task = self.create_task(self._keep_warm(), name="keep_warm")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame; on an interruption, close the streams it cut short."""
        if not isinstance(frame, InterruptionFrame):
            await super().process_frame(frame, direction)
            return
        self._interruptions += 1
        cut_short = list(self._streams)
        # Pipecat cancels the running utterance here, and that normally closes
        # its HTTP stream too. But an utterance parked between two frames only
        # closes once its generator is finalised, which waits on garbage
        # collection if anything still refers to it, and until then the server
        # keeps generating into a slot nobody will hear. Close it now.
        await super().process_frame(frame, direction)
        for response in cut_short:
            await response.aclose()
        if cut_short:
            self._start_warm_up()  # those connections are gone; open the next one now

    def _requested_server_rate(self) -> int | None:
        """The ``sample_rate`` to put in the request, or ``None`` to leave it out."""
        if self._server_rate_refused:
            return None
        if self._server_rate_option == "auto":
            return self.sample_rate if self.sample_rate in SERVER_SAMPLE_RATES else None
        return self._server_rate_option

    @traced_tts
    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        """Synthesize ``text`` and yield audio frames as they stream in."""
        if not text.strip():
            return
        self.last_error = None
        epoch = self._interruptions
        rate = self.sample_rate
        frame_bytes = max(1, round(rate * FRAME_SECS)) * 2
        prebuffer_bytes = round(rate * self._prebuffer_secs) * 2
        pending = bytearray()  # resampled audio not pushed yet: < one frame once playing
        playing = prebuffer_bytes == 0
        carry = b""
        received = False
        try:
            await self.start_ttfb_metrics()
            self._last_activity = time.monotonic()  # keep-warm stands aside while this runs
            response = await self._post_speech(text)
            self._streams.add(response)
            try:
                if response.is_error:
                    await response.aread()
                    response.raise_for_status()
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
                        await self.stop_ttfb_metrics()  # TTFB = first audio byte received
                    if resampler:
                        data = resampler.resample_chunk(np.frombuffer(data, dtype="<i2")).tobytes()
                    pending += data
                    if not playing:
                        if len(pending) < prebuffer_bytes:
                            continue
                        playing = True
                    # Push whole frames; a burst of several seconds becomes many
                    # 40 ms frames, and only the remainder stays here.
                    whole = len(pending) - len(pending) % frame_bytes
                    for offset in range(0, whole, frame_bytes):
                        audio = bytes(pending[offset : offset + frame_bytes])
                        yield TTSAudioRawFrame(audio, rate, 1, context_id=context_id)
                        if self._interruptions != epoch or self._closing:
                            return  # cut short while parked here: push nothing more
                    del pending[:whole]
                if carry:
                    raise ValueError("audio ended with an incomplete PCM sample")
                if not received:
                    raise ValueError("no audio returned")
                if resampler:
                    pending += resampler.resample_chunk(np.empty(0, dtype=np.int16), last=True).tobytes()
                for offset in range(0, len(pending), frame_bytes):
                    audio = bytes(pending[offset : offset + frame_bytes])
                    yield TTSAudioRawFrame(audio, rate, 1, context_id=context_id)
                    if self._interruptions != epoch or self._closing:
                        return
            finally:
                self._streams.discard(response)
                self._last_activity = time.monotonic()
                await response.aclose()
        except httpx.HTTPStatusError as exc:
            message = f"Mirai TTS returned HTTP {exc.response.status_code}"
            detail = _error_detail(exc.response)
            if detail:
                message += f": {detail}"
            self.last_error = message
            yield ErrorFrame(error=message, exception=exc)
        except (httpx.HTTPError, ValueError) as exc:
            self.last_error = f"Mirai TTS: {exc}"
            yield ErrorFrame(error=self.last_error, exception=exc)
        except (asyncio.CancelledError, GeneratorExit):
            # Interrupted. Closing an unfinished response drops its connection,
            # so open the next one now, while the caller is still talking.
            self._start_warm_up()
            raise
        finally:
            # An interruption closes this generator: the HTTP stream is closed
            # with it and nothing already-interrupted is retried or flushed.
            await self.stop_ttfb_metrics()

    async def _post_speech(self, text: str) -> httpx.Response:
        """Send the utterance and return the open, streaming response.

        A server that doesn't take ``sample_rate`` may answer 400. The request
        is then sent once more without it. If that one gets past validation,
        the field is left out for the rest of the session and the server's
        48 kHz audio is resampled locally. If it is refused too, the request
        itself is at fault: that error is reported and nothing is remembered.
        """
        rate = self._requested_server_rate()
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

    async def _send_speech(self, text: str, rate: int | None) -> httpx.Response:
        body = {
            "model": self._settings.model,
            "voice": self._settings.voice,
            "input": text,
            "response_format": "pcm",
        }
        if rate is not None:
            body["sample_rate"] = rate
        request = self._http.build_request("POST", self._speech_url, headers=self._headers, json=body)
        return await self._http.send(request, stream=True)

    def _start_warm_up(self):
        """Open a connection to Mirai in the background, unless one is opening."""
        if not self._warm_connection or self._closing or self._http.is_closed:
            return
        if self._warm_task is not None and not self._warm_task.done():
            return
        try:
            self._warm_task = self.create_task(self._warm_up(), name="warm_up")
        except Exception as exc:  # e.g. no task manager yet; warming is only an optimisation
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
            logger.warning(f"{self}: Mirai rejected the API key (HTTP {response.status_code})")
        else:
            logger.debug(f"{self}: connection to Mirai open (GET /models: HTTP {response.status_code})")

    async def _keep_warm(self):
        """While the pipeline runs, keep the idle connection from timing out."""
        interval = self._keep_warm_secs
        while True:
            await asyncio.sleep(max(1.0, self._last_activity + interval - time.monotonic()))
            if self._streams or time.monotonic() - self._last_activity < interval:
                continue  # in use, or used since we went to sleep
            if self._warm_task is not None and not self._warm_task.done():
                continue
            await self._warm_up()

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
        self._closing = True
        for response in list(self._streams):
            await response.aclose()
        tasks = (self._warm_task, self._keep_warm_task)
        self._warm_task = self._keep_warm_task = None
        for task in tasks:
            if task is not None and not task.done():
                await self.cancel_task(task)
        if self._owns_http and not self._http.is_closed:
            await self._http.aclose()


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


def _error_detail(response: httpx.Response) -> str | None:
    """The ``error.message`` of a JSON error body, if there is one."""
    try:
        detail = response.json().get("error", {}).get("message")
    except (ValueError, httpx.HTTPError, AttributeError):
        return None
    return detail[:500] if isinstance(detail, str) else None
