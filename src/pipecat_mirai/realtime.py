#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai's Realtime API for Pipecat.

Mirai's Realtime API speaks the OpenAI Realtime protocol: one WebSocket session in
which Mirai runs speech recognition, turn detection, the language model and speech
synthesis next to each other. Pipecat's stock ``OpenAIRealtimeLLMService`` can
connect to it, but a few things only work through this subclass:

- the session URL is built for you: ``agent_id``, ``variables``, ``metadata``,
  ``webhook_url`` and ``max_duration_secs`` are query parameters, and the
  sandbox and production addresses are named;
- the ``mirai`` block (model temperature, the caller's language, tool timeout and
  so on), which stock ``SessionProperties`` silently drops, reaches the server;
- Mirai's own events (``mirai.session.applied``, ``mirai.turn.metrics``) are
  handled instead of crashing stock Pipecat's event parser, and each turn's
  latency and providers arrive through the ``on_turn_metrics`` event handler;
- caller audio at any sample rate is resampled to the 24 kHz the protocol
  expects, so an 8 kHz phone pipeline needs no ``audio_in_sample_rate`` change.

Everything else (tools, interruptions, transcripts, context) is Pipecat's own
``OpenAIRealtimeLLMService``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import soxr
from loguru import logger
from pipecat.frames.frames import InputAudioRawFrame
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

SANDBOX_REALTIME_URL = "wss://sandbox.voice.miraiminds.co/v2/realtime"
PRODUCTION_REALTIME_URL = "wss://prod.voice.miraiminds.co/v2/realtime"
REALTIME_MODEL = "mira-realtime"
PROTOCOL_SAMPLE_RATE = 24000

# Keys the server accepts in session.mirai. It rejects anything else with
# `unknown_parameter`, which stock Pipecat treats as fatal, so check them here.
MIRAI_BLOCK_KEYS = frozenset(
    {
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "llm",
        "language",
        "stt",
        "tts",
        "tool_timeout_ms",
        "events",
    }
)


@dataclass
class MiraiTurnMetrics:
    """One turn's numbers, as Mirai measured them on its side.

    Parameters:
        response_id: The ``response.done`` this turn belongs to.
        v2v_ms: End of the caller's speech to the first audio of the reply.
            Add your own network time for what the caller hears.
        stt_ms: Speech recognition, to its first result for the turn.
        llm_ms: The model, to its first output.
        tts_ms: Speech synthesis, to its first audio.
        providers: Which provider served ``stt``, ``llm`` and ``tts``
            (``"mira"`` for Mirai's own).
        fallback: True when any step ran on a fallback provider.
        failovers: Provider switches during the turn, each
            ``{"kind", "from", "to", "reason"}``.
    """

    response_id: str | None = None
    v2v_ms: float | None = None
    stt_ms: float | None = None
    llm_ms: float | None = None
    tts_ms: float | None = None
    providers: dict[str, str] = field(default_factory=dict)
    fallback: bool = False
    failovers: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_event(cls, evt: Mapping[str, Any]) -> MiraiTurnMetrics:
        """From a ``mirai.turn.metrics`` event."""
        return cls(
            response_id=evt.get("response_id"),
            v2v_ms=_number(evt.get("v2v_ms")),
            stt_ms=_number(evt.get("stt_ms")),
            llm_ms=_number(evt.get("llm_ms")),
            tts_ms=_number(evt.get("tts_ms")),
            providers=dict(evt.get("providers") or {}),
            fallback=bool(evt.get("fallback")),
            failovers=list(evt.get("failovers") or []),
        )

    @classmethod
    def from_response_metadata(cls, response_id: str | None, metadata: Mapping[str, Any]) -> MiraiTurnMetrics:
        """From ``response.done``'s ``response.metadata`` (string values)."""
        providers = {
            kind: str(metadata[f"mirai_{kind}_provider"])
            for kind in ("stt", "llm", "tts")
            if metadata.get(f"mirai_{kind}_provider")
        }
        return cls(
            response_id=response_id,
            v2v_ms=_number(metadata.get("mirai_v2v_ms")),
            stt_ms=_number(metadata.get("mirai_stt_ms")),
            llm_ms=_number(metadata.get("mirai_llm_ms")),
            tts_ms=_number(metadata.get("mirai_tts_ms")),
            providers=providers,
            fallback=str(metadata.get("mirai_fallback", "")).lower() == "true",
        )


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


class _MiraiEventTap:
    """Wraps the session socket: Mirai's own events go to the service, the rest pass through.

    Stock Pipecat parses every server event against OpenAI's event types and stops
    reading the socket on one it does not know. Taking ``mirai.*`` events out
    before that parser sees them keeps the session alive.
    """

    def __init__(self, socket: Any, service: MiraiRealtimeLLMService):
        self._socket = socket
        self._service = service

    def __getattr__(self, name: str) -> Any:
        return getattr(self._socket, name)

    async def send(self, message: Any) -> None:
        await self._socket.send(message)

    async def close(self, *args: Any, **kwargs: Any) -> None:
        await self._socket.close(*args, **kwargs)

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        async for message in self._socket:
            try:
                evt = json.loads(message)
            except (TypeError, ValueError):
                yield message
                continue
            kind = evt.get("type") if isinstance(evt, dict) else None
            ours = isinstance(kind, str) and kind.startswith("mirai.")
            try:
                if ours:
                    await self._service._handle_mirai_event(evt)
                elif kind == "response.done":
                    await self._service._handle_response_done_metadata(evt)
            except Exception as exc:  # never let our extras end the session
                logger.warning(f"{self._service}: could not handle {kind}: {exc}")
            if not ours:  # stock Pipecat stops reading on an event type it does not know
                yield message


class MiraiRealtimeLLMService(OpenAIRealtimeLLMService):
    """Pipecat's OpenAI Realtime service, set up for Mirai's Realtime API.

    Replaces your STT, LLM and TTS services. Your transport and context
    aggregators stay as they are::

        llm = MiraiRealtimeLLMService(voice="shruti", language="hi")
        pipeline = Pipeline([transport.input(), ctx.user(), llm, transport.output(), ctx.assistant()])

    Event handlers, in addition to the stock ones:

    - ``on_turn_metrics(service, metrics: MiraiTurnMetrics)`` after each reply.
    - ``on_session_applied(service, session: dict, changed: list, ignored: list)``
      when the session starts and after each settings change, with what is
      actually running and every setting the server did not apply.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SANDBOX_REALTIME_URL,
        agent_id: str | None = None,
        variables: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        webhook_url: str | None = None,
        max_duration_secs: int | None = None,
        voice: str | None = None,
        language: str | None = None,
        mirai: Mapping[str, Any] | None = None,
        settings: OpenAIRealtimeLLMService.Settings | None = None,
        **kwargs: Any,
    ):
        """Initialize the Mirai Realtime service.

        Args:
            api_key: Mirai API key. Defaults to ``MIRAI_API_KEY`` (or ``MIRA_API_KEY``).
            base_url: ``SANDBOX_REALTIME_URL`` (default, for sandbox keys) or
                ``PRODUCTION_REALTIME_URL`` once your company has gone live.
            agent_id: Start from one of your agents: its prompt, tools, voice and language.
            variables: Values for the agent's ``{{placeholders}}``.
            metadata: Echoed on the call and in every webhook; never shown to the model.
            webhook_url: Where this session's webhooks go.
            max_duration_secs: End the session after this many seconds (30 to 1800).
            voice: ``ashu``, ``neha``, ``shruti`` or ``sameer``. Defaults to the agent's
                voice with ``agent_id``, otherwise ``neha``.
            language: The caller's and the agent's language, such as ``"hi"``.
                Leave it out to detect the caller's language.
            mirai: Extra ``session.mirai`` settings, such as
                ``{"temperature": 0.3, "tool_timeout_ms": 15000}``.
            settings: Pipecat Realtime settings; its ``session_properties`` win over
                the ``voice`` default.
            **kwargs: Passed to ``OpenAIRealtimeLLMService``.
        """
        key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
        if not key:
            raise ValueError("Set MIRAI_API_KEY or pass api_key to MiraiRealtimeLLMService.")
        if max_duration_secs is not None and not 30 <= int(max_duration_secs) <= 1800:
            raise ValueError("max_duration_secs must be between 30 and 1800.")

        block = dict(mirai or {})
        unknown = set(block) - MIRAI_BLOCK_KEYS
        if unknown:
            raise ValueError(f"Unknown mirai settings: {', '.join(sorted(unknown))}")
        if language:
            block.setdefault("language", language)
        # Mirai's events carry the turn metrics; the tap below keeps them away
        # from stock Pipecat's parser.
        block.setdefault("events", True)
        self._mirai_block = block

        # With agent_id and no voice, the agent's own voice is used.
        out_voice = voice or (None if agent_id else "neha")
        session = events.SessionProperties(
            audio=events.AudioConfiguration(
                input=events.AudioInput(
                    # server_vad: Mirai's own turn detection decides when the caller is done.
                    turn_detection=events.TurnDetection(),
                ),
                output=events.AudioOutput(voice=out_voice) if out_voice else None,
            ),
        )
        defaults = OpenAIRealtimeLLMService.Settings(model=REALTIME_MODEL, session_properties=session)
        if settings is not None:
            defaults.apply_update(settings)

        super().__init__(api_key=key, base_url=base_url, settings=defaults, **kwargs)

        query: dict[str, Any] = {"model": REALTIME_MODEL, "mirai_events": "1"}
        if agent_id:
            query["agent_id"] = agent_id
        if variables:
            query["variables"] = json.dumps(dict(variables), ensure_ascii=False, separators=(",", ":"))
        if metadata:
            query["metadata"] = json.dumps(dict(metadata), ensure_ascii=False, separators=(",", ":"))
        if webhook_url:
            query["webhook_url"] = webhook_url
        if max_duration_secs is not None:
            query["max_duration_secs"] = str(int(max_duration_secs))
        # Pipecat appends "?model=…" to base_url; build the query properly instead.
        self.base_url = _with_query(base_url, query)

        self.last_turn_metrics: MiraiTurnMetrics | None = None
        self._reported_responses: set[str] = set()
        self._resampler: soxr.ResampleStream | None = None
        self._resampler_rate: int | None = None
        self._register_event_handler("on_turn_metrics")
        self._register_event_handler("on_session_applied")

    # -- connection -------------------------------------------------------------

    async def _connect(self):
        await super()._connect()
        # The receive task was created inside super()._connect() and has not run
        # yet (no await since), so it iterates the tap from its first message.
        if self._websocket is not None and not isinstance(self._websocket, _MiraiEventTap):
            self._websocket = _MiraiEventTap(self._websocket, self)

    # -- outbound ---------------------------------------------------------------

    async def send_client_event(self, event: events.ClientEvent):
        """Send an event; ``session.update`` gains the ``mirai`` block."""
        if isinstance(event, events.SessionUpdateEvent) and self._mirai_block:
            payload = event.model_dump(exclude_none=True)
            payload.setdefault("session", {})["mirai"] = self._mirai_block
            await self._ws_send(payload)
            return
        await super().send_client_event(event)

    async def _send_user_audio(self, frame):
        # The protocol carries 16-bit PCM at 24 kHz. Resample anything else, with
        # one streaming resampler per input rate so chunk edges stay clean.
        if frame.sample_rate and frame.sample_rate != PROTOCOL_SAMPLE_RATE:
            if self._resampler is None or self._resampler_rate != frame.sample_rate:
                self._resampler = soxr.ResampleStream(
                    frame.sample_rate, PROTOCOL_SAMPLE_RATE, 1, dtype="int16", quality="HQ"
                )
                self._resampler_rate = frame.sample_rate
            audio = self._resampler.resample_chunk(np.frombuffer(frame.audio, dtype="<i2")).tobytes()
            if not audio:
                return
            frame = InputAudioRawFrame(audio=audio, sample_rate=PROTOCOL_SAMPLE_RATE, num_channels=1)
        await super()._send_user_audio(frame)

    # -- Mirai's events ---------------------------------------------------------

    async def _handle_mirai_event(self, evt: dict[str, Any]):
        kind = evt.get("type")
        if kind == "mirai.turn.metrics":
            metrics = MiraiTurnMetrics.from_event(evt)
            if metrics.response_id:
                self._reported_responses.add(metrics.response_id)
            await self._report_turn(metrics)
        elif kind == "mirai.session.applied":
            ignored = list(evt.get("ignored") or [])
            for item in ignored:
                logger.debug(f"{self}: Mirai did not apply {item.get('field')}: {item.get('reason')}")
            await self._call_event_handler(
                "on_session_applied", dict(evt.get("session") or {}), list(evt.get("changed") or []), ignored
            )
        else:
            logger.trace(f"{self}: {kind}")

    async def _handle_response_done_metadata(self, evt: dict[str, Any]):
        # Without mirai.turn.metrics (events turned off), the same numbers arrive
        # as strings in response.metadata, which stock Pipecat discards.
        response = evt.get("response") or {}
        response_id = response.get("id")
        metadata = response.get("metadata") or {}
        if not metadata or (response_id and response_id in self._reported_responses):
            if response_id:
                self._reported_responses.discard(response_id)
            return
        await self._report_turn(MiraiTurnMetrics.from_response_metadata(response_id, metadata))

    async def _report_turn(self, metrics: MiraiTurnMetrics):
        self.last_turn_metrics = metrics
        logger.debug(
            f"{self}: turn v2v={metrics.v2v_ms} stt={metrics.stt_ms} llm={metrics.llm_ms} "
            f"tts={metrics.tts_ms} providers={metrics.providers} fallback={metrics.fallback}"
        )
        await self._call_event_handler("on_turn_metrics", metrics)


def _with_query(url: str, params: Mapping[str, Any]) -> str:
    """``url`` with ``params`` merged into its query string (``params`` win)."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({k: str(v) for k, v in params.items()})
    return urlunsplit(parts._replace(query=urlencode(query)))
