#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Keep a phone call's audio a little ahead of real time.

Pipecat's websocket output transports (FastAPI, websocket server, websocket
client) pace outgoing audio at exactly real time: each chunk is written, then the
transport sleeps for that chunk's duration. The telephony provider therefore
never holds more than one chunk (~40 ms). When the backend's event loop stalls -
a synchronous call, a large JSON parse, many concurrent calls on one CPU - the
provider runs out of audio and the caller hears a break. Any TTS vendor is
affected the same way.

:func:`apply_output_lead` lets the transport run up to ``seconds`` ahead of real
time instead. Stalls shorter than the lead are absorbed. On an interruption
pipecat already sends the provider a clear (``InterruptionFrame`` through the
serializer), so at most ``seconds`` of queued audio is discarded, which is what
should happen on a barge-in anyway.

Measured on 8 kHz Twilio-protocol calls with 50-250 ms event-loop stalls every
~3 s per call (see ``benchmarks/phone-breaks``): stock pacing stretched speech
by 6.6% at 3 concurrent calls and 29% at 10; with a 0.4 s lead, 0.08% and 0.01%.
"""

from __future__ import annotations

import asyncio
import time
import types

from loguru import logger

DEFAULT_LEAD_SECS = 0.4
_REQUIRED = ("_write_audio_sleep", "_next_send_time", "_send_interval")


def apply_output_lead(transport, seconds: float = DEFAULT_LEAD_SECS):
    """Let ``transport``'s audio output run up to ``seconds`` ahead of real time.

    Works with Pipecat's ``FastAPIWebsocketTransport``, ``WebsocketServerTransport``
    and ``WebsocketClientTransport`` (or their ``output()`` processors). Call it
    once, after creating the transport and before running the pipeline. Other
    transports (WebRTC, Daily, LiveKit) pace audio differently and are returned
    unchanged with a warning.

    Args:
        transport: A Pipecat transport, or its output processor.
        seconds: Maximum lead in seconds. ``0`` keeps Pipecat's stock pacing.

    Returns:
        The output processor that was configured.
    """
    if seconds < 0:
        raise ValueError("seconds must be >= 0")
    output = transport.output() if hasattr(transport, "output") and callable(transport.output) else transport
    if not all(hasattr(output, name) for name in _REQUIRED):
        logger.warning(
            f"apply_output_lead: {type(output).__name__} does not use websocket real-time "
            "pacing; leaving it unchanged"
        )
        return output
    if seconds == 0:
        return output

    async def _write_audio_sleep(self, _lead=float(seconds)):
        # Same clock as Pipecat's, but sleep only once the provider already holds
        # `_lead` seconds of audio. `_next_send_time` is reset to 0 by Pipecat on
        # every interruption; restart the clock from now when it is unset or stale.
        now = time.monotonic()
        if self._next_send_time == 0 or self._next_send_time < now:
            self._next_send_time = now
        await asyncio.sleep(max(0.0, self._next_send_time - _lead - now))
        self._next_send_time += self._send_interval

    output._write_audio_sleep = types.MethodType(_write_audio_sleep, output)
    output._mirai_output_lead_secs = float(seconds)
    logger.debug(f"{output}: audio output lead {seconds:.2f}s")
    return output


def _is_websocket_output(processor) -> bool:
    return all(hasattr(processor, name) for name in _REQUIRED)


def _is_output_transport(processor) -> bool:
    try:
        from pipecat.transports.base_output import BaseOutputTransport
    except ImportError:  # pragma: no cover - every supported Pipecat has it
        return False
    return isinstance(processor, BaseOutputTransport)


def _downstream(processor):
    """The processor a frame pushed downstream from ``processor`` reaches next.

    Inside a pipeline that is the next linked processor. A pipeline's last
    processor is its sink, which hands frames to the pipeline itself (its
    ``push_frame``), and from there they go to whatever follows the pipeline.
    """
    following = getattr(processor, "_next", None)
    if following is not None:
        return following
    owner = getattr(getattr(processor, "_downstream_push_frame", None), "__self__", None)
    if owner is not None and owner is not processor:
        return getattr(owner, "_next", None)
    return None


def find_output_transport(processor, max_hops: int = 512):
    """The first output transport downstream of ``processor``, or ``None``."""
    seen = set()
    node = _downstream(processor)
    while node is not None and id(node) not in seen and len(seen) < max_hops:
        seen.add(id(node))
        if _is_websocket_output(node) or _is_output_transport(node):
            return node
        node = _downstream(node)
    return None


_said: set[str] = set()


def _once(key: str, log, message: str):
    """Log ``message`` the first time ``key`` comes up in this process, at debug level after that."""
    if key in _said:
        logger.debug(message)
    else:
        _said.add(key)
        log(message)


def ensure_output_lead(processor, seconds: float | None = DEFAULT_LEAD_SECS):
    """Give the output transport downstream of ``processor`` an audio output lead.

    What the Mirai TTS services do when their pipeline starts (``output_lead_secs``).
    It finds the first output transport downstream and applies
    :func:`apply_output_lead` to it, unless a lead is already set there (by
    you, or by another service in the same pipeline). Transports that pace
    audio their own way (WebRTC, Daily, LiveKit) are left alone, and so is a
    pipeline with no output transport after the service. Never raises.

    Returns:
        The output processor that has the lead, or ``None``.
    """
    if not seconds:
        return None
    try:
        output = find_output_transport(processor)
        if output is None:
            _once(
                "lead-none",
                logger.warning,
                f"{processor}: no output transport after this service in the pipeline, so no audio "
                f"output lead was set; if this is a phone call, call apply_output_lead(transport)",
            )
            return None
        if not _is_websocket_output(output):
            name = type(output).__name__
            _once(
                f"lead-other-{name}",
                logger.debug,
                f"{processor}: {name} paces audio itself; no output lead needed",
            )
            return None
        if getattr(output, "_mirai_output_lead_secs", None) is not None:
            return output  # set already: by the caller, or by another service in this pipeline
        apply_output_lead(output, seconds)
        name = type(output).__name__
        _once(
            f"lead-set-{name}",
            logger.info,
            f"{processor}: {name} may now send up to {seconds:g} s of audio ahead of real time, so a "
            f"busy server doesn't break up the call's audio (output_lead_secs=None turns this off)",
        )
        return output
    except Exception as exc:  # never let an optimisation break a call
        logger.warning(f"{processor}: could not set an audio output lead: {exc!r}")
        return None
