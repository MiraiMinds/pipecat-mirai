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
