#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Send speech to a phone media stream at real time, a little ahead.

A media stream (Twilio, Plivo, Exotel, Telnyx, ...) plays audio as it comes.
Sent exactly at real time, any pause on the sending server (a busy event
loop, a garbage collection, a slow sentence) is a gap the caller hears. In
our phone benchmark, audio sent at real time was stretched 6.6% at three
concurrent calls and 29% at ten; with a 0.4 s lead, under 0.1%. Sent
all at once, the far end holds seconds of reply, and a caller who barges in
hears it all unless it is cleared. :class:`Pacer` keeps a small lead buffered
at the far end (0.4 s by default): enough to ride out a stall, little enough
that a barge-in cuts the reply off within ``lead_secs`` (at once if the far
end is also told to clear its buffer).

Example, with a :class:`~fourier_frequency.TTSTurn`::

    pacer = Pacer(sample_rate=8000, encoding="mulaw")
    async for audio in pacer.pace(turn):
        await phone.send_media(audio)
    ...
    pacer.reset()  # the caller barged in and the media stream was cleared
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, AsyncIterator

DEFAULT_LEAD_SECS = 0.4


class Pacer:
    """Hands out audio no further ahead of real time than ``lead_secs``."""

    def __init__(
        self, sample_rate: int = 8000, *, encoding: str = "pcm_s16le", lead_secs: float = DEFAULT_LEAD_SECS
    ):
        """Pace audio of ``sample_rate`` Hz in ``encoding`` (``"pcm_s16le"``, ``"mulaw"`` or ``"alaw"``)."""
        if not lead_secs >= 0:
            raise ValueError(f"lead_secs must be >= 0; got {lead_secs!r}")
        self._bytes_per_sec = sample_rate * (2 if encoding == "pcm_s16le" else 1)
        self._lead = float(lead_secs)
        # When the far end will have played everything sent so far (monotonic).
        self._played_at = 0.0

    @property
    def buffered_secs(self) -> float:
        """Seconds of audio sent and not played yet, by this clock."""
        return max(0.0, self._played_at - time.monotonic())

    async def wait(self, nbytes: int):
        """Wait until ``nbytes`` more may be sent, and count them as sent."""
        now = time.monotonic()
        if self._played_at < now:
            self._played_at = now  # the far end ran dry (between replies): start over
        ahead = self._played_at - now - self._lead
        if ahead > 0:
            await asyncio.sleep(ahead)
        self._played_at += nbytes / self._bytes_per_sec

    async def pace(self, audio: AsyncIterable[bytes]) -> AsyncIterator[bytes]:
        """Yield each piece of ``audio`` when it is time to send it."""
        async for piece in audio:
            await self.wait(len(piece))
            yield piece

    def reset(self):
        """The far end's buffer was cleared (barge-in): the next audio goes at once."""
        self._played_at = 0.0
