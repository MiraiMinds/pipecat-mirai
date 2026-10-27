"""Pacer: audio for a phone media stream at real time plus a small lead."""

import time

from fourier_frequency.pacer import Pacer


async def pieces(n, size):
    for _ in range(n):
        yield b"\0" * size


async def test_one_second_goes_out_in_one_second_less_the_lead():
    pacer = Pacer(8000, encoding="mulaw", lead_secs=0.4)
    started = time.monotonic()
    sent = [p async for p in pacer.pace(pieces(25, 320))]  # 25 x 40 ms of 8 kHz mu-law
    took = time.monotonic() - started
    assert len(sent) == 25
    assert 0.55 <= took < 0.75  # the last 40 ms piece leaves 0.4 s before the second is over
    assert 0.35 <= pacer.buffered_secs <= 0.45


async def test_the_lead_goes_out_at_once():
    pacer = Pacer(8000, encoding="pcm_s16le", lead_secs=0.4)
    started = time.monotonic()
    [p async for p in pacer.pace(pieces(10, 640))]  # 0.4 s of 16-bit PCM
    assert time.monotonic() - started < 0.05


async def test_reset_after_a_barge_in_sends_the_next_reply_at_once():
    pacer = Pacer(8000, encoding="mulaw", lead_secs=0.0)
    [p async for p in pacer.pace(pieces(5, 320))]
    pacer.reset()
    started = time.monotonic()
    await pacer.wait(320)
    assert time.monotonic() - started < 0.02
