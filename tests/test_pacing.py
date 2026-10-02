import time
from unittest.mock import MagicMock

import pytest
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from pipecat_mirai import apply_output_lead

FRAME = 0.02  # 20 ms chunks


class FakeOutput:
    """The three attributes Pipecat's websocket output transports pace with."""

    def __init__(self):
        self._next_send_time = 0
        self._send_interval = FRAME

    async def _write_audio_sleep(self):  # stock pacing, never used once patched
        raise AssertionError("stock pacing should have been replaced")


async def send(output, n):
    t0 = time.monotonic()
    for _ in range(n):
        await output._write_audio_sleep()
    return time.monotonic() - t0


async def test_first_lead_seconds_go_out_immediately_then_real_time():
    out = apply_output_lead(FakeOutput(), 0.4)
    assert await send(out, 20) < 0.05  # 0.4 s of audio, no waiting
    took = await send(out, 10)  # next 0.2 s paced at real time
    assert 0.15 < took < 0.3


async def test_interruption_restarts_the_clock():
    out = apply_output_lead(FakeOutput(), 0.2)
    await send(out, 10)
    out._next_send_time = 0  # what Pipecat does on InterruptionFrame
    assert await send(out, 10) < 0.05


async def test_stale_clock_after_a_pause_does_not_burst_beyond_lead():
    out = apply_output_lead(FakeOutput(), 0.2)
    await send(out, 10)
    out._next_send_time = time.monotonic() - 5  # long silence between turns
    assert await send(out, 10) < 0.05
    took = await send(out, 5)
    assert took > 0.07  # back to real time after the lead, no 5 s catch-up burst


def test_zero_keeps_stock_pacing():
    fake = FakeOutput()
    original = fake._write_audio_sleep
    assert apply_output_lead(fake, 0)._write_audio_sleep == original


def test_negative_lead_rejected():
    with pytest.raises(ValueError):
        apply_output_lead(FakeOutput(), -0.1)


def test_non_websocket_transport_left_unchanged():
    other = object()
    assert apply_output_lead(other) is other


def test_real_fastapi_websocket_transport_is_supported():
    # Guards against Pipecat renaming the pacing internals we rely on.
    transport = FastAPIWebsocketTransport(
        websocket=MagicMock(), params=FastAPIWebsocketParams(audio_out_enabled=True)
    )
    out = apply_output_lead(transport, 0.4)
    assert out is transport.output()
    assert out._mirai_output_lead_secs == 0.4
