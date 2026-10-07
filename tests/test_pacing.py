import time
from unittest.mock import MagicMock

import pytest
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import run_test
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

from pipecat_mirai import MiraiTTSService, MiraiWebsocketTTSService, apply_output_lead
from pipecat_mirai.pacing import ensure_output_lead, find_output_transport

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


# --- the services set the lead themselves (output_lead_secs) ---------------------------------


class PassThrough(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class WebsocketLikeOutput(PassThrough):
    """A processor with the pacing internals of Pipecat's websocket outputs."""

    def __init__(self):
        super().__init__()
        self._next_send_time = 0
        self._send_interval = FRAME

    async def _write_audio_sleep(self):  # stock pacing
        pass


def http_tts(**kwargs):
    return MiraiTTSService(api_key="k", warm_connection=False, **kwargs)


def test_the_lead_goes_on_the_first_output_after_the_service():
    tts, out = http_tts(), WebsocketLikeOutput()
    Pipeline([tts, PassThrough(), out, PassThrough()])
    assert ensure_output_lead(tts, 0.4) is out
    assert out._mirai_output_lead_secs == 0.4


def test_the_output_is_found_past_the_end_of_a_nested_pipeline():
    tts, out = http_tts(), WebsocketLikeOutput()
    Pipeline([PassThrough(), Pipeline([PassThrough(), tts]), PassThrough(), out])
    assert find_output_transport(tts) is out


def test_no_output_after_the_service_changes_nothing():
    tts, out = http_tts(), WebsocketLikeOutput()
    Pipeline([out, tts, PassThrough()])  # the output is before it, not after
    assert ensure_output_lead(tts, 0.4) is None
    assert not hasattr(out, "_mirai_output_lead_secs")


def test_a_lead_set_by_the_caller_is_kept():
    tts, out = http_tts(), WebsocketLikeOutput()
    Pipeline([tts, out])
    apply_output_lead(out, 0.2)
    patched = out._write_audio_sleep
    assert ensure_output_lead(tts, 0.4) is out
    assert out._mirai_output_lead_secs == 0.2 and out._write_audio_sleep == patched


def test_two_services_before_one_transport_set_it_once():
    first, second, out = http_tts(), http_tts(), WebsocketLikeOutput()
    Pipeline([first, second, out])
    ensure_output_lead(first, 0.4)
    patched = out._write_audio_sleep
    ensure_output_lead(second, 0.6)
    assert out._mirai_output_lead_secs == 0.4 and out._write_audio_sleep == patched


def test_transports_that_pace_themselves_are_left_alone():
    # WebRTC, Daily, LiveKit: output transports without websocket pacing.
    tts, out = http_tts(), BaseOutputTransport(TransportParams(audio_out_enabled=True))
    Pipeline([tts, out])
    assert ensure_output_lead(tts, 0.4) is None
    assert not hasattr(out, "_mirai_output_lead_secs")


def test_a_real_fastapi_websocket_output_gets_the_lead():
    transport = FastAPIWebsocketTransport(
        websocket=MagicMock(), params=FastAPIWebsocketParams(audio_out_enabled=True)
    )
    tts = http_tts()
    Pipeline([transport.input(), tts, transport.output()])
    assert ensure_output_lead(tts, 0.4) is transport.output()
    assert transport.output()._mirai_output_lead_secs == 0.4


@pytest.mark.parametrize("cls", [MiraiTTSService, MiraiWebsocketTTSService])
@pytest.mark.parametrize("lead", [0.4, None])
async def test_the_services_set_the_lead_when_the_pipeline_starts(cls, lead):
    kwargs = (
        {"url": "ws://127.0.0.1:9/none", "http_fallback": False} if cls is MiraiWebsocketTTSService else {}
    )
    tts, out = cls(api_key="k", output_lead_secs=lead, **kwargs), WebsocketLikeOutput()
    await run_test(
        Pipeline([tts, out]),
        frames_to_send=[],
        pipeline_params=PipelineParams(audio_out_sample_rate=8000),
        start_timeout=10.0,
    )
    assert getattr(out, "_mirai_output_lead_secs", None) == lead
