"""What a customer's worker meets in the wild: busy servers, dropped connections, a missing endpoint."""

import asyncio

import pytest
from fake_mirai_ws import FakeMiraiWS
from pipecat.frames.frames import (
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test
from test_tts import TEXT, FakeMirai, Recorder, audio_of, errors_in, speak, tone

from pipecat_mirai import MiraiHttpTTSService, MiraiWebsocketTTSService, close_shared_connections
from pipecat_mirai import tts as tts_module

KEY = "sk_test"


@pytest.fixture(autouse=True)
async def _no_shared_connections_left():
    yield
    await close_shared_connections()


def posts(fake):
    return [r for r in fake.requests if r["method"] == "POST"]


def mirai_error(up):
    """The one error the service reported (Pipecat may add "completed with no audio")."""
    (error,) = [e for e in errors_in(up) if e.error.startswith("Mirai TTS")]
    return error


def http_tts(url, **kwargs):
    return MiraiHttpTTSService(api_key=KEY, base_url=url, **kwargs)


# --- HTTP: Mirai busy, rate limits, server errors ------------------------------------------


async def test_a_busy_mirai_is_asked_again_after_retry_after():
    fake = FakeMirai(script=[{"status": 429, "retry_after": 0.2, "code": "at_capacity"}])
    async with fake.serve() as url:
        down, up = await speak(http_tts(url), 8000)
    assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    first, second = posts(fake)
    assert second["t"] - first["t"] >= 0.2


async def test_busy_for_longer_than_the_retry_budget_is_reported_with_what_to_do(monkeypatch):
    monkeypatch.setattr(tts_module, "RETRY_BUDGET_SECS", 0.5)
    busy = {"status": 429, "retry_after": 0.2, "code": "at_capacity", "message": "no slot"}
    fake = FakeMirai(script=[busy] * 5)
    async with fake.serve() as url:
        down, up = await speak(http_tts(url), 8000)
    assert len(posts(fake)) == 3  # 0.2 + 0.2 s of waiting fit the budget; a third wait doesn't
    error = mirai_error(up)
    assert "HTTP 429: no slot" in error.error and "still after 0.4 s of retries" in error.error
    assert "[request req_3]" in error.error and "concurrency limit" in error.error
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down)


async def test_a_retry_after_longer_than_the_budget_is_reported_at_once():
    fake = FakeMirai(script=[{"status": 429, "retry_after": 5, "code": "rate_limited"}])
    async with fake.serve() as url:
        _, up = await speak(http_tts(url), 8000)
    assert len(posts(fake)) == 1
    error = mirai_error(up)
    assert "rate limit" in error.error and "raise it" in error.error


@pytest.mark.parametrize("status", [502, 503, 504])
async def test_a_server_error_is_retried_once(status):
    fake = FakeMirai(script=[{"status": status}])
    async with fake.serve() as url:
        down, up = await speak(http_tts(url), 8000)
    assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    assert len(posts(fake)) == 2


async def test_a_server_error_twice_is_reported():
    fake = FakeMirai(script=[{"status": 502}, {"status": 502}])
    async with fake.serve() as url:
        _, up = await speak(http_tts(url), 8000)
    error = mirai_error(up)
    assert "HTTP 502" in error.error and "also on a retry" in error.error and "[request" in error.error


async def test_a_connection_dropped_before_any_response_is_retried_on_a_new_one():
    fake = FakeMirai(script=[{"reset": True}])
    async with fake.serve() as url:
        down, up = await speak(http_tts(url), 8000)
    assert audio_of(down, 8000) == tone(0.5, 8000) and not errors_in(up)
    assert fake.connections == 2 and len(posts(fake)) == 2


async def test_a_connection_dropped_twice_is_reported():
    fake = FakeMirai(script=[{"reset": True}, {"reset": True}])
    async with fake.serve() as url:
        _, up = await speak(http_tts(url), 8000)
    error = mirai_error(up)
    assert "could not reach" in error.error and "also on a retry" in error.error


async def test_a_stream_cut_mid_sentence_plays_what_arrived_and_says_so():
    fake = FakeMirai(seconds=1.0, reads=(1600,), script=[{"cut_after": 8000}])  # 0.5 s of 8 kHz
    rec = Recorder()
    async with fake.serve() as url:
        started = asyncio.get_running_loop().time()
        _, up = await run_test(
            Pipeline([http_tts(url), rec]),
            frames_to_send=[TTSSpeakFrame(TEXT)],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
        took = asyncio.get_running_loop().time() - started
    assert b"".join(f.audio for _, f in rec.audio) == tone(1.0, 8000)[:8000]  # every byte that came
    error = mirai_error(up)
    assert "broke off after 0.5 s of audio" in error.error and "[request req_1]" in error.error
    assert len([f for _, f in rec.seen if isinstance(f, TTSStoppedFrame)]) == 1  # the turn ends: no hang
    assert took < 5


async def test_errors_name_the_request_and_say_what_to_do():
    fake = FakeMirai(script=[{"status": 402, "code": "insufficient_balance", "message": "wallet is empty"}])
    async with fake.serve() as url:
        _, up = await speak(http_tts(url), 8000)
    error = mirai_error(up)
    assert error.error == (
        "Mirai TTS returned HTTP 402: wallet is empty [request req_1]. "
        "The workspace is out of credit. Top it up in the Mirai console."
    )


# --- WebSocket: an endpoint that isn't there ------------------------------------------------


def ws_url_of(http_url):
    return http_url.replace("http://", "ws://") + "/audio/speech/stream"


async def test_a_missing_streaming_endpoint_falls_back_to_http():
    fake = FakeMirai()  # answers the WebSocket handshake with 404, serves POST /audio/speech
    async with fake.serve() as url:
        tts = MiraiWebsocketTTSService(api_key=KEY, url=ws_url_of(url))
        down, up = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(TEXT), SleepFrame(0.2), TTSSpeakFrame("दूसरा")],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
    assert tts._fallback == "HTTP 404"
    assert not errors_in(up)
    assert [(b["input"], b["sample_rate"]) for b in fake.speech] == [(TEXT, 8000), ("दूसरा", 8000)]
    assert audio_of(down, 8000) == tone(0.5, 8000) * 2
    assert len([f for f in down if isinstance(f, TTSStoppedFrame)]) == 2


async def test_the_http_fallback_cuts_streamed_tokens_into_sentences():
    fake = FakeMirai()
    async with fake.serve() as url:
        tts = MiraiWebsocketTTSService(
            api_key=KEY, url=ws_url_of(url), text_aggregation_mode=TextAggregationMode.TOKEN
        )
        tokens = ["पहला ", "वाक्य। ", "दूसरा", " वाक्य"]
        down, up = await run_test(
            tts,
            frames_to_send=[
                LLMFullResponseStartFrame(),
                *(TextFrame(t) for t in tokens),
                LLMFullResponseEndFrame(),
            ],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
    assert not errors_in(up)
    assert [b["input"].strip() for b in fake.speech] == ["पहला वाक्य।", "दूसरा वाक्य"]
    assert audio_of(down, 8000) == tone(0.5, 8000) * 2


async def test_an_interruption_stops_the_http_fallback_at_once():
    fake = FakeMirai(seconds={"लंबा": 20.0, "छोटा": 0.5}, reads=(3200,), pace=0.05)
    rec = Recorder()
    async with fake.serve() as url:
        tts = MiraiWebsocketTTSService(api_key=KEY, url=ws_url_of(url))
        await run_test(
            Pipeline([tts, rec]),
            frames_to_send=[
                TTSSpeakFrame("लंबा"),
                SleepFrame(0.6),
                InterruptionFrame(),
                SleepFrame(0.3),
                TTSSpeakFrame("छोटा"),
                SleepFrame(0.5),
            ],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
    interrupted_at = next(t for t, f in rec.seen if isinstance(f, InterruptionFrame))
    long, short = posts(fake)
    assert long["cut_at"] is not None and long["cut_at"] - interrupted_at < 0.5
    after = [f for t, f in rec.audio if t > interrupted_at]
    assert b"".join(f.audio for f in after) == tone(0.5, 8000)


async def test_a_refused_api_key_is_reported_not_worked_around():
    fake = FakeMiraiWS(refuse_status=401)
    async with fake.serve() as url:
        tts = MiraiWebsocketTTSService(api_key=KEY, url=url)
        _, up = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(TEXT)],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
    assert tts._fallback is None
    assert any("HTTP 401" in e.error and "Check MIRAI_API_KEY" in e.error for e in errors_in(up))


async def test_http_fallback_false_reports_a_missing_endpoint():
    fake = FakeMirai()
    async with fake.serve() as url:
        tts = MiraiWebsocketTTSService(api_key=KEY, url=ws_url_of(url), http_fallback=False)
        _, up = await run_test(
            tts,
            frames_to_send=[TTSSpeakFrame(TEXT)],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
    assert tts._fallback is None and not fake.speech
    assert any("HTTP 404" in e.error for e in errors_in(up))
