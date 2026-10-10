"""MiraiSTTService against a fake of Mirai's streaming STT gateway (tests/fake_mirai_stt.py)."""

import asyncio
import json

import numpy as np
import pytest
from fake_mirai_stt import FakeMiraiSTT, alaw2lin, dominant_hz, ulaw2lin
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    STTMetadataFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import ProcessingMetricsData, STTUsageMetricsData, TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies

from pipecat_mirai import MiraiSTTService, MiraiSTTSettings, close_shared_connections, prewarm
from pipecat_mirai import stt as stt_module
from pipecat_mirai.stt import lin2alaw, lin2ulaw

KEY = "sk_test"


@pytest.fixture(autouse=True)
async def _close_pools():
    yield
    await close_shared_connections()


@pytest.fixture
def logs():
    seen = []
    sink = logger.add(lambda m: seen.append((m.record["level"].name, m.record["message"])))
    yield seen
    logger.remove(sink)


# ---- audio ----


def tone(secs: float, rate: int, hz: float, amp: int = 8000) -> bytes:
    t = np.arange(int(round(secs * rate))) / rate
    return (np.sin(2 * np.pi * hz * t) * amp).astype("<i2").tobytes()


def quiet(secs: float, rate: int) -> bytes:
    return bytes(2 * int(round(secs * rate)))


def frames(pcm: bytes, rate: int, ms: int = 20) -> list[InputAudioRawFrame]:
    step = rate * ms // 1000 * 2
    return [
        InputAudioRawFrame(audio=pcm[i : i + step], sample_rate=rate, num_channels=1)
        for i in range(0, len(pcm), step)
    ]


def utterance(hz: int, rate: int = 8000, *, secs=0.6, lead=0.3, tail=0.4, vad_lag=0.2) -> list:
    """Silence, a tone (the pipeline's VAD fires vad_lag into it), silence, the VAD's stop."""
    return [
        *frames(quiet(lead, rate), rate),
        *frames(tone(vad_lag, rate, hz), rate),
        VADUserStartedSpeakingFrame(),
        *frames(tone(secs - vad_lag, rate, hz), rate),
        *frames(quiet(tail, rate), rate),
        VADUserStoppedSpeakingFrame(stop_secs=tail),
    ]


def stt(url, **kwargs) -> MiraiSTTService:
    return MiraiSTTService(api_key=KEY, url=url, **kwargs)


async def call(service, frames_to_send, rate=8000, metrics=False):
    return await run_test(
        service,
        frames_to_send=frames_to_send,
        pipeline_params=PipelineParams(
            audio_in_sample_rate=rate, enable_metrics=metrics, enable_usage_metrics=metrics
        ),
        start_timeout=10.0,
    )


def finals(down) -> list[TranscriptionFrame]:
    return [f for f in down if type(f) is TranscriptionFrame]


def texts(down) -> list[str]:
    return [f.text for f in finals(down)]


def errors_in(up) -> list[str]:
    return [f.error for f in up if isinstance(f, ErrorFrame)]


def metrics_of(down, kind) -> list:
    return [d for f in down if isinstance(f, MetricsFrame) for d in f.data if isinstance(d, kind)]


def hz_of(text: str) -> int:
    return int(text.split(" hertz")[0]) if text else 0


# ---- the basics -------------------------------------------------------------------------


async def test_manual_endpointing_sends_each_utterance_and_gets_one_final_each():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, [SleepFrame(0.3), *utterance(440), *utterance(880)])
    assert not errors_in(up)
    assert [hz_of(t) for t in texts(down)] == [440, 880]
    assert all(f.finalized for f in finals(down))
    assert any(isinstance(f, InterimTranscriptionFrame) and "440 hertz" in f.text for f in down)
    (conn,) = fake.conns
    assert [e["event"] for e in conn.received if e["event"] != "audio_input"] == [
        "speech_start",
        "speech_end",
        "speech_start",
        "speech_end",
        "end",
    ]
    assert conn.query == {
        "model": "mira-stt",
        "encoding": "linear16",
        "sample_rate": "8000",
        "endpointing": "manual",
        "language_code": "hi-IN",
    }
    assert conn.headers["authorization"] == f"Bearer {KEY}"
    # Audio goes in pieces of 60 ms (the last of an utterance may be shorter).
    sizes = [e["bytes"] for e in conn.received if e["event"] == "audio_input"]
    assert max(sizes) == 960
    assert service.session_id == "sttws_1" and service.connected_url == url
    assert service.endpointing == "manual" and service.wire_sample_rate == 8000


async def test_the_preroll_carries_the_start_of_the_word_and_nothing_between_turns_is_sent():
    fake = FakeMiraiSTT()
    rate = 8000
    lead = quiet(1.0, rate)
    head = tone(0.2, rate, 440)
    body = tone(0.4, rate, 440)
    tail = quiet(0.3, rate)
    async with fake.serve() as url:
        await call(
            stt(url, endpointing="manual"),
            [
                *frames(lead, rate),
                *frames(head, rate),
                VADUserStartedSpeakingFrame(),
                *frames(body, rate),
                *frames(tail, rate),
                VADUserStoppedSpeakingFrame(),
                *frames(quiet(1.0, rate), rate),
            ],
        )
    (conn,) = fake.conns
    (utt,) = conn.utterances
    # 8 kHz passes through untouched: the utterance is exactly the last 0.5 s
    # before the VAD fired, then everything up to its stop.
    expected = (lead + head)[-int(0.5 * rate) * 2 :] + body + tail
    assert bytes(utt.audio) == expected
    assert bytes(conn.audio) == expected  # nothing outside the utterance


@pytest.mark.parametrize(
    "pipeline_rate, option, wire",
    [(16000, "auto", 16000), (48000, "auto", 16000), (24000, "auto", 16000), (16000, 8000, 8000)],
)
async def test_other_rates_are_resampled(pipeline_rate, option, wire):
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        service = stt(url, endpointing="manual", sample_rate=option)
        down, up = await call(service, utterance(660, pipeline_rate), rate=pipeline_rate)
    assert not errors_in(up)
    (conn,) = fake.conns
    assert conn.query["sample_rate"] == str(wire) and service.wire_sample_rate == wire
    (utt,) = conn.utterances
    seconds = len(utt.audio) / 2 / wire
    assert (
        abs(seconds - (0.5 + 0.4 + 0.4)) < 0.08
    )  # pre-roll + rest of the tone + tail (less the filter's delay)
    assert [hz_of(t) for t in texts(down)] == [660]


async def test_16k_pipeline_passes_through_unchanged():
    fake = FakeMiraiSTT()
    rate = 16000
    pcm = quiet(0.3, rate) + tone(0.6, rate, 500) + quiet(0.4, rate)
    async with fake.serve() as url:
        await call(stt(url, endpointing="vad"), frames(pcm, rate), rate=rate)
    (conn,) = fake.conns
    assert bytes(conn.audio) == pcm


async def test_server_endpointing_streams_everything_and_proposes_turns():
    fake = FakeMiraiSTT()
    rate = 8000
    pcm = quiet(0.3, rate) + tone(0.5, rate, 440) + quiet(0.5, rate) + tone(0.5, rate, 990) + quiet(0.5, rate)
    async with fake.serve() as url:
        service = stt(url)  # auto: the test pipeline has no VAD
        down, up = await call(service, [*frames(pcm, rate), SleepFrame(0.3)], rate=rate)
    assert not errors_in(up)
    assert service.endpointing == "vad"
    (conn,) = fake.conns
    assert conn.query["endpointing"] == "vad"
    assert bytes(conn.audio).startswith(pcm)  # every byte, continuously (then keepalive silence, if any)
    assert [hz_of(t) for t in texts(down)] == [440, 990]
    assert sum(isinstance(f, ProposedUserStartedSpeakingFrame) for f in down) == 2
    assert sum(isinstance(f, ProposedUserStoppedSpeakingFrame) for f in down) == 2
    (meta,) = [f for f in down if isinstance(f, STTMetadataFrame)]
    assert isinstance(meta.user_turn_strategies, ExternalUserTurnStrategies)


async def test_manual_endpointing_leaves_the_turn_strategies_alone():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        down, _ = await call(stt(url, endpointing="manual"), utterance(440))
    (meta,) = [f for f in down if isinstance(f, STTMetadataFrame)]
    assert meta.user_turn_strategies is None and meta.ttfs_p99_latency == stt_module.DEFAULT_TTFS_P99


async def test_an_empty_final_still_resolves_its_utterance():
    fake = FakeMiraiSTT()
    rate = 8000
    silent_turn = [
        *frames(quiet(0.3, rate), rate),
        VADUserStartedSpeakingFrame(),
        *frames(quiet(0.4, rate), rate),
        VADUserStoppedSpeakingFrame(),
    ]
    async with fake.serve() as url:
        down, up = await call(stt(url, endpointing="manual"), [*silent_turn, *utterance(770)])
    assert not errors_in(up)
    assert texts(down)[0] == "" and hz_of(texts(down)[1]) == 770
    assert all(f.finalized for f in finals(down))


# ---- metrics ----------------------------------------------------------------------------


async def test_ttfb_is_speech_end_to_final_and_usage_is_audio_seconds_sent():
    fake = FakeMiraiSTT(slow_final_s=0.15)
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, [*utterance(440), SleepFrame(0.4), *utterance(550)], metrics=True)
    assert not errors_in(up)
    ttfb = [d.value for d in metrics_of(down, TTFBMetricsData) if d.value > 0]
    assert len(ttfb) == 2 and all(0.14 <= v < 0.5 for v in ttfb), ttfb
    processing = [d.value for d in metrics_of(down, ProcessingMetricsData) if d.value > 0]
    assert len(processing) == 2 and all(0.14 <= v < 0.5 for v in processing), processing
    usage = sum(d.value.audio_seconds for d in metrics_of(down, STTUsageMetricsData))
    (conn,) = fake.conns
    assert abs(usage - conn.audio_secs) < 0.01 and abs(service.audio_seconds_sent - conn.audio_secs) < 0.01
    assert 0.14 <= service.last_final_latency_ms / 1000 < 0.5
    assert all(f.result["latency_ms"] >= 140 for f in finals(down))


# ---- endpointing auto -------------------------------------------------------------------


class _VADHolder(FrameProcessor):
    """Stands in for a user aggregator that runs a VAD (``_params.vad_analyzer``)."""

    class _Params:
        vad_analyzer = object()

    def __init__(self):
        super().__init__()
        self._params = self._Params()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


def test_auto_endpointing_finds_a_vad_linked_to_the_service():
    service = stt("ws://127.0.0.1:1/v1/audio/transcriptions/stream")
    Pipeline([Pipeline([FrameProcessor(), service]), FrameProcessor(), _VADHolder()])
    assert service._pipeline_has_vad()
    lone = stt("ws://127.0.0.1:1/v1/audio/transcriptions/stream")
    Pipeline([FrameProcessor(), lone, FrameProcessor()])
    assert not lone._pipeline_has_vad()


async def test_auto_endpointing_is_manual_in_a_pipeline_with_a_vad():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        service = stt(url)
        down, up = await call(Pipeline([service, _VADHolder()]), utterance(440))
    assert service.endpointing == "manual" and fake.conns[0].query["endpointing"] == "manual"
    assert [hz_of(t) for t in texts(down)] == [440]


# ---- failures ---------------------------------------------------------------------------


async def test_a_dropped_socket_is_redialled_and_the_unanswered_audio_replayed(logs):
    fake = FakeMiraiSTT(drop_after_s=0.4)
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, [*utterance(440, secs=1.0), SleepFrame(0.3), *utterance(880)])
    assert not errors_in(up)
    assert [hz_of(t) for t in texts(down)] == [440, 880]
    assert len(fake.conns) == 2
    first = fake.conns[1].utterances[0]
    assert abs(len(first.audio) / 2 / 8000 - (0.5 + 0.8 + 0.4)) < 0.03  # the whole utterance again
    assert any("connection lost" in m for level, m in logs if level == "WARNING")
    assert any("replaying" in m for _, m in logs)


async def test_a_drop_under_server_endpointing_replays_too():
    fake = FakeMiraiSTT(drop_after_s=0.5)
    rate = 8000
    pcm = quiet(0.2, rate) + tone(0.8, rate, 440) + quiet(0.5, rate)
    async with fake.serve() as url:
        down, up = await call(stt(url, endpointing="vad"), [*frames(pcm, rate), SleepFrame(0.3)], rate=rate)
    assert not errors_in(up)
    assert [hz_of(t) for t in texts(down)] == [440]
    assert len(fake.conns) == 2


async def test_when_the_redial_fails_too_the_call_gets_an_error_frame():
    fake = FakeMiraiSTT(drop_after_s=0.4, refuse_attempts={2})
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, utterance(440, secs=1.0))
    (error,) = errors_in(up)
    assert "HTTP 503" in error and "not transcribed" in error
    assert texts(down) == []


async def test_a_fatal_server_error_is_an_error_frame_and_ends_the_service():
    fake = FakeMiraiSTT(fatal_error={"code": "insufficient_balance", "message": "wallet empty"})
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, [*utterance(440), SleepFrame(0.3), *utterance(880)])
    (error,) = errors_in(up)
    assert "insufficient_balance" in error and "wallet empty" in error and "Top it up" in error
    assert len(fake.handshakes) == 1  # no point in another socket
    assert texts(down) == []


async def test_a_refused_key_is_reported_once():
    fake = FakeMiraiSTT(auth=lambda headers: False)
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, [*utterance(440), SleepFrame(0.2), *utterance(880)])
    (error,) = errors_in(up)
    assert "HTTP 401" in error and "invalid API key" in error and "Check MIRAI_API_KEY" in error
    assert len(fake.handshakes) == 1
    assert not service.is_usable


async def test_429_is_retried_before_it_is_reported():
    fake = FakeMiraiSTT(rate_limit_attempts={1, 2})
    async with fake.serve() as url:
        down, up = await call(stt(url, endpointing="manual"), utterance(440))
    assert not errors_in(up)
    assert len(fake.handshakes) == 3 and [hz_of(t) for t in texts(down)] == [440]


async def test_a_server_at_capacity_for_long_is_reported(monkeypatch):
    monkeypatch.setattr(stt_module, "CONNECT_RETRY_SECS", 0.5)
    fake = FakeMiraiSTT(refuse_status=429)
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        _, up = await call(service, utterance(440))
    (error,) = errors_in(up)
    assert "HTTP 429" in error
    assert service.is_usable  # crowding passes: the next turn tries again


async def test_a_missing_final_does_not_hang_the_call():
    fake = FakeMiraiSTT(no_final=True)
    async with fake.serve() as url:
        down, up = await call(stt(url, endpointing="manual"), utterance(440))
    assert texts(down) == [] and not errors_in(up)


# ---- the socket's life ------------------------------------------------------------------


async def test_a_quiet_socket_gets_a_ping_and_a_little_silence(monkeypatch):
    monkeypatch.setattr(stt_module, "KEEPALIVE_SECS", 0.2)
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        await call(stt(url, endpointing="manual"), [SleepFrame(0.9)])
    (conn,) = fake.conns
    pings = conn.events("ping")
    assert 3 <= len(pings) <= 5
    assert {e["bytes"] for e in conn.events("audio_input")} == {int(8000 * 0.04) * 2}


async def test_an_old_session_is_replaced_at_a_pause():
    fake = FakeMiraiSTT(session_limits={"max_session_secs": 1.2, "idle_timeout_secs": 120})
    async with fake.serve() as url:
        service = stt(url, endpointing="manual")
        down, up = await call(service, [*utterance(440), SleepFrame(1.0), *utterance(880)])
    assert not errors_in(up)
    assert [hz_of(t) for t in texts(down)] == [440, 880]
    assert len(fake.conns) == 2
    assert [len(c.utterances) for c in fake.conns] == [1, 1]  # the second utterance went to the new session
    assert service.session_id == "sttws_2"


async def test_a_draining_session_hands_an_open_utterance_to_a_new_one():
    fake = FakeMiraiSTT(drain_after_s=0.3, drain_grace_s=2.0)
    rate = 8000
    long_turn = [
        *frames(quiet(0.2, rate), rate),
        VADUserStartedSpeakingFrame(),
        *frames(tone(0.3, rate, 440), rate),
        SleepFrame(0.6),  # Mirai asks to end the session while the caller speaks
        *frames(tone(0.3, rate, 440), rate),
        *frames(quiet(0.3, rate), rate),
        VADUserStoppedSpeakingFrame(),
    ]
    async with fake.serve() as url:
        down, up = await call(stt(url, endpointing="manual"), long_turn)
    assert not errors_in(up)
    assert [hz_of(t) for t in texts(down)] == [440]
    assert len(fake.conns) == 2 and len(fake.conns[1].utterances) == 1


async def test_language_changes_on_the_open_socket():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        down, _ = await call(
            stt(url, endpointing="manual"),
            [
                *utterance(440),
                SleepFrame(0.3),
                STTUpdateSettingsFrame(delta=MiraiSTTSettings(language="gu-IN")),
                SleepFrame(0.1),
                *utterance(880),
            ],
        )
    (conn,) = fake.conns
    assert conn.events("config.update") == [{"event": "config.update", "language_code": "gu-IN"}]
    assert [f.language.value for f in finals(down)] == ["hi-IN", "gu-IN"]


@pytest.mark.parametrize("encoding, decode", [("mulaw", ulaw2lin), ("alaw", alaw2lin)])
async def test_g711_encodings(encoding, decode):
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        down, up = await call(
            stt(url, endpointing="manual", encoding=encoding), [SleepFrame(0.3), *utterance(440)]
        )
    assert not errors_in(up)
    (conn,) = fake.conns
    assert conn.query["encoding"] == encoding and conn.query["sample_rate"] == "8000"
    assert [hz_of(t) for t in texts(down)] == [440]
    assert max(e["bytes"] for e in conn.events("audio_input")) == 480  # 60 ms, a byte a sample


def test_g711_encoders_match_the_standard():
    pcm = np.array([0, 1, -1, 100, -100, 1000, -1000, 8000, -8000, 32767, -32768], dtype=np.int16)
    assert lin2ulaw(np.array([0], dtype=np.int16)).tobytes() == b"\xff"
    assert lin2alaw(np.array([0], dtype=np.int16)).tobytes() == b"\xd5"
    for encode, decode in ((lin2ulaw, ulaw2lin), (lin2alaw, alaw2lin)):
        back = decode(encode(pcm).tobytes()).astype(np.int32)
        assert np.all(np.abs(back - pcm) <= np.maximum(16, np.abs(pcm.astype(np.int32)) // 16)), (
            encode,
            back,
        )
    t = np.arange(8000) / 8000
    wave = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    assert dominant_hz(ulaw2lin(lin2ulaw(wave).tobytes()), 8000) == 440


# ---- the pool ---------------------------------------------------------------------------


async def test_a_waiting_socket_is_taken_and_moved_to_the_pipelines_language():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        result = await prewarm(api_key=KEY, connections=0, websocket=0, stt_websockets=1, stt_url=url)
        assert (result.stt_websockets, result.errors) == (1, [])
        (waiting,) = fake.conns
        assert waiting.query["language_code"] == "hi-IN" and waiting.query["endpointing"] == "manual"
        service = stt(url, endpointing="manual", language="gu-IN")
        down, up = await call(service, utterance(440))
        assert not errors_in(up)
        assert [hz_of(t) for t in texts(down)] == [440]
        assert waiting.events("config.update") == [{"event": "config.update", "language_code": "gu-IN"}]
        assert waiting.utterances and service.session_id == "sttws_1"
        await asyncio.sleep(0.3)
    assert len(fake.conns) == 2  # the pool opened a replacement


async def test_a_waiting_socket_in_the_same_language_needs_no_update():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=0, stt_websockets=1, stt_url=url)
        down, _ = await call(stt(url, endpointing="manual"), utterance(440))
    assert fake.conns[0].events("config.update") == [] and fake.conns[0].utterances
    assert [hz_of(t) for t in texts(down)] == [440]


async def test_waiting_sockets_only_go_to_pipelines_with_the_same_fixed_settings():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=0, stt_websockets=1, stt_url=url)  # 8 kHz, manual
        await call(stt(url, endpointing="manual"), utterance(440, 16000), rate=16000)
    assert not fake.conns[0].utterances and fake.conns[-1].query["sample_rate"] == "16000"


async def test_waiting_stt_sockets_get_keepalives(monkeypatch):
    from pipecat_mirai import pool as pool_module

    monkeypatch.setattr(pool_module, "STT_KEEPALIVE_SECS", 0.2)
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        await prewarm(api_key=KEY, connections=0, websocket=0, stt_websockets=1, stt_url=url)
        await asyncio.sleep(0.75)
        (conn,) = fake.conns
        assert 3 <= len(conn.events("ping")) <= 4
        assert all(e["bytes"] == 640 for e in conn.events("audio_input"))  # 40 ms at 8 kHz


# ---- options ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "https://example/v1/audio/transcriptions/stream"},
        {"url": "wss://example/v1/audio/transcriptions/stream?language_code=hi-IN"},
        {"sample_rate": 44100},
        {"encoding": "opus"},
        {"encoding": "mulaw", "sample_rate": 16000},
        {"endpointing": "server"},
        {"edge": "yes"},
    ],
)
def test_invalid_options_are_refused(kwargs):
    with pytest.raises(ValueError):
        MiraiSTTService(api_key=KEY, **kwargs)


def test_an_api_key_is_required(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MiraiSTTService()


def test_languages_are_mirais_codes():
    from pipecat.transcriptions.language import Language

    service = MiraiSTTService(api_key=KEY, language=Language.GU)
    assert service._settings.language == "gu-IN"
    assert MiraiSTTService(api_key=KEY)._settings.language == "hi-IN"
    assert (
        MiraiSTTService(api_key=KEY, settings=MiraiSTTSettings(language="ta-IN"))._settings.language
        == "ta-IN"
    )


def test_the_key_is_never_in_the_url():
    service = MiraiSTTService(api_key=KEY)
    service._wire_rate, service._endpointing = 8000, "manual"
    service._build_urls()
    assert KEY not in service._url and KEY not in service._pool_url
    assert json.dumps(stt_module.stt_query(service._url)).find(KEY) == -1
