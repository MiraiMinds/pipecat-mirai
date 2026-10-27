"""fourier_frequency.STTStream on its own, against the fake STT gateway (tests/fake_mirai_stt.py).

Nothing here imports Pipecat: this is how a team with its own orchestrator uses the engine.
"""

import asyncio
import logging
import subprocess
import sys

import numpy as np
import pytest
from fake_mirai_stt import FakeMiraiSTT, dominant_hz, ulaw2lin
from fourier_frequency import stt as stt_module
from fourier_frequency.errors import FrequencyError
from fourier_frequency.pool import close_shared_connections
from fourier_frequency.stt import (
    SpeechStarted,
    SpeechStopped,
    StreamClosed,
    StreamError,
    STTListener,
    STTStream,
    Transcript,
)

KEY = "sk_test"
RATE = 8000


@pytest.fixture(autouse=True)
async def _close_pools():
    yield
    await close_shared_connections()


@pytest.fixture
def logs():
    """What the stream logged during the test, at any level: (level, message)."""
    seen = []

    class Keep(logging.Handler):
        def emit(self, record):
            seen.append((record.levelname, record.getMessage()))

    log = logging.getLogger("fourier_frequency.stt")
    handler, level = Keep(logging.DEBUG), log.level
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    yield seen
    log.removeHandler(handler)
    log.setLevel(level)


# ---- audio ----


def tone(secs: float, hz: float, rate: int = RATE, amp: int = 8000) -> bytes:
    t = np.arange(int(round(secs * rate))) / rate
    return (np.sin(2 * np.pi * hz * t) * amp).astype("<i2").tobytes()


def quiet(secs: float, rate: int = RATE) -> bytes:
    return bytes(2 * int(round(secs * rate)))


def send(stt: STTStream, pcm: bytes, rate: int = RATE, ms: int = 20):
    """``pcm`` in 20 ms pieces, as a phone's media stream hands it over."""
    step = rate * ms // 1000 * 2
    for i in range(0, len(pcm), step):
        stt.send(pcm[i : i + step])


async def utterance(stt: STTStream, hz: int, rate: int = RATE, *, secs=0.6, lead=0.3, tail=0.4, vad_lag=0.2):
    """Silence, a tone (your VAD fires vad_lag into it), silence, your VAD's stop."""
    send(stt, quiet(lead, rate) + tone(vad_lag, hz, rate), rate)
    await stt.speech_started()
    send(stt, tone(secs - vad_lag, hz, rate) + quiet(tail, rate), rate)
    await stt.speech_stopped()


def stream(url, **kwargs) -> STTStream:
    return STTStream(api_key=KEY, url=url, **{"input_sample_rate": RATE, **kwargs})


async def connected(stt: STTStream):
    """Wait for the socket (audio sent before it is up goes as one replay, in 0.5 s pieces)."""
    for _ in range(100):
        if stt.connected_url:
            return
        await asyncio.sleep(0.02)
    raise TimeoutError("the stream never connected")


async def all_events(stt: STTStream) -> list:
    return [event async for event in stt.events()]


def finals(events) -> list[Transcript]:
    return [e for e in events if isinstance(e, Transcript) and e.final]


def errors_in(events) -> list[FrequencyError]:
    return [e.error for e in events if isinstance(e, StreamError)]


def hz_of(text: str) -> int:
    return int(text.split(" hertz")[0]) if text else 0


# ---- endpointing ------------------------------------------------------------------------


async def test_mirais_vad_ends_utterances_and_partials_and_finals_come_back():
    fake = FakeMiraiSTT()
    pcm = quiet(0.3) + tone(0.6, 440) + quiet(0.5) + tone(0.5, 990) + quiet(0.5)
    async with fake.serve() as url:
        async with stream(url, language="hi-IN", endpointing="vad") as stt:
            send(stt, pcm)
            await asyncio.sleep(0.3)
        events = await all_events(stt)
    assert not errors_in(events)
    assert [hz_of(f.text) for f in finals(events)] == [440, 990]
    assert any(isinstance(e, Transcript) and not e.final and "440 hertz" in e.text for e in events)
    first = finals(events)[0]
    assert first.language == "hi-IN" and first.latency_ms is not None and first.latency_ms >= 0
    assert first.event["session_id"] == "sttws_1" and first.event["request_id"] == "req_1"
    # Mirai's VAD says where each utterance began and ended, each final after its end.
    turns = [e for e in events if isinstance(e, SpeechStarted | SpeechStopped)]
    assert [type(e) for e in turns] == [SpeechStarted, SpeechStopped, SpeechStarted, SpeechStopped]
    stops = [events.index(e) for e in turns[1::2]]
    assert all(stop < events.index(final) for stop, final in zip(stops, finals(events), strict=True))
    assert isinstance(events[-1], StreamClosed)
    (conn,) = fake.conns
    assert conn.query == {
        "model": "mira-stt",
        "encoding": "linear16",
        "sample_rate": "8000",
        "endpointing": "vad",
        "language_code": "hi-IN",
    }
    assert conn.headers["authorization"] == f"Bearer {KEY}"
    assert bytes(conn.audio).startswith(pcm)  # every byte, continuously
    assert stt.endpointing == "vad" and stt.wire_sample_rate == 8000 and stt.connected_url == url


async def test_your_vad_ends_utterances_and_the_preroll_carries_the_start_of_the_word():
    fake = FakeMiraiSTT()
    lead, head, body, tail = quiet(1.0), tone(0.2, 440), tone(0.4, 440), quiet(0.3)
    async with fake.serve() as url:
        async with stream(url, endpointing="manual") as stt:
            await connected(stt)
            send(stt, lead + head)
            await stt.speech_started()
            send(stt, body + tail)
            await stt.speech_stopped()
            send(stt, quiet(1.0))
            await utterance(stt, 880)
        events = await all_events(stt)
    assert not errors_in(events)
    assert [hz_of(f.text) for f in finals(events)] == [440, 880]
    assert not any(isinstance(e, SpeechStarted | SpeechStopped) for e in events)  # those are Mirai's VAD's
    (conn,) = fake.conns
    assert conn.query["endpointing"] == "manual"
    assert [e["event"] for e in conn.received if e["event"] != "audio_input"] == [
        "speech_start",
        "speech_end",
        "speech_start",
        "speech_end",
        "end",
    ]
    # 8 kHz passes through untouched: the utterance is exactly the last 0.5 s
    # before speech_started, then everything up to speech_stopped.
    assert bytes(conn.utterances[0].audio) == (lead + head)[-int(0.5 * RATE) * 2 :] + body + tail
    assert max(e["bytes"] for e in conn.events("audio_input")) == 960  # 60 ms pieces


async def test_other_input_rates_are_resampled_to_16k():
    fake = FakeMiraiSTT()
    rate = 48000
    async with fake.serve() as url:
        async with stream(url, input_sample_rate=rate, endpointing="manual") as stt:
            await utterance(stt, 660, rate)
        events = await all_events(stt)
    assert stt.wire_sample_rate == 16000 and fake.conns[0].query["sample_rate"] == "16000"
    assert [hz_of(f.text) for f in finals(events)] == [660]


# ---- the wire ---------------------------------------------------------------------------


@pytest.mark.parametrize("input_rate", [8000, 16000])
async def test_mulaw_goes_on_the_wire_at_8k_a_byte_a_sample(input_rate):
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        async with stream(url, input_sample_rate=input_rate, encoding="mulaw", endpointing="manual") as stt:
            await connected(stt)
            await utterance(stt, 440, input_rate)
        events = await all_events(stt)
    assert not errors_in(events)
    (conn,) = fake.conns
    assert conn.query["encoding"] == "mulaw" and conn.query["sample_rate"] == "8000"
    assert stt.encoding == "mulaw" and stt.wire_sample_rate == 8000
    assert max(e["bytes"] for e in conn.events("audio_input")) == 480  # 60 ms, a byte a sample
    assert dominant_hz(ulaw2lin(bytes(conn.utterances[0].audio)), 8000) == 440
    assert [hz_of(f.text) for f in finals(events)] == [440]


async def test_settings_change_on_the_open_socket():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        async with stream(url, endpointing="manual") as stt:
            await utterance(stt, 440)
            await asyncio.sleep(0.3)
            with pytest.raises(ValueError, match="spells language detection 'auto'"):
                await stt.update(language="multi")
            assert await stt.update(language="gu", secondary_languages=["en-IN"]) == {
                "language",
                "secondary_languages",
            }
            assert stt.language == "gu-IN" and stt.secondary_languages == ("en-IN",)
            assert await stt.update(language="gu-IN") == set()  # nothing changed, nothing sent
            await asyncio.sleep(0.1)
            await utterance(stt, 880)
        events = await all_events(stt)
    (conn,) = fake.conns
    assert conn.events("config.update") == [
        {"event": "config.update", "language_code": "gu-IN", "secondary_languages": "en-IN"}
    ]
    assert [f.language for f in finals(events)] == ["hi-IN", "gu-IN"]


# ---- failures ---------------------------------------------------------------------------


async def test_a_dropped_socket_is_redialled_and_the_unanswered_audio_replayed(logs):
    fake = FakeMiraiSTT(drop_after_s=0.4)
    async with fake.serve() as url:
        async with stream(url, endpointing="manual") as stt:
            await utterance(stt, 440, secs=1.0)
            await asyncio.sleep(0.3)
            await utterance(stt, 880)
        events = await all_events(stt)
    assert not errors_in(events)
    assert [hz_of(f.text) for f in finals(events)] == [440, 880]
    assert len(fake.conns) == 2
    first = fake.conns[1].utterances[0]
    assert abs(len(first.audio) / 2 / RATE - (0.5 + 0.8 + 0.4)) < 0.03  # the whole utterance again
    assert any(level == "WARNING" and "connection lost" in m for level, m in logs)
    assert any("replaying" in m for _, m in logs)


async def test_a_refused_key_is_a_permanent_error_and_the_stream_stops_trying():
    fake = FakeMiraiSTT(auth=lambda headers: False)
    async with fake.serve() as url:
        async with stream(url, endpointing="manual") as stt:
            await utterance(stt, 440)
            await asyncio.sleep(0.3)
            await utterance(stt, 880)
            await asyncio.sleep(0.1)
        events = await all_events(stt)
    (error,) = errors_in(events)
    assert isinstance(error, FrequencyError) and error.permanent
    assert (error.category, error.status, error.code) == ("authentication", 401, "unauthorized")
    assert (
        "HTTP 401" in str(error) and "invalid API key" in str(error) and "Check MIRAI_API_KEY" in str(error)
    )
    assert stt.last_error == str(error)
    assert len(fake.handshakes) == 1  # no point in another socket
    assert finals(events) == [] and isinstance(events[-1], StreamClosed)


async def test_a_server_at_capacity_is_reported_but_the_next_turn_tries_again(monkeypatch):
    monkeypatch.setattr(stt_module, "CONNECT_RETRY_SECS", 0.5)
    fake = FakeMiraiSTT(refuse_status=429)
    async with fake.serve() as url:
        async with stream(url, endpointing="manual") as stt:
            await utterance(stt, 440)
            await asyncio.sleep(1.0)
        events = await all_events(stt)
    (error,) = errors_in(events)
    assert (error.category, error.status, error.permanent) == ("rate_limit", 429, False)
    assert "HTTP 429" in str(error) and "not transcribed" in str(error)


async def test_a_fatal_server_error_names_the_code_and_the_session():
    fake = FakeMiraiSTT(fatal_error={"code": "insufficient_balance", "message": "wallet empty"})
    async with fake.serve() as url:
        async with stream(url, endpointing="manual") as stt:
            await utterance(stt, 440)
            await asyncio.sleep(0.3)
            await utterance(stt, 880)
        events = await all_events(stt)
    (error,) = errors_in(events)
    assert (error.category, error.code, error.request_id, error.permanent) == (
        "quota",
        "insufficient_balance",
        "req_1",
        True,
    )
    assert "wallet empty" in str(error) and "Top it up" in str(error) and "sttws_1" in str(error)
    assert len(fake.handshakes) == 1 and finals(events) == []


# ---- closing ----------------------------------------------------------------------------


async def test_a_graceful_close_ends_the_open_utterance_and_waits_for_its_final_and_the_bill():
    fake = FakeMiraiSTT(slow_final_s=0.2)
    async with fake.serve() as url:
        stt = stream(url, endpointing="manual")
        await stt.connect()
        send(stt, quiet(0.3) + tone(0.2, 440))
        await stt.speech_started()
        send(stt, tone(0.6, 440))
        await stt.close()  # no speech_stopped: the close ends the utterance
        events = await all_events(stt)
    assert [hz_of(f.text) for f in finals(events)] == [440] and isinstance(events[-1], StreamClosed)
    (conn,) = fake.conns
    assert [e["event"] for e in conn.received if e["event"] != "audio_input"] == [
        "speech_start",
        "speech_end",
        "end",
    ]
    assert stt.audio_seconds_billed >= 1 and stt.cost_paise == 1
    assert abs(stt.audio_seconds_sent - conn.audio_secs) < 0.01
    assert stt.last_final_latency_ms is not None and stt.last_final_latency_ms >= 190
    await stt.close()  # twice: nothing more happens
    assert await all_events(stt) == []


async def test_a_vad_stream_closed_mid_utterance_still_gets_its_final():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        async with stream(url, endpointing="vad") as stt:
            send(stt, quiet(0.2) + tone(0.5, 550))  # still speaking when the call ends
        events = await all_events(stt)
    assert [hz_of(f.text) for f in finals(events)] == [550]
    assert fake.conns[0].events("end") and stt.audio_seconds_billed >= 1


async def test_close_without_grace_drops_the_open_utterance():
    fake = FakeMiraiSTT()
    async with fake.serve() as url:
        stt = stream(url, endpointing="manual")
        await stt.connect()
        await stt.speech_started()
        send(stt, tone(0.5, 440))
        await asyncio.sleep(0.3)
        await stt.close(graceful=False)
        events = await all_events(stt)
        await asyncio.sleep(0.1)
    assert finals(events) == [] and events[-1] == StreamClosed()
    received = [e["event"] for e in fake.conns[0].received if e["event"] != "audio_input"]
    assert received == ["speech_start"]
    send(stt, tone(0.1, 440))  # dropped: the stream is closed
    with pytest.raises(RuntimeError):
        await stt.connect()


# ---- a listener of your own -------------------------------------------------------------


class Recorder(STTListener):
    def __init__(self):
        self.calls: list[tuple] = []

    async def on_connected(self):
        self.calls.append(("connected",))

    async def on_disconnected(self):
        self.calls.append(("disconnected",))

    async def on_partial(self, text, language, event):
        self.calls.append(("partial", text, language))

    async def on_final_latency(self, started, received):
        assert received >= started
        self.calls.append(("latency",))

    async def on_usage(self, audio_seconds):
        self.calls.append(("usage", audio_seconds))

    async def on_final(self, text, language, event):
        self.calls.append(("final", text, language))

    async def on_closed(self):
        self.calls.append(("closed",))


async def test_a_listener_of_your_own_is_told_everything_in_order():
    fake = FakeMiraiSTT()
    recorder = Recorder()
    async with fake.serve() as url:
        async with stream(url, endpointing="manual", language="ta", listener=recorder, name="call-42") as stt:
            await utterance(stt, 440)
        with pytest.raises(RuntimeError):
            await all_events(stt)
    names = [call[0] for call in recorder.calls]
    assert names[0] == "connected" and names[-2:] == ["disconnected", "closed"]
    at = names.index("final")
    assert names[at - 2 : at + 1] == ["latency", "usage", "final"]
    assert recorder.calls[at][1:] == ("440 hertz, 1.3 seconds", "ta-IN")
    assert recorder.calls[at - 1][1] > 0 and str(stt) == "call-42"


async def test_auto_names_each_transcript_by_the_language_heard_or_none():
    fake = FakeMiraiSTT(heard={440: "gu-IN"}, language_field="language")
    async with fake.serve() as url:
        async with stream(url, endpointing="manual", language="auto") as stt:
            await utterance(stt, 440)
        events = await all_events(stt)
    assert fake.conns[0].query["language_code"] == "auto"
    assert [f.language for f in finals(events)] == ["gu-IN"]
    fake = FakeMiraiSTT(language_field="unused")
    async with fake.serve() as url:
        async with stream(url, endpointing="manual", language="auto") as stt:
            await utterance(stt, 440)
        events = await all_events(stt)
    assert [f.language for f in finals(events)] == [None]


# ---- options ----------------------------------------------------------------------------


def test_the_api_key_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MIRAI_API_KEY"):
        STTStream()
    monkeypatch.setenv("MIRA_API_KEY", "sk_env")
    stt = STTStream()
    assert stt._headers == {"Authorization": "Bearer sk_env"}
    assert stt.endpointing == "vad" and stt.language == "hi-IN"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"url": "https://example/v1/audio/transcriptions/stream"},
        {"url": "wss://example/v1/audio/transcriptions/stream?language_code=hi-IN"},
        {"sample_rate": 44100},
        {"encoding": "opus"},
        {"encoding": "mulaw", "sample_rate": 16000},
        {"endpointing": "auto"},  # a host decides "auto" itself
        {"edge": "yes"},
        {"input_sample_rate": 0},
        {"language": "en-US"},
        {"language": "auto", "secondary_languages": ["en-IN"]},
    ],
)
def test_invalid_options_are_refused(kwargs):
    with pytest.raises(ValueError):
        STTStream(api_key=KEY, **kwargs)


def test_the_core_imports_without_pipecat():
    code = (
        "import sys, fourier_frequency.stt\n"
        "hosts = ('pipecat', 'pipecat_mirai', 'loguru')\n"
        "found = sorted(m for m in sys.modules if m.split('.')[0] in hosts)\n"
        "sys.exit(f'imported {found}' if found else 0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
