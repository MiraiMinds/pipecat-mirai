import asyncio
import time

import numpy as np
import pytest
import soxr
from fake_mirai_ws import SR, FakeMiraiWS, tone
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData, TTSUsageMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.tts_service import TextAggregationMode
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_mirai import MiraiWebsocketTTSService

TEXT = "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?"
FRAME_MS = 40


class Recorder(FrameProcessor):
    """Passes frames on, noting when each one went by."""

    def __init__(self):
        super().__init__()
        self.seen = []  # (time.monotonic(), frame)

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        self.seen.append((time.monotonic(), frame))
        await self.push_frame(frame, direction)

    @property
    def audio(self):
        return [(t, f) for t, f in self.seen if isinstance(f, TTSAudioRawFrame)]

    def of(self, kind):
        return [f for _, f in self.seen if isinstance(f, kind)]


async def run(tts, rate, frames, *, metrics=False):
    rec = Recorder()
    down, up = await run_test(
        Pipeline([tts, rec]),
        frames_to_send=frames,
        pipeline_params=PipelineParams(
            audio_out_sample_rate=rate, enable_metrics=metrics, enable_usage_metrics=metrics
        ),
        start_timeout=10.0,
    )
    return rec, down, up


def ws_tts(url, **kwargs):
    return MiraiWebsocketTTSService(api_key="sk_test", url=url, **kwargs)


def audio_of(frames, rate):
    out = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert out, "no audio frames"
    assert all(f.sample_rate == rate and f.num_channels == 1 for f in out)
    return b"".join(f.audio for f in out)


def errors_in(frames):
    return [f for f in frames if isinstance(f, ErrorFrame)]


def llm_turn(*tokens):
    return [LLMFullResponseStartFrame(), *(TextFrame(t) for t in tokens), LLMFullResponseEndFrame()]


# --- session and sample rate ---------------------------------------------------------------


@pytest.mark.parametrize("rate", [8000, 16000, 24000, 48000])
async def test_asks_for_the_pipeline_rate_and_passes_audio_through(rate):
    fake = FakeMiraiWS(reads=(1, 4097, 3, 9600))
    async with fake.serve() as url:
        tts = ws_tts(url)
        rec, down, up = await run(tts, rate, [TTSSpeakFrame(TEXT)])
    (conn,) = fake.conns
    assert conn.headers["authorization"] == "Bearer sk_test"
    assert conn.received[0] == {
        "type": "session.update",
        "voice": "neha",
        "model": "mira-tts",
        "response_format": "pcm",
        "audio_transport": "binary",
        "sample_rate": rate,
    }
    assert fake.messages("text") == [
        {"type": "text", "context_id": fake.messages("text")[0]["context_id"], "text": TEXT, "flush": True}
    ]
    # Mirai sent the pipeline's rate: not resampled, byte-exact despite odd-sized frames.
    assert audio_of(down, rate) == tone(0.5, rate)
    assert tts.last_server_sample_rate == rate
    assert tts.session_id == "ttsws_1"
    assert len(rec.of(TTSStartedFrame)) == 1 and len(rec.of(TTSStoppedFrame)) == 1
    assert not errors_in(up)


async def test_server_that_sends_another_rate_is_resampled():
    # A server that ignores sample_rate: audio.start says 48 kHz.
    fake = FakeMiraiWS(seconds=1.0, honours_rate=False)
    async with fake.serve() as url:
        tts = ws_tts(url)
        _, down, up = await run(tts, 8000, [TTSSpeakFrame(TEXT)])
    assert fake.messages("session.update")[0]["sample_rate"] == 8000
    assert tts.last_server_sample_rate == SR
    got = np.frombuffer(audio_of(down, 8000), dtype="<i2").astype(int)
    want = soxr.resample(np.frombuffer(tone(1.0), dtype="<i2"), SR, 8000, quality="HQ").astype(int)
    assert len(got) == len(want) == 8000
    assert np.abs(got - want).max() <= 4
    assert not errors_in(up)


@pytest.mark.parametrize("rate, option", [(22050, "auto"), (8000, None)])
async def test_rate_left_out_means_48k_resampled(rate, option):
    fake = FakeMiraiWS(seconds=1.0)
    async with fake.serve() as url:
        tts = ws_tts(url, server_sample_rate=option)
        _, down, _ = await run(tts, rate, [TTSSpeakFrame(TEXT)])
    assert "sample_rate" not in fake.messages("session.update")[0]
    assert tts.last_server_sample_rate == SR
    assert abs(len(audio_of(down, rate)) // 2 - rate) <= 1


async def test_explicit_server_sample_rate():
    fake = FakeMiraiWS(seconds=1.0)
    async with fake.serve() as url:
        tts = ws_tts(url, server_sample_rate=16000)
        _, down, _ = await run(tts, 8000, [TTSSpeakFrame(TEXT)])
    assert fake.messages("session.update")[0]["sample_rate"] == 16000
    assert tts.last_server_sample_rate == 16000
    assert len(audio_of(down, 8000)) // 2 == 8000


@pytest.mark.parametrize(
    "bad",
    [
        {"server_sample_rate": 22050},
        {"server_sample_rate": True},
        {"prebuffer_secs": -1},
        {"keepalive_secs": 0},
        {"url": "https://sandbox.voice.miraiminds.co/v1/audio/speech/stream"},
    ],
)
def test_invalid_options_are_refused(bad):
    with pytest.raises(ValueError):
        MiraiWebsocketTTSService(api_key="k", **bad)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MiraiWebsocketTTSService()


# --- streaming text ------------------------------------------------------------------------

SENTENCES = ["आपका order कल deliver होगा।", "क्या मैं कुछ और बता सकती हूँ?", "धन्यवाद, आपका दिन शुभ हो।"]
TOKENS = [
    "आपका order",
    " कल deliver",
    " होगा। क्या",
    " मैं कुछ और",
    " बता सकती हूँ? ",
    "धन्यवाद,",
    " आपका दिन शुभ हो।",
]


async def test_sentences_stream_on_one_context_and_socket():
    # Bursts several times faster than real time, sentences pipelined.
    fake = FakeMiraiWS(seconds=lambda s: 0.6 if s in SENTENCES else 0.5, reads=(1000, 333), first_byte=0.05)
    async with fake.serve() as url:
        tts = ws_tts(url)
        rec, down, up = await run(tts, 16000, [*llm_turn(*TOKENS), TTSSpeakFrame("हाँ जी")])
    assert len(fake.conns) == 1  # one socket for everything
    texts = fake.messages("text")
    # Default (sentence) aggregation: one message per sentence, flushed, one
    # Mirai context for the turn; then the TTSSpeakFrame on its own context.
    assert [m["text"].strip() for m in texts] == [*SENTENCES, "हाँ जी"]
    assert all(m["flush"] for m in texts)
    turn_ids = {m["context_id"] for m in texts[:3]}
    assert len(turn_ids) == 1 and texts[3]["context_id"] not in turn_ids
    assert [s for _, _, s in fake.spoken] == [*SENTENCES, "हाँ जी"]
    want = b"".join(tone(0.6, 16000) for _ in SENTENCES) + tone(0.5, 16000)
    assert audio_of(down, 16000) == want
    assert len(rec.of(TTSStartedFrame)) == 2 and len(rec.of(TTSStoppedFrame)) == 2
    spoken_text = " ".join(f.text for f in rec.of(TTSTextFrame))
    assert all(s in spoken_text for s in SENTENCES)
    assert not errors_in(up)


async def test_token_mode_streams_tokens_and_flushes_at_the_end():
    fake = FakeMiraiWS(seconds={s: 0.4 for s in SENTENCES}, billing=lambda s: 11)
    async with fake.serve() as url:
        tts = ws_tts(url, text_aggregation_mode=TextAggregationMode.TOKEN)
        rec, down, up = await run(tts, 8000, llm_turn(*TOKENS), metrics=True)
    texts = fake.messages("text")
    assert "".join(m["text"] for m in texts) == "".join(TOKENS)
    assert not any(m.get("flush") for m in texts)
    assert fake.messages("flush") == [{"type": "flush", "context_id": texts[0]["context_id"]}]
    # Mirai cut the sentences itself.
    assert [s for _, _, s in fake.spoken] == SENTENCES
    assert audio_of(down, 8000) == b"".join(tone(0.4, 8000) for _ in SENTENCES)
    assert len(rec.of(TTSStoppedFrame)) == 1
    # The assistant's transcript still gets the whole reply.
    assert "".join(f.text for f in rec.of(TTSTextFrame)).split() == "".join(TOKENS).split()
    # Usage is what Mirai billed, once per sentence, and nothing else.
    usage = [d.value for f in rec.of(MetricsFrame) for d in f.data if isinstance(d, TTSUsageMetricsData)]
    assert usage == [11, 11, 11]
    assert not errors_in(up)


# --- delivery: framing, first-audio buffer, metrics ----------------------------------------


@pytest.mark.parametrize("reads", [(1,), (7, 333), (16000,), (40000, 3)])
async def test_frames_are_40_ms_whatever_the_socket_frames(reads):
    fake = FakeMiraiWS(seconds=1.0, reads=reads)
    async with fake.serve() as url:
        _, down, _ = await run(ws_tts(url), 8000, [TTSSpeakFrame(TEXT)])
    frames = [f.audio for f in down if isinstance(f, TTSAudioRawFrame)]
    size = 8000 * FRAME_MS // 1000 * 2
    assert all(len(a) == size for a in frames[:-1]) and 0 < len(frames[-1]) <= size
    assert b"".join(frames) == tone(1.0, 8000)


async def test_a_resampled_burst_is_framed_too():
    fake = FakeMiraiWS(seconds=3.0, honours_rate=False, reads=(3 * SR * 2,))
    async with fake.serve() as url:
        _, down, _ = await run(ws_tts(url), 16000, [TTSSpeakFrame(TEXT)])
    frames = [f.audio for f in down if isinstance(f, TTSAudioRawFrame)]
    size = 16000 * FRAME_MS // 1000 * 2
    assert len(frames) >= 75 and all(len(a) == size for a in frames[:-1])
    assert abs(sum(map(len, frames)) // 2 - 3 * 16000) <= 1


def ttfb_of(rec):
    return [
        d.value
        for f in rec.of(MetricsFrame)
        for d in f.data
        if isinstance(d, TTFBMetricsData) and d.value > 0
    ]


@pytest.mark.parametrize("first_ms", [14, 100])
async def test_prebuffer_waits_out_a_short_first_chunk(first_ms):
    # Mirai's first frame can be 14-133 ms of audio, then nothing for up to
    # 200 ms: starting playback on it would play that much and stall. Nothing
    # is pushed until 150 ms is in hand.
    fake = FakeMiraiWS(seconds=1.0, reads=(first_ms * 16, 1600), first_gap=0.2)
    async with fake.serve() as url:
        rec, _, _ = await run(ws_tts(url), 8000, [TTSSpeakFrame(TEXT)], metrics=True)
    first_binary = fake.conns[0].binary[0][0]
    audio = rec.audio
    assert audio[0][0] - first_binary >= 0.19
    first_push = [f for t, f in audio if t - audio[0][0] < 0.005]
    assert sum(len(f.audio) for f in first_push) >= 0.15 * 8000 * 2
    assert b"".join(f.audio for _, f in audio) == tone(1.0, 8000)
    ttfb = ttfb_of(rec)
    assert ttfb and ttfb[0] < 0.15  # TTFB is the first byte received, not the first push


async def test_prebuffer_zero_pushes_the_first_frame_at_once():
    fake = FakeMiraiWS(seconds=1.0, reads=(48 * 16, 1600), first_gap=0.2)  # 48 ms, then the gap
    async with fake.serve() as url:
        rec, _, _ = await run(ws_tts(url, prebuffer_secs=0), 8000, [TTSSpeakFrame(TEXT)])
    assert rec.audio[0][0] - fake.conns[0].binary[0][0] < 0.15
    assert len(rec.audio[0][1].audio) == 8000 * FRAME_MS // 1000 * 2


async def test_metrics_ttfb_and_billed_usage():
    fake = FakeMiraiWS(seconds=0.5, first_byte=0.1, billing=lambda s: 1000 + len(s))
    async with fake.serve() as url:
        rec, _, _ = await run(ws_tts(url), 8000, [TTSSpeakFrame(TEXT)], metrics=True)
    ttfb = ttfb_of(rec)
    assert len(ttfb) == 1 and 0.09 <= ttfb[0] < 0.5
    usage = [d for f in rec.of(MetricsFrame) for d in f.data if isinstance(d, TTSUsageMetricsData)]
    assert [(d.value, d.model) for d in usage] == [(1000 + len(TEXT), "mira-tts")]


# --- interruptions -------------------------------------------------------------------------


async def test_interruption_cancels_once_and_nothing_leaks_into_the_next_context():
    # 20 s of audio at ~4x real time; after a cancel arrives the server keeps
    # streaming for another 200 ms (frames already in flight) before its ack.
    fake = FakeMiraiWS(seconds={"लंबा जवाब।": 20.0, "छोटा।": 0.5}, reads=(3200,), pace=0.05, cancel_delay=0.2)
    async with fake.serve() as url:
        rec, _, up = await run(
            ws_tts(url),
            8000,
            [
                TTSSpeakFrame("लंबा जवाब।"),
                SleepFrame(0.6),
                InterruptionFrame(),
                InterruptionFrame(),  # a second interruption sends nothing more
                SleepFrame(0.1),
                TTSSpeakFrame("छोटा।"),
                SleepFrame(0.5),
            ],
        )
    long_id = fake.messages("text")[0]["context_id"]
    assert fake.messages("cancel") == [{"type": "cancel", "context_id": long_id}]
    conn = fake.conns[0]
    cancelled_at = next(t for t, kind, e in conn.events if kind == "context.cancelled")
    # The server really did keep sending the long sentence after the cancel...
    assert any(t > cancelled_at - 0.15 for t, _ in conn.binary if t < cancelled_at)
    interrupted_at = next(t for t, f in rec.seen if isinstance(f, InterruptionFrame))
    before = {f.context_id for t, f in rec.audio if t < interrupted_at}
    after = [f for t, f in rec.audio if t > interrupted_at]
    assert before == {long_id}
    # ...but after the interruption only the next context's audio, all of it.
    assert after and {f.context_id for f in after} == {fake.messages("text")[1]["context_id"]}
    assert b"".join(f.audio for f in after) == tone(0.5, 8000)
    assert not errors_in(up)


async def test_interruption_cancels_queued_sentences_too():
    fake = FakeMiraiWS(seconds={s: 3.0 for s in SENTENCES}, reads=(3200,), pace=0.05)
    async with fake.serve() as url:
        rec, _, _ = await run(
            ws_tts(url), 8000, [*llm_turn(*TOKENS), SleepFrame(0.5), InterruptionFrame(), SleepFrame(0.5)]
        )
    (cancel,) = fake.messages("cancel")
    assert cancel["context_id"] == fake.messages("text")[0]["context_id"]
    assert fake.spoken == []  # neither the playing sentence nor the two queued ones finished
    interrupted_at = next(t for t, f in rec.seen if isinstance(f, InterruptionFrame))
    assert not [f for t, f in rec.audio if t > interrupted_at]


async def test_interruption_before_any_text_sends_no_cancel():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await run(ws_tts(url), 8000, [SleepFrame(0.2), InterruptionFrame(), TTSSpeakFrame(TEXT)])
    assert fake.messages("cancel") == []


# --- errors and retries --------------------------------------------------------------------


async def test_capacity_error_is_retried_once_after_retry_after():
    second = SENTENCES[1]
    fake = FakeMiraiWS(seconds={s: 0.3 for s in SENTENCES}, reads=(800,), pace=0.01, capacity={second: 0.4})
    async with fake.serve() as url:
        rec, down, up = await run(ws_tts(url), 8000, llm_turn(*TOKENS))
    texts = fake.messages("text")
    first_id = texts[0]["context_id"]
    # The refused context is cancelled (so the server doesn't go on without the
    # refused sentence), and what wasn't spoken is resent under a new id.
    assert fake.messages("cancel") == [{"type": "cancel", "context_id": first_id}]
    retry = [m for m in texts if m["context_id"] != first_id]
    assert retry and retry[0]["context_id"].startswith(first_id + "#")
    assert " ".join(m["text"] for m in retry).split() == " ".join(SENTENCES[1:]).split()
    conn = fake.conns[0]
    refused_at = next(t for t, kind, e in conn.events if kind == "error")
    resent_at = next(t for t, kind, e in conn.events if kind == "audio.start" and e["text"] == second)
    assert resent_at - refused_at >= 0.4
    # Every sentence is heard once, in order, and nothing is reported.
    assert [s for _, c, s in fake.spoken if c == first_id] == SENTENCES[:1]
    assert [s for _, c, s in fake.spoken if c == retry[0]["context_id"]] == SENTENCES[1:]
    assert audio_of(down, 8000) == b"".join(tone(0.3, 8000) for _ in SENTENCES)
    assert len(rec.of(TTSStoppedFrame)) == 1
    assert not errors_in(up)


async def test_a_second_capacity_error_is_reported_and_the_reply_goes_on():
    second = SENTENCES[1]
    fake = FakeMiraiWS(
        seconds={s: 0.3 for s in SENTENCES}, reads=(800,), pace=0.01, always_refuse={second: 0.2}
    )
    async with fake.serve() as url:
        tts = ws_tts(url)
        _, down, up = await run(tts, 8000, llm_turn(*TOKENS))
    assert len(fake.messages("cancel")) == 1  # retried once only
    errors = errors_in(up)
    assert len(errors) == 1 and "at_capacity" in errors[0].error
    retry_id = fake.messages("text")[-1]["context_id"]
    assert [s for _, c, s in fake.spoken if c == retry_id] == [SENTENCES[2]]
    assert audio_of(down, 8000) == tone(0.3, 8000) * 2
    assert tts.last_error and "at_capacity" in tts.last_error


async def test_a_capacity_retry_is_dropped_when_interrupted():
    second = SENTENCES[1]
    fake = FakeMiraiWS(seconds={s: 0.3 for s in SENTENCES}, reads=(800,), pace=0.01, capacity={second: 1.0})
    async with fake.serve() as url:
        rec, down, _ = await run(
            ws_tts(url), 8000, [*llm_turn(*TOKENS), SleepFrame(0.6), InterruptionFrame(), SleepFrame(1.0)]
        )
    first_id = fake.messages("text")[0]["context_id"]
    assert [m["context_id"] for m in fake.messages("text")] == [first_id] * 3  # nothing resent
    assert audio_of(down, 8000) == tone(0.3, 8000)  # the first sentence, and nothing after it


async def test_error_event_becomes_error_frame():
    fake = FakeMiraiWS(seconds={s: 0.3 for s in SENTENCES}, fail={SENTENCES[0]: "upstream_error"})
    async with fake.serve() as url:
        _, down, up = await run(ws_tts(url), 8000, llm_turn(*TOKENS))
    (error,) = errors_in(up)
    assert "upstream_error" in error.error and "the speech node failed" in error.error
    assert [s for _, _, s in fake.spoken] == SENTENCES[1:]
    assert len(fake.messages("cancel")) == 0


async def test_refused_handshake_is_reported():
    fake = FakeMiraiWS(refuse_status=401)
    async with fake.serve() as url:
        tts = ws_tts(url)
        _, down, up = await run(tts, 8000, [TTSSpeakFrame(TEXT)])
    errors = errors_in(up)
    assert errors and "HTTP 401" in errors[0].error and "invalid API key" in errors[0].error
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down)


# --- connection: reconnect, keepalive, settings --------------------------------------------


@pytest.mark.parametrize("heard", [True, False])
async def test_a_dropped_socket_is_reopened_and_the_reply_resent(heard):
    # The connection drops in the first sentence: after 0.3 s of its audio
    # (past the 150 ms first-audio buffer, so the caller has started hearing
    # it), or after 0.1 s (nothing pushed yet).
    fake = FakeMiraiWS(
        seconds={s: 1.0 for s in SENTENCES}, reads=(800,), pace=0.02, drop_after_bytes=4800 if heard else 1600
    )
    async with fake.serve() as url:
        rec, down, up = await run(ws_tts(url), 8000, [*llm_turn(*TOKENS), SleepFrame(2.0)])
    assert len(fake.conns) == 2
    second = fake.conns[1]
    assert second.received[0]["type"] == "session.update" and second.received[0]["sample_rate"] == 8000
    # A sentence the caller had started hearing is not repeated; the rest is.
    rest = SENTENCES[1:] if heard else SENTENCES
    resent = [m for m in second.received if m["type"] == "text"]
    assert " ".join(m["text"] for m in resent).split() == " ".join(rest).split()
    assert [s for c, _, s in fake.spoken] == rest
    assert len(rec.of(TTSStoppedFrame)) == 1
    assert not errors_in(up)


async def test_a_socket_reopened_during_reconnect_backoff_is_read_at_once():
    # The socket drops; the reconnect attempt right after it is refused, so
    # Pipecat's reconnect loop sleeps (4 s or more) before trying again. The
    # next utterance opens a socket itself, and its audio must not wait for
    # that sleep to end.
    fake = FakeMiraiWS(close_after=0.2, refuse_attempts={2})
    async with fake.serve() as url:
        rec, down, _ = await run(ws_tts(url), 8000, [SleepFrame(0.6), TTSSpeakFrame(TEXT), SleepFrame(0.5)])
    assert fake.attempts == 3 and len(fake.conns) == 2
    spoke_at = next(t for t, f in rec.seen if isinstance(f, TTSStartedFrame))
    assert rec.audio and rec.audio[0][0] - spoke_at < 1.0
    assert audio_of(down, 8000) == tone(0.5, 8000)


@pytest.mark.parametrize("settle", [0.0, 0.05])
async def test_text_sent_on_a_dead_socket_is_resent_on_a_new_one(settle):
    # The socket dies just as a sentence is sent: whichever notices first
    # (the send, or the receive task), the sentence is spoken once.
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        tts = ws_tts(url)

        async def kill_on_first_text():
            while tts._websocket is None:
                await asyncio.sleep(0.005)
            ws = tts._websocket
            real_send = ws.send

            async def send(message):
                ws.send = real_send
                if '"type": "text"' in message:
                    ws.transport.abort()
                    await asyncio.sleep(settle)
                return await real_send(message)

            ws.send = send

        killer = asyncio.create_task(kill_on_first_text())
        _, down, up = await run(tts, 8000, [SleepFrame(0.2), TTSSpeakFrame(TEXT), SleepFrame(0.5)])
        killer.cancel()
    assert len(fake.conns) == 2
    assert not fake.messages("text", conn=1)
    assert [m["text"] for m in fake.messages("text", conn=2)] == [TEXT]
    assert audio_of(down, 8000) == tone(0.5, 8000)
    assert not errors_in(up)


async def test_a_dead_socket_found_by_a_send_settles_the_reply_on_it():
    # The socket dies while the receive task is busy, so the next sentence's
    # send is what finds it dead. The reply that was on it must not be lost:
    # it is resent on the new socket, ahead of the new sentence.
    first, second = "पहला वाक्य यह है।", "दूसरा वाक्य यह है।"
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        tts = ws_tts(url)
        release = asyncio.Event()
        on_audio_start = tts._on_audio_start

        async def die_while_busy(event):
            if not release.is_set():
                tts._websocket.transport.abort()
                await release.wait()
            await on_audio_start(event)

        async def release_later():
            await asyncio.sleep(0.6)
            release.set()

        tts._on_audio_start = die_while_busy
        releaser = asyncio.create_task(release_later())
        _, down, up = await run(
            tts, 8000, [TTSSpeakFrame(first), SleepFrame(0.2), TTSSpeakFrame(second), SleepFrame(1.0)]
        )
        releaser.cancel()
    assert len(fake.conns) == 2
    assert [m["text"] for m in fake.messages("text", conn=2)] == [first, second]
    assert audio_of(down, 8000) == tone(0.5, 8000) * 2
    assert not errors_in(up)


async def test_base64_audio_events_are_played_too():
    fake = FakeMiraiWS(base64=True, reads=(1001, 640))
    async with fake.serve() as url:
        _, down, up = await run(ws_tts(url), 16000, [TTSSpeakFrame(TEXT)])
    assert audio_of(down, 16000) == tone(0.5, 16000)
    assert not errors_in(up)


async def test_a_server_close_reconnects_for_the_next_utterance():
    fake = FakeMiraiWS(close_after=0.3)
    async with fake.serve() as url:
        _, down, up = await run(ws_tts(url), 8000, [SleepFrame(0.6), TTSSpeakFrame(TEXT)])
    assert len(fake.conns) == 2
    assert fake.messages("text", conn=2) and not fake.messages("text", conn=1)
    assert audio_of(down, 8000) == tone(0.5, 8000)


async def test_keepalive_only_while_idle_and_running():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        await run(ws_tts(url, keepalive_secs=0.3), 8000, [SleepFrame(1.1)])
        at_stop = len(fake.messages())
        await asyncio.sleep(0.7)
    keepalives = [m for m in fake.messages("session.update") if m == {"type": "session.update"}]
    assert 2 <= len(keepalives) <= 4
    assert len(fake.messages()) == at_stop


async def test_voice_change_updates_the_session():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        tts = ws_tts(url, settings=MiraiWebsocketTTSService.Settings(voice="shruti"))
        await run(
            tts,
            8000,
            [
                TTSSpeakFrame(TEXT),
                TTSUpdateSettingsFrame(delta=MiraiWebsocketTTSService.Settings(voice="sameer")),
                TTSSpeakFrame(TEXT),
            ],
        )
    updates = [m for m in fake.messages("session.update") if "voice" in m]
    assert [u["voice"] for u in updates] == ["shruti", "sameer"]
    assert updates[1]["sample_rate"] == 8000
