import asyncio
import json
import time
from contextlib import asynccontextmanager
from http import HTTPStatus

import httpx
import numpy as np
import pytest
import soxr
from pipecat.frames.frames import (
    ErrorFrame,
    InterruptionFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.tests.utils import SleepFrame, run_test

from pipecat_mirai import MiraiHttpTTSService

SR = 48000
TEXT = "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?"
ODD_READS = (1, 4097, 3, 9600)
FRAME_MS = 40


def tone(seconds: float, rate: int = SR) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()


class FakeMirai:
    """Mirai's ``/v1/audio/speech`` and ``/v1/models``, answering like a given server version.

    By default it is the new server: it produces the requested ``sample_rate`` and
    says so in ``X-Sample-Rate``. ``honours_rate=False`` is today's server (ignores
    the field, always 48 kHz); ``rejects_rate=True`` is a strict old server that
    answers 400 to it. Use ``client()`` for an httpx mock transport that controls
    exactly where network reads end, or ``serve()`` for a real local HTTP/1.1
    server that counts TCP connections and sees the client hang up.

    ``reads`` are the sizes of the pieces the body is sent in, ``pace`` the pause
    after each piece and ``first_gap`` the pause after the first one. ``seconds``
    is the length of the audio, or a dict from input text to length.

    ``script`` (``serve()`` only) scripts successive speech requests, one dict
    each, after which requests are answered normally: ``{"status": 429,
    "retry_after": 0.2, "code": "at_capacity"}`` refuses one, ``{"reset":
    True}`` drops the connection without answering, ``{"cut_after": n}``
    sends ``n`` bytes of audio and then drops it. Every response carries an
    ``X-Request-Id``.
    """

    def __init__(
        self,
        *,
        seconds=0.5,
        honours_rate=True,
        rejects_rate=False,
        rate_header=True,
        headers=None,
        status=200,
        error="bad request",
        reads=ODD_READS,
        pace=0.0,
        first_gap=0.0,
        models_error=None,
        script=None,
    ):
        self.seconds = seconds
        self.honours_rate = honours_rate
        self.rejects_rate = rejects_rate
        self.rate_header = rate_header
        self.headers = headers or {}
        self.status = status
        self.error = error
        self.reads = reads
        self.pace = pace
        self.first_gap = first_gap
        self.models_error = models_error
        self.script = list(script or [])
        self.requests = []  # {"method", "path", "headers", "json", "conn", "t", "sent", "cut_at"}
        self.connections = 0

    @property
    def speech(self):
        """JSON bodies of the speech requests, in order."""
        return [r["json"] for r in self.requests if r["path"].endswith("/audio/speech")]

    @property
    def speech_requests_t(self):
        """When each speech request arrived (time.monotonic())."""
        return [r["t"] for r in self.requests if r["path"].endswith("/audio/speech")]

    def _respond(self, path, body):
        if path.endswith("/models"):
            return 200, {"content-type": "application/json"}, b'{"object": "list", "data": []}'
        if not path.endswith("/audio/speech"):  # e.g. a WebSocket handshake to /audio/speech/stream
            return 404, {"content-type": "application/json"}, _error_body("not found")
        req = json.loads(body)
        if self.rejects_rate and "sample_rate" in req:
            message = 'json: unknown field "sample_rate"'
            return 400, {"content-type": "application/json"}, _error_body(message)
        if self.status != 200:
            return self.status, {"content-type": "application/json"}, _error_body(self.error)
        rate = req.get("sample_rate", SR) if self.honours_rate else SR
        headers = {"content-type": "audio/pcm"}
        if self.rate_header:
            headers["x-sample-rate"] = str(rate)
        if self.honours_rate:
            headers["x-audio-encoding"] = "pcm_s16le"
        headers.update(self.headers)
        seconds = self.seconds[req["input"]] if isinstance(self.seconds, dict) else self.seconds
        return 200, headers, tone(seconds, rate)

    def _pieces(self, body):
        """(piece, pause after it) pairs."""
        pos, i = 0, 0
        while pos < len(body):
            n = self.reads[i % len(self.reads)]
            yield body[pos : pos + n], self.first_gap if i == 0 and self.first_gap else self.pace
            pos += n
            i += 1

    def _record(self, method, path, headers, body, conn):
        self.requests.append(
            {
                "method": method,
                "path": path,
                "headers": headers,
                "json": json.loads(body) if body else None,
                "conn": conn,
                "t": time.monotonic(),
                "sent": 0,
                "cut_at": None,
            }
        )
        return self.requests[-1]

    def client(self) -> httpx.AsyncClient:
        async def handler(request: httpx.Request) -> httpx.Response:
            self._record(request.method, request.url.path, dict(request.headers), request.content, 1)
            if request.url.path.endswith("/models") and self.models_error:
                raise self.models_error
            status, headers, body = self._respond(request.url.path, request.content)
            if headers["content-type"] != "audio/pcm":
                return httpx.Response(status, headers=headers, content=body)

            async def stream():
                for piece, pause in self._pieces(body):
                    yield piece
                    if pause:
                        await asyncio.sleep(pause)

            return httpx.Response(status, headers=headers, content=stream())

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @asynccontextmanager
    async def serve(self):
        """Serve on a local port; yields the base URL (``.../v1``)."""
        writers = []

        async def handle(reader, writer):
            writers.append(writer)
            self.connections += 1
            conn = self.connections
            try:
                while line := await reader.readline():
                    method, target, _ = line.decode("latin-1").split(" ", 2)
                    headers = {}
                    while (h := await reader.readline()) not in (b"\r\n", b""):
                        k, _, v = h.decode("latin-1").partition(":")
                        headers[k.strip().lower()] = v.strip()
                    body = await reader.readexactly(int(headers.get("content-length", "0")))
                    record = self._record(method, target, headers, body, conn)
                    action = self.script.pop(0) if self.script and target.endswith("/audio/speech") else {}
                    if action.get("reset"):
                        writer.transport.abort()  # no response at all
                        return
                    status, rheaders, payload = self._respond(target, body)
                    rheaders = {**rheaders, "x-request-id": f"req_{len(self.requests)}"}
                    if action.get("status"):
                        status = action["status"]
                        rheaders["content-type"] = "application/json"
                        if "retry_after" in action:
                            rheaders["retry-after"] = str(action["retry_after"])
                        error = {
                            "code": action.get("code", "error"),
                            "message": action.get("message", "busy"),
                        }
                        payload = json.dumps({"error": error}).encode()
                    head = [f"HTTP/1.1 {status} {HTTPStatus(status).phrase}", "transfer-encoding: chunked"]
                    head += [f"{k}: {v}" for k, v in rheaders.items()]
                    writer.write(("\r\n".join(head) + "\r\n\r\n").encode())
                    try:
                        for piece, pause in self._pieces(payload):
                            if "cut_after" in action and record["sent"] >= action["cut_after"]:
                                await writer.drain()
                                writer.transport.abort()  # mid-stream: the network dropped
                                return
                            writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                            await writer.drain()
                            record["sent"] += len(piece)
                            if pause:
                                await asyncio.sleep(pause)
                        writer.write(b"0\r\n\r\n")
                        await writer.drain()
                    except ConnectionError:
                        record["cut_at"] = time.monotonic()  # the client hung up mid-stream
                        raise
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        try:
            yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        finally:
            server.close()
            for w in writers:
                w.close()
            await server.wait_closed()


def _error_body(message):
    return json.dumps({"error": {"code": "invalid_request", "message": message}}).encode()


def audio_of(frames, rate):
    out = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert out, "no audio frames"
    assert all(f.sample_rate == rate and f.num_channels == 1 for f in out)
    assert all(len(f.audio) % 2 == 0 for f in out)
    return b"".join(f.audio for f in out)


def errors_in(frames):
    return [f for f in frames if isinstance(f, ErrorFrame)]


async def speak(service, rate, *texts, before=(), after=(), metrics=False):
    down, up = await run_test(
        service,
        frames_to_send=[*before, *(TTSSpeakFrame(t) for t in texts or (TEXT,)), *after],
        pipeline_params=PipelineParams(audio_out_sample_rate=rate, enable_metrics=metrics),
        start_timeout=10.0,  # the first pipeline on a cold CI runner can take >1 s to start
    )
    return down, up


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


def tts_for(fake, **kwargs):
    return MiraiHttpTTSService(api_key="k", http_client=fake.client(), **kwargs)


# --- server-side sample rate -------------------------------------------------------------


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 24000, 44100, 48000])
async def test_asks_for_the_pipeline_rate_and_passes_audio_through(rate):
    fake = FakeMirai()
    tts = tts_for(fake)
    down, up = await speak(tts, rate)
    assert fake.speech == [
        {"model": "mira-tts", "voice": "neha", "input": TEXT, "response_format": "pcm", "sample_rate": rate}
    ]
    # Mirai sent the pipeline's rate: not resampled, byte-exact despite odd-sized reads.
    assert audio_of(down, rate) == tone(0.5, rate)
    assert tts.last_server_sample_rate == rate
    assert any(isinstance(f, TTSStartedFrame) for f in down)
    assert any(isinstance(f, TTSStoppedFrame) for f in down)
    assert not errors_in(up)


async def test_passthrough_carries_odd_bytes_across_every_read():
    fake = FakeMirai(reads=(1, 3, 5, 7, 4095, 333))
    down, _ = await speak(tts_for(fake), 8000)
    assert audio_of(down, 8000) == tone(0.5, 8000)


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
async def test_server_that_ignores_sample_rate_is_resampled(rate):
    # Today's server: ignores the field and sends 48 kHz with X-Sample-Rate: 48000.
    fake = FakeMirai(seconds=1.0, honours_rate=False)
    tts = tts_for(fake)
    down, up = await speak(tts, rate)
    assert fake.speech[0]["sample_rate"] == rate
    assert tts.last_server_sample_rate == SR
    got = np.frombuffer(audio_of(down, rate), dtype="<i2").astype(int)
    want = soxr.resample(np.frombuffer(tone(1.0), dtype="<i2"), SR, rate, quality="HQ").astype(int)
    assert len(got) == len(want) == rate  # one second in, one second out
    assert np.abs(got - want).max() <= 4  # streaming == one-shot resampling, to a few LSB
    assert not errors_in(up)


async def test_48k_is_byte_exact_despite_odd_reads():
    fake = FakeMirai(honours_rate=False)
    down, _ = await speak(tts_for(fake), 48000)
    assert audio_of(down, 48000) == tone(0.5)


async def test_missing_rate_header_means_48k():
    fake = FakeMirai(seconds=1.0, honours_rate=False, rate_header=False)
    tts = tts_for(fake)
    down, up = await speak(tts, 8000)
    assert len(audio_of(down, 8000)) // 2 == 8000
    assert tts.last_server_sample_rate == SR
    assert not errors_in(up)


async def test_rate_mirai_does_not_serve_leaves_the_field_out():
    fake = FakeMirai(seconds=1.0)
    down, _ = await speak(tts_for(fake), 32000)
    assert "sample_rate" not in fake.speech[0]
    assert abs(len(audio_of(down, 32000)) // 2 - 32000) <= 1


async def test_server_sample_rate_none_sends_the_0_2_request():
    fake = FakeMirai(seconds=1.0)
    tts = tts_for(fake, server_sample_rate=None)
    down, _ = await speak(tts, 8000, "हाँ जी")
    assert fake.speech == [{"model": "mira-tts", "voice": "neha", "input": "हाँ जी", "response_format": "pcm"}]
    assert tts.last_server_sample_rate == SR
    assert len(audio_of(down, 8000)) // 2 == 8000


async def test_explicit_server_sample_rate_is_resampled_to_the_output_rate():
    fake = FakeMirai(seconds=1.0)
    tts = tts_for(fake, server_sample_rate=16000)
    down, _ = await speak(tts, 8000)
    assert fake.speech[0]["sample_rate"] == 16000
    assert tts.last_server_sample_rate == 16000
    assert len(audio_of(down, 8000)) // 2 == 8000


@pytest.mark.parametrize("bad", [11025, 32000, "fast", True, 8000.0])
def test_invalid_server_sample_rate_is_refused(bad):
    with pytest.raises(ValueError, match="server_sample_rate"):
        MiraiHttpTTSService(api_key="k", server_sample_rate=bad)


async def test_unreadable_rate_header_is_an_error():
    fake = FakeMirai(headers={"x-sample-rate": "fast"})
    down, up = await speak(tts_for(fake), 8000)
    assert any("X-Sample-Rate" in f.error for f in errors_in(up))
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down)


async def test_non_pcm_encoding_is_an_error():
    fake = FakeMirai(headers={"x-audio-encoding": "mulaw"})
    _, up = await speak(tts_for(fake), 8000)
    assert any("pcm_s16le" in f.error and "mulaw" in f.error for f in errors_in(up))


async def test_server_that_rejects_sample_rate_is_retried_once_and_remembered():
    fake = FakeMirai(seconds=1.0, rejects_rate=True)
    async with fake.serve() as url:
        tts = MiraiHttpTTSService(api_key="k", base_url=url, warm_connection=False)
        down, up = await speak(tts, 8000, "पहला", "दूसरा")
    assert [b.get("sample_rate") for b in fake.speech] == [8000, None, None]
    assert [b["input"] for b in fake.speech] == ["पहला", "पहला", "दूसरा"]
    assert not errors_in(up)
    assert len(audio_of(down, 8000)) // 2 == 2 * 8000  # two seconds of 48 kHz, resampled
    assert fake.connections == 1  # the 400's body was read, so its connection was reused


async def test_bad_request_is_reported_and_sample_rate_kept():
    fake = FakeMirai(status=400, error="voice must be one of ashu, neha, shruti, sameer")
    _, up = await speak(tts_for(fake), 8000, "पहला", "दूसरा")
    # Each utterance: once with sample_rate, once without; both refused, so nothing is remembered.
    assert [b.get("sample_rate") for b in fake.speech] == [8000, None, 8000, None]
    errors = errors_in(up)
    assert len(errors) == 2 and all("HTTP 400" in e.error and "voice must be" in e.error for e in errors)


# --- errors --------------------------------------------------------------------------------


async def test_request_shape_and_settings():
    fake = FakeMirai()
    tts = MiraiHttpTTSService(
        api_key="sk_test",
        base_url="https://example.test/v1/",
        http_client=fake.client(),
        settings=MiraiHttpTTSService.Settings(voice="shruti"),
    )
    await speak(tts, 8000, "हाँ जी")
    req = next(r for r in fake.requests if r["method"] == "POST")
    assert req["path"] == "/v1/audio/speech"
    assert req["headers"]["authorization"] == "Bearer sk_test"
    assert req["json"] == {
        "model": "mira-tts",
        "voice": "shruti",
        "input": "हाँ जी",
        "response_format": "pcm",
        "sample_rate": 8000,
    }


async def test_http_error_becomes_error_frame():
    fake = FakeMirai(status=402, error="wallet is empty")
    down, up = await speak(tts_for(fake), 8000)
    errors = errors_in(up)
    assert errors and "HTTP 402" in errors[0].error and "wallet is empty" in errors[0].error
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down)
    assert len(fake.speech) == 1  # only a 400 is retried


async def test_wrong_content_type_is_an_error():
    fake = FakeMirai(headers={"content-type": "audio/mpeg"})
    _, up = await speak(tts_for(fake), 8000)
    assert any("audio/pcm" in f.error for f in errors_in(up))


async def test_truncated_sample_is_an_error():
    fake = FakeMirai(honours_rate=False)
    fake._respond = lambda path, body: (200, {"content-type": "audio/pcm"}, tone(0.1) + b"\x01")
    _, up = await speak(tts_for(fake, warm_connection=False), 48000)
    assert any("incomplete PCM sample" in f.error for f in errors_in(up))


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MiraiHttpTTSService()


# --- connection reuse ----------------------------------------------------------------------


async def test_utterances_share_one_connection():
    fake = FakeMirai()
    async with fake.serve() as url:
        tts = MiraiHttpTTSService(api_key="k", base_url=url, warm_connection=False)
        down, up = await speak(tts, 8000, "पहला", "दूसरा")
    assert [(r["method"], r["path"], r["conn"]) for r in fake.requests] == [
        ("POST", "/v1/audio/speech", 1),
        ("POST", "/v1/audio/speech", 1),
    ]
    assert fake.connections == 1  # keep-alive: no second TCP/TLS handshake
    assert audio_of(down, 8000) == tone(0.5, 8000) * 2
    assert not errors_in(up)


@pytest.mark.parametrize("shared_pool", [True, False])
async def test_warm_up_opens_the_connection_before_the_first_sentence(monkeypatch, shared_pool):
    monkeypatch.setenv("MIRAI_WARM_CONNECTIONS", "1")  # the shared pool keeps one open
    fake = FakeMirai()
    async with fake.serve() as url:
        tts = MiraiHttpTTSService(api_key="sk_test", base_url=url, shared_pool=shared_pool)
        _, up = await speak(tts, 8000, "पहला", "दूसरा", before=[SleepFrame(0.3)])
    assert [(r["method"], r["path"], r["conn"]) for r in fake.requests] == [
        ("GET", "/v1/models", 1),
        ("POST", "/v1/audio/speech", 1),
        ("POST", "/v1/audio/speech", 1),
    ]
    assert fake.requests[0]["headers"]["authorization"] == "Bearer sk_test"
    assert fake.connections == 1
    assert not errors_in(up)


async def test_warm_up_failure_is_not_fatal():
    fake = FakeMirai(models_error=httpx.ConnectError("no route to host"))
    down, up = await speak(tts_for(fake), 8000, before=[SleepFrame(0.1)])
    assert fake.requests[0]["path"] == "/v1/models"
    assert audio_of(down, 8000) == tone(0.5, 8000)
    assert not errors_in(up)


async def test_warm_up_can_be_turned_off():
    fake = FakeMirai()
    await speak(tts_for(fake, warm_connection=False), 8000, before=[SleepFrame(0.1)])
    assert [r["method"] for r in fake.requests] == ["POST"]


@pytest.mark.parametrize("bad", [{"prebuffer_secs": -0.1}, {"keep_warm_secs": 0}, {"keep_warm_secs": -5}])
def test_invalid_buffering_options_are_refused(bad):
    with pytest.raises(ValueError):
        MiraiHttpTTSService(api_key="k", **bad)


async def test_keep_warm_pings_an_idle_connection_only_while_the_pipeline_runs():
    # A client of the service's own (the shared pool refreshes its connections itself).
    fake = FakeMirai()
    async with fake.serve() as url:
        tts = MiraiHttpTTSService(api_key="k", base_url=url, keep_warm_secs=1.0, shared_pool=False)
        await speak(tts, 8000, before=[SleepFrame(0.3)], after=[SleepFrame(2.6)])
        at_stop = len(fake.requests)
        await asyncio.sleep(1.5)
    methods = [r["method"] for r in fake.requests]
    assert methods[:2] == ["GET", "POST"]  # warm-up, then the sentence
    assert methods[2:] and set(methods[2:]) == {"GET"}  # idle pings during the 2.6 s pause
    assert {r["conn"] for r in fake.requests} == {1}  # ...which kept the one connection open
    assert len(fake.requests) == at_stop  # and stop once the pipeline has ended


# --- delivery: framing, first-audio buffer, bursts, interruptions --------------------------


@pytest.mark.parametrize("reads", [(1,), (7, 333), (16000,), (40000, 3)])
async def test_frames_are_40_ms_whatever_the_network_reads(reads):
    fake = FakeMirai(seconds=1.0, reads=reads)
    down, _ = await speak(tts_for(fake), 8000)
    frames = [f.audio for f in down if isinstance(f, TTSAudioRawFrame)]
    size = 8000 * FRAME_MS // 1000 * 2
    assert all(len(a) == size for a in frames[:-1]) and 0 < len(frames[-1]) <= size
    assert b"".join(frames) == tone(1.0, 8000)


async def test_a_resampled_burst_is_framed_too():
    # Three seconds of 48 kHz audio in a single read, resampled to 16 kHz.
    fake = FakeMirai(seconds=3.0, honours_rate=False, reads=(3 * SR * 2,))
    down, _ = await speak(tts_for(fake), 16000)
    frames = [f.audio for f in down if isinstance(f, TTSAudioRawFrame)]
    size = 16000 * FRAME_MS // 1000 * 2
    assert len(frames) >= 75 and all(len(a) == size for a in frames[:-1])
    assert abs(sum(map(len, frames)) // 2 - 3 * 16000) <= 1


def spy_appends(tts):
    """Record when the service hands each audio frame to Pipecat.

    The frames of one push are appended back to back, so their timestamps
    are microseconds apart. Downstream, Pipecat's queues can spread them out
    by several ms on a busy machine, so a push is judged here, not there.
    """
    appended = []
    append = tts.append_to_audio_context

    async def spy(context_id, frame):
        if isinstance(frame, TTSAudioRawFrame):
            appended.append((time.monotonic(), frame))
        await append(context_id, frame)

    tts.append_to_audio_context = spy
    return appended


def first_push(appended):
    """The audio frames of the service's first push."""
    return [f for t, f in appended if t - appended[0][0] < 0.005]


async def run_timed(fake, **kwargs):
    rec = Recorder()
    tts = tts_for(fake, warm_connection=False, **kwargs)
    appended = spy_appends(tts)
    await run_test(
        Pipeline([tts, rec]),
        frames_to_send=[TTSSpeakFrame(TEXT)],
        pipeline_params=PipelineParams(audio_out_sample_rate=8000, enable_metrics=True),
        start_timeout=10.0,
    )
    ttfb = [
        d.value
        for _, f in rec.seen
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, TTFBMetricsData) and d.value > 0
    ]
    return fake.speech_requests_t[0], rec.audio, ttfb, appended


@pytest.mark.parametrize("first_ms", [14, 100])
async def test_prebuffer_waits_out_a_short_first_chunk(first_ms):
    # Mirai's first chunk can be 14-133 ms of audio, then nothing for up to 200 ms:
    # starting playback on it would play that much and stall. Nothing is pushed
    # until 150 ms is in hand.
    fake = FakeMirai(seconds=1.0, reads=(first_ms * 16, 1600), first_gap=0.2)
    asked, audio, ttfb, appended = await run_timed(fake)
    assert audio[0][0] - asked >= 0.19
    assert sum(len(f.audio) for f in first_push(appended)) >= 0.15 * 8000 * 2
    assert b"".join(f.audio for _, f in audio) == tone(1.0, 8000)
    # TTFB is still the first byte received, not the first audio pushed.
    assert ttfb and ttfb[0] < 0.15


async def test_prebuffer_zero_pushes_the_first_frame_at_once():
    fake = FakeMirai(seconds=1.0, reads=(48 * 16, 1600), first_gap=0.2)  # 48 ms, then the gap
    asked, audio, _, _ = await run_timed(fake, prebuffer_secs=0)
    assert audio[0][0] - asked < 0.15
    assert len(audio[0][1].audio) == 8000 * FRAME_MS // 1000 * 2


async def test_interruption_cuts_the_stream_and_nothing_leaks_into_the_next_sentence():
    # A server that bursts: 20 s of audio at ~4x real time (200 ms every 50 ms).
    fake = FakeMirai(seconds={"लंबा": 20.0, "छोटा": 0.5}, reads=(3200,), pace=0.05)
    rec = Recorder()
    async with fake.serve() as url:
        tts = MiraiHttpTTSService(api_key="k", base_url=url, shared_pool=False)
        await run_test(
            Pipeline([tts, rec]),
            frames_to_send=[
                SleepFrame(0.3),
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
    long, short = (r for r in fake.requests if r["method"] == "POST")
    interrupted_at = next(t for t, f in rec.seen if isinstance(f, InterruptionFrame))

    # The server saw the client hang up at once, with most of the 20 s unsent.
    assert long["cut_at"] is not None and long["cut_at"] - interrupted_at < 0.5
    assert long["sent"] < len(tone(20.0, 8000)) // 2

    # After the interruption: only the next sentence's audio, all of it.
    before = {f.context_id for t, f in rec.audio if t < interrupted_at}
    after = [f for t, f in rec.audio if t > interrupted_at]
    assert before and after
    assert not before & {f.context_id for f in after}
    assert b"".join(f.audio for f in after) == tone(0.5, 8000)

    # The cut connection is replaced straight away and the next sentence uses the new one.
    rewarm = next(r for r in fake.requests if r["method"] == "GET" and r["t"] > interrupted_at)
    assert rewarm["conn"] != long["conn"]
    assert short["conn"] == rewarm["conn"]


async def test_interruption_closes_a_stream_paused_between_frames():
    # If Pipecat is busy elsewhere when the interruption lands (a slow or bounded
    # audio queue), the utterance is parked between two frames and the cancellation
    # never reaches its HTTP read. Here something also still holds the parked
    # generator (as a stored traceback or a tracer can), so it isn't finalised
    # either. The stream must still close at once.
    fake = FakeMirai(seconds=20.0, reads=(3200,), pace=0.05)
    rec = Recorder()
    held = []
    async with fake.serve() as url:
        tts = MiraiHttpTTSService(api_key="k", base_url=url, warm_connection=False)
        append, run_tts = tts.append_to_audio_context, tts.run_tts

        async def slow_append(context_id, frame):
            await asyncio.sleep(0.02)
            await append(context_id, frame)

        def held_run_tts(text, context_id):
            held.append(run_tts(text, context_id))
            return held[-1]

        tts.append_to_audio_context, tts.run_tts = slow_append, held_run_tts
        await run_test(
            Pipeline([tts, rec]),
            frames_to_send=[TTSSpeakFrame(TEXT), SleepFrame(0.6), InterruptionFrame(), SleepFrame(1.0)],
            pipeline_params=PipelineParams(audio_out_sample_rate=8000),
            start_timeout=10.0,
        )
        # Resumed after the interruption, the old utterance ends without pushing
        # another frame (and without reading the stream that was closed under it).
        for gen in held:
            with pytest.raises(StopAsyncIteration):
                await gen.__anext__()
    interrupted_at = next(t for t, f in rec.seen if isinstance(f, InterruptionFrame))
    (long,) = fake.requests
    assert long["cut_at"] is not None and long["cut_at"] - interrupted_at < 0.5
