import json

import httpx
import numpy as np
import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.tests.utils import run_test

from pipecat_mirai import MiraiTTSService

SR = 48000


def tone(seconds: float, rate: int = SR) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()


def mock_client(body: bytes, *, status=200, headers=None, chunks=(1, 4097, 3, 9600), seen=None):
    """httpx client whose /audio/speech streams `body` in awkward (odd-sized) reads."""
    headers = headers or {"content-type": "audio/pcm", "x-sample-rate": "48000"}

    async def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)

        async def stream():
            pos, i = 0, 0
            while pos < len(body):
                n = chunks[i % len(chunks)]
                yield body[pos : pos + n]
                pos += n
                i += 1

        if status != 200:
            return httpx.Response(status, headers={"content-type": "application/json"}, content=body)
        return httpx.Response(200, headers=headers, content=stream())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def audio_of(frames, rate):
    out = [f for f in frames if isinstance(f, TTSAudioRawFrame)]
    assert out, "no audio frames"
    assert all(f.sample_rate == rate and f.num_channels == 1 for f in out)
    assert all(len(f.audio) % 2 == 0 for f in out)
    return b"".join(f.audio for f in out)


async def speak(service, rate, text="नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?"):
    down, up = await run_test(
        service,
        frames_to_send=[TTSSpeakFrame(text)],
        pipeline_params=PipelineParams(audio_out_sample_rate=rate),
        start_timeout=10.0,  # the first pipeline on a cold CI runner can take >1 s to start
    )
    return down, up


@pytest.mark.parametrize("rate", [8000, 16000, 24000])
async def test_resamples_to_pipeline_rate(rate):
    tts = MiraiTTSService(api_key="k", http_client=mock_client(tone(1.0)))
    down, _ = await speak(tts, rate)
    assert any(isinstance(f, TTSStartedFrame) for f in down)
    assert any(isinstance(f, TTSStoppedFrame) for f in down)
    samples = len(audio_of(down, rate)) // 2
    assert abs(samples - rate) <= rate * 0.01  # one second in, one second out


async def test_48k_is_byte_exact_despite_odd_reads():
    body = tone(0.5)
    tts = MiraiTTSService(api_key="k", http_client=mock_client(body))
    down, _ = await speak(tts, 48000)
    assert audio_of(down, 48000) == body


async def test_request_shape_and_settings():
    seen = []
    tts = MiraiTTSService(
        api_key="sk_test",
        base_url="https://example.test/v1/",
        http_client=mock_client(tone(0.2), seen=seen),
        settings=MiraiTTSService.Settings(voice="shruti"),
    )
    await speak(tts, 8000, text="हाँ जी")
    req = seen[0]
    assert str(req.url) == "https://example.test/v1/audio/speech"
    assert req.headers["authorization"] == "Bearer sk_test"
    assert json.loads(req.content) == {
        "model": "mira-tts",
        "voice": "shruti",
        "input": "हाँ जी",
        "response_format": "pcm",
    }


async def test_http_error_becomes_error_frame():
    err = json.dumps({"error": {"code": "insufficient_balance", "message": "wallet is empty"}}).encode()
    tts = MiraiTTSService(api_key="k", http_client=mock_client(err, status=402))
    down, up = await speak(tts, 8000)
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert errors and "HTTP 402" in errors[0].error and "wallet is empty" in errors[0].error
    assert not any(isinstance(f, TTSAudioRawFrame) for f in down)


async def test_wrong_content_type_is_an_error():
    tts = MiraiTTSService(
        api_key="k", http_client=mock_client(tone(0.1), headers={"content-type": "audio/mpeg"})
    )
    _, up = await speak(tts, 8000)
    assert any(isinstance(f, ErrorFrame) and "audio/pcm" in f.error for f in up)


async def test_truncated_sample_is_an_error():
    tts = MiraiTTSService(api_key="k", http_client=mock_client(tone(0.1) + b"\x01"))
    _, up = await speak(tts, 48000)
    assert any(isinstance(f, ErrorFrame) and "incomplete PCM sample" in f.error for f in up)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MiraiTTSService()
