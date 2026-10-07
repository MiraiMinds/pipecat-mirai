"""MiraiTTSService is the one service (WebSocket: edge, then the API, then HTTP); 0.4 HTTP code still runs."""

import httpx
import pytest
from fake_mirai_ws import FakeMiraiWS, tone
from pipecat.frames.frames import TTSSpeakFrame
from test_tts_websocket import TEXT, audio_of, run

import pipecat_mirai
from pipecat_mirai import MiraiHttpTTSService, MiraiTTSService, MiraiWebsocketTTSService
from pipecat_mirai.tts_websocket import _stream_url


def test_one_service_and_the_old_names():
    assert MiraiWebsocketTTSService is MiraiTTSService
    assert MiraiHttpTTSService is not MiraiTTSService
    assert MiraiTTSService.Settings is MiraiHttpTTSService.Settings
    assert {"MiraiTTSService", "MiraiWebsocketTTSService", "MiraiHttpTTSService"} <= set(
        pipecat_mirai.__all__
    )


@pytest.mark.parametrize(
    "base, url",
    [
        (
            "https://sandbox.voice.miraiminds.co/v1",
            "wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream",
        ),
        (
            "https://sandbox.voice.miraiminds.co/v1/",
            "wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream",
        ),
        ("http://127.0.0.1:8000/v1", "ws://127.0.0.1:8000/v1/audio/speech/stream"),
    ],
)
def test_base_url_gives_the_streaming_endpoint(base, url):
    assert _stream_url(base) == url
    assert MiraiTTSService(api_key="k", base_url=base)._url == url


def test_defaults_and_url_wins():
    assert MiraiTTSService(api_key="k")._url == "wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream"
    tts = MiraiTTSService(api_key="k", base_url="https://a/v1", url="wss://b/v1/audio/speech/stream")
    assert tts._url == "wss://b/v1/audio/speech/stream"


def test_http_era_options_are_accepted():
    client = httpx.AsyncClient()
    tts = MiraiTTSService(api_key="k", http_client=client, warm_connection=False, keep_warm_secs=12.0)
    assert tts._keepalive_secs == 12.0
    assert tts._client() is client  # the HTTP fallback uses the caller's client
    assert MiraiTTSService(api_key="k", keep_warm_secs=None)._keepalive_secs is None
    assert MiraiTTSService(api_key="k")._keepalive_secs == 30.0


async def test_code_written_for_the_http_service_now_streams_over_the_websocket():
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        base = url.replace("ws://", "http://").removesuffix("/audio/speech/stream")
        # Exactly as 0.4's README wrote it for the HTTP service.
        tts = MiraiTTSService(
            api_key="sk_test",
            base_url=base,
            settings=MiraiTTSService.Settings(voice="shruti"),
            warm_connection=True,
            keep_warm_secs=30.0,
        )
        rec, down, _ = await run(tts, 8000, [TTSSpeakFrame(TEXT)])
    assert len(fake.handshakes) == 1
    assert audio_of(down, 8000) == tone(0.5, 8000)
