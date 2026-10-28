"""A workspace's own voice (cv_...) streams from the gateway's socket, never from the edge."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from fake_mirai_gateway import FakeGateway
from fake_mirai_http import FakeMirai
from fake_mirai_ws import FakeMiraiWS, tone
from fourier_frequency import FrequencyError, TTSStream, close_shared_connections, is_custom_voice
from fourier_frequency import edge as edge_module

KEY = "sk_test"
VOICE = "cv_01m608ea5ewtee7gaz9dvzfe2z"
FIRST = "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?"
SECOND = "आपका order कल deliver होगा।"


@pytest.fixture(autouse=True)
async def _edge_on(monkeypatch):
    monkeypatch.delenv("MIRAI_TTS_EDGE", raising=False)  # the default: "auto"
    edge_module._reset()
    yield
    await close_shared_connections()
    edge_module._reset()


@dataclass
class Stack:
    url: str  # the gateway's streaming endpoint
    gateway: FakeGateway
    api: FakeMiraiWS  # the gateway's own streaming server
    edge: FakeMiraiWS


@asynccontextmanager
async def stack(**edge_options):
    """A gateway that offers an edge; the edge takes the API key, as Mirai's does."""
    api = FakeMiraiWS()
    gateway = FakeGateway(api, key=KEY)
    edge = FakeMiraiWS(auth=lambda headers: headers.get("Authorization") == f"Bearer {KEY}", **edge_options)
    async with edge.serve() as edge_url, gateway.serve() as url:
        gateway.edge_url = edge_url
        yield Stack(url, gateway, api, edge)


async def speak(tts: TTSStream, text: str) -> tuple[bytes, list]:
    turn = tts.turn()
    await turn.send(text)
    await turn.end()
    audio = b"".join([piece async for piece in turn])
    return audio, turn.errors


def voices(fake: FakeMiraiWS) -> list[str]:
    return [m["voice"] for m in fake.messages("session.update") if "voice" in m]


def test_custom_voice_ids():
    assert is_custom_voice(VOICE) and not is_custom_voice("shruti")
    assert not is_custom_voice(None) and not is_custom_voice("CV_x")


async def test_a_built_in_voice_opens_on_the_edge():
    async with stack() as s, TTSStream(api_key=KEY, url=s.url, voice="shruti", sample_rate=8000) as tts:
        audio, errors = await speak(tts, FIRST)
        assert tts.connected_url != s.url
    assert audio == tone(0.5, 8000) and not errors
    assert voices(s.edge) == ["shruti"] and not s.api.conns


async def test_a_custom_voice_never_opens_on_the_edge():
    async with stack() as s, TTSStream(api_key=KEY, url=s.url, voice=VOICE, sample_rate=8000) as tts:
        audio, errors = await speak(tts, FIRST)
        assert tts.connected_url == s.url
    assert audio == tone(0.5, 8000) and not errors
    assert voices(s.api) == [VOICE]
    assert not s.edge.handshakes and not s.gateway.mints  # the edge isn't even asked about


async def test_a_forced_edge_still_leaves_a_custom_voice_on_the_gateway():
    async with stack() as s:
        edge_url = s.gateway.edge_url
        async with TTSStream(api_key=KEY, url=s.url, voice=VOICE, edge=edge_url, sample_rate=8000) as tts:
            audio, errors = await speak(tts, FIRST)
    assert audio == tone(0.5, 8000) and not errors
    assert not s.edge.handshakes and voices(s.api) == [VOICE]


async def test_switching_to_a_custom_voice_moves_the_stream_off_the_edge():
    async with stack() as s, TTSStream(api_key=KEY, url=s.url, voice="shruti", sample_rate=8000) as tts:
        first, errors_before = await speak(tts, FIRST)
        await tts.update(voice=VOICE)
        assert tts.connected_url == s.url
        second, errors_after = await speak(tts, SECOND)
    assert first == second == tone(0.5, 8000) and not errors_before and not errors_after
    assert voices(s.edge) == ["shruti"]  # the custom voice never reached the edge
    assert voices(s.api) == [VOICE]
    assert [sentence for _, _, sentence in s.edge.spoken] == [FIRST]
    assert [sentence for _, _, sentence in s.api.spoken] == [SECOND]


async def test_switching_back_to_a_built_in_voice_stays_on_the_open_socket():
    async with stack() as s, TTSStream(api_key=KEY, url=s.url, voice=VOICE, sample_rate=8000) as tts:
        await speak(tts, FIRST)
        await tts.update(voice="neha")
        audio, errors = await speak(tts, SECOND)
    assert audio == tone(0.5, 8000) and not errors
    assert voices(s.api) == [VOICE, "neha"] and len(s.api.conns) == 1 and not s.edge.handshakes


async def test_an_edge_that_refuses_a_custom_voice_moves_the_stream_to_the_gateway():
    async with stack(refuse_voice=is_custom_voice) as s:
        async with TTSStream(api_key=KEY, url=s.url, voice="shruti", sample_rate=8000) as tts:
            await speak(tts, FIRST)
            # As if the voice changed under an open edge socket without update():
            # the edge refuses it, and the stream carries on from the gateway.
            tts.voice = VOICE
            await tts._send(tts._session_update())
            for _ in range(100):
                if tts.connected_url == s.url and tts.is_open():
                    break
                await asyncio.sleep(0.02)
            audio, errors = await speak(tts, SECOND)
    assert audio == tone(0.5, 8000) and not errors
    assert voices(s.api) == [VOICE] and [sentence for _, _, sentence in s.api.spoken] == [SECOND]


async def test_a_custom_voice_over_http_names_the_voice():
    fake = FakeMirai()  # no WebSocket: the call speaks over HTTP
    async with fake.serve() as url:
        ws_url = url.replace("http://", "ws://") + "/audio/speech/stream"
        async with TTSStream(api_key=KEY, url=ws_url, voice=VOICE, sample_rate=8000) as tts:
            audio, errors = await speak(tts, FIRST)
    assert audio == tone(0.5, 8000) and not errors
    assert [b["voice"] for b in fake.speech] == [VOICE]


async def test_a_custom_voice_mirai_doesnt_know_is_never_spoken_in_another_voice():
    api = FakeMiraiWS(unknown_voice=is_custom_voice, first_byte=0.2)
    async with api.serve() as url, TTSStream(api_key=KEY, url=url, voice=VOICE, sample_rate=8000) as tts:
        audio, errors = await speak(tts, FIRST)
        with pytest.raises(FrequencyError) as refused:
            await tts.turn().send(SECOND)
        await tts.update(voice="neha")  # another voice: the stream speaks again
        after, errors_after = await speak(tts, SECOND)
    assert audio == b""
    (error,) = errors
    assert error.permanent and error.category == "invalid_request" and error.code == "invalid_request"
    assert str(error).startswith(f"Mirai TTS refused voice {VOICE}: voice must be one of")
    assert str(refused.value) == str(error) and refused.value.permanent
    assert api.messages("cancel")  # what reached Mirai before the refusal was cancelled
    assert after == tone(0.5, 8000) and not errors_after
    assert [sentence for _, _, sentence in api.spoken] == [SECOND]


async def test_a_built_in_voice_refusal_is_reported_as_before():
    api = FakeMiraiWS(unknown_voice=lambda voice: voice == "nobody")
    async with api.serve() as url, TTSStream(api_key=KEY, url=url, voice="nobody", sample_rate=8000) as tts:
        audio, errors = await speak(tts, FIRST)
    assert audio == tone(0.5, 8000) and not errors  # spoken in the session's voice, a warning logged
