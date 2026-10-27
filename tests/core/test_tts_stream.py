"""fourier-frequency's text to speech on its own: what an in-house orchestrator gets, no Pipecat."""

import asyncio
import subprocess
import sys

import numpy as np
import pytest
from fake_mirai_http import FakeMirai
from fake_mirai_ws import FakeMiraiWS, tone
from fourier_frequency import FrequencyError, HttpTTS, TTSListener, TTSStream, close_shared_connections
from fourier_frequency.audio import lin2ulaw

KEY = "sk_test"
TEXT = "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?"
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


@pytest.fixture(autouse=True)
async def _no_shared_connections_left():
    yield
    await close_shared_connections()


def stream(url, **kwargs) -> TTSStream:
    kwargs.setdefault("sample_rate", 8000)
    return TTSStream(api_key=KEY, url=url, **kwargs)


async def speak(tts: TTSStream, *texts: str) -> tuple[bytes, list]:
    """Send ``texts`` as one turn and read its audio to the end."""
    turn = tts.turn()
    for text in texts:
        await turn.send(text)
    await turn.end()
    audio = b"".join([piece async for piece in turn])
    return audio, turn.errors


def test_the_package_never_imports_pipecat():
    script = (
        "import sys, pkgutil, importlib\n"
        "sys.modules['pipecat'] = None\n"  # any `import pipecat...` now raises
        "import fourier_frequency\n"
        "for m in pkgutil.iter_modules(fourier_frequency.__path__):\n"
        "    importlib.import_module('fourier_frequency.' + m.name)\n"
        "assert not any(name.startswith('pipecat') and sys.modules[name] for name in sys.modules)\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True)


async def test_a_turn_speaks_at_the_rate_asked_for():
    fake = FakeMiraiWS()
    async with fake.serve() as url, stream(url) as tts:
        audio, errors = await speak(tts, TEXT)
    assert audio == tone(0.5, 8000) and not errors
    (update,) = fake.messages("session.update")
    assert update["sample_rate"] == 8000 and update["voice"] == "neha"
    (text,) = fake.messages("text")
    assert text["text"] == TEXT and text["flush"] is True


async def test_tokens_are_sent_as_they_come_and_flushed_at_the_end():
    fake = FakeMiraiWS(seconds={s: 0.3 for s in SENTENCES})
    async with fake.serve() as url, stream(url, text_mode="token") as tts:
        audio, errors = await speak(tts, *TOKENS)
    assert audio == tone(0.3, 8000) * 3 and not errors
    assert all("flush" not in m for m in fake.messages("text"))
    assert len(fake.messages("flush")) == 1
    assert [s for _, _, s in fake.spoken] == SENTENCES


async def test_mulaw_for_a_phone_media_stream():
    fake = FakeMiraiWS()
    async with fake.serve() as url, stream(url, encoding="mulaw") as tts:
        audio, _ = await speak(tts, TEXT)
    expected = lin2ulaw(np.frombuffer(tone(0.5, 8000), dtype="<i2")).tobytes()
    assert audio == expected and len(audio) == 4000


async def test_audio_comes_in_40_ms_pieces():
    fake = FakeMiraiWS(reads=(1, 999, 5000))
    async with fake.serve() as url, stream(url) as tts:
        turn = tts.turn()
        await turn.send(TEXT)
        await turn.end()
        pieces = [piece async for piece in turn]
    assert {len(p) for p in pieces[:-1]} == {640}  # 40 ms of 8 kHz 16-bit
    assert b"".join(pieces) == tone(0.5, 8000)


@pytest.mark.parametrize("heard", [False, True])
async def test_a_dropped_socket_is_reopened_and_the_rest_of_the_turn_resent(heard):
    fake = FakeMiraiWS(
        seconds={s: 1.0 for s in SENTENCES}, reads=(800,), pace=0.02, drop_after_bytes=4800 if heard else 1600
    )
    async with fake.serve() as url, stream(url, text_mode="token") as tts:
        _, errors = await asyncio.wait_for(speak(tts, *TOKENS), 10)
    assert len(fake.conns) == 2 and not errors
    rest = SENTENCES[1:] if heard else SENTENCES
    resent = [m for m in fake.conns[1].received if m["type"] == "text"]
    assert " ".join(m["text"] for m in resent).split() == " ".join(rest).split()


async def test_a_sentence_refused_for_capacity_is_resent_after_retry_after():
    second = SENTENCES[1]
    fake = FakeMiraiWS(seconds={s: 0.3 for s in SENTENCES}, reads=(800,), pace=0.01, capacity={second: 0.4})
    async with fake.serve() as url, stream(url, text_mode="token") as tts:
        audio, errors = await asyncio.wait_for(speak(tts, *TOKENS), 10)
    assert audio == tone(0.3, 8000) * 3 and not errors
    assert len(fake.messages("cancel")) == 1


async def test_cancelling_a_turn_stops_its_audio_and_cancels_it_on_the_server():
    fake = FakeMiraiWS(seconds={"लंबा जवाब।": 20.0, "छोटा।": 0.5}, reads=(3200,), pace=0.05)
    async with fake.serve() as url, stream(url) as tts:
        long = tts.turn()
        await long.send("लंबा जवाब।")
        await long.end()
        heard = 0
        async for piece in long:
            heard += len(piece)
            if heard >= 3200:
                await long.cancel()
        short_audio, _ = await speak(tts, "छोटा।")
    assert heard < 16000  # far less than the 20 s it would have been
    assert [m["context_id"] for m in fake.messages("cancel")] == [fake.messages("text")[0]["context_id"]]
    assert short_audio == tone(0.5, 8000)  # nothing of the cancelled turn leaks into the next


async def test_interrupt_stops_every_turn():
    fake = FakeMiraiWS(seconds=20.0, reads=(3200,), pace=0.05)
    async with fake.serve() as url, stream(url) as tts:
        turn = tts.turn()
        await turn.send(TEXT)
        reader = asyncio.create_task(asyncio.wait_for(_drain(turn), 5))
        await asyncio.sleep(0.3)
        await tts.interrupt()
        await reader
    assert turn.cancelled and fake.messages("cancel")


async def _drain(turn):
    return [piece async for piece in turn]


async def test_a_refused_key_raises_what_to_do():
    fake = FakeMiraiWS(refuse_status=401)
    async with fake.serve() as url:
        tts = stream(url, shared_pool=False, edge=False)
        await tts.connect()
        with pytest.raises(FrequencyError) as refused:
            await tts.turn().send(TEXT)
        await tts.close()
    assert refused.value.category == "authentication"
    assert "HTTP 401" in str(refused.value) and "Check MIRAI_API_KEY" in str(refused.value)


async def test_a_listener_drives_the_stream_by_turn_id():
    class Collect(TTSListener):
        def __init__(self):
            self.audio, self.done, self.usage = bytearray(), [], []

        async def on_audio(self, turn_id, audio):
            self.audio += audio

        async def on_turn_done(self, turn_id):
            self.done.append(turn_id)

        async def on_usage(self, chars):
            self.usage.append(chars)

    listener = Collect()
    fake = FakeMiraiWS()
    async with fake.serve() as url:
        tts = stream(url, listener=listener)
        await tts.connect()
        await tts.send_text("reply-1", TEXT)
        await tts.end_turn("reply-1")
        for _ in range(100):
            if listener.done:
                break
            await asyncio.sleep(0.05)
        await tts.close()
    assert listener.done == ["reply-1"] and bytes(listener.audio) == tone(0.5, 8000)
    assert listener.usage == [len(TEXT)]


async def test_an_endpoint_without_websockets_speaks_over_http():
    fake = FakeMirai()  # answers the WebSocket handshake with 404, serves POST /audio/speech
    async with fake.serve() as url:
        ws_url = url.replace("http://", "ws://") + "/audio/speech/stream"
        async with stream(ws_url) as tts:
            audio, errors = await speak(tts, TEXT)
            assert tts.fallback == "HTTP 404"
    assert audio == tone(0.5, 8000) and not errors
    assert [(b["input"], b["sample_rate"]) for b in fake.speech] == [(TEXT, 8000)]


async def test_http_speaks_a_sentence():
    fake = FakeMirai()
    async with fake.serve() as url, HttpTTS(api_key=KEY, base_url=url, sample_rate=8000) as tts:
        audio = b"".join([piece async for piece in tts.speak(TEXT)])
    assert audio == tone(0.5, 8000)


async def test_http_out_of_credit_is_a_permanent_error():
    fake = FakeMirai(script=[{"status": 402, "code": "insufficient_balance", "message": "wallet is empty"}])
    async with fake.serve() as url, HttpTTS(api_key=KEY, base_url=url, sample_rate=8000) as tts:
        with pytest.raises(FrequencyError) as refused:
            async for _ in tts.speak(TEXT):
                pass
    error = refused.value
    assert error.permanent and error.category == "quota" and error.status == 402
    assert str(error) == (
        "Mirai TTS returned HTTP 402: wallet is empty [request req_1]. "
        "The workspace is out of credit. Top it up in the Mirai console."
    )


def test_an_api_key_is_required(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MIRAI_API_KEY"):
        TTSStream()
    with pytest.raises(ValueError, match="MIRAI_API_KEY"):
        HttpTTS()
