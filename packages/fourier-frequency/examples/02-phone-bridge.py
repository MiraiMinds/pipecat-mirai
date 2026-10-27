# /// script
# requires-python = ">=3.11"
# dependencies = ["fourier-frequency", "fastapi", "uvicorn[standard]"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""A phone agent on Twilio Media Streams with your own orchestration, no framework.

    export MIRAI_API_KEY=sk_live_...
    uv run packages/fourier-frequency/examples/02-phone-bridge.py

Point a Twilio number's TwiML at it (``<Connect><Stream url="wss://<host>/media"/>``).
Plivo and Exotel streams work the same way; only their JSON field names differ.

Per call: one STTStream (Mirai's VAD ends each utterance) and one TTSStream
(8 kHz mu-law straight into the media stream). ``answer()`` is where your
orchestrator goes: an LLM, a flow, a tool call; it yields text as it has it.
When the caller talks over the bot, the reply is cancelled on Mirai's side,
Twilio is told to clear what it holds, and the bot listens.
"""

import asyncio
import base64
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fourier_frequency import (
    DEFAULT_STT_WEBSOCKET_URL,
    DEFAULT_WEBSOCKET_URL,
    Pacer,
    SpeechStarted,
    STTStream,
    Transcript,
    TTSStream,
    close_shared_connections,
    prewarm,
)
from fourier_frequency.audio import ulaw2lin

TTS_URL = os.getenv("MIRAI_TTS_URL", DEFAULT_WEBSOCKET_URL)
STT_URL = os.getenv("MIRAI_STT_URL", DEFAULT_STT_WEBSOCKET_URL)


async def answer(heard: str) -> AsyncIterator[str]:
    """Your orchestrator: what the bot says to ``heard``, yielded as it is ready."""
    yield "जी, समझ गई। "
    await asyncio.sleep(0.2)  # an LLM thinking
    yield f"आपने कहा: {heard}। "
    yield "मैं और कैसे मदद कर सकती हूँ?"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sockets waiting for the first calls; each Call's streams find them on their own.
    await prewarm(
        websocket=4, websocket_url=TTS_URL, stt_websockets=4, stt_url=STT_URL, stt_endpointing="vad"
    )
    yield
    await close_shared_connections()


app = FastAPI(lifespan=lifespan)


class Call:
    def __init__(self, phone: WebSocket, stream_sid: str):
        self.phone = phone
        self.stream_sid = stream_sid
        self.tts = TTSStream(
            url=TTS_URL, voice="shruti", sample_rate=8000, encoding="mulaw", text_mode="token"
        )
        # 16-bit PCM up, so it can take one of prewarm()'s sockets (those wait as linear16).
        self.stt = STTStream(url=STT_URL, language="hi-IN", input_sample_rate=8000, endpointing="vad")
        self.pacer = Pacer(8000, encoding="mulaw")
        self.speaking: asyncio.Task | None = None

    async def run(self):
        async with self.tts, self.stt:
            try:
                await self.say(["नमस्ते! मैं Mirai से बोल रही हूँ। बताइए, मैं क्या मदद करूँ?"])
                async for event in self.stt.events():
                    if isinstance(event, SpeechStarted):
                        await self.barge_in()
                    elif isinstance(event, Transcript) and event.final and event.text:
                        await self.say(answer(event.text))
            finally:
                await self.stop_speaking()

    def hear(self, payload: str):
        self.stt.send(ulaw2lin(base64.b64decode(payload)).tobytes())

    async def say(self, text):
        await self.barge_in()
        self.speaking = asyncio.create_task(self._speak(text))

    async def _speak(self, text):
        turn = self.tts.turn()

        async def send():
            if isinstance(text, list):
                for part in text:
                    await turn.send(part)
            else:
                async for part in text:
                    await turn.send(part)
            await turn.end()

        sender = asyncio.create_task(send())
        try:
            async for audio in self.pacer.pace(turn):
                media = {"payload": base64.b64encode(audio).decode()}
                await self.phone.send_json({"event": "media", "streamSid": self.stream_sid, "media": media})
        finally:
            sender.cancel()
            if not turn.done:
                await turn.cancel()

    async def stop_speaking(self) -> bool:
        """Cancel the reply being spoken; whether there was one."""
        speaking, self.speaking = self.speaking, None
        if speaking is None or speaking.done():
            return False
        speaking.cancel()
        await asyncio.gather(speaking, return_exceptions=True)
        return True

    async def barge_in(self):
        if await self.stop_speaking():
            # What Twilio holds of the reply (the pacer's lead) is dropped too.
            await self.phone.send_json({"event": "clear", "streamSid": self.stream_sid})
            self.pacer.reset()


@app.websocket("/media")
async def media(phone: WebSocket):
    await phone.accept()
    call: Call | None = None
    runner: asyncio.Task | None = None
    try:
        async for raw in phone.iter_text():
            message = json.loads(raw)
            if message["event"] == "start":
                call = Call(phone, message["start"]["streamSid"])
                runner = asyncio.create_task(call.run())
            elif message["event"] == "media" and call is not None:
                call.hear(message["media"]["payload"])
            elif message["event"] == "stop":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if runner is not None:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
