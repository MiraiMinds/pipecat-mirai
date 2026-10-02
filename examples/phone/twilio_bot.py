#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""A Twilio phone bot that speaks with Mirai.

    export MIRAI_API_KEY=sk_live_...
    uv run --with "pipecat-ai[websocket]" --with uvicorn examples/phone/twilio_bot.py

Expose port 8765 publicly (for example `ngrok http 8765`) and point your Twilio
number's voice webhook at `https://<host>/twiml`. The bot answers and speaks two
lines. Add your STT, LLM and context aggregators between `transport.input()` and
`tts` to make it conversational; nothing about the Mirai part changes.

`apply_output_lead` is the important line for phone calls: it keeps the call
smooth when the server is busy. See the README.
"""

import json
import os

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from pipecat_mirai import MiraiTTSService, apply_output_lead

app = FastAPI()


@app.post("/twiml")
async def twiml(request: Request):
    host = request.headers.get("host")
    xml = f'<Response><Connect><Stream url="wss://{host}/ws"/></Connect></Response>'
    return Response(content=xml, media_type="application/xml")


@app.websocket("/ws")
async def call(websocket: WebSocket):
    await websocket.accept()
    await websocket.receive_text()  # {"event": "connected"}
    start = json.loads(await websocket.receive_text())["start"]  # {"event": "start", ...}

    can_hang_up = bool(os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"))
    serializer = TwilioFrameSerializer(
        stream_sid=start["streamSid"],
        call_sid=start.get("callSid"),
        account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        params=TwilioFrameSerializer.InputParams(auto_hang_up=can_hang_up),
    )
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )
    apply_output_lead(transport)  # send up to 0.4 s ahead so a busy server never starves the call

    tts = MiraiTTSService(settings=MiraiTTSService.Settings(voice="shruti"))
    worker = PipelineWorker(
        Pipeline([transport.input(), tts, transport.output()]),
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
    )
    await worker.queue_frames(
        [
            TTSSpeakFrame("नमस्ते! मैं Mirai से बोल रही हूँ।"),
            TTSSpeakFrame("यह call Pipecat और Mirai TTS से चल रही है। धन्यवाद!"),
            EndFrame(),
        ]
    )
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
