#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""A Twilio phone bot on Mirai's Realtime API.

    export MIRAI_API_KEY=sk_live_...
    uv run --with "pipecat-ai[websocket]" --with uvicorn examples/phone/twilio_realtime_bot.py

Expose port 8765 publicly (for example `ngrok http 8765`) and point your Twilio
number's voice webhook at `https://<host>/twiml`. Mirai hears the caller, decides
when they have finished, answers and speaks; this process only moves audio.

The phone line is 8 kHz. MiraiRealtimeLLMService resamples the caller's audio
to the 24 kHz the protocol uses, and the transport resamples the replies back,
so the pipeline stays at 8 kHz throughout.
"""

import json
import os

import uvicorn
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport
from pipecat.workers.runner import WorkerRunner

from pipecat_mirai import MiraiRealtimeLLMService, apply_output_lead

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
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=TwilioFrameSerializer(
                stream_sid=start["streamSid"],
                call_sid=start.get("callSid"),
                account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
                auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
                params=TwilioFrameSerializer.InputParams(auto_hang_up=can_hang_up),
            ),
        ),
    )
    apply_output_lead(transport)  # keep the line fed even when this server is busy

    llm = MiraiRealtimeLLMService(
        agent_id=os.getenv("MIRAI_AGENT_ID"),  # optional: start from one of your agents
        voice="shruti",
        language="hi",
        metadata={"twilio_call_sid": start.get("callSid")},
    )
    context = LLMContext(
        [
            {
                "role": "system",
                "content": "You are Mira, a polite phone assistant. Keep answers short and spoken.",
            },
            {"role": "user", "content": "Greet the caller in one short sentence."},
        ]
    )
    aggregators = LLMContextAggregatorPair(context)
    worker = PipelineWorker(
        Pipeline([transport.input(), aggregators.user(), llm, transport.output(), aggregators.assistant()]),
        params=PipelineParams(audio_in_sample_rate=8000, audio_out_sample_rate=8000),
    )
    await worker.queue_frame(LLMRunFrame())  # the bot speaks first
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
