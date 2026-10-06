# /// script
# requires-python = ">=3.11"
# dependencies = ["pipecat-mirai>=0.3.0"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Stream a reply into Mirai over one WebSocket, as an LLM would, and save it.

    export MIRAI_API_KEY=sk_live_...
    uv run examples/foundational/02-websocket-say-hello.py --voice shruti

From a checkout of this repository, `uv run python examples/foundational/02-websocket-say-hello.py`
uses the local package instead.

The text goes to `MiraiWebsocketTTSService` a few words at a time, the way an
LLM's tokens arrive. Mirai cuts it into sentences and streams each one back on
the same socket while synthesising the next. The script prints how soon the
first audio came back and saves the reply to hello-websocket.wav. Pass
--tokens to send every chunk as it comes instead of whole sentences.

No room, microphone, LLM or STT is needed. Each sentence is billed like any
other synthesis.
"""

import argparse
import asyncio
import time
import wave

from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.tts_service import TextAggregationMode
from pipecat.workers.runner import WorkerRunner

from pipecat_mirai import DEFAULT_WEBSOCKET_URL, MiraiWebsocketTTSService

REPLY = "नमस्ते, मैं Mirai से बोल रही हूँ। आपका order कल शाम तक deliver हो जाएगा। क्या मैं आपकी कुछ और मदद कर सकती हूँ?"


class Capture(FrameProcessor):
    """Collects the audio and notes when it started."""

    def __init__(self):
        super().__init__()
        self.audio = bytearray()
        self.sent_at = 0.0
        self.first_audio_at = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSStartedFrame) and not self.sent_at:
            self.sent_at = time.monotonic()
        if isinstance(frame, TTSAudioRawFrame):
            if self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
            self.audio.extend(frame.audio)
        await self.push_frame(frame, direction)


def words(text, per_token=2):
    """Split text into chunks of a couple of words, like streamed LLM tokens."""
    parts = text.split(" ")
    for i in range(0, len(parts), per_token):
        yield " ".join(parts[i : i + per_token]) + " "


async def main(args):
    tts = MiraiWebsocketTTSService(
        url=args.url,
        settings=MiraiWebsocketTTSService.Settings(voice=args.voice),
        text_aggregation_mode=TextAggregationMode.TOKEN if args.tokens else None,
    )

    @tts.event_handler("on_connected")
    async def on_connected(service):
        print(f"Connected to {args.url}")

    capture = Capture()
    worker = PipelineWorker(
        Pipeline([tts, capture]),
        params=PipelineParams(audio_out_sample_rate=args.sample_rate, enable_metrics=True),
    )
    await worker.queue_frames(
        [LLMFullResponseStartFrame(), *(TextFrame(t) for t in words(args.text)), LLMFullResponseEndFrame()]
    )
    await worker.stop_when_done()
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)
    await runner.run()
    if tts.last_error or not capture.audio:
        raise SystemExit(tts.last_error or "No audio received - check MIRAI_API_KEY.")
    with wave.open(args.output, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(args.sample_rate)
        w.writeframes(capture.audio)
    seconds = len(capture.audio) / 2 / args.sample_rate
    first = (capture.first_audio_at - capture.sent_at) * 1000 if capture.first_audio_at else float("nan")
    print(f"First audio pushed {first:.0f} ms after the first words went out (after the first-audio buffer)")
    print(f"Saved {args.output}: {seconds:.2f}s at {args.sample_rate} Hz (session {tts.session_id})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default="neha", help="ashu, neha, shruti or sameer")
    p.add_argument("--url", default=DEFAULT_WEBSOCKET_URL, help="Mirai's streaming TTS WebSocket URL")
    p.add_argument("--text", default=REPLY)
    p.add_argument("--tokens", action="store_true", help="send every chunk as it comes (token mode)")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--output", default="hello-websocket.wav")
    asyncio.run(main(p.parse_args()))
