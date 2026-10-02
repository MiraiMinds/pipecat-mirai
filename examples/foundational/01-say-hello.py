#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Speak one line with Mirai through a real Pipecat pipeline and save hello.wav.

    export MIRAI_API_KEY=sk_live_...
    uv run examples/foundational/01-say-hello.py --voice shruti

No room, microphone, LLM or STT is needed. The request is billed like any other
synthesis.
"""

import argparse
import asyncio
import wave

from pipecat.frames.frames import TTSAudioRawFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_mirai import MiraiTTSService


class Capture(FrameProcessor):
    """Collects the audio the TTS service produces."""

    def __init__(self):
        super().__init__()
        self.audio = bytearray()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TTSAudioRawFrame):
            self.audio.extend(frame.audio)
        await self.push_frame(frame, direction)


async def main(args):
    tts = MiraiTTSService(settings=MiraiTTSService.Settings(voice=args.voice))
    capture = Capture()
    worker = PipelineWorker(
        Pipeline([tts, capture]), params=PipelineParams(audio_out_sample_rate=args.sample_rate)
    )
    await worker.queue_frame(TTSSpeakFrame(args.text))
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
    print(f"Saved {args.output}: {len(capture.audio) / 2 / args.sample_rate:.2f}s at {args.sample_rate} Hz")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default="neha", help="ashu, neha, shruti or sameer")
    p.add_argument("--text", default="नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?")
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument("--output", default="hello.wav")
    asyncio.run(main(p.parse_args()))
