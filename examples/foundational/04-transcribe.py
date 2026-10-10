# /// script
# requires-python = ">=3.11"
# dependencies = ["pipecat-mirai>=0.6.0"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Transcribe a recording with Mirai's streaming speech-to-text and print what it hears.

    export MIRAI_API_KEY=sk_live_...
    uv run examples/foundational/04-transcribe.py call.wav --language hi-IN

From a checkout of this repository, `uv run python examples/foundational/04-transcribe.py call.wav`
uses the local package instead.

The WAV (mono, 16-bit; 8 kHz is sent as it is, anything else is resampled to 16 kHz) is
fed to `MiraiSTTService` in 20 ms frames, at real time with --realtime or as fast as
the pipeline takes it. Mirai finds where each utterance ends (`endpointing="vad"`, since
there is no VAD in this pipeline), so you see partial transcripts grow and then one
final transcript per utterance, with how long the final took after the speaker stopped.

No room, microphone or LLM is needed. The audio is billed like any other streaming
transcription (per second of audio sent).
"""

import argparse
import asyncio
import time
import wave

from pipecat.frames.frames import InputAudioRawFrame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

from pipecat_mirai import DEFAULT_STT_WEBSOCKET_URL, MiraiSTTService


class Printer(FrameProcessor):
    """Prints partial transcripts on one line and each final on its own."""

    def __init__(self):
        super().__init__()
        self.finals = []

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterimTranscriptionFrame):
            print(f"\r  ... {frame.text}", end="\033[K", flush=True)
        elif isinstance(frame, TranscriptionFrame):
            took = (frame.result or {}).get("latency_ms")
            print(f"\r{len(self.finals) + 1:>3}. {frame.text or '(nothing heard)'}", end="\033[K")
            print(f"   [{took:.0f} ms after the speaker stopped]" if took is not None else "")
            self.finals.append(frame.text)
        await self.push_frame(frame, direction)


def read_wav(path):
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise SystemExit(f"{path}: need a mono, 16-bit WAV")
        return w.readframes(w.getnframes()), w.getframerate()


async def main(args):
    pcm, rate = read_wav(args.wav)
    stt = MiraiSTTService(url=args.url, language=args.language, endpointing="vad")
    printer = Printer()
    worker = PipelineWorker(
        Pipeline([stt, printer]),
        params=PipelineParams(audio_in_sample_rate=rate, enable_metrics=True),
    )
    step = rate // 50 * 2  # 20 ms
    silence = bytes(rate * 2)  # one second, so the last utterance ends
    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(worker)

    async def feed():
        started = time.monotonic()
        for n, i in enumerate(range(0, len(pcm) + len(silence), step)):
            data = (pcm + silence)[i : i + step]
            await worker.queue_frame(InputAudioRawFrame(audio=data, sample_rate=rate, num_channels=1))
            if args.realtime:
                await asyncio.sleep(max(0.0, started + (n + 1) * 0.02 - time.monotonic()))
        await worker.stop_when_done()

    feeder = asyncio.create_task(feed())
    await runner.run()
    await feeder
    if stt.last_error:
        raise SystemExit(stt.last_error)
    print(
        f"\n{len(printer.finals)} utterances, {stt.audio_seconds_sent:.1f} s of audio sent ({stt.session_id})"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", help="a mono 16-bit WAV file")
    p.add_argument(
        "--language",
        default="hi-IN",
        help="hi-IN, en-IN, gu-IN, mr-IN, pa-IN, bn-IN, ta-IN, te-IN, kn-IN or ml-IN",
    )
    p.add_argument("--url", default=DEFAULT_STT_WEBSOCKET_URL, help="Mirai's streaming STT WebSocket URL")
    p.add_argument("--realtime", action="store_true", help="feed the audio at real time instead of at once")
    asyncio.run(main(p.parse_args()))
