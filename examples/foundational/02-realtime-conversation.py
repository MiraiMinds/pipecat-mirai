# /// script
# requires-python = ">=3.11"
# dependencies = ["pipecat-mirai", "pipecat-ai[local]"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Talk to a Mirai Realtime session from your microphone.

Mirai runs speech recognition, turn detection, the model and the voice on its
side; this process only moves audio. Put on headphones (the local transport has
no echo cancellation) and run:

    export MIRAI_API_KEY=sk_live_...
    uv run examples/foundational/02-realtime-conversation.py --voice shruti --language hi

Each reply prints Mirai's own timing for the turn. The microphone needs PortAudio
(`brew install portaudio` on macOS). Pass --agent-id to start from one of your
agents, and --production once your company has gone live.
"""

import argparse
import asyncio

from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

from pipecat_mirai import PRODUCTION_REALTIME_URL, SANDBOX_REALTIME_URL, MiraiRealtimeLLMService

INSTRUCTIONS = (
    "You are Mira, a friendly voice assistant. Keep every answer to one or two short "
    "sentences, because it will be spoken. No markdown, lists or emoji. Reply in the "
    "language the caller uses."
)


async def main(args):
    transport = LocalAudioTransport(LocalAudioTransportParams(audio_in_enabled=True, audio_out_enabled=True))
    llm = MiraiRealtimeLLMService(
        base_url=PRODUCTION_REALTIME_URL if args.production else SANDBOX_REALTIME_URL,
        agent_id=args.agent_id,
        voice=args.voice,
        language=args.language,
    )

    @llm.event_handler("on_turn_metrics")
    async def on_turn_metrics(service, m):
        print(
            f"  turn: {m.v2v_ms or 0:.0f} ms voice-to-voice on Mirai's side "
            f"(stt {m.stt_ms or 0:.0f}, llm {m.llm_ms or 0:.0f}, tts {m.tts_ms or 0:.0f})"
            + (" · fallback" if m.fallback else "")
        )

    # The agent speaks first: the first run happens once the session is set up.
    messages = [] if args.agent_id else [{"role": "system", "content": INSTRUCTIONS}]
    messages.append({"role": "user", "content": "Say hello in one short sentence."})
    aggregators = LLMContextAggregatorPair(LLMContext(messages))

    worker = PipelineWorker(
        Pipeline([transport.input(), aggregators.user(), llm, transport.output(), aggregators.assistant()]),
        params=PipelineParams(audio_in_sample_rate=24000, audio_out_sample_rate=24000, enable_metrics=True),
    )
    await worker.queue_frame(LLMRunFrame())
    runner = WorkerRunner(handle_sigint=True)
    await runner.add_workers(worker)
    print("Talking to Mira. Press Ctrl+C to end the session.")
    await runner.run()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--voice", default=None, help="ashu, neha, shruti or sameer")
    p.add_argument("--language", default=None, help="the caller's language, such as hi or en")
    p.add_argument("--agent-id", default=None, help="start from one of your agents")
    p.add_argument("--production", action="store_true", help="use the production address")
    asyncio.run(main(p.parse_args()))
