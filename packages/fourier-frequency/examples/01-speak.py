# /// script
# requires-python = ">=3.11"
# dependencies = ["fourier-frequency"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Speak a reply the way an LLM produces it, token by token, and save reply.wav.

    export MIRAI_API_KEY=sk_live_...
    uv run packages/fourier-frequency/examples/01-speak.py --voice shruti

No framework: one TTSStream per call, one TTSTurn per reply. The tokens go
out as they come; the audio comes back while they are still being sent, and
is written as it arrives. The request is billed like any other synthesis.
"""

import argparse
import asyncio
import time
import wave

from fourier_frequency import DEFAULT_WEBSOCKET_URL, TTSStream

REPLY = ["नमस्ते! ", "आपका order ", "कल शाम तक ", "deliver हो जाएगा। ", "क्या मैं ", "कुछ और ", "बता सकती हूँ?"]


async def send_tokens(turn):
    for token in REPLY:
        await turn.send(token)
        await asyncio.sleep(0.03)  # an LLM's pace
    await turn.end()


async def main(args):
    async with TTSStream(
        url=args.url, voice=args.voice, sample_rate=args.sample_rate, text_mode="token"
    ) as tts:
        started = time.monotonic()
        turn = tts.turn()
        sender = asyncio.create_task(send_tokens(turn))
        audio = bytearray()
        async for piece in turn:
            if not audio:
                print(f"first audio after {(time.monotonic() - started) * 1000:.0f} ms")
            audio += piece
        await sender
    with wave.open(args.out, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(args.sample_rate)
        out.writeframes(bytes(audio))
    seconds = len(audio) / 2 / args.sample_rate
    print(f"wrote {args.out}: {seconds:.1f} s at {args.sample_rate} Hz (session {tts.session_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--voice", default="shruti")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--url", default=DEFAULT_WEBSOCKET_URL)
    parser.add_argument("--out", default="reply.wav")
    asyncio.run(main(parser.parse_args()))
