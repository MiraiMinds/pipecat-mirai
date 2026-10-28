# /// script
# requires-python = ">=3.11"
# dependencies = ["fourier-frequency"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Speak a reply in your own voice over the WebSocket, token by token, and save it as a WAV.

    uv run speak.py                      # the voice in .voice.json (from clone_voice.py)
    uv run speak.py --voice cv_... --text "Hello! Your order ships tomorrow."

One TTSStream per call, one turn per reply, exactly as with a built-in voice.
A custom voice streams from Mirai's API socket (built-in voices may use the
edge); the stream picks that by itself.
"""

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from pathlib import Path

from fourier_frequency import DEFAULT_WEBSOCKET_URL, FrequencyError, TTSStream

REPLY = ["नमस्ते! ", "आपका order ", "कल शाम तक ", "deliver हो जाएगा। ", "क्या मैं ", "कुछ और ", "बता सकती हूँ?"]


def tokens(text: str | None) -> list[str]:
    """``text`` as an LLM would stream it (a word at a time), or the built-in reply."""
    if not text:
        return REPLY
    words = text.split(" ")
    return [w + " " for w in words[:-1]] + [words[-1]]


async def send(turn, pieces):
    for piece in pieces:
        if turn.done:
            return  # stopped by an error, such as a voice Mirai refused: it is in turn.errors
        await turn.send(piece)
        await asyncio.sleep(0.03)  # an LLM's pace
    await turn.end()


async def main(args):
    voice = args.voice or json.loads(Path(".voice.json").read_text())["id"]
    async with TTSStream(url=args.url, voice=voice, sample_rate=args.sample_rate, text_mode="token") as tts:
        where = "Mirai's API socket" if tts.connected_url == tts.url else f"the edge ({tts.connected_url})"
        print(f"voice {voice} on {where}")
        started = time.monotonic()
        turn = tts.turn()
        sender = asyncio.create_task(send(turn, tokens(args.text)))
        audio = bytearray()
        async for piece in turn:
            if not audio:
                print(f"first audio after {(time.monotonic() - started) * 1000:.0f} ms")
            audio += piece
        try:
            await sender
        except FrequencyError as exc:  # a refused voice stops the turn; sending more fails the same way
            if not turn.errors:
                turn.errors.append(exc)
        session = tts.session_id
    for error in turn.errors:
        print(f"error: {error} (category {error.category}, code {error.code})", file=sys.stderr)
    if not audio:
        sys.exit(1)
    with wave.open(args.out, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(args.sample_rate)
        out.writeframes(bytes(audio))
    seconds = len(audio) / 2 / args.sample_rate
    print(f"wrote {args.out}: {seconds:.1f} s at {args.sample_rate} Hz (session {session})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--voice", help="a custom voice id (cv_...) or a built-in voice; default: .voice.json"
    )
    parser.add_argument("--text", help="what to say (default: a Hindi order update)")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--url", default=DEFAULT_WEBSOCKET_URL)
    parser.add_argument("--key-file", help="a file holding the API key (else MIRAI_API_KEY)")
    parser.add_argument("--out", default="reply.wav")
    args = parser.parse_args()
    if args.key_file:
        os.environ["MIRAI_API_KEY"] = Path(args.key_file).read_text().strip()
    asyncio.run(main(args))
