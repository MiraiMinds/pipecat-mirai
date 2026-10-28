# /// script
# requires-python = ">=3.11"
# dependencies = ["fourier-frequency"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Make an instant voice from a recording of one consenting speaker; save its id to .voice.json.

    export MIRAI_API_KEY=sk_live_...
    uv run clone_voice.py --clip asha.wav --name Asha --language hi \\
        --consent-name "Asha Mehta" --relationship employee --i-have-consent

The recording: 10 to 60 seconds of one person speaking naturally, no music,
background noise or echo (30 to 60 seconds keeps an accent best). Use only a
recording of someone who agreed to a synthetic version of their voice being
made; Mirai keeps it as the consent record. The voice is ready in about a
minute, and its id works as ``voice=`` everywhere.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from fourier_frequency import CONSENT_RELATIONSHIPS, DEFAULT_BASE_URL, CustomVoices, FrequencyError


async def main(args):
    async with CustomVoices(base_url=args.base_url) as voices:
        try:
            voice = await voices.clone(
                args.clip,
                name=args.name,
                language=args.language,
                consent_name=args.consent_name,
                consent_relationship=args.relationship,
                consent_attested=args.i_have_consent,
                accent=args.accent,
                gender=args.gender,
            )
        except FrequencyError as exc:
            sys.exit(f"Not made: {exc}")
    Path(args.out).write_text(json.dumps(voice.raw, ensure_ascii=False, indent=2) + "\n")
    length = f", {voice.clip_seconds:g} s of recording" if voice.clip_seconds else ""
    print(f"{voice.id}  {voice.name} ({voice.language}{length}), {voice.status}")
    print(f"saved to {args.out}; next: uv run speak.py --voice {voice.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--clip", required=True, help="the recording (WAV, MP3, FLAC, OGG, WEBM)")
    parser.add_argument("--name", required=True, help="what to call the voice")
    parser.add_argument(
        "--language", required=True, help="the language it will speak: hi, en, gu, ta, te, ..."
    )
    parser.add_argument("--consent-name", required=True, help="the name of the person speaking")
    parser.add_argument("--relationship", required=True, choices=CONSENT_RELATIONSHIPS)
    parser.add_argument(
        "--i-have-consent",
        action="store_true",
        help="the speaker agreed to a synthetic version of their voice being made",
    )
    parser.add_argument("--accent", help="optional: australian, british, indian, ...")
    parser.add_argument(
        "--gender", choices=("female", "male"), help="optional: picks a phone call's backup voice"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key-file", help="a file holding the API key (else MIRAI_API_KEY)")
    parser.add_argument("--out", default=".voice.json")
    args = parser.parse_args()
    if not args.i_have_consent:
        parser.error("--i-have-consent is required: clone only a speaker who agreed to it")
    if args.key_file:
        os.environ["MIRAI_API_KEY"] = Path(args.key_file).read_text().strip()
    asyncio.run(main(args))
