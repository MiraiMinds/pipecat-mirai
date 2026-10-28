# /// script
# requires-python = ">=3.11"
# dependencies = ["fourier-frequency"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""List the workspace's own voices, or delete one (and its recording).

    uv run cleanup.py --list
    uv run cleanup.py --voice cv_...       # or the voice in .voice.json

A voice an agent still speaks with can't be deleted: Mirai names the agents,
and you move them to another voice first.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from fourier_frequency import DEFAULT_BASE_URL, CustomVoices, FrequencyError


async def main(args):
    async with CustomVoices(base_url=args.base_url) as voices:
        if args.list:
            for voice in await voices.list():
                length = f"{voice.clip_seconds:g} s" if voice.clip_seconds else "–"
                print(f"{voice.id}  {voice.name:<20} {voice.language:<3} {voice.kind:<8} {length}")
            return
        voice_id = args.voice or json.loads(Path(".voice.json").read_text())["id"]
        try:
            await voices.delete(voice_id)
        except FrequencyError as exc:
            sys.exit(f"Not deleted: {exc}")
    print(f"deleted {voice_id}")
    if not args.voice and Path(".voice.json").exists():
        Path(".voice.json").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--voice", help="the voice to delete (default: .voice.json)")
    parser.add_argument("--list", action="store_true", help="list the workspace's voices instead")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key-file", help="a file holding the API key (else MIRAI_API_KEY)")
    args = parser.parse_args()
    if args.key_file:
        os.environ["MIRAI_API_KEY"] = Path(args.key_file).read_text().strip()
    asyncio.run(main(args))
