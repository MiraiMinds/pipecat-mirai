# /// script
# requires-python = ">=3.11"
# dependencies = ["fourier-frequency"]
# ///
#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Evaluate a voice the way a phone agent uses it: speed, intelligibility, likeness, against a built-in voice.

    uv run evaluate.py                                   # the voice in .voice.json, against shruti
    uv run evaluate.py --voice cv_... --runs 3 --clip asha.wav
    uv run evaluate.py --voice cv_... --transport http --baseline none

Each run speaks the same phone-agent lines (greetings, numbers, dates, money,
brand names) in every voice under test, one sentence at a time on one open
stream per voice, and records for each sentence:

- first audio: from sending the sentence to its first audio byte, and total
  time to its last; the real-time factor (synthesis time over audio length);
- characters billed;
- what Mira STT hears in the audio, as character and word error rates
  against the text sent (``--no-asr`` skips it);
- with ``--clip`` (the recording the voice was made from) and ``resemblyzer``
  installed, how alike the speaker sounds to the recording (cosine similarity
  of speaker embeddings, 1.0 = identical).

Results go to ``--out``: every sentence in ``results.jsonl``, the WAVs under
``audio/``, and ``report.md``; the summary is printed. Everything is billed as
ordinary speech and transcription (a 3-run evaluation of 12 lines in two voices
is about 4 minutes of audio).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import wave
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import evalkit
import numpy as np
from fourier_frequency import (
    DEFAULT_BASE_URL,
    CustomVoices,
    FrequencyError,
    HttpTTS,
    STTListener,
    STTStream,
    TTSListener,
    TTSStream,
    is_custom_voice,
)

SENTENCE_TIMEOUT_SECS = 30.0
TRANSCRIPT_TIMEOUT_SECS = 20.0
PAD_SECS = 0.25  # silence around each clip sent to speech-to-text


@dataclass
class Take:
    """One sentence being spoken."""

    started: float = field(default_factory=time.monotonic)
    first: float | None = None
    ended: float | None = None
    audio: bytearray = field(default_factory=bytearray)
    chars: int = 0
    errors: list = field(default_factory=list)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class Collector(TTSListener):
    """Times each sentence: its first audio byte, its audio, what it was billed, what went wrong."""

    def __init__(self):
        self.takes: dict[str, Take] = {}
        self.current: Take | None = None

    def begin(self, turn_id: str) -> Take:
        take = self.takes[turn_id] = self.current = Take()
        return take

    async def on_first_audio(self, turn_id: str) -> None:
        take = self.takes.get(turn_id)
        if take is not None and take.first is None:
            take.first = time.monotonic()

    async def on_audio(self, turn_id: str, audio: bytes) -> None:
        if turn_id in self.takes:
            self.takes[turn_id].audio += audio

    async def on_turn_done(self, turn_id: str) -> None:
        if turn_id in self.takes:
            self.takes[turn_id].done.set()

    async def on_usage(self, chars: int) -> None:
        if self.current is not None:
            self.current.chars += chars

    async def on_text_billed(self, text: str) -> None:
        if self.current is not None:
            self.current.chars += len(text)

    async def on_error(self, error: FrequencyError, turn_id: str | None = None) -> None:
        take = self.takes.get(turn_id) if turn_id else self.current
        if take is not None:
            take.errors.append(error)
            if error.permanent:
                take.done.set()


class Voice:
    """One voice under test and its open stream (or HTTP engine)."""

    def __init__(self, name: str, role: str, args):
        self.name, self.role = name, role
        self.collector = Collector()
        self.transport = args.transport
        common = dict(voice=name, sample_rate=args.sample_rate, prebuffer_secs=0.0, listener=self.collector)
        if args.transport == "websocket":
            self.engine = TTSStream(base_url=args.base_url, **common)
        else:
            self.engine = HttpTTS(base_url=args.base_url, **common)

    async def open(self):
        if isinstance(self.engine, TTSStream):
            await self.engine.connect()
        else:
            self.engine.start()

    @property
    def socket(self) -> str:
        """Where the audio came from: Mirai's API socket, the edge, or HTTP."""
        tts = self.engine
        if not isinstance(tts, TTSStream) or tts.fallback:
            return "http"
        return "api" if tts.connected_url == tts.url else "edge"

    async def speak(self, turn_id: str, text: str) -> Take:
        take = self.collector.begin(turn_id)
        try:
            if isinstance(self.engine, TTSStream):
                await self.engine.send_text(turn_id, text)
                await self.engine.end_turn(turn_id)
                await asyncio.wait_for(take.done.wait(), SENTENCE_TIMEOUT_SECS)
            else:
                async for piece in self.engine.speak(text, turn_id):
                    take.audio += piece
        except FrequencyError as exc:
            if exc not in take.errors:
                take.errors.append(exc)
        except TimeoutError:
            take.errors.append(f"no end within {SENTENCE_TIMEOUT_SECS:g} s")
            await self.engine.cancel_turn(turn_id)
        take.ended = time.monotonic()
        return take

    async def close(self):
        if isinstance(self.engine, TTSStream):
            await self.engine.close()
        else:
            await self.engine.aclose()


class Finals(STTListener):
    def __init__(self):
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.errors: list[FrequencyError] = []

    async def on_final(self, text: str, language: str | None, event: dict) -> None:
        self.queue.put_nowait(text)

    async def on_error(self, error: FrequencyError) -> None:
        self.errors.append(error)


async def transcribe(rows: list[dict], language: str, rate: int, url: str) -> str:
    """Fill each row's transcript, CER and WER from Mira STT; say how it went."""
    finals = Finals()
    stt = STTStream(
        url=url, language=f"{language}-IN", input_sample_rate=rate, endpointing="manual", listener=finals
    )
    await stt.connect()
    pad = bytes(int(rate * PAD_SECS) * 2)
    heard = 0
    try:
        for row in rows:
            if not row.get("pcm"):
                continue
            await stt.speech_started()
            stt.send(pad + row["pcm"] + pad)
            await stt.speech_stopped()
            try:
                text = await asyncio.wait_for(finals.queue.get(), TRANSCRIPT_TIMEOUT_SECS)
            except TimeoutError:
                row["transcript_error"] = f"no transcript within {TRANSCRIPT_TIMEOUT_SECS:g} s"
                continue
            heard += 1
            row["transcript"] = text
            row["cer"] = evalkit.cer(row["heard"], text)
            row["wer"] = evalkit.wer(row["heard"], text)
    finally:
        await stt.close()
    problems = f"; errors: {'; '.join(map(str, finals.errors))}" if finals.errors else ""
    clips = sum(1 for r in rows if r.get("pcm"))
    return f"Mira STT ({language}-IN), {heard} of {clips} clips transcribed{problems}"


def similarity_scorer(clip: str | None):
    """A function from (pcm, rate) to speaker similarity with ``clip``, and how it is scored (or why not)."""
    if not clip:
        return None, "not scored (pass --clip, the recording the voice was made from)"
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return None, "not scored (pip install resemblyzer to score it)"
    encoder = VoiceEncoder(verbose=False)
    reference = encoder.embed_utterance(preprocess_wav(Path(clip)))

    def score(pcm: bytes, rate: int) -> float:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return float(np.dot(reference, encoder.embed_utterance(preprocess_wav(samples, source_sr=rate))))

    return score, f"speaker-embedding cosine similarity with {Path(clip).name}"


def stt_url(base_url: str) -> str:
    """``https://host/v1`` -> ``wss://host/v1/audio/transcriptions/stream``."""
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/v1/audio/transcriptions/stream"


def write_wav(path: Path, pcm: bytes, rate: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm)


async def language_of(voice: str, args) -> str:
    """The language to test in: the custom voice's own, else ``--language``."""
    if args.language:
        return args.language
    if not is_custom_voice(voice):
        return "hi"
    async with CustomVoices(base_url=args.base_url) as voices:
        try:
            found = await voices.get(voice)
        except FrequencyError as exc:
            sys.exit(f"Can't read voice {voice}: {exc}")
    print(f"{found.id}: {found.name}, {found.language}, {found.kind}")
    return found.language


async def main(args):
    voice = args.voice or json.loads(Path(".voice.json").read_text())["id"]
    language = await language_of(voice, args)
    if args.sentences:
        lines = evalkit.read_sentences(Path(args.sentences).read_text())
    else:
        lines, built_in = evalkit.sentences_for(language)
        if built_in != language.split("-", 1)[0]:
            print(f"no built-in lines in {language}; using English (pass --sentences for your own)")
    lines = lines[: args.limit] if args.limit else lines
    baseline = args.baseline
    if baseline == "auto":
        baseline = "neha" if voice == "shruti" else "shruti"
    voices = [Voice(voice, "voice under test", args)]
    if baseline and baseline != "none" and baseline != voice:
        voices.append(Voice(baseline, "baseline", args))
    out = Path(args.out or f"eval-results/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for v in voices:
        await v.open()
    try:
        if not args.no_warmup:
            for v in voices:  # one untimed sentence: the first one on a socket isn't what a call hears
                await v.speak(f"{v.name}-warmup", lines[0][0])
        for run in range(1, args.runs + 1):
            for v in voices:
                for index, (text, heard) in enumerate(lines, 1):
                    take = await v.speak(f"{v.name}-r{run}-s{index}", text)
                    audio_secs = len(take.audio) / 2 / args.sample_rate
                    total = (take.ended - take.started) if take.ended else None
                    error = "; ".join(map(str, take.errors)) if not take.audio and take.errors else None
                    row = {
                        "voice": v.name,
                        "role": v.role,
                        "transport": v.transport,
                        "socket": v.socket,
                        "run": run,
                        "index": index,
                        "text": text,
                        "heard": heard,
                        "first_audio_ms": (take.first - take.started) * 1000 if take.first else None,
                        "total_ms": total * 1000 if total is not None else None,
                        "audio_secs": audio_secs,
                        "rtf": total / audio_secs if total is not None and audio_secs > 0 else None,
                        "chars_billed": take.chars,
                        "error": error,
                        "warnings": [str(e) for e in take.errors] if take.audio and take.errors else [],
                        "pcm": bytes(take.audio),
                    }
                    if take.audio:
                        row["wav"] = f"audio/{v.name}/r{run}-s{index:02d}.wav"
                        write_wav(out / row["wav"], row["pcm"], args.sample_rate)
                    rows.append(row)
                    first = row["first_audio_ms"]
                    print(
                        f"{v.name:<30} run {run} line {index:>2}: "
                        + (f"first audio {first:.0f} ms, {audio_secs:.1f} s" if first else f"error: {error}")
                    )
    finally:
        for v in voices:
            await v.close()

    asr = "skipped (--no-asr)"
    if not args.no_asr:
        asr = await transcribe(rows, language.split("-", 1)[0], args.sample_rate, stt_url(args.base_url))
    score, similarity = similarity_scorer(args.clip)
    if score is not None:
        for row in rows:
            if row.get("pcm"):
                row["similarity"] = score(row["pcm"], args.sample_rate)

    for row in rows:
        row.pop("pcm", None)
    summary = evalkit.summarize(rows)
    meta = {
        "when": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "transport": args.transport,
        "sample_rate": args.sample_rate,
        "runs": args.runs,
        "sentence_count": len(lines),
        "language": language,
        "asr": asr,
        "similarity": similarity,
        "voices": [v.name for v in voices],
    }
    with open(out / "results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary}, ensure_ascii=False, indent=2)
    )
    (out / "report.md").write_text(evalkit.report(summary, rows, meta))
    print()
    print(evalkit.table(summary))
    print(f"\n{asr}\nsimilarity: {similarity}")
    print("sockets: " + ", ".join(f"{s['voice']} → {'/'.join(s['sockets'])}" for s in summary))
    print(f"results in {out}/ (results.jsonl, report.md, audio/)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--voice", help="the voice to evaluate (default: .voice.json)")
    parser.add_argument("--baseline", default="auto", help="a built-in voice to compare with, or none")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--transport", choices=("websocket", "http"), default="websocket")
    parser.add_argument("--clip", help="the recording the voice was made from (speaker similarity)")
    parser.add_argument("--language", help="the lines' language (default: the custom voice's own, else hi)")
    parser.add_argument(
        "--sentences", help="a file of your own lines, one per line (optionally: text<TAB>how it is heard)"
    )
    parser.add_argument("--limit", type=int, help="use only the first N lines")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--no-asr", action="store_true", help="skip the speech-to-text check")
    parser.add_argument("--no-warmup", action="store_true", help="time the first sentence on each socket too")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--key-file", help="a file holding the API key (else MIRAI_API_KEY)")
    parser.add_argument("--out", help="where the results go (default: eval-results/<time>)")
    args = parser.parse_args()
    if args.key_file:
        os.environ["MIRAI_API_KEY"] = Path(args.key_file).read_text().strip()
    asyncio.run(main(args))
