# Cookbook: your own voice, streamed and measured

Make an instant voice from a short recording, speak with it over the
WebSocket the way a phone agent does, and measure it against a built-in Mira
TTS voice before you put it in front of callers. Four small scripts, each
runnable on its own:

| Step | Script | What it does |
| --- | --- | --- |
| 1 | `clone_voice.py` | uploads the recording, gets back a voice id (`cv_...`) |
| 2 | `speak.py` | streams a reply in that voice, token by token, to a WAV |
| 3 | `evaluate.py` | times, transcribes and compares it on phone-agent lines |
| 4 | `cleanup.py` | lists the workspace's voices, deletes one |

```bash
export MIRAI_API_KEY=sk_live_...      # or pass --key-file to any script
cd packages/fourier-frequency/cookbook/instant-voice
```

The scripts carry their dependencies inline, so `uv run <script>` installs
`fourier-frequency` for them. From a checkout of this repository, `uv sync`
at its root and run them with `../../../../.venv/bin/python <script>` instead.

## Before you record: consent

Clone only a person who agreed to a synthetic version of their voice being
made: yourself, an employee or contractor, or a customer, with their
permission. The API asks who they are (`--consent-name`), how you know them
(`--relationship`: self, employee, contractor, customer or other) and for your
confirmation (`--i-have-consent`), and keeps the recording as the consent
record. Never clone a public figure or a voice you found online.

What makes a good recording:

- **10 to 60 seconds** of one person talking naturally; 30 to 60 seconds keeps
  an accent best. A WAV shorter than 5 seconds is refused outright.
- **One speaker**, no music, no background noise, no echo. A quiet room and a
  phone held close beat a good microphone in a big room.
- **The language the voice will speak.** A Hindi voice from Hindi speech; read
  something with numbers and English words in it if your agent says those.
- WAV, MP3, FLAC, OGG, WEBM, at most 16 MB.

## 1. Make the voice

```bash
uv run clone_voice.py --clip asha.wav --name Asha --language hi \
    --consent-name "Asha Mehta" --relationship employee --i-have-consent --gender female
```

```
cv_01m6...  Asha (hi, 41.2 s of recording), ready
saved to .voice.json; next: uv run speak.py --voice cv_01m6...
```

It takes about a minute. The id is what `voice=` takes from now on, in this
package, in `pipecat-mirai` and as an agent's `voice.voice_id` on phone and
web calls. `--gender` picks the built-in voice a phone call falls back to.

In your own code it is one call:

```python
from fourier_frequency import CustomVoices

async with CustomVoices() as voices:
    voice = await voices.clone(
        "asha.wav",
        name="Asha",
        language="hi",
        consent_name="Asha Mehta",
        consent_relationship="employee",
        consent_attested=True,
    )
    sample = await voices.preview(voice.id, "नमस्ते, मैं आपकी कैसे मदद कर सकती हूँ?")  # WAV bytes
```

A refusal is a `FrequencyError` whose `code` and message say what to change:
`clip_rejected` (too short, too long, not one clear speaker: the message
says which), `file_too_large`, `unsupported_file_type`, `limit_reached` (10
voices per workspace: delete one) or, briefly, `upstream_error` (try again in
a few minutes; `error.permanent` is `False`).

## 2. Speak with it

```bash
uv run speak.py                                      # the voice in .voice.json
uv run speak.py --text "Your order ships tomorrow. Anything else I can help with?"
```

```
voice cv_01m6... on Mirai's API socket
first audio after 310 ms
wrote reply.wav: 4.6 s at 24000 Hz (session ttsws_01M6...)
```

Nothing changes in your code but the voice:

```python
async with TTSStream(voice=voice.id, sample_rate=8000, encoding="mulaw", text_mode="token") as tts:
    turn = tts.turn()
    ...
```

A custom voice streams from Mirai's API socket; built-in voices may use the
edge next to the GPUs. `TTSStream` picks by itself, and moves an open stream
to the API socket if you switch to a custom voice mid-call with
`await tts.update(voice=...)`. If Mirai doesn't know the voice (another
workspace's, or deleted), the stream says so with a `FrequencyError` and
speaks nothing rather than speak in a different voice; `update()` to a
voice it knows and it carries on.

## 3. Evaluate it

```bash
uv run evaluate.py --clip asha.wav                   # .voice.json against shruti, 3 runs
uv run evaluate.py --voice cv_... --baseline neha --runs 5 --transport http
```

Each run speaks the same phone-agent lines (greetings, an order number, a
date, rupee amounts, a time, brand names, the English words Indian agents mix
in) in the voice under test and in a built-in baseline voice, one sentence
at a time on one open stream each, as a call would. For every sentence it
records:

- **first audio**: from sending the sentence to its first audio byte, the
  silence a caller hears before the agent speaks; and **total**, to its last;
- **RTF**: total time over the audio's length (below 1: audio arrives faster
  than it plays, so a reply never stalls);
- **CER and WER**: what Mira STT hears in the audio against how the line
  should be heard (numbers as words, English words in the line's script);
- **similarity** (with `--clip` and `pip install resemblyzer`): how alike the
  speaker sounds to the recording, from speaker embeddings (1.0 = identical);
- characters billed, and which socket served it.

Lines are built in for Hindi, English and Gujarati and picked from the
voice's language; `--sentences my-lines.txt` uses yours (one per line,
optionally `text<TAB>how it is heard`). Results go to `eval-results/<time>/`:
every sentence in `results.jsonl`, each clip under `audio/` to listen to, and
`report.md` with every transcript next to what it should have been.

A run against two built-in voices, two runs of the twelve Hindi lines:

```
voice   role              n   errors  first audio p50 ms  p95 ms  total p50 ms  RTF p50  CER   WER   similarity
------  ----------------  --  ------  ------------------  ------  ------------  -------  ----  ----  ----------
shruti  voice under test  24  0       303                 840     2102          0.52     0.5%  2.1%  –
neha    baseline          24  0       281                 438     1776          0.46     0.5%  1.8%  –
```

Reading it:

- **First audio** is what a caller notices. Compare p95 as well as p50: one
  slow sentence in twenty is a pause in every other call.
- **CER/WER near the baseline's** means the voice is as easy to understand.
  Read the lines that differ in `report.md` before you trust a number: a
  voice that reads 48213 as "चार आठ दो एक तीन" where the line expects the
  English digits ("फोर एट टू वन थ्री") scores worse while being understood,
  and that may still be a change your callers notice. Fix it in your
  prompt's wording, or pick the reading you want with `--sentences`.
- **Similarity** is a rough guide: around 0.75 and up usually sounds like the
  same person; much lower, record again (longer, quieter, one speaker). Compare with the baseline
  row: a built-in voice scores low against your recording, as it should.
- **Listen.** Ten clips from `audio/` tell you more about warmth and pace
  than any number here.

An evaluation is billed as ordinary speech and transcription: three runs of
twelve lines in two voices is about four minutes of audio.

## 4. Clean up

```bash
uv run cleanup.py --list
uv run cleanup.py                                    # deletes the voice in .voice.json
```

Deleting removes the voice and its recording. Mirai refuses (`voice_in_use`)
while an agent speaks with it, and names the agents; move them first.
