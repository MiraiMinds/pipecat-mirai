# Streaming STT end-to-end test

What a caller's words cost in time on Mirai's streaming transcription socket,
measured the way a phone pipeline uses it. Each call opens one WebSocket on
`/v1/audio/transcriptions/stream` (8 kHz PCM16, manual endpointing) and plays
audio at exactly real time, as a carrier would: `speech_start` with 0.5 s of
pre-roll, the utterance in 60 ms pieces, `speech_end`, a pause, the next
utterance. `procs x calls` of them run at once.

It records, per call and overall:

- connect to `session.begin`, the cost of a new socket;
- `speech_end` to `transcript.final`, the number the caller feels. p50, p95 and
  p99 are nearest-rank, and always printed together;
- partial transcripts: how many, time from `speech_start` to the first, gaps
  between them;
- missing finals (an utterance sent that never got its final), errors, and the
  event loop's lag in the process that drove the calls.

It talks to the socket directly with `websockets`, so the numbers are the
service's and the network's, not a client library's.

| Flag | |
| --- | --- |
| `--url URL` | the streaming endpoint (default: Mirai's sandbox) |
| `--key-file FILE` | file holding the API key (else `MIRAI_API_KEY`) |
| `--procs N`, `--calls N` | worker processes, and calls in each (all at once) |
| `--duration S` | seconds of audio per call (default 30; `0` plays a whole `--audio-dir` file) |
| `--edge` / `--no-edge` | `--edge` mints a token on the API and connects to the edge it names (fails if none is offered); `--edge-url URL` names one yourself. Default is `--no-edge`: `--url` as given |
| `--language`, `--model`, `--sample-rate 8000\|16000` | the socket's settings (defaults `hi-IN`, `mira-stt`, `8000`) |
| `--endpointing manual\|vad` | `manual`: the harness sends `speech_start`/`speech_end` and measures from `speech_end`. `vad`: it streams everything and measures from Mirai's `vad.speech_end` |
| `--stagger S` | seconds between call starts within a process (default 0.2) |
| `--final-p95-ms MS` | the `final_p95_under_ms` limit (default 100) |
| `--audio-dir DIR` | real audio instead of the built-in generator (see below) |
| `--standin` | run against `tests/fake_mirai_stt.py` on a local port, no key |
| `--out DIR` | where `result.json` and `callNN.json` go (default `results/stt-e2e`) |

```bash
unset VIRTUAL_ENV
# against the local stand-in; no key:
uv run python benchmarks/stt-e2e/run.py --standin --procs 2 --calls 5 --duration 20 --out /tmp/stt
# Mirai's API, then the same call mix through its edge:
uv run python benchmarks/stt-e2e/run.py --key-file key.txt --no-edge --procs 4 --calls 5 --out results/stt-api
uv run python benchmarks/stt-e2e/run.py --key-file key.txt --edge --procs 4 --calls 5 --out results/stt-edge
# 10 processes x 5 calls for 10 minutes of audio each, with recorded calls:
uv run python benchmarks/stt-e2e/run.py --key-file key.txt --edge --procs 10 --calls 5 \
    --duration 600 --audio-dir calls/ --out results/stt-soak
```

## Audio

By default each call gets its own synthetic audio from a seeded generator:
speech-like voiced bursts of 1 to 3 seconds with pauses of 0.7 to 2 seconds.
The transcripts are meaningless but every utterance gets a final, which is all
latency needs. (An empty final still counts as one.)

For real audio, put mono 16-bit WAVs in `--audio-dir` (resampled to
`--sample-rate` if they differ). Next to `call.wav`, `call.json` says where the
utterances are, in seconds:

```json
{"utterances": [[1.2, 3.8], [5.1, 6.4]]}
```

Without a sidecar the harness finds utterances by energy, which is good enough
for clean recordings. Calls take the files in turn.

## Result

`result.json` holds the settings, the package version, every summary above, the
flags and the list of errors. `callNN.json` has each call's raw latencies. The
run exits `1` if a flag fails:

| Flag | Passes when |
| --- | --- |
| `no_errors` | no connection, protocol or socket error, every worker exited cleanly, every call reported |
| `no_missing_finals` | every utterance sent got its final (and at least one did) |
| `final_p95_under_ms` | the `speech_end` to final p95 is under `--final-p95-ms` |
