# Phone-call breaks benchmark

Do callers hear the bot's voice break up when the bot's server is busy? This
harness measures it end to end. A real Pipecat bot (`FastAPIWebsocketTransport` +
`TwilioFrameSerializer`, 8 kHz μ-law, `MiraiTTSService`) talks to a simulated phone
provider in a separate process. The provider plays the received audio in real
time from a 60 ms jitter buffer and records every moment it has nothing to play.

The bot's event loop gets two kinds of load, per call:

- steady work: 1 ms of CPU every 20 ms, like VAD or STT processing
- stalls: on average every 3 s, the loop is blocked for 50–250 ms, the way a
  synchronous HTTP call, a large JSON parse or a CPU-heavy step blocks a real bot

Each call speaks seven lines (about 45 s of speech) at fixed offsets over 80 s.

## Results

Pipecat 1.8.1, real Mirai audio (recorded engine streams replayed on their
original timing), 4-core arm64 Linux host:

| Concurrent calls | Output pacing | Breaks per sentence | Silence inserted per sentence | Speech stretched |
|---|---|---|---|---|
| 3 | stock Pipecat | 3.46 | 365 ms | 6.6% |
| 3 | `apply_output_lead(0.4)` | 0.04 | 4 ms | **0.08%** |
| 10 | stock Pipecat | 14.54 | 1,608 ms | 29.1% |
| 10 | `apply_output_lead(0.4)` | 0.01 | 1 ms | **0.01%** |
| 3 | stock, with the stand-in server below | 3.88 | 383 ms | 6.3% |
| 3 | `apply_output_lead(0.4)`, with the stand-in | 0 | 0 ms | **0%** |

Stock Pipecat sends audio at exactly real time, so the provider never holds more
than one 40 ms chunk, and every stall longer than that is heard as a gap. The
damage grows with the number of calls, because each call adds stalls to the same
event loop. This doesn't depend on the TTS vendor.

## Running it

```bash
uv venv && uv pip install -e ../.. "pipecat-ai[websocket]" uvicorn aiohttp websockets

# Mirai stand-in: streams a 48 kHz mono WAV (10 s or more) with Mirai's delivery timing.
python phone_breaks.py standin --wav speech48k.wav &
# Or skip the stand-in: export MIRAI_API_KEY=... and pass --tts-url https://sandbox.voice.miraiminds.co/v1

python phone_breaks.py bot --port 8765 --lead 0 &      # stock Pipecat pacing
python phone_breaks.py bot --port 8766 --lead 0.4 &    # with apply_output_lead

python phone_breaks.py phone --ws-url ws://127.0.0.1:8765/ws --calls 3 --out results/stock_c3
python phone_breaks.py phone --ws-url ws://127.0.0.1:8766/ws --calls 3 --out results/lead_c3
```

Each run prints a JSON summary, writes `result.json` with per-sentence numbers,
and saves `caller_hears_call0.wav`, which is what the first caller heard.
