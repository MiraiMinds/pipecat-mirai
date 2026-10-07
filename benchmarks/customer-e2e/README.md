# Customer end-to-end test

What a customer's phone bot gets from pipecat-mirai, measured where the caller
is. Each call runs the bot from our README: a fake LLM streams a Hindi/Hinglish
reply token by token (~40 tokens/s) into `MiraiWebsocketTTSService` (or
`MiraiTTSService` with `--tts http`) in an 8 kHz pipeline, out through
`FastAPIWebsocketTransport` with the Twilio serializer (μ-law) and
`apply_output_lead(transport, 0.4)`. A phone simulator plays what arrives at
exactly real time, as a carrier would, and saves what the caller heard.

`--defaults` builds the bot the way a customer who has only installed the
package writes it: `MiraiTTSService(api_key=key, voice=voice)` (or
`MiraiWebsocketTTSService(...)`) and nothing else, and no `apply_output_lead`
call (whatever lead the library applies on its own is reported as "seen on the
transport"). A URL is passed only with `--standin`, `--url` or `--http-url`:
otherwise the library's default endpoint (production) is used. The run records
the mode and the `pipecat_mirai.__version__` the bot process imported.

Turn loop: a reply, then 2–6 s of the caller speaking, for `--duration`.
`--barge-in 0.2` interrupts a fifth of the replies mid-way (an `InterruptionFrame`,
so Pipecat sends the carrier a `clear`). The first reply is the greeting: with
`--greeting-now` (on by default with `--defaults`) it starts the moment the call
connects, is never interrupted, and its TTFB is reported apart from the later
turns'.

Bots run in `--procs` processes (uvicorn + FastAPI); the phone runs in the parent
process so a stalled bot can't skew the carrier's clock.

| Flag | |
| --- | --- |
| `--defaults` | customer-default construction, no harness lead (see above) |
| `--greeting-now` / `--no-greeting-now` | greeting at connect instead of after a 1–2 s answer delay |
| `--stagger S` | seconds between call starts, from one shared instant; `0` starts every call at once |
| `--seed-call S`, `--seed-gap G` | first one S-second call per bot process, then G s idle (default 5), then the burst; seed calls are reported apart and left out of the stats |
| `--stall-every S`, `--stall-ms LO,HI` | in each bot, block the event loop for LO–HI ms about every S s (±30%), like a busy server; counted in `bot_procN.json` and the summary |
| `--max-runtime S` | cap on the whole run (default 150 s): calls still running are hung up, the bots killed, and the `within_max_runtime` flag fails |
| `--standin` | local fakes: `tests/fake_mirai_ws.py` for `--tts ws`, an HTTP `/v1/audio/speech` + `/v1/models` fake for `--tts http` (150 ms to first byte, then `--standin-speed` × real time) |

```bash
unset VIRTUAL_ENV
# against the stand-ins; no key:
uv run --with uvicorn python benchmarks/customer-e2e/run.py --standin --calls 12 --duration 60 --out /tmp/e2e
uv run --with uvicorn python benchmarks/customer-e2e/run.py --standin --defaults --tts http \
    --calls 4 --stagger 0 --seed-call 5 --duration 15 --out /tmp/e2e-http
# what a customer gets from the defaults, against production: 10 calls at once
# on a server that served one call 5 s before
uv run --with uvicorn python benchmarks/customer-e2e/run.py --defaults --tts http --calls 10 --procs 1 \
    --stagger 0 --seed-call 10 --duration 100 --barge-in 0.2 --key-file key.txt --out results/x
# the same on a busy server
uv run --with uvicorn python benchmarks/customer-e2e/run.py --defaults --tts http --calls 10 --procs 1 \
    --stagger 0 --seed-call 10 --duration 100 --barge-in 0.2 --stall-every 2 --stall-ms 150,300 \
    --key-file key.txt --out results/x-stalls
# against a given server:
uv run --with uvicorn python benchmarks/customer-e2e/run.py \
    --url ws://localhost:8130/v1/audio/speech/stream --key-file key.txt \
    --calls 12 --duration 100 --out results/e2e
```

Per call and overall: TTFB (the LLM's first token to the first audio at the
phone; it includes sentence aggregation and the 150 ms first-audio buffer) for
all turns, for the greeting (turn 0) and for later turns, plus the greeting's
time from the phone connecting to its first audio; audible gaps (a 20 ms tick
with too little audio while more of the reply was still coming) and the silence
they add, stretch %, audio arriving after a `clear` (must be 0), sentence order
(WebSocket only), ErrorFrames, capacity retries, characters sent vs characters
Mirai billed, the bots' injected stalls, and the CPU and event-loop lag of every
process. Percentiles are nearest-rank (with ~10 samples, p95 is the maximum).
Flags: no errors, no audible gaps, no audio after a clear, sentences in order,
every reply heard, client CPU under 60% of the host, finished within
`--max-runtime`. `result.json` has every turn; `callNN.wav` is what each caller
heard (`seedN.wav` for seed calls).
`--standin-speed 0.8` (slower than real time) must fail the gap flag: a check
that the measurement works.
