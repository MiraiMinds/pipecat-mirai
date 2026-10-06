# Customer end-to-end test

What a customer's phone bot gets from pipecat-mirai, measured where the caller
is. Each call runs the bot from our README: a fake LLM streams a Hindi/Hinglish
reply token by token (~40 tokens/s) into `MiraiWebsocketTTSService` (or
`MiraiTTSService` with `--tts http`) in an 8 kHz pipeline, out through
`FastAPIWebsocketTransport` with the Twilio serializer (μ-law) and
`apply_output_lead(transport, 0.4)`. A phone simulator plays what arrives at
exactly real time, as a carrier would, and saves what the caller heard.

Turn loop: a reply, then 2–6 s of the caller speaking, for `--duration`.
`--barge-in 0.2` interrupts a fifth of the replies mid-way (an `InterruptionFrame`,
so Pipecat sends the carrier a `clear`).

Bots run in `--procs` processes (uvicorn + FastAPI); the phone runs in the parent
process so a stalled bot can't skew the carrier's clock.

```bash
unset VIRTUAL_ENV
# against the protocol stand-in (tests/fake_mirai_ws.py); no key:
uv run --with uvicorn python benchmarks/customer-e2e/run.py --standin --calls 12 --duration 60 --out /tmp/e2e
# against a server:
uv run --with uvicorn python benchmarks/customer-e2e/run.py \
    --url ws://localhost:8130/v1/audio/speech/stream --key-file key.txt \
    --calls 12 --duration 180 --out results/e2e
```

Per call and overall: TTFB (the LLM's first token to the first audio at the
phone; it includes sentence aggregation and the 150 ms first-audio buffer),
audible gaps (a 20 ms tick with too little audio while more of the reply was
still coming) and the silence they add, stretch %, audio arriving after a
`clear` (must be 0), sentence order (WebSocket only), ErrorFrames, capacity
retries, characters sent vs characters Mirai billed, and the CPU and event-loop
lag of every process. Flags: no errors, no audible gaps, no audio after a clear,
sentences in order, every reply heard, client CPU under 60% of the host.
`result.json` has every turn; `callNN.wav` is what each caller heard.
`--standin-speed 0.8` (slower than real time) must fail the gap flag: a check
that the measurement works.
