# fourier-frequency

Mirai's streaming text-to-speech and speech-to-text for **any Python voice
agent**: your own orchestrator, a FastAPI phone bridge, a worker you wrote
years ago. Natural voices in Hindi, Hinglish, Gujarati,
English, Tamil and Telugu; transcripts in ten Indian languages.

This is the engine inside [`pipecat-mirai`](https://pypi.org/project/pipecat-mirai/),
without Pipecat. Same code, same defaults, same tests, so a bot you build here
holds up under load the way a Pipecat bot does:

- **Fast first audio.** Sockets open on Mirai's edge next to the speech GPUs,
  and a process-wide pool keeps a few open and authenticated ahead of need, so
  a call that starts takes one that is already there.
- **Calls that survive the network.** A dropped socket is reopened and the
  reply it cut short is resent from the first sentence the caller hasn't heard;
  speech-to-text replays the audio Mirai hadn't answered. A busy Mirai (429) is
  waited out for a moment, not reported at once. If WebSockets can't get
  through at all, text-to-speech carries on over HTTP.
- **Phone audio as it is.** Ask for 8 kHz mu-law and get 8 kHz mu-law, in
  40 ms pieces; a `Pacer` sends it to a media stream at real time with 0.4 s
  to spare.

```bash
uv add fourier-frequency     # or: pip install fourier-frequency
export MIRAI_API_KEY=sk_live_...
```

Python 3.11 or later. Dependencies: httpx, websockets, numpy, soxr. No
framework.

## Text to speech

One `TTSStream` per call, one `TTSTurn` per reply. Send text while you read
audio; the audio starts long before the text ends.

```python
import asyncio
from fourier_frequency import TTSStream


async def reply(tts: TTSStream, tokens, phone):
    turn = tts.turn()

    async def send():
        async for token in tokens:  # your LLM's stream
            await turn.send(token)
        await turn.end()  # speak what's left

    sender = asyncio.create_task(send())
    async for audio in turn:  # 8 kHz mu-law, 40 ms at a time
        await phone.send(audio)
    await sender


async with TTSStream(voice="shruti", sample_rate=8000, encoding="mulaw", text_mode="token") as tts:
    await reply(tts, llm_tokens(), phone)
```

- **`text_mode`**: `"sentence"` (default) when you send whole sentences, each
  spoken at once; `"token"` when you send LLM tokens as they come (Mirai cuts
  the sentences, `turn.end()` speaks the rest).
- **Barge-in**: `await turn.cancel()` cancels the reply on Mirai's side and
  stops its audio here; `await tts.interrupt()` does it for every turn.
- **Voices**: `ashu`, `neha`, `shruti`, `sameer`. `await tts.update(voice=...)`
  applies from the next sentence.
- **Rates**: any `sample_rate`. Mirai is asked for it directly when it serves
  it (8, 16, 22.05, 24, 44.1 or 48 kHz); anything else is resampled here.
- **Errors**: every error is a `FrequencyError` with a `category`
  (`authentication`, `quota`, `rate_limit`, `connectivity`, ...), Mirai's
  `code`, the `request_id` and whether it is `permanent`. A sentence Mirai
  refuses lands in `turn.errors` and the turn goes on; a refused key or an
  empty wallet raises from `turn.send()`.

`HttpTTS` speaks one sentence per HTTP request (`async for audio in
tts.speak(text)`), for when you want no sockets at all.

### Your own listener

`TTSStream` tells a `TTSListener` everything: audio for a turn id, the turn
done, first audio (time to first byte), characters billed, errors, connects
and disconnects. Hand it one and drive the stream by your own ids instead of
`TTSTurn` objects; this is how the Pipecat service is built:

```python
from fourier_frequency import TTSListener, TTSStream


class ToMyPipeline(TTSListener):
    async def on_audio(self, turn_id, audio): ...
    async def on_turn_done(self, turn_id): ...
    async def on_usage(self, chars): ...
    async def on_error(self, error, turn_id=None): ...


tts = TTSStream(listener=ToMyPipeline(), sample_rate=16000)
await tts.connect()
await tts.send_text("reply-1", "नमस्ते! आपका order कल deliver होगा।")
await tts.end_turn("reply-1")
```

## Speech to text

One `STTStream` per call. `send()` never blocks: audio is held until the
socket is up and replayed on a new one after a drop. Transcripts come back as
events (or to an `STTListener`).

```python
from fourier_frequency import SpeechStarted, STTStream, Transcript

async with STTStream(language="hi-IN", input_sample_rate=8000) as stt:
    # elsewhere, as the caller's audio arrives: stt.send(pcm16)
    async for event in stt.events():
        if isinstance(event, SpeechStarted):
            ...  # the caller started talking: stop the bot (barge-in)
        elif isinstance(event, Transcript) and event.final:
            print(event.text, event.language, event.latency_ms)
```

- **Languages**: `hi-IN` (Hindi and Hinglish), `en-IN`, `gu-IN`, `mr-IN`,
  `pa-IN`, `bn-IN`, `ta-IN`, `te-IN`, `kn-IN`, `ml-IN`. Add
  `secondary_languages=["en-IN"]` when callers switch between known ones;
  `language="auto"` when you don't know. `await stt.update(language=...)`
  changes it on the open socket.
- **Endpointing**: `"vad"` (default): Mirai's VAD ends each utterance and
  says so (`SpeechStarted`, `SpeechStopped`). `"manual"`: your VAD; call
  `await stt.speech_started()` and `await stt.speech_stopped()` (the last
  0.5 s before the start goes with it, so the first word isn't cut).
- **Audio**: 16-bit PCM at `input_sample_rate`. 8 kHz goes as it is;
  anything else is resampled to 16 kHz. `encoding="mulaw"` (or `"alaw"`)
  halves the upload at 8 kHz. `fourier_frequency.audio.ulaw2lin` decodes a
  phone stream's mu-law for `send()`.
- **Close**: `close()` (or leaving `async with`) waits briefly for the last
  final and Mirai's bill (`audio_seconds_billed`, `cost_paise`);
  `close(graceful=False)` doesn't.
- **Errors**: a `StreamError(error)` event with a `FrequencyError`; after a
  `permanent` one the stream has stopped.

## Phone media streams

[`examples/02-phone-bridge.py`](examples/02-phone-bridge.py) is a whole
phone agent on Twilio Media Streams in about 170 lines of FastAPI: caller
audio into `STTStream`, your orchestrator's reply out of `TTSStream` as 8 kHz
mu-law, paced by `Pacer`, with barge-in (cancel the turn, tell Twilio to
`clear`, reset the pacer). Plivo and Exotel streams differ only in their JSON
field names.

```python
pacer = Pacer(8000, encoding="mulaw")  # 0.4 s ahead of real time
async for audio in pacer.pace(turn):
    await ws.send_json(
        {"event": "media", "streamSid": sid, "media": {"payload": base64.b64encode(audio).decode()}}
    )
```

Sent exactly at real time, any stall on your server is a gap the caller
hears; sent all at once, a barge-in can't stop what the far end already
holds. The pacer keeps 0.4 s buffered there: in our phone benchmark that took
stretched audio from 6.6% (3 calls at once) and 29% (10 calls) to under 0.1%.

## Before the first call

```python
from fourier_frequency import prewarm

await prewarm(websocket=8, stt_websockets=8)  # in your server's startup, same event loop
```

opens sockets for the calls you expect to start together, so none of their
first sentences waits for a TCP and TLS handshake. Streams with the same URL,
key and fixed settings take them as they connect (STT sockets wait as
linear16 with the default model; pass `stt_endpointing` and
`stt_sample_rate` to match your calls). Without it, the first stream to
connect starts the pool and it keeps more sockets ready than the recent peak
of calls (none for text-to-speech when your workspace is billed per
connected minute, unless you ask with `prewarm`).
`shared_connection_stats()` shows what is open; `close_shared_connections()`
closes it all at shutdown.

## Logging

Everything is logged to the standard library's `fourier_frequency` logger
(warnings and errors only reach stderr until you configure logging).
`logging.getLogger("fourier_frequency").setLevel(logging.DEBUG)` shows every
connect, sentence and retry.

## Pipecat

Use [`pipecat-mirai`](https://pypi.org/project/pipecat-mirai/): its
`MiraiTTSService` and `MiraiSTTService` are this package's `TTSStream` and
`STTStream` as Pipecat services. The two are released together and
`pipecat-mirai` pins the matching version.

## License

BSD 2-Clause. See [LICENSE](LICENSE).
