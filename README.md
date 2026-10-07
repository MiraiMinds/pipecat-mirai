# pipecat-mirai

[Mirai](https://miraiminds.co) for [Pipecat](https://github.com/pipecat-ai/pipecat)
voice agents: natural Hindi, Hinglish and Gujarati voices.

- **`MiraiTTSService`**: a Pipecat TTS service for Mirai's streaming API, about
  100 ms to first audio. Mirai sends audio at your pipeline's rate (8 kHz for
  phone calls), with no voice registration or sample-rate workarounds.
- **`MiraiRealtimeLLMService`**: hand Mirai the whole turn. Speech recognition,
  turn detection, the model and the voice run together on Mirai's side, and
  this one service replaces your STT, LLM and TTS (see [Realtime](#realtime)).
- **`apply_output_lead()`**: stops audio breaking up on phone calls when your
  server is busy (see [Phone calls](#phone-calls)). This works with any TTS service.

**Demo:** [a 68-second conversation in Hindi with an interruption](https://github.com/MiraiMinds/pipecat-mirai/releases/download/v0.1.0/pipecat-mirai-demo.mp4)
(Pipecat 1.12, Sarvam STT, `MiraiTTSService`).

## Installation

```bash
uv add pipecat-mirai     # or: pip install pipecat-mirai
```

## Prerequisites

- A Mirai API key (`sk_live_…`) from the [Mirai console](https://sandbox.voice.miraiminds.co).
  Set it as `MIRAI_API_KEY`, or pass `api_key=`.
- Pipecat 1.8.1 or newer and Python 3.11+.

## Usage

```python
from pipecat_mirai import MiraiTTSService

tts = MiraiTTSService(settings=MiraiTTSService.Settings(voice="shruti"))

pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
```

| Voice | |
|---|---|
| `ashu` | male |
| `neha` | female (default) |
| `shruti` | female |
| `sameer` | male |

Write Hindi in Devanagari and English words in Latin script, as people actually
type Hinglish: `आपका order कल deliver होगा।`

Change the voice mid-call with Pipecat's standard settings frame:

```python
await task.queue_frame(TTSUpdateSettingsFrame(delta=MiraiTTSService.Settings(voice="sameer")))
```

### Parameters

| Argument | Default | |
|---|---|---|
| `api_key` | `$MIRAI_API_KEY` | Your Mirai API key |
| `settings` | `voice="neha"`, `model="mira-tts"` | `MiraiTTSService.Settings(...)` |
| `voice`, `model` | | Shortcuts for the same settings |
| `sample_rate` | pipeline `audio_out_sample_rate` | Output rate |
| `server_sample_rate` | `"auto"` | Rate to ask Mirai for (see [Sample rate](#sample-rate)) |
| `prebuffer_secs` | `0.15` | Audio collected before an utterance starts playing (see [Delivery](#delivery)) |
| `warm_connection` | `True` | Open the connection to Mirai while the pipeline starts (see [Connections](#connections)) |
| `keep_warm_secs` | `30` | Keep an idle connection open while the pipeline runs; `None` turns it off |
| `base_url` | `https://sandbox.voice.miraiminds.co/v1` | API base URL |
| `http_client` | own client | An `httpx.AsyncClient` you manage |

The service reports time-to-first-byte and character usage metrics and supports
Pipecat tracing. Interrupting the bot closes the HTTP stream at once.

### Sample rate

The service asks Mirai for audio at your pipeline's output rate when Mirai serves
it (8000, 16000, 22050, 24000, 44100 or 48000 Hz), so nothing is converted on your side. On a
phone pipeline at 8 kHz that is 128 kbit/s per call instead of 768 kbit/s at
48 kHz. Bandwidth matters here: at 48 kHz, six concurrent calls on an ordinary
link already receive audio slower than real time, and callers hear gaps.

The service reads the rate Mirai actually sent (`X-Sample-Rate`) and resamples
only if it differs from the output rate, so it also works with servers that
always send 48 kHz. `tts.last_server_sample_rate` shows the rate of the latest
utterance.

| `server_sample_rate` | Request | |
|---|---|---|
| `"auto"` (default) | the output rate, if Mirai serves it | Otherwise Mirai sends 48 kHz and it is resampled |
| `8000`, `16000`, `24000`, `48000` | that rate | Resampled to the output rate if they differ |
| `None` | no rate | Mirai's default 48 kHz, resampled (the 0.2 behaviour) |

If a server answers HTTP 400 to the `sample_rate` field, the request is sent once
more without it. If that one succeeds, the field is left out for the rest of the
session.

### Connections

All requests from one service share a kept-alive HTTPS connection, so only the
first pays for the TCP and TLS handshake (0.4–1 s from India on a fresh
connection, and occasionally more when a connection attempt is retried). With
`warm_connection=True` (the default), the service:

- opens that connection with a `GET /v1/models` as soon as the pipeline starts,
  before the bot's first sentence;
- repeats the request when the connection has been idle for `keep_warm_secs`
  (30 s), but only while the pipeline runs. Mirai closes connections that have
  been idle for 75 s, and this keeps one open through long pauses in a call. The
  client drops idle connections after 70 s, so it never sends a request on one
  the server is closing;
- opens a new connection straight away when an interruption cuts a sentence off
  mid-stream (which drops that sentence's connection), while the caller is still
  talking.

These requests are never billed. A failed one is logged and ignored, and the
next sentence connects on its own. `warm_connection=False` turns all three off.

### Delivery

Audio is pushed downstream in 40 ms frames, whatever size the network reads
are, and as fast as Mirai sends it. When Mirai sends audio several times faster
than real time, the service reads the stream to the end straight away (so the
server's slot is freed sooner) and the audio waits in Pipecat's queues, as it
would for any TTS service. Between reads, the service itself holds less than
one frame (or, before playback starts, the first-audio buffer).

Mirai's first chunk is sometimes short (14–133 ms of audio) and followed by a
pause of up to 200 ms. If playback started on it, the caller would hear a sliver
of speech, a gap, then the rest. So the service collects `prebuffer_secs`
(150 ms) before it pushes an utterance's first frame. Time to first byte is
still measured at the first byte received. `prebuffer_secs=0` pushes audio as
soon as a frame is in hand.

When the bot is interrupted, the service closes the utterance's HTTP stream at
once, so Mirai stops generating it. Audio not yet pushed is dropped, and nothing
from the interrupted utterance reaches the next one.

## Realtime

Mirai's [Realtime API](https://docs.miraiminds.co/v2/realtime) speaks the OpenAI
Realtime protocol. `MiraiRealtimeLLMService` is Pipecat's own
`OpenAIRealtimeLLMService`, set up for Mirai. It replaces your STT, LLM and TTS
services, and your transport and context aggregators stay as they are:

```python
from pipecat_mirai import MiraiRealtimeLLMService

llm = MiraiRealtimeLLMService(voice="shruti", language="hi")

pipeline = Pipeline([transport.input(), user_aggregator, llm, transport.output(), assistant_aggregator])
```

Each turn makes one round trip to your server instead of three, and a session is
billed per minute like a browser call. On top of the stock service, it:

- builds the session URL for you: `agent_id`, `variables`, `metadata`,
  `webhook_url` and `max_duration_secs` are plain arguments;
- sends the `mirai` settings block (`language`, `temperature`, `tool_timeout_ms`,
  ...), which stock `SessionProperties` drops;
- handles Mirai's own events, which make stock Pipecat stop reading the socket,
  and reports each turn's timing through `on_turn_metrics`;
- resamples caller audio to the 24 kHz the protocol uses, so an 8 kHz phone
  pipeline works without changing `audio_in_sample_rate`.

```python
llm = MiraiRealtimeLLMService(
    agent_id="agt_...",                    # start from one of your agents
    variables={"customer_name": "Rahul"},  # its {{placeholders}}
    mirai={"temperature": 0.3},
)

@llm.event_handler("on_turn_metrics")
async def on_turn_metrics(service, m):
    print(f"{m.v2v_ms} ms voice to voice, providers {m.providers}")
```

It connects to the sandbox (`SANDBOX_REALTIME_URL`) by default, which is what a
sandbox key needs. Pass `base_url=PRODUCTION_REALTIME_URL` once your company has
gone live.

## Phone calls

Pipecat's websocket transports (`FastAPIWebsocketTransport` with Twilio, Plivo,
Exotel or Telnyx serializers, `WebsocketServerTransport`, `WebsocketClientTransport`)
send audio at exactly real time. The phone provider then never holds more than
about 40 ms of audio. If your server's event loop stalls for longer (a blocking
call, a heavy parse, many calls on one CPU), the provider runs out and the caller
hears the voice break up. This happens with every TTS vendor, and it gets worse
as you add concurrent calls.

`apply_output_lead()` lets the transport send up to 0.4 s ahead, so short stalls
go unnoticed:

```python
from pipecat_mirai import MiraiTTSService, apply_output_lead

transport = FastAPIWebsocketTransport(websocket, FastAPIWebsocketParams(
    audio_out_enabled=True, add_wav_header=False, serializer=serializer, ...))
apply_output_lead(transport)              # default 0.4 s; apply_output_lead(transport, 0.6) for more
```

We measured this on 8 kHz Twilio-protocol calls, with Pipecat 1.8.1, real Mirai audio
and 50–250 ms event-loop stalls every ~3 s per call
([benchmarks/phone-breaks](benchmarks/phone-breaks)):

| Concurrent calls | Stock Pipecat: speech stretched by breaks | With `apply_output_lead` |
|---|---|---|
| 3 | 6.6% (3.5 breaks per sentence) | **0.08%** |
| 10 | 29.1% (14.5 breaks per sentence) | **0.01%** |

When the caller interrupts, Pipecat still tells the provider to clear its queue, so
at most the lead (0.4 s) of already-sent audio is discarded. One side effect:
Pipecat's "bot stopped speaking" event fires up to the lead earlier than the caller
actually stops hearing the bot.

**Recommended phone setup:** `apply_output_lead(transport)` on the transport, an
8 kHz pipeline (`audio_out_sample_rate=8000`) and `MiraiTTSService` with its
default `server_sample_rate="auto"`. The lead absorbs stalls on your server, and
8 kHz audio from Mirai keeps each call's download at a sixth of 48 kHz, so it
keeps up even when many calls share one link.

## Examples

- [`examples/foundational/01-say-hello.py`](examples/foundational/01-say-hello.py):
  a minimal Pipecat pipeline that speaks one line and saves `hello.wav`.
- [`examples/foundational/02-realtime-conversation.py`](examples/foundational/02-realtime-conversation.py):
  talk to a Realtime session from your microphone, with per-turn timing.
- [`examples/phone/twilio_bot.py`](examples/phone/twilio_bot.py): a Twilio Media
  Streams bot with `MiraiTTSService` and `apply_output_lead`.
- [`examples/phone/twilio_realtime_bot.py`](examples/phone/twilio_realtime_bot.py):
  the same phone line on the Realtime API.

## Compatibility

Tested with Pipecat 1.8.1 and 1.12.0 on Python 3.11–3.12. `apply_output_lead` changes how
Pipecat's websocket output transports time their writes. A test in this repo
fails if a new Pipecat release changes that, and the function logs a warning and
does nothing for transports it doesn't support.

## License

BSD-2-Clause. Maintained by [Mirai](https://miraiminds.co) (Sona Labs Pvt Ltd).
This is a community integration, not maintained by the Pipecat team.
