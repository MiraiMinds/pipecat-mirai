# pipecat-mirai

[Mirai](https://miraiminds.co) for [Pipecat](https://github.com/pipecat-ai/pipecat)
voice agents: natural Hindi, Hinglish and Gujarati voices.

- **`MiraiTTSService`**: a Pipecat TTS service for Mirai's streaming API, about
  100 ms to first audio. Audio is resampled to your pipeline's rate (8 kHz for
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
| `sample_rate` | pipeline `audio_out_sample_rate` | Output rate; Mirai's 48 kHz audio is resampled to it |
| `base_url` | `https://sandbox.voice.miraiminds.co/v1` | API base URL |
| `http_client` | own client | An `httpx.AsyncClient` you manage |

The service reports time-to-first-byte and character usage metrics and supports
Pipecat tracing. Interrupting the bot closes the HTTP stream at once.

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
