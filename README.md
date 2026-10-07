# pipecat-mirai

[Mirai](https://miraiminds.co) for [Pipecat](https://github.com/pipecat-ai/pipecat)
voice agents: natural Hindi, Hinglish and Gujarati voices.

- **`MiraiTTSService`**: a Pipecat TTS service for Mirai's streaming API, about
  100 ms to first audio. Mirai sends audio at your pipeline's rate (8 kHz for
  phone calls), with no voice registration or sample-rate workarounds.
- **`MiraiWebsocketTTSService`**: the same voices over one WebSocket for the
  whole call, so no sentence waits for a connection (see
  [WebSocket](#websocket-lowest-latency)).
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
| `shared_pool` | `True` | Share connections with every other `MiraiTTSService` in the process (see [Load tests and many agents per process](#load-tests-and-many-agents-per-process)); `False` gives each service its own |
| `http_client` | shared client | An `httpx.AsyncClient` you manage (overrides `shared_pool`) |

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

Requests go over kept-alive HTTPS connections, so only a new connection pays
for the TCP and TLS handshake (0.4–1 s from India, and occasionally more when a
connection attempt is retried). The connections are shared by every
`MiraiTTSService` for the same `base_url` in the process, so a connection one
call opened serves the next call too. With `warm_connection=True` (the
default), the service:

- opens a connection with a `GET /v1/models` as soon as the pipeline starts,
  before the bot's first sentence, unless an idle one is already open;
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

When many pipelines start at once, starting the warm-up with each pipeline is
too late: the greeting is sent at the same moment and opens its own connection.
For that, call [`prewarm()`](#load-tests-and-many-agents-per-process) when your
server starts.

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

## WebSocket (lowest latency)

`MiraiTTSService` sends one HTTPS request per sentence. It keeps the connection
alive, but whenever a request finds it gone, that sentence waits for a new TCP
and TLS handshake (0.4–1 s from India). `MiraiWebsocketTTSService` opens one
WebSocket to Mirai when the pipeline starts and sends the whole call over it,
so that cost is paid once, before the bot says anything.

```python
from pipecat_mirai import MiraiWebsocketTTSService

tts = MiraiWebsocketTTSService(settings=MiraiWebsocketTTSService.Settings(voice="shruti"))

pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
```

It takes the place of `MiraiTTSService` with nothing else changed: the same
voices and settings, audio at your pipeline's rate, 40 ms frames and the 150 ms
first-audio buffer (here, at the start of each sentence). Billing is the same
too: each sentence Mirai delivers is billed per character, as on the HTTP
endpoint.

**How text reaches Mirai.** By default Pipecat collects the LLM's reply a
sentence at a time, and each sentence is sent the moment it is complete. Mirai
speaks them in order and starts synthesising the next one while the current
one plays, so there is no pause between sentences. With
`text_aggregation_mode=TextAggregationMode.TOKEN`, every LLM token goes to
Mirai as it arrives and Mirai cuts the sentences itself. It knows the danda
(।), abbreviations like "Rs." and "Dr.", and numbers like 3.5.

**What the service handles for you:**

- **Interruptions.** The reply is cancelled on Mirai straight away, and you
  aren't billed for the sentence that was cut off. Audio from it that is still
  on its way is dropped, so none of it plays after the interruption.
- **Capacity.** If Mirai is briefly at capacity and says when to try again (up
  to 5 s), the service waits that long and sends the rest of the reply once
  more. Otherwise the error goes up the pipeline as an `ErrorFrame` and Mirai
  carries on with the next sentence.
- **Dropped connections.** The socket is reopened at once (Pipecat's reconnect,
  with backoff after a failed attempt). A reply the drop cut short is sent
  again once, from the first sentence the caller hadn't started hearing.
- **Quiet calls.** Mirai closes a socket that has sent nothing for 120 s. After
  `keepalive_secs` (30 s) of quiet, the service sends an empty
  `session.update`, which changes nothing and keeps the socket open.
- **Metrics.** Time to first byte is measured at the first audio byte of each
  reply. Usage metrics are the characters Mirai billed for each sentence.

| Argument | Default | |
|---|---|---|
| `api_key` | `$MIRAI_API_KEY` | Sent as `Authorization: Bearer` when the socket opens |
| `url` | `wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream` | The streaming endpoint |
| `settings`, `voice`, `model` | `voice="neha"`, `model="mira-tts"` | As for `MiraiTTSService`. A new voice applies from the next sentence |
| `sample_rate`, `server_sample_rate` | pipeline rate, `"auto"` | As for `MiraiTTSService`. Each sentence's rate is read from Mirai's `audio.start` |
| `prebuffer_secs` | `0.15` | Audio collected before a sentence starts playing |
| `keepalive_secs` | `30` | Keep a quiet socket open; `None` turns it off |
| `shared_pool` | `True` | Take a socket `prewarm()` opened, when one is waiting; `False` always connects |
| `text_aggregation_mode` | sentence | `TextAggregationMode.TOKEN` sends every token as it arrives |

`tts.session_id` is the socket's id (`ttsws_…`), and `tts.last_server_sample_rate`
the rate of the latest sentence. Pipecat's `on_connected`, `on_disconnected` and
`on_connection_error` events fire as the socket opens and closes.

## Load tests and many agents per process

A connection to Mirai costs a TCP and TLS handshake before the first audio byte.
One call pays it once and reuses the connection. But when many calls start at
the same moment (a load test that starts a dozen agents at once, or a burst of
real calls), each one pays it on its first sentence, the greeting, because
every pipeline opens its connection as it starts. Measured from India: 343 ms
to first byte (p50) on a new connection against 240 ms on an open one, and
about 800 ms at p95 when a SYN has to be retransmitted.

Call `prewarm()` once when your server or worker starts, in the event loop that
will run the pipelines, with the number of calls you expect to start together:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pipecat_mirai import prewarm

@asynccontextmanager
async def lifespan(app: FastAPI):
    await prewarm(connections=12)              # MiraiTTSService (HTTP)
    # await prewarm(connections=0, websocket=12)  # MiraiWebsocketTTSService
    yield

app = FastAPI(lifespan=lifespan)
```

Nothing else changes: the services find the connections on their own.

- **HTTP.** Every `MiraiTTSService` in the process (with the same `base_url`)
  shares one connection pool. `prewarm(connections=N)` opens N connections in
  it (one `GET /v1/models` each, never billed) and refreshes them every 45 s,
  so Mirai's 75 s idle timeout never closes them. N calls that start together
  each find an open connection for their greeting.
- **WebSocket.** `prewarm(websocket=M)` opens M sockets and leaves them
  waiting, authenticated, at `session.ready`, with an empty `session.update`
  every 30 s against Mirai's 120 s idle timeout. A `MiraiWebsocketTTSService`
  takes one as its pipeline starts (and sends its own voice and sample rate),
  so the call starts without a handshake, and the pool opens a replacement in
  the background. A socket belongs to one call and is closed when that call
  ends, never handed to another. If no socket is waiting, the service connects
  as it always has. Waiting sockets are never billed.

One process pays the handshakes once, at start-up, and its calls reuse them
from then on. Several worker processes each have their own pool (connections
can't be shared between processes), so each calls `prewarm()` at start-up with
its own share of the calls. The same goes for event loops: connections belong
to the loop that opened them, and `prewarm()` must run in the loop that runs
the pipelines (not in a separate `asyncio.run()` before the server starts).

| `prewarm()` argument | Default | |
|---|---|---|
| `api_key` | `$MIRAI_API_KEY` | The key the services use; waiting sockets only go to services with this key |
| `base_url` | `https://sandbox.voice.miraiminds.co/v1` | As given to `MiraiTTSService` |
| `connections` | `8` | HTTP connections to keep open (0–64); `0` if you only use WebSocket |
| `websocket` | `0` | Sockets to keep waiting for `MiraiWebsocketTTSService` |
| `websocket_url` | `base_url` as `wss://…/audio/speech/stream` | As given to `MiraiWebsocketTTSService` |
| `timeout` | `10` | Seconds to wait for them to open |

It returns a `PrewarmResult` (`http_connections`, `websockets`, `errors`) and
never raises on a network failure: the pools keep trying in the background,
and a service whose pool is empty connects on its own. Call it again to change
the numbers. `shared_connection_stats()` shows what is open, and
`close_shared_connections()` closes it all (it also closes when the event loop
ends). `shared_pool=False` on a service opts it out (the 0.3.0 behaviour).

Measured on our production API from a server in Germany, 12 calls starting at
the same instant ([how to run it yourself](benchmarks/burst-start/)):

<!-- burst-table -->

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
8 kHz pipeline (`audio_out_sample_rate=8000`) and `MiraiWebsocketTTSService`
with its default `server_sample_rate="auto"`. The lead absorbs stalls on your
server, one socket per call removes a connection handshake per sentence, and
8 kHz audio from Mirai keeps each call's download at a sixth of 48 kHz, so it
keeps up even when many calls share one link. `MiraiTTSService` (HTTP) with the
same lead and rate also works. Measured on our production API: 12 simultaneous
calls through `MiraiWebsocketTTSService`, 194 turns with barge-ins, no audible
gaps, no errors, every sentence in order ([how to run it
yourself](benchmarks/customer-e2e/)).

## Examples

- [`examples/foundational/01-say-hello.py`](examples/foundational/01-say-hello.py):
  a minimal Pipecat pipeline that speaks one line and saves `hello.wav`.
- [`examples/foundational/03-websocket-say-hello.py`](examples/foundational/03-websocket-say-hello.py):
  stream a reply into `MiraiWebsocketTTSService` a few words at a time, as an
  LLM would, and save it.
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
