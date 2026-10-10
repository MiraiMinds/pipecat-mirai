# pipecat-mirai

[Mirai](https://miraiminds.co) for [Pipecat](https://github.com/pipecat-ai/pipecat)
voice agents: natural Hindi, Hinglish and Gujarati voices.

- **`MiraiTTSService`**: Mirai's voices in your Pipecat pipeline. One service,
  defaults that hold up under load: it streams over one WebSocket per call from
  Mirai's edge next to the speech GPUs, and finds its own way back to Mirai's
  API (WebSocket, then HTTP) if the edge can't be reached. Audio comes at your
  pipeline's rate (8 kHz for phone calls).
- **`MiraiSTTService`**: Mirai's streaming speech-to-text for the same
  pipeline: partial and final transcripts over one WebSocket per call, from the
  same edge, at 8 kHz as it comes off the phone (see
  [Speech-to-text](#speech-to-text)).
- **`MiraiRealtimeLLMService`**: hand Mirai the whole turn. Speech recognition,
  turn detection, the model and the voice run together on Mirai's side, and
  this one service replaces your STT, LLM and TTS (see [Realtime](#realtime)).
- **`apply_output_lead()`**: stops audio breaking up on phone calls when your
  server is busy (see [Phone calls](#phone-calls)). `MiraiTTSService` applies
  it to your output transport for you; call it yourself only for another TTS
  vendor.

**Demo:** [a 68-second conversation in Hindi with an interruption](https://github.com/MiraiMinds/pipecat-mirai/releases/download/v0.1.0/pipecat-mirai-demo.mp4)
(Pipecat 1.12, Sarvam STT, Mirai TTS).

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

That's the whole setup. Keep the defaults: they are what we load-test in
production (10 calls starting at once, interruptions, a busy event loop).

| Voice | |
|---|---|
| `ashu` | male |
| `neha` | female (default) |
| `shruti` | female |
| `sameer` | male |

Write Hindi in Devanagari and English words in Latin script, as people actually
type Hinglish: `आपका order कल deliver होगा।`

Change the voice mid-call with Pipecat's standard settings frame (it applies
from the next sentence):

```python
await task.queue_frame(TTSUpdateSettingsFrame(delta=MiraiTTSService.Settings(voice="sameer")))
```

### What it does for you

The service opens one WebSocket to Mirai when the pipeline starts and carries
the whole call over it, so no sentence waits for a connection.

- **The edge.** Each socket opens on Mirai's edge, next to the speech GPUs,
  with your API key (in the header, never in a URL). Measured in production
  from India, 10 calls at once: Pipecat's time to first byte p50 156 ms, p95
  285 ms, p99 344 ms. If the edge can't be reached or isn't ready within 3 s,
  the socket opens on Mirai's API instead and the edge is left alone for a
  minute (doubling, up to 16). A rejected key, an empty wallet or a rate limit
  comes back with the API's own error message.
- **HTTP when WebSockets can't get through.** If the streaming endpoint can't be
  reached at all (a proxy that doesn't pass WebSockets, say), the call speaks
  over Mirai's HTTP endpoint instead and the service logs it once.
- **Text as it comes.** By default Pipecat collects the LLM's reply a sentence
  at a time, and each sentence goes to Mirai the moment it is complete. Mirai
  synthesises the next sentence while the current one plays, so there is no
  pause between them. With `text_aggregation_mode=TextAggregationMode.TOKEN`,
  every LLM token goes as it arrives and Mirai cuts the sentences itself (it
  knows the danda (।), "Rs.", "Dr." and numbers like 3.5).
- **Interruptions.** The reply is cancelled on Mirai straight away, and you
  aren't billed for the sentence that was cut off. Audio from it that is still
  on its way is dropped, so none of it plays after the interruption.
- **Capacity.** If Mirai is briefly at capacity and says when to try again (up
  to 5 s), the service waits that long and sends the rest of the reply once
  more. Otherwise the error goes up the pipeline as an `ErrorFrame`.
- **Dropped connections.** The socket is reopened at once, and a reply the drop
  cut short is sent again from the first sentence the caller hadn't started
  hearing.
- **Quiet calls.** Mirai closes a socket that has sent nothing for 120 s. After
  `keepalive_secs` (30 s) of quiet, the service sends an empty `session.update`,
  which changes nothing and keeps the socket open.
- **Many calls at once.** Sockets are opened ahead of need by a pool shared by
  every service in the process, so a call usually starts on a socket that is
  already open (see [Load tests](#load-tests-and-many-agents-per-process)).
- **Phone calls.** It lets Pipecat's websocket output transport run 0.4 s ahead
  of real time, so a busy server doesn't break up the caller's audio (see
  [Phone calls](#phone-calls)).
- **Metrics.** Time to first byte is measured at the first audio byte of each
  reply; usage metrics are the characters Mirai billed. Pipecat tracing works.

Billing is per character, as on Mirai's HTTP endpoint.

### Parameters

| Argument | Default | |
|---|---|---|
| `api_key` | `$MIRAI_API_KEY` | Your Mirai API key |
| `settings` | `voice="neha"`, `model="mira-tts"` | `MiraiTTSService.Settings(...)` |
| `voice`, `model` | | Shortcuts for the same settings |
| `sample_rate` | pipeline `audio_out_sample_rate` | Output rate |
| `server_sample_rate` | `"auto"` | Rate to ask Mirai for (see [Sample rate](#sample-rate)) |
| `prebuffer_secs` | `0.15` | Audio collected before a sentence starts playing (see [Delivery](#delivery)) |
| `keepalive_secs` | `30` | Keep a quiet socket open; `None` turns it off |
| `url` | `wss://sandbox.voice.miraiminds.co/v1/audio/speech/stream` | Mirai's streaming endpoint |
| `base_url` | | The API base URL (`https://…/v1`), as the HTTP service took it; the streaming endpoint is derived from it |
| `edge` | `"auto"` | Stream from Mirai's edge when it can; `False` (or `MIRAI_TTS_EDGE=off`) always uses `url` |
| `http_fallback` | `True` | Speak over HTTP if WebSockets can't reach Mirai; `False` reports the error instead |
| `shared_pool` | `True` | Take a socket the process's pool opened ahead of need; `False` always connects |
| `output_lead_secs` | `0.4` | Output lead on phone transports; `None` turns it off |
| `text_aggregation_mode` | sentence | `TextAggregationMode.TOKEN` sends every token as it arrives |

`tts.connected_url` shows where the call is streaming from (the edge or `url`),
`tts.session_id` the socket's id (`ttsws_…`) and `tts.last_server_sample_rate`
the rate of the latest sentence. Pipecat's `on_connected`, `on_disconnected` and
`on_connection_error` events fire as the socket opens and closes.

### Limits

Mirai sets two limits per workspace, shared by every process that uses your key
([full table](https://docs.miraiminds.co/v2/limits)):

- **Sentences in flight at once: 10.** A call holds a slot while one of its
  sentences is streaming, which is most of the time its bot is speaking, and
  none while the caller talks. So 10 slots carry 10 bots speaking at the same
  moment (more live calls than that, since nobody speaks all the time). When
  more sentences than that start together, the extra ones wait for a slot (up
  to 3 s), which shows up as a longer time to first byte, and past that the
  service reports `at_capacity`.
- **Open sockets.** One per live call, plus the few each process keeps waiting
  (2, or the recent peak of calls plus 2).

If you plan a load test with more than 10 calls starting together, tell us the
number and we raise your workspace's limits before you start.

### Sample rate

The service asks Mirai for audio at your pipeline's output rate when Mirai
serves it (8000, 16000, 24000 or 48000 Hz), so nothing is converted on your
side. On a phone pipeline at 8 kHz that is 128 kbit/s per call instead of
768 kbit/s at 48 kHz. Bandwidth matters here: at 48 kHz, six concurrent calls on
an ordinary link already receive audio slower than real time, and callers hear
gaps.

Each sentence says the rate Mirai actually sent, and audio is resampled only if
it differs from the output rate.

| `server_sample_rate` | Request | |
|---|---|---|
| `"auto"` (default) | the output rate, if Mirai serves it | Otherwise Mirai sends 48 kHz and it is resampled |
| `8000`, `16000`, `24000`, `48000` | that rate | Resampled to the output rate if they differ |
| `None` | no rate | Mirai's default 48 kHz, resampled |

### Delivery

Audio is pushed downstream in 40 ms frames, whatever size the network reads
are, and as fast as Mirai sends it. Mirai's first chunk of a sentence is
sometimes short (14–133 ms of audio) and followed by a pause of up to 200 ms.
If playback started on it, the caller would hear a sliver of speech, a gap, then
the rest. So the service collects `prebuffer_secs` (150 ms) before it pushes a
sentence's first frame. Time to first byte is still measured at the first byte
received. `prebuffer_secs=0` pushes audio as soon as a frame is in hand.

### Upgrading

- **From 0.4 or 0.3 with `MiraiWebsocketTTSService`:** nothing to change. It is
  now another name for `MiraiTTSService`, the same class.
- **From `MiraiTTSService` in 0.4 and earlier (HTTP):** nothing to change either.
  Your code now streams over a WebSocket from the edge. `base_url`,
  `http_client` (used for the HTTP fallback), `warm_connection` and
  `keep_warm_secs` (now `keepalive_secs`) are still accepted.
- **If you want HTTP only**, use `MiraiHttpTTSService`: the 0.4 HTTP service
  under a new name, one streaming request per sentence over shared, self-warming
  keep-alive connections, with the same arguments as before.

## Load tests and many agents per process

A connection to Mirai costs a TCP and TLS handshake before the first audio byte.
One call pays it once and reuses the connection. But when many calls start at
the same moment (a load test that starts a dozen agents at once, or a burst of
real calls), each one pays it on its first sentence, the greeting, because
every pipeline opens its connection as it starts. Measured from India: 343 ms
to first byte (p50) on a new connection against 240 ms on an open one, and
about 800 ms at p95 when a SYN has to be retransmitted.

You don't have to do anything: the first service to start sets the process's
socket pool going, and it keeps more sockets waiting than the recent peak of
calls. To have the very first burst after start-up find open sockets too, call
`prewarm()` once when your server or worker starts, in the event loop that will
run the pipelines, with the number of calls you expect to start together:

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pipecat_mirai import prewarm

@asynccontextmanager
async def lifespan(app: FastAPI):
    await prewarm(websocket=12)                    # MiraiTTSService
    # await prewarm(connections=12, websocket=0)   # MiraiHttpTTSService
    yield

app = FastAPI(lifespan=lifespan)
```

Nothing else changes: the services find the connections on their own.

- **HTTP** (`MiraiHttpTTSService`). Every one in the process (with the same `base_url`)
  shares one connection pool. `prewarm(connections=N)` opens N connections in
  it (one `GET /v1/models` each, never billed) and refreshes them every 45 s,
  so Mirai's 75 s idle timeout never closes them. N calls that start together
  each find an open connection for their greeting.
- **WebSocket** (`MiraiTTSService`). `prewarm(websocket=M)` opens M sockets and leaves them
  waiting, authenticated, at `session.ready`, with an empty `session.update`
  every 30 s against Mirai's 120 s idle timeout. A `MiraiTTSService`
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
| `base_url` | `https://sandbox.voice.miraiminds.co/v1` | The API base URL |
| `connections` | `0` | HTTP connections to keep open for `MiraiHttpTTSService` (0–64) |
| `websocket` | `8` | Sockets to keep waiting for `MiraiTTSService` (0–64) |
| `websocket_url` | `base_url` as `wss://…/audio/speech/stream` | As given to `MiraiTTSService` as `url` |
| `edge` | `"auto"` | As given to `MiraiTTSService`; waiting sockets open on the edge |
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

## Speech-to-text

`MiraiSTTService` streams the caller's audio to Mirai over one WebSocket and
turns what comes back into Pipecat's transcription frames: partial transcripts
as `InterimTranscriptionFrame`, and one `TranscriptionFrame` per utterance.

```python
from pipecat_mirai import MiraiSTTService

stt = MiraiSTTService(language="hi-IN")  # api_key from MIRAI_API_KEY

pipeline = Pipeline([transport.input(), stt, user_aggregator, llm, tts, transport.output(), assistant_aggregator])
```

Nothing else needs setting for a phone bot:

- **Rate.** A pipeline at 8 kHz sends its audio as it is; any other rate is
  resampled to 16 kHz. `sample_rate=8000` or `16000` forces one, and
  `encoding="mulaw"` / `"alaw"` halves the bytes for telephony audio.
- **Who ends the utterance** (`endpointing`). With a VAD in your pipeline
  (a `vad_analyzer` on the user aggregator), the VAD's start and stop become
  `speech_start` and `speech_end`, and the service sends the 0.5 s of audio
  before the VAD fired, so the first word isn't cut. The final comes back
  within tens of milliseconds of `speech_end`, and nothing is sent between turns
  but a keepalive. Without a VAD, Mirai's does the job and the service proposes
  the turn boundaries to your aggregator. `endpointing="manual"` or `"vad"`
  forces one.
- **Where it connects.** Mirai's STT edge, next to the GPUs, when it is offered;
  otherwise the API. A process keeps two sockets waiting, so a call's first audio
  doesn't wait for a handshake (`prewarm(stt_websockets=…)` or
  `MIRAI_WARM_STT_WEBSOCKETS` sets how many). `edge=False` always uses `url`,
  and `MIRAI_STT_EDGE=off` turns the edge off for the process.
- **When something goes wrong.** A dropped socket is redialled once and the
  audio Mirai hadn't answered (up to 30 s) is replayed; a fatal error from Mirai
  (a revoked key, an empty wallet) is an `ErrorFrame`. A `429` is retried for a
  few seconds before it is reported.

Metrics: time to first byte is `speech_end` sent to the final, and usage is the
audio seconds sent. `language` and the VAD tuning (`settings=MiraiSTTSettings(...)`)
change on the open socket, through Pipecat's `STTUpdateSettingsFrame`.
[`benchmarks/stt-e2e`](benchmarks/stt-e2e/) measures it under load, and
[`examples/foundational/04-transcribe.py`](examples/foundational/04-transcribe.py)
transcribes a recording.

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
go unnoticed. **`MiraiTTSService` (and `MiraiHttpTTSService`) do this for you** when the pipeline starts (`output_lead_secs=0.4`; `None` turns
it off). Call it yourself only with another TTS vendor:

```python
from pipecat_mirai import apply_output_lead

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

**Recommended phone setup:** an 8 kHz pipeline (`audio_out_sample_rate=8000`) and
`MiraiTTSService(api_key=..., voice=...)` with its defaults. Nothing else to
configure: the service streams from Mirai's edge, applies the 0.4 s output lead
to your transport, asks Mirai for 8 kHz audio, keeps a pool of ready sockets
shared by every call in the process, and races a second TCP connect when one
stalls.

Measured on our production API with only those defaults: 10 phone calls started
at the same instant in one Pipecat process, about 100 turns with a fifth of the
replies interrupted, and the bot's event loop deliberately stalled 150–300 ms
every ~2 s: no errors, nothing played after an interruption, every sentence in
order, and from India a time to first audio (LLM first token to the phone) of
p50 422 ms, p95 641 ms after the greeting
([how to run it yourself](benchmarks/customer-e2e/)).

## Examples

- [`examples/foundational/01-say-hello.py`](examples/foundational/01-say-hello.py):
  a minimal Pipecat pipeline that speaks one line and saves `hello.wav`.
- [`examples/foundational/03-websocket-say-hello.py`](examples/foundational/03-websocket-say-hello.py):
  stream a reply into `MiraiTTSService` a few words at a time, as an
  LLM would, and save it.
- [`examples/foundational/02-realtime-conversation.py`](examples/foundational/02-realtime-conversation.py):
  talk to a Realtime session from your microphone, with per-turn timing.
- [`examples/foundational/04-transcribe.py`](examples/foundational/04-transcribe.py):
  transcribe a WAV with `MiraiSTTService` and print the partials and finals.
- [`examples/phone/twilio_bot.py`](examples/phone/twilio_bot.py): a Twilio Media
  Streams bot with `MiraiTTSService`.
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
