# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.4.0] - Unreleased

### Added

- `MiraiWebsocketTTSService` streams from Mirai's edge, next to the speech
  GPUs, which reaches the first audio byte in about half the time of the
  gateway. Nothing to change: each socket (at the start of a call, on a
  reconnect, or in the pool) opens on the edge with the API key in the
  `Authorization` header, exactly as on the gateway, so the first sentence
  pays one connection, not two. The gateway is asked which edge it offers
  (`edge_url` from `POST /v2/tts/stream/tokens`) in the background and its
  answer is reused for 10 minutes; when it stops offering the edge, sockets go
  back to the gateway. The key is never put in a URL.
- Safe fallback: if Mirai offers no edge (or the gateway has no token route),
  or the edge refuses, can't be reached or doesn't
  send `session.ready` within 3 s, that socket opens on the gateway exactly as
  before. A failure is logged once per process and keeps sockets off the edge
  for 60 s, doubling up to 16 min, so a dead edge doesn't slow every call. An
  edge socket that drops mid-call reconnects (edge first, then the gateway) and
  resends the rest of the reply, as before.
- `edge=` on `MiraiWebsocketTTSService` and `prewarm()`: `"auto"` (default),
  `False` (the gateway only, the 0.3.1 behaviour) or a `wss://` URL to use
  instead of the edge Mirai names. `MIRAI_TTS_EDGE=off` turns `"auto"` off for
  the process.
- The WebSocket pool opens its waiting sockets on the edge too, and moves
  sockets it opened on the gateway (while the edge was unavailable) back to
  the edge one at a time once it is reachable. `shared_connection_stats()`
  reports `edge`, the waiting sockets on the edge, and the service's
  `connected_url` says where its socket is.

## [0.3.1] - 2026-10-07

### Added

- `prewarm()`: open connections to Mirai when your server starts, so calls that
  start together don't each pay a TCP and TLS handshake on their greeting.
  `connections=N` opens N keep-alive HTTP connections (one `GET /v1/models`
  each) and refreshes them every 45 s, inside Mirai's 75 s idle timeout.
  `websocket=M` keeps M authenticated sockets waiting at `session.ready`, with
  an empty `session.update` every 30 s against the 120 s idle timeout. Returns
  a `PrewarmResult`; network failures are logged and returned, never raised.
- `MiraiWebsocketTTSService` takes a waiting socket when its pipeline starts
  (sending its own voice and rate), and connects as before when none is
  waiting. A socket serves one call and is closed when it ends; the pool opens
  a replacement in the background. A waiting socket closed by Mirai, or near
  Mirai's maximum session length, is replaced. `shared_pool=False` always
  connects.
- `shared_connection_stats()` and `close_shared_connections()`.
- Hedged connects: a TCP connect to Mirai that hasn't completed after 300 ms
  starts a second attempt (and a third at 1 s); the first to connect is used and
  the rest are closed. A burst of new connections, as when many calls start at
  once, can lose SYNs on some network paths, and each lost SYN otherwise costs a
  1 s (then 3 s) retransmit. `MIRAI_CONNECT_HEDGE_MS` sets the delay; `0` turns
  it off. Not used for WebSockets when an HTTP(S) proxy is configured.
- The output lead (0.4 s) is applied automatically to the pipeline's output
  transport when the service starts; `apply_output_lead` is no longer needed
  (calling it as well is harmless). `output_lead_secs=None` turns it off.
- `benchmarks/burst-start`: K pipelines starting at the same instant, in one or
  more processes, with each call's greeting and later-sentence TTFB.

### Changed

- `MiraiTTSService` uses one HTTP client per `base_url` and event loop, shared
  by every service in the process, instead of one per service. A connection
  opened for one call serves the next, the client (and its TLS context) is
  created once instead of per call, and up to 64 idle connections are kept.
  It closes 70 s after the last service using it stops, unless `prewarm()`
  keeps it. Nothing is shared between event loops. The start-of-call warm-up
  is skipped when an idle connection is already open, so it never ties up a
  connection the greeting needs; one keep-warm request serves every service.
  `shared_pool=False` restores the 0.3.0 behaviour (a client per service),
  and `http_client=` still overrides both.

## [0.3.0] - 2026-10-07

### Added

- `MiraiWebsocketTTSService`: Mirai TTS over one WebSocket for the whole call
  (`/v1/audio/speech/stream`), so no sentence waits for a TCP and TLS
  handshake. Each sentence is sent as soon as Pipecat has it (or each token,
  with `TextAggregationMode.TOKEN`, and Mirai cuts the sentences), and Mirai
  synthesises the next sentence while the current one plays. Same voices,
  settings, output-rate audio, 40 ms framing and first-audio buffer as
  `MiraiTTSService`. An interruption cancels the reply on Mirai and drops any
  of its audio still arriving. A capacity error with a retry time of up to 5 s
  is retried once. A dropped socket is reopened and a reply it cut short is
  resent once. A quiet socket is kept open with an empty `session.update`
  every `keepalive_secs` (30 s). TTFB is measured at the first audio byte;
  usage metrics are the characters Mirai billed.
- `DEFAULT_WEBSOCKET_URL`, and the example
  `examples/foundational/03-websocket-say-hello.py`.
- `MiraiTTSService` asks Mirai for audio at the pipeline's output rate (8, 16,
  22.05, 24, 44.1 or 48 kHz) with the request's `sample_rate` field. An 8 kHz phone pipeline now
  downloads 128 kbit/s per call instead of 768 kbit/s.
- `server_sample_rate` (`"auto"`, one of 8000/16000/22050/24000/44100/48000, or `None` for the
  0.2 behaviour) chooses the rate to ask for.
- `warm_connection` (on by default): the connection to Mirai opens with a
  `GET /v1/models` when the pipeline starts, and reopens right after an
  interruption drops it, so sentences don't wait for a TCP and TLS handshake.
  Failures are logged and ignored.
- `keep_warm_secs` (30 s by default): while the pipeline runs, an idle
  connection is kept open with the same request, so a long pause in a call
  doesn't cost a new handshake. Mirai closes idle connections after 75 s; the
  client now drops them after 70 s (was 120 s), so a request is never sent on
  one the server is closing.
- `prebuffer_secs` (150 ms by default): audio collected before an utterance's
  first frame is pushed, so a short first chunk followed by a pause doesn't
  start playback and then stall. TTFB is still measured at the first byte.
- `MiraiTTSService.last_server_sample_rate`: the rate Mirai sent for the latest
  utterance.

### Changed

- The service reads the rate Mirai sent from `X-Sample-Rate` (48 kHz when the
  header is missing) and resamples only when it differs from the output rate.
  A rate other than 48 kHz is no longer an error.
- Audio is pushed in 40 ms frames whatever the size of the network reads. A
  burst of several seconds becomes many 40 ms frames instead of one large one.
- An interruption closes the utterance's HTTP stream explicitly, even when the
  utterance is paused between frames, so the server stops generating it at
  once. An utterance cut short never pushes another frame.
- A response whose `X-Audio-Encoding` is not `pcm_s16le` is reported as an
  error instead of being played.
- If the server answers HTTP 400 to `sample_rate`, the request is retried once
  without it. If the retry gets past validation, the field is left out for the
  rest of the session; if not, the error is reported and nothing changes.

## [0.2.0] - 2026-10-06

### Added

- `MiraiRealtimeLLMService`: Mirai's Realtime API in Pipecat. It's Pipecat's
  `OpenAIRealtimeLLMService`, set up for Mirai: session URL parameters (`agent_id`,
  `variables`, `metadata`, `webhook_url`, `max_duration_secs`), the `mirai`
  settings block, Mirai's own events handled (stock Pipecat stops reading on
  them), per-turn timing through `on_turn_metrics`, and caller audio resampled
  to 24 kHz from any pipeline rate.
- `MiraiTurnMetrics`, `SANDBOX_REALTIME_URL` and `PRODUCTION_REALTIME_URL`.
- Examples: a microphone conversation on the Realtime API, and a Twilio phone
  bot on the Realtime API.

## [0.1.0] - 2026-10-02

### Added

- `MiraiTTSService`: streaming Mirai text-to-speech for Pipecat. Audio is
  resampled from Mirai's 48 kHz to the pipeline's output rate per utterance, and
  network reads that end mid-sample are handled. It reports TTFB and usage
  metrics, supports tracing and `TTSUpdateSettingsFrame`, and closes the HTTP
  stream on interruption.
- `apply_output_lead()`: lets Pipecat's websocket output transports (FastAPI,
  websocket server and websocket client) send up to 0.4 s ahead of real time, so
  event-loop stalls no longer break audio on phone calls.
- Examples: a foundational speech check and a Twilio phone bot.
- `benchmarks/phone-breaks`: a reproducible harness that measures breaks on phone
  calls under event-loop stalls.

Tested with Pipecat 1.8.1 and 1.12.0.
