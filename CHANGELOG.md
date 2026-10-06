# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.3.0] - Unreleased

### Added

- `MiraiTTSService` asks Mirai for audio at the pipeline's output rate (8, 16, 24
  or 48 kHz) with the request's `sample_rate` field. An 8 kHz phone pipeline now
  downloads 128 kbit/s per call instead of 768 kbit/s.
- `server_sample_rate` (`"auto"`, one of 8000/16000/24000/48000, or `None` for the
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
