# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
