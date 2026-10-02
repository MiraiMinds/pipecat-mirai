# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

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
