#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech, speech-to-text and Realtime API for Pipecat."""

from pipecat_mirai._logging import forward_to_loguru

forward_to_loguru()

# ruff: noqa: E402
from pipecat_mirai.pacing import DEFAULT_LEAD_SECS, apply_output_lead
from pipecat_mirai.pool import close_shared_connections, shared_connection_stats
from pipecat_mirai.prewarm import PrewarmResult, prewarm
from pipecat_mirai.realtime import (
    PRODUCTION_REALTIME_URL,
    SANDBOX_REALTIME_URL,
    MiraiRealtimeLLMService,
    MiraiTurnMetrics,
)
from pipecat_mirai.stt import DEFAULT_STT_WEBSOCKET_URL, STT_LANGUAGES, MiraiSTTService, MiraiSTTSettings
from pipecat_mirai.tts import VOICES, MiraiHttpTTSService, MiraiTTSSettings
from pipecat_mirai.tts_websocket import DEFAULT_WEBSOCKET_URL, MiraiTTSService, MiraiWebsocketTTSService

__version__ = "0.8.0"

__all__ = [
    "DEFAULT_LEAD_SECS",
    "DEFAULT_STT_WEBSOCKET_URL",
    "DEFAULT_WEBSOCKET_URL",
    "PRODUCTION_REALTIME_URL",
    "SANDBOX_REALTIME_URL",
    "STT_LANGUAGES",
    "VOICES",
    "MiraiHttpTTSService",
    "MiraiRealtimeLLMService",
    "MiraiSTTService",
    "MiraiSTTSettings",
    "MiraiTTSService",
    "MiraiTTSSettings",
    "MiraiTurnMetrics",
    "MiraiWebsocketTTSService",
    "PrewarmResult",
    "__version__",
    "apply_output_lead",
    "close_shared_connections",
    "prewarm",
    "shared_connection_stats",
]
