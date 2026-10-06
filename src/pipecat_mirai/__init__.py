#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech and Realtime API for Pipecat."""

from pipecat_mirai.pacing import DEFAULT_LEAD_SECS, apply_output_lead
from pipecat_mirai.realtime import (
    PRODUCTION_REALTIME_URL,
    SANDBOX_REALTIME_URL,
    MiraiRealtimeLLMService,
    MiraiTurnMetrics,
)
from pipecat_mirai.tts import VOICES, MiraiTTSService, MiraiTTSSettings
from pipecat_mirai.tts_websocket import DEFAULT_WEBSOCKET_URL, MiraiWebsocketTTSService

__version__ = "0.3.0"

__all__ = [
    "DEFAULT_LEAD_SECS",
    "DEFAULT_WEBSOCKET_URL",
    "PRODUCTION_REALTIME_URL",
    "SANDBOX_REALTIME_URL",
    "VOICES",
    "MiraiRealtimeLLMService",
    "MiraiTTSService",
    "MiraiTTSSettings",
    "MiraiTurnMetrics",
    "MiraiWebsocketTTSService",
    "__version__",
    "apply_output_lead",
]
