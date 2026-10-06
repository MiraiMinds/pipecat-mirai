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

__version__ = "0.2.0"

__all__ = [
    "DEFAULT_LEAD_SECS",
    "PRODUCTION_REALTIME_URL",
    "SANDBOX_REALTIME_URL",
    "VOICES",
    "MiraiRealtimeLLMService",
    "MiraiTTSService",
    "MiraiTTSSettings",
    "MiraiTurnMetrics",
    "__version__",
    "apply_output_lead",
]
