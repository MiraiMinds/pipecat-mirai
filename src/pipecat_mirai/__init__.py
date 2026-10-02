#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai text-to-speech for Pipecat."""

from pipecat_mirai.pacing import DEFAULT_LEAD_SECS, apply_output_lead
from pipecat_mirai.tts import VOICES, MiraiTTSService, MiraiTTSSettings

__version__ = "0.1.0"

__all__ = [
    "DEFAULT_LEAD_SECS",
    "VOICES",
    "MiraiTTSService",
    "MiraiTTSSettings",
    "__version__",
    "apply_output_lead",
]
