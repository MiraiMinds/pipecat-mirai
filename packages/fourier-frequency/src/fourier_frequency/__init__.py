#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai's streaming text-to-speech and speech-to-text, for any Python voice agent."""

from fourier_frequency.errors import FrequencyError
from fourier_frequency.pool import close_shared_connections, shared_connection_stats
from fourier_frequency.tts import (
    DEFAULT_BASE_URL,
    DEFAULT_WEBSOCKET_URL,
    VOICES,
    HttpTTS,
    TTSListener,
    TTSStream,
    TTSTurn,
)

__version__ = "0.8.0"

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_WEBSOCKET_URL",
    "VOICES",
    "FrequencyError",
    "HttpTTS",
    "TTSListener",
    "TTSStream",
    "TTSTurn",
    "__version__",
    "close_shared_connections",
    "shared_connection_stats",
]
