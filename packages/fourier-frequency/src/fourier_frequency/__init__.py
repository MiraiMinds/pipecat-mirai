#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Mirai's streaming text-to-speech and speech-to-text, for any Python voice agent."""

from fourier_frequency.errors import FrequencyError
from fourier_frequency.pacer import Pacer
from fourier_frequency.pool import close_shared_connections, shared_connection_stats
from fourier_frequency.prewarm import PrewarmResult, prewarm
from fourier_frequency.stt import (
    DEFAULT_STT_WEBSOCKET_URL,
    STT_LANGUAGES,
    SpeechStarted,
    SpeechStopped,
    StreamClosed,
    StreamError,
    STTEvent,
    STTListener,
    STTStream,
    Transcript,
    stt_language,
    stt_secondary_languages,
)
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
    "DEFAULT_STT_WEBSOCKET_URL",
    "DEFAULT_WEBSOCKET_URL",
    "STT_LANGUAGES",
    "VOICES",
    "FrequencyError",
    "HttpTTS",
    "Pacer",
    "PrewarmResult",
    "STTEvent",
    "STTListener",
    "STTStream",
    "SpeechStarted",
    "SpeechStopped",
    "StreamClosed",
    "StreamError",
    "TTSListener",
    "TTSStream",
    "TTSTurn",
    "Transcript",
    "__version__",
    "close_shared_connections",
    "prewarm",
    "shared_connection_stats",
    "stt_language",
    "stt_secondary_languages",
]
