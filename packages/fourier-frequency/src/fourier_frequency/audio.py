#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""G.711 (mu-law and A-law) to and from 16-bit PCM, for phone audio.

Telephony media streams (Twilio, Plivo, Exotel, ...) carry 8 kHz mu-law or
A-law. Speech-to-text can take either on the wire as it is; text-to-speech
produces 16-bit PCM and converts at the end when asked for a G.711 encoding.
"""

from __future__ import annotations

import numpy as np


def lin2ulaw(pcm: np.ndarray) -> np.ndarray:
    """16-bit PCM to G.711 mu-law bytes."""
    x = pcm.astype(np.int32)
    sign = (x < 0).astype(np.int32) << 7
    magnitude = np.minimum(np.abs(x), 32635) + 0x84
    exponent = np.clip(np.floor(np.log2(magnitude)).astype(np.int32) - 7, 0, 7)
    mantissa = (magnitude >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa) & 0xFF).astype(np.uint8)


def lin2alaw(pcm: np.ndarray) -> np.ndarray:
    """16-bit PCM to G.711 A-law bytes."""
    x = pcm.astype(np.int32)
    sign = (x >= 0).astype(np.int32) << 7
    magnitude = np.where(x < 0, -x - 1, x) >> 3  # 13-bit magnitude, 0..4095
    segment = np.where(magnitude >= 32, np.floor(np.log2(np.maximum(magnitude, 1))).astype(np.int32) - 4, 0)
    shift = np.maximum(segment, 1)
    mantissa = (magnitude >> shift) & 0x0F
    return ((sign | (segment << 4) | mantissa) ^ 0x55).astype(np.uint8)


def ulaw2lin(data: bytes) -> np.ndarray:
    """G.711 mu-law bytes to 16-bit PCM."""
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(sign, -magnitude, magnitude).astype(np.int16)


def alaw2lin(data: bytes) -> np.ndarray:
    """G.711 A-law bytes to 16-bit PCM."""
    a = np.frombuffer(data, dtype=np.uint8).astype(np.int32) ^ 0x55
    sign = a & 0x80
    exponent = (a >> 4) & 0x07
    mantissa = a & 0x0F
    magnitude = np.where(exponent == 0, (mantissa << 4) + 8, ((mantissa << 4) + 0x108) << (exponent - 1))
    return np.where(sign, magnitude, -magnitude).astype(np.int16)
