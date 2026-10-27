#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Moved to :mod:`fourier_frequency.pool`; this name is that same module."""

import sys

from fourier_frequency import pool as _module

sys.modules[__name__] = _module
