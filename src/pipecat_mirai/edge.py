#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Moved to :mod:`fourier_frequency.edge`; this name is that same module."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("fourier_frequency.edge")
