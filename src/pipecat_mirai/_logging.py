#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Pipecat logs with loguru; fourier-frequency logs with the standard library.

This sends fourier-frequency's records to loguru, every level, under the module
that logged them, so a Pipecat bot sees the same lines it always has.
"""

import inspect
import logging

from loguru import logger


class _ToLoguru(logging.Handler):
    def emit(self, record: logging.LogRecord):
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        # Point loguru at the frame that called logging, not at this handler.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def forward_to_loguru(name: str = "fourier_frequency"):
    """Send the ``name`` logger's records to loguru (once), and only there."""
    log = logging.getLogger(name)
    if any(isinstance(handler, _ToLoguru) for handler in log.handlers):
        return
    log.addHandler(_ToLoguru())
    log.setLevel(logging.DEBUG)
    log.propagate = False
