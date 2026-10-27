#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""What went wrong, in one exception type.

Every error a stream reports (an event to its listener, or an exception from
an awaited call) is a :class:`FrequencyError`. ``str()`` says what happened
and what to do about it; ``category`` says what kind of problem it is, in the
words Pipecat's ``ErrorCategory`` uses, so an orchestrator can decide whether
to retry, alert or give up without parsing the message.
"""

from __future__ import annotations

# What kind of problem an error is. The values are Pipecat's ErrorCategory values.
AUTHENTICATION = "authentication"  # the API key is wrong
AUTHORIZATION = "authorization"  # the key is revoked or not allowed this
INVALID_REQUEST = "invalid_request"  # a setting Mirai refuses
QUOTA = "quota"  # the workspace is out of credit
RATE_LIMIT = "rate_limit"  # too many requests or sessions at once
CONNECTIVITY = "connectivity"  # the network, or a dropped socket
SERVER = "server"  # Mirai failed

CATEGORIES = (AUTHENTICATION, AUTHORIZATION, INVALID_REQUEST, QUOTA, RATE_LIMIT, CONNECTIVITY, SERVER)


class FrequencyError(Exception):
    """An error from Mirai, or on the way to it. ``str()`` says what to do.

    Attributes:
        category: One of :data:`CATEGORIES`.
        code: Mirai's error code (``"at_capacity"``, ``"insufficient_balance"``,
            ...), when Mirai sent one.
        status: The HTTP status of a refused request or handshake.
        request_id: Mirai's request id, for support.
        permanent: Retrying the same thing would fail the same way (a bad key,
            no credit, a refused setting); the stream has stopped trying.
    """

    def __init__(
        self,
        message: str,
        *,
        category: str = SERVER,
        code: str | None = None,
        status: int | None = None,
        request_id: str | None = None,
        permanent: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.code = code
        self.status = status
        self.request_id = request_id
        self.permanent = permanent
