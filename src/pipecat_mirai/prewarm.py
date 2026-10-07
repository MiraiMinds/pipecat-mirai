#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Open connections to Mirai before the first call: :func:`prewarm`."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field

from loguru import logger

from pipecat_mirai.pool import shared_http_client, websocket_pool
from pipecat_mirai.tts import DEFAULT_BASE_URL
from pipecat_mirai.tts_websocket import MAX_MESSAGE_BYTES


@dataclass
class PrewarmResult:
    """What :func:`prewarm` has open.

    Parameters:
        http_connections: Idle keep-alive HTTP connections open to ``base_url``.
        websockets: Pre-authenticated sockets waiting for a pipeline.
        errors: What failed, if anything. The pools keep trying in the
            background either way.
    """

    http_connections: int = 0
    websockets: int = 0
    errors: list[str] = field(default_factory=list)


async def prewarm(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    connections: int = 8,
    websocket: int = 0,
    websocket_url: str | None = None,
    timeout: float = 10.0,
) -> PrewarmResult:
    """Open connections to Mirai now, for every pipeline this event loop runs later.

    Call it once when your server or worker starts, in the event loop that
    will run the pipelines (in FastAPI, the lifespan handler), and give it the
    number of calls you expect to start at the same moment. The calls that
    start together then find their connections already open, and none of
    their first sentences waits for a TCP and TLS handshake.

    - ``connections`` HTTP keep-alive connections are opened (one
      authenticated ``GET /v1/models`` each, never billed) in the client every
      :class:`~pipecat_mirai.MiraiTTSService` for ``base_url`` in this loop
      shares, and refreshed every 45 s, so Mirai's 75 s idle timeout never
      closes them.
    - ``websocket`` sockets to the streaming endpoint are opened and left
      waiting at ``session.ready``, with an empty ``session.update`` every
      30 s against Mirai's 120 s idle timeout. Each
      :class:`~pipecat_mirai.MiraiWebsocketTTSService` with the same URL and
      API key takes one when its pipeline starts, and a replacement is opened
      in the background. A socket serves one pipeline and is closed when it
      ends.

    Calling it again changes the numbers; ``0`` stops keeping that kind open.
    Network failures are logged and returned in :attr:`PrewarmResult.errors`,
    never raised: a service whose pool is empty connects on its own, as it
    would without this.

    Args:
        api_key: Mirai API key. Defaults to ``MIRAI_API_KEY`` (or ``MIRA_API_KEY``).
        base_url: API base URL, as given to ``MiraiTTSService``.
        connections: HTTP connections to keep open (0 to 64).
        websocket: WebSocket sockets to keep waiting.
        websocket_url: The streaming endpoint, as given to
            ``MiraiWebsocketTTSService``. Defaults to ``base_url`` with
            ``wss://`` and ``/audio/speech/stream``.
        timeout: Seconds to wait for the connections to open.

    Returns:
        How many connections and sockets are open, and any errors.
    """
    key = api_key or os.getenv("MIRAI_API_KEY") or os.getenv("MIRA_API_KEY")
    if not key:
        raise ValueError("Set MIRAI_API_KEY or pass api_key to prewarm().")
    if type(connections) is not int or not 0 <= connections <= 64:
        raise ValueError(f"connections must be an int from 0 to 64; got {connections!r}")
    if type(websocket) is not int or not 0 <= websocket <= 64:
        raise ValueError(f"websocket must be an int from 0 to 64; got {websocket!r}")
    if not timeout > 0:
        raise ValueError(f"timeout must be > 0; got {timeout!r}")
    base = base_url.rstrip("/")
    ws_url = websocket_url or _websocket_url(base)
    if not ws_url.startswith(("ws://", "wss://")):
        raise ValueError(f"websocket_url must be a ws:// or wss:// URL; got {ws_url!r}")
    headers = {"Authorization": f"Bearer {key}"}

    result = PrewarmResult()

    async def http():
        shared = shared_http_client(base, create=connections > 0)
        if shared is None:
            return
        shared.set_floor(connections, headers)
        if connections:
            result.http_connections = await shared.wait_warm(connections, timeout)
            if result.http_connections < connections and shared.last_error:
                result.errors.append(shared.last_error)

    async def ws():
        pool = websocket_pool(ws_url, headers, MAX_MESSAGE_BYTES, create=websocket > 0)
        if pool is None:
            return
        pool.set_floor(websocket)
        if websocket:
            result.websockets = await pool.wait_ready(websocket, timeout)
            if result.websockets < websocket and pool.last_error:
                result.errors.append(f"WebSocket {ws_url}: {pool.last_error}")

    await asyncio.gather(http(), ws())
    logger.debug(
        f"Mirai prewarm: {result.http_connections}/{connections} HTTP connections to {base}, "
        f"{result.websockets}/{websocket} sockets to {ws_url}"
    )
    return result


def _websocket_url(base: str) -> str:
    """``https://host/v1`` -> ``wss://host/v1/audio/speech/stream``."""
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            return ws + base[len(http) :] + "/audio/speech/stream"
    return base + "/audio/speech/stream"
