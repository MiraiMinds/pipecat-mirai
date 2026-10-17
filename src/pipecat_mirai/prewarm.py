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

from pipecat_mirai.edge import EdgeRoute, edge_mode
from pipecat_mirai.pool import STT_PROTOCOL, shared_http_client, websocket_pool
from pipecat_mirai.stt import MAX_MESSAGE_BYTES as STT_MAX_MESSAGE_BYTES
from pipecat_mirai.stt import WIRE_SAMPLE_RATES, pool_url
from pipecat_mirai.stt import _http_base as _stt_http_base
from pipecat_mirai.tts import DEFAULT_BASE_URL
from pipecat_mirai.tts_websocket import MAX_MESSAGE_BYTES, _http_base


@dataclass
class PrewarmResult:
    """What :func:`prewarm` has open.

    Parameters:
        http_connections: Idle keep-alive HTTP connections open to ``base_url``.
        websockets: Pre-authenticated TTS sockets waiting for a pipeline.
        errors: What failed, if anything. The pools keep trying in the
            background either way.
        stt_websockets: Pre-authenticated STT sockets waiting for a pipeline.
    """

    http_connections: int = 0
    websockets: int = 0
    errors: list[str] = field(default_factory=list)
    stt_websockets: int = 0


async def prewarm(
    *,
    api_key: str | None = None,
    base_url: str = DEFAULT_BASE_URL,
    connections: int = 0,
    websocket: int = 8,
    websocket_url: str | None = None,
    edge: bool | str = "auto",
    timeout: float = 10.0,
    stt_websockets: int = 0,
    stt_url: str | None = None,
    stt_sample_rate: int = 8000,
    stt_endpointing: str = "manual",
) -> PrewarmResult:
    """Open connections to Mirai now, for every pipeline this event loop runs later.

    Call it once when your server or worker starts, in the event loop that
    will run the pipelines (in FastAPI, the lifespan handler), and give it the
    number of calls you expect to start at the same moment. The calls that
    start together then find their connections already open, and none of
    their first sentences waits for a TCP and TLS handshake.

    - ``connections`` HTTP keep-alive connections are opened (one
      authenticated ``GET /v1/models`` each, never billed) in the client every
      :class:`~pipecat_mirai.MiraiHttpTTSService` for ``base_url`` in this loop
      shares, and refreshed every 45 s, so Mirai's 75 s idle timeout never
      closes them.
    - ``websocket`` sockets to the streaming endpoint are opened and left
      waiting at ``session.ready``, with an empty ``session.update`` every
      30 s against Mirai's 120 s idle timeout. Each
      :class:`~pipecat_mirai.MiraiTTSService` with the same URL and
      API key takes one when its pipeline starts, and a replacement is opened
      in the background. A socket serves one pipeline and is closed when it
      ends. Sockets open on Mirai's edge when it is available, as the
      service's own do.
    - ``stt_websockets`` sockets to the streaming STT endpoint are opened and
      left waiting at ``session.begin`` (a ping and 40 ms of silence every
      20 s). A :class:`~pipecat_mirai.MiraiSTTService` takes one when its
      pipeline starts if its fixed settings match (``stt_sample_rate``,
      ``stt_endpointing``, the default model and linear16), and moves it to
      its own language with ``config.update`` if that differs.

    Calling it again changes the numbers; ``0`` stops keeping that kind open.
    Network failures are logged and returned in :attr:`PrewarmResult.errors`,
    never raised: a service whose pool is empty connects on its own, as it
    would without this.

    Args:
        api_key: Mirai API key. Defaults to ``MIRAI_API_KEY`` (or ``MIRA_API_KEY``).
        base_url: API base URL, as given to the services.
        connections: HTTP connections to keep open (0 to 64).
        websocket: WebSocket sockets to keep waiting.
        websocket_url: The streaming endpoint, as given to
            ``MiraiTTSService``. Defaults to ``base_url`` with
            ``wss://`` and ``/audio/speech/stream``.
        edge: As given to ``MiraiTTSService`` (``"auto"``,
            ``False`` or an edge URL); waiting sockets go only to services
            with the same setting.
        timeout: Seconds to wait for the connections to open.
        stt_websockets: STT sockets to keep waiting (0 to 64).
        stt_url: The STT streaming endpoint, as given to ``MiraiSTTService``
            as ``url``. Defaults to ``base_url`` with ``wss://`` and
            ``/audio/transcriptions/stream``.
        stt_sample_rate: The rate the services that take these sockets send
            at: 8000 (phone pipelines, the default) or 16000.
        stt_endpointing: ``"manual"`` (the pipeline has a VAD; the default)
            or ``"vad"``, as the services resolve it.

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
    if type(stt_websockets) is not int or not 0 <= stt_websockets <= 64:
        raise ValueError(f"stt_websockets must be an int from 0 to 64; got {stt_websockets!r}")
    if stt_sample_rate not in WIRE_SAMPLE_RATES:
        raise ValueError(f"stt_sample_rate must be 8000 or 16000; got {stt_sample_rate!r}")
    if stt_endpointing not in ("manual", "vad"):
        raise ValueError(f"stt_endpointing must be 'manual' or 'vad'; got {stt_endpointing!r}")
    if not timeout > 0:
        raise ValueError(f"timeout must be > 0; got {timeout!r}")
    base = base_url.rstrip("/")
    ws_url = websocket_url or _websocket_url(base)
    if not ws_url.startswith(("ws://", "wss://")):
        raise ValueError(f"websocket_url must be a ws:// or wss:// URL; got {ws_url!r}")
    headers = {"Authorization": f"Bearer {key}"}
    mode = edge_mode(edge)
    route = EdgeRoute(ws_url, headers, mode, _http_base(ws_url)) if mode else None
    stt_base = (stt_url or _websocket_url(base, "/audio/transcriptions/stream")).rstrip("/")
    if not stt_base.startswith(("ws://", "wss://")):
        raise ValueError(f"stt_url must be a ws:// or wss:// URL; got {stt_base!r}")
    stt_pool_url = pool_url(stt_base, sample_rate=stt_sample_rate, endpointing=stt_endpointing)
    stt_mode = edge_mode(edge, "stt")
    stt_route = (
        EdgeRoute(stt_pool_url, headers, stt_mode, _stt_http_base(stt_base), "stt") if stt_mode else None
    )

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
        pool = websocket_pool(ws_url, headers, MAX_MESSAGE_BYTES, edge=route, create=websocket > 0)
        if pool is None:
            return
        pool.set_floor(websocket)
        if websocket:
            result.websockets = await pool.wait_ready(websocket, timeout)
            if result.websockets < websocket and pool.last_error:
                result.errors.append(f"WebSocket {ws_url}: {pool.last_error}")

    async def stt():
        pool = websocket_pool(
            stt_pool_url,
            headers,
            STT_MAX_MESSAGE_BYTES,
            edge=stt_route,
            create=stt_websockets > 0,
            protocol=STT_PROTOCOL,
        )
        if pool is None:
            return
        pool.set_floor(stt_websockets)
        if stt_websockets:
            result.stt_websockets = await pool.wait_ready(stt_websockets, timeout)
            if result.stt_websockets < stt_websockets and pool.last_error:
                result.errors.append(f"WebSocket {stt_base}: {pool.last_error}")

    await asyncio.gather(http(), ws(), stt())
    logger.debug(
        f"Mirai prewarm: {result.http_connections}/{connections} HTTP connections to {base}, "
        f"{result.websockets}/{websocket} sockets to {ws_url}, "
        f"{result.stt_websockets}/{stt_websockets} sockets to {stt_base}"
    )
    return result


def _websocket_url(base: str, path: str = "/audio/speech/stream") -> str:
    """``https://host/v1`` -> ``wss://host/v1/audio/speech/stream`` (or another ``path``)."""
    for http, ws in (("https://", "wss://"), ("http://", "ws://")):
        if base.startswith(http):
            return ws + base[len(http) :] + path
    return base + path
