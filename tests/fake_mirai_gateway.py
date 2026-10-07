"""A fake of Mirai's API gateway for the edge tests: the token route and the streaming socket.

As in production, one origin serves both: ``POST /v2/tts/stream/tokens`` is
answered here, and a WebSocket handshake is passed on, bytes and all, to a
:class:`~fake_mirai_ws.FakeMiraiWS` (the gateway's streaming endpoint). The
edge is a second ``FakeMiraiWS`` whose ``auth`` is :meth:`FakeGateway.spend`:
it accepts only tokens this gateway issued, each once.

Knobs: ``edge_url`` (what the token answer names; ``None`` offers no edge),
``edge_field`` (False: answer without the ``edge_url`` field, as today's
gateway does), ``token_status`` (e.g. 404 for a gateway without the route, or
503) and ``token_delay`` (seconds before answering).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from contextlib import asynccontextmanager
from http import HTTPStatus
from urllib.parse import urlparse

TOKEN_PATH = "/v2/tts/stream/tokens"


class FakeGateway:
    def __init__(
        self,
        stream,
        *,
        key="sk_test",
        edge_url=None,
        edge_field=True,
        token_status=201,
        token_delay=0.0,
        batch=False,
    ):
        self.stream = stream  # the FakeMiraiWS behind the gateway's streaming endpoint
        self.key = key
        self.edge_url = edge_url
        self.edge_field = edge_field
        self.token_status = token_status
        self.token_delay = token_delay
        self.batch = batch  # answer `count` with a `tokens` list (gateways since the batch change)
        self.mints: list[dict] = []  # {"path", "headers", "conn", "t", "token"}
        self.issued: dict[str, bool] = {}  # token -> spent
        self.connections = 0  # TCP connections, token requests and handshakes alike

    # ---- the edge's side ----

    def spend(self, headers) -> bool:
        """Whether a handshake's ``Authorization`` is a token issued here and not yet used."""
        token = (headers.get("Authorization") or "").removeprefix("Bearer ")
        if self.issued.get(token) is False:
            self.issued[token] = True
            return True
        return False

    @property
    def tokens(self) -> list[str]:
        return [m["token"] for m in self.mints if m["token"]]

    # ---- server ----

    async def _answer_token(self, writer, path, headers, conn):
        record = {"path": path, "headers": headers, "conn": conn, "t": time.monotonic(), "token": None}
        self.mints.append(record)
        if self.token_delay:
            await asyncio.sleep(self.token_delay)
        if headers.get("authorization") != f"Bearer {self.key}":
            status, body = 401, {"error": {"code": "unauthorized", "message": "invalid API key"}}
        elif path.split("?", 1)[0] != TOKEN_PATH or self.token_status == 404:
            status, body = 404, {"error": {"code": "not_found", "message": "not found"}}
        elif self.token_status not in (200, 201):
            status, body = self.token_status, {"error": {"code": "unavailable", "message": "try again"}}
        else:
            count = 1
            if self.batch and "count=" in path:
                with contextlib.suppress(ValueError):
                    count = max(1, min(20, int(path.split("count=", 1)[1].split("&", 1)[0])))
            batch = [f"te1.{secrets.token_urlsafe(12)}.sig" for _ in range(count)]
            token = batch[0]
            record["token"] = token
            record["batch"] = batch
            for t in batch:
                self.issued[t] = False
            status = self.token_status
            body = {"object": "tts_stream_token", "token": token, "expires_at": int(time.time()) + 60}
            if self.batch:
                body["tokens"] = batch
            if self.edge_field:
                body["edge_url"] = self.edge_url
        payload = json.dumps(body).encode()
        head = (
            f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n"
            f"content-type: application/json\r\ncontent-length: {len(payload)}\r\n\r\n"
        )
        writer.write(head.encode() + payload)
        await writer.drain()

    @asynccontextmanager
    async def serve(self):
        """Serve the gateway; yields its streaming URL (``ws://127.0.0.1:<port>/v1/audio/speech/stream``)."""
        writers = []

        async def pipe(reader, writer):
            try:
                while data := await reader.read(65536):
                    writer.write(data)
                    await writer.drain()
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    writer.close()

        async with self.stream.serve() as stream_url:
            upstream = urlparse(stream_url)

            async def handle(reader, writer):
                writers.append(writer)
                self.connections += 1
                conn = self.connections
                try:
                    while True:
                        try:
                            head = await reader.readuntil(b"\r\n\r\n")
                        except (asyncio.IncompleteReadError, ConnectionError):
                            return
                        lines = head.decode("latin-1").split("\r\n")
                        method, path, _ = lines[0].split(" ", 2)
                        headers = {}
                        for line in lines[1:]:
                            if ":" in line:
                                k, _, v = line.partition(":")
                                headers[k.strip().lower()] = v.strip()
                        if method == "POST":
                            await reader.readexactly(int(headers.get("content-length", "0")))
                            await self._answer_token(writer, path, headers, conn)
                            continue
                        # A WebSocket handshake: the streaming endpoint takes it from here.
                        up_reader, up_writer = await asyncio.open_connection(upstream.hostname, upstream.port)
                        writers.append(up_writer)
                        up_writer.write(head)
                        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
                        return
                except Exception:
                    pass
                finally:
                    with contextlib.suppress(Exception):
                        writer.close()

            server = await asyncio.start_server(handle, "127.0.0.1", 0)
            port = server.sockets[0].getsockname()[1]
            try:
                yield f"ws://127.0.0.1:{port}/v1/audio/speech/stream"
            finally:
                server.close()
                for w in writers:
                    with contextlib.suppress(Exception):
                        w.close()
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(server.wait_closed(), 2.0)
