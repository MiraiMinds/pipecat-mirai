"""A fake of Mirai's HTTP speech endpoint (``POST /v1/audio/speech``), for tests. No Pipecat."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from http import HTTPStatus

import httpx
import numpy as np

SR = 48000
ODD_READS = (1, 4097, 3, 9600)


def tone(seconds: float, rate: int = SR) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes()


class FakeMirai:
    """Mirai's ``/v1/audio/speech`` and ``/v1/models``, answering like a given server version.

    By default it is the new server: it produces the requested ``sample_rate`` and
    says so in ``X-Sample-Rate``. ``honours_rate=False`` is today's server (ignores
    the field, always 48 kHz); ``rejects_rate=True`` is a strict old server that
    answers 400 to it. Use ``client()`` for an httpx mock transport that controls
    exactly where network reads end, or ``serve()`` for a real local HTTP/1.1
    server that counts TCP connections and sees the client hang up.

    ``reads`` are the sizes of the pieces the body is sent in, ``pace`` the pause
    after each piece and ``first_gap`` the pause after the first one. ``seconds``
    is the length of the audio, or a dict from input text to length.

    ``script`` (``serve()`` only) scripts successive speech requests, one dict
    each, after which requests are answered normally: ``{"status": 429,
    "retry_after": 0.2, "code": "at_capacity"}`` refuses one, ``{"reset":
    True}`` drops the connection without answering, ``{"cut_after": n}``
    sends ``n`` bytes of audio and then drops it. Every response carries an
    ``X-Request-Id``.
    """

    def __init__(
        self,
        *,
        seconds=0.5,
        honours_rate=True,
        rejects_rate=False,
        rate_header=True,
        headers=None,
        status=200,
        error="bad request",
        reads=ODD_READS,
        pace=0.0,
        first_gap=0.0,
        models_error=None,
        script=None,
    ):
        self.seconds = seconds
        self.honours_rate = honours_rate
        self.rejects_rate = rejects_rate
        self.rate_header = rate_header
        self.headers = headers or {}
        self.status = status
        self.error = error
        self.reads = reads
        self.pace = pace
        self.first_gap = first_gap
        self.models_error = models_error
        self.script = list(script or [])
        self.requests = []  # {"method", "path", "headers", "json", "conn", "t", "sent", "cut_at"}
        self.connections = 0

    @property
    def speech(self):
        """JSON bodies of the speech requests, in order."""
        return [r["json"] for r in self.requests if r["path"].endswith("/audio/speech")]

    @property
    def speech_requests_t(self):
        """When each speech request arrived (time.monotonic())."""
        return [r["t"] for r in self.requests if r["path"].endswith("/audio/speech")]

    def _respond(self, path, body):
        if path.endswith("/models"):
            return 200, {"content-type": "application/json"}, b'{"object": "list", "data": []}'
        if not path.endswith("/audio/speech"):  # e.g. a WebSocket handshake to /audio/speech/stream
            return 404, {"content-type": "application/json"}, _error_body("not found")
        req = json.loads(body)
        if self.rejects_rate and "sample_rate" in req:
            message = 'json: unknown field "sample_rate"'
            return 400, {"content-type": "application/json"}, _error_body(message)
        if self.status != 200:
            return self.status, {"content-type": "application/json"}, _error_body(self.error)
        rate = req.get("sample_rate", SR) if self.honours_rate else SR
        headers = {"content-type": "audio/pcm"}
        if self.rate_header:
            headers["x-sample-rate"] = str(rate)
        if self.honours_rate:
            headers["x-audio-encoding"] = "pcm_s16le"
        headers.update(self.headers)
        seconds = self.seconds[req["input"]] if isinstance(self.seconds, dict) else self.seconds
        return 200, headers, tone(seconds, rate)

    def _pieces(self, body):
        """(piece, pause after it) pairs."""
        pos, i = 0, 0
        while pos < len(body):
            n = self.reads[i % len(self.reads)]
            yield body[pos : pos + n], self.first_gap if i == 0 and self.first_gap else self.pace
            pos += n
            i += 1

    def _record(self, method, path, headers, body, conn):
        self.requests.append(
            {
                "method": method,
                "path": path,
                "headers": headers,
                "json": json.loads(body) if body else None,
                "conn": conn,
                "t": time.monotonic(),
                "sent": 0,
                "cut_at": None,
            }
        )
        return self.requests[-1]

    def client(self) -> httpx.AsyncClient:
        async def handler(request: httpx.Request) -> httpx.Response:
            self._record(request.method, request.url.path, dict(request.headers), request.content, 1)
            if request.url.path.endswith("/models") and self.models_error:
                raise self.models_error
            status, headers, body = self._respond(request.url.path, request.content)
            if headers["content-type"] != "audio/pcm":
                return httpx.Response(status, headers=headers, content=body)

            async def stream():
                for piece, pause in self._pieces(body):
                    yield piece
                    if pause:
                        await asyncio.sleep(pause)

            return httpx.Response(status, headers=headers, content=stream())

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    @asynccontextmanager
    async def serve(self):
        """Serve on a local port; yields the base URL (``.../v1``)."""
        writers = []

        async def handle(reader, writer):
            writers.append(writer)
            self.connections += 1
            conn = self.connections
            try:
                while line := await reader.readline():
                    method, target, _ = line.decode("latin-1").split(" ", 2)
                    headers = {}
                    while (h := await reader.readline()) not in (b"\r\n", b""):
                        k, _, v = h.decode("latin-1").partition(":")
                        headers[k.strip().lower()] = v.strip()
                    body = await reader.readexactly(int(headers.get("content-length", "0")))
                    record = self._record(method, target, headers, body, conn)
                    action = self.script.pop(0) if self.script and target.endswith("/audio/speech") else {}
                    if action.get("reset"):
                        writer.transport.abort()  # no response at all
                        return
                    status, rheaders, payload = self._respond(target, body)
                    rheaders = {**rheaders, "x-request-id": f"req_{len(self.requests)}"}
                    if action.get("status"):
                        status = action["status"]
                        rheaders["content-type"] = "application/json"
                        if "retry_after" in action:
                            rheaders["retry-after"] = str(action["retry_after"])
                        error = {
                            "code": action.get("code", "error"),
                            "message": action.get("message", "busy"),
                        }
                        payload = json.dumps({"error": error}).encode()
                    head = [f"HTTP/1.1 {status} {HTTPStatus(status).phrase}", "transfer-encoding: chunked"]
                    head += [f"{k}: {v}" for k, v in rheaders.items()]
                    writer.write(("\r\n".join(head) + "\r\n\r\n").encode())
                    try:
                        for piece, pause in self._pieces(payload):
                            if "cut_after" in action and record["sent"] >= action["cut_after"]:
                                await writer.drain()
                                writer.transport.abort()  # mid-stream: the network dropped
                                return
                            writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                            await writer.drain()
                            record["sent"] += len(piece)
                            if pause:
                                await asyncio.sleep(pause)
                        writer.write(b"0\r\n\r\n")
                        await writer.drain()
                    except ConnectionError:
                        record["cut_at"] = time.monotonic()  # the client hung up mid-stream
                        raise
            except (ConnectionError, asyncio.IncompleteReadError):
                pass
            finally:
                writer.close()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        try:
            yield f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}/v1"
        finally:
            server.close()
            for w in writers:
                w.close()
            await server.wait_closed()


def _error_body(message):
    return json.dumps({"error": {"code": "invalid_request", "message": message}}).encode()
