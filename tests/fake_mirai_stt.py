"""A fake of Mirai's streaming STT socket (``/v1/audio/transcriptions/stream``), for tests.

It implements the protocol from the server's side, as Mirai's STT gateway
speaks it (Sarvam-v3 compatible). The query (``language_code``, ``model``,
``encoding``, ``sample_rate``, ``endpointing``, tuning) is validated before the
upgrade: an unknown parameter is a 400 ``invalid_config``. Then
``session.begin`` {request_id, session_id, config, capabilities, limits}, and:

- ``audio_input``: audio is decoded (linear16, mulaw or alaw) and kept.
  Under ``endpointing=manual`` it belongs to the utterance open between
  ``speech_start`` and ``speech_end``; under ``vad`` a simple energy VAD on the
  server finds utterances itself and sends ``vad.speech_start`` /
  ``vad.speech_end``.
- ``transcript.partial`` every ``partial_every_s`` of an utterance's audio, and
  exactly one ``transcript.final`` per utterance, in order, ``slow_final_s``
  after it ends. The text names the utterance's loudest frequency and its
  length (``"440 hertz, 0.8 seconds"``), so a test knows which utterance a
  final is for whichever socket heard it; silence gives an empty final.
- ``flush`` finalises the open utterance; ``config.update`` answers
  ``config.updated``; ``ping`` answers ``pong``; ``end`` finalises what is
  open, waits for the finals, sends ``session.end`` and closes.

Knobs: ``auth`` (a callable given each handshake's headers: False answers
401), ``refuse_status`` (answer every handshake with that HTTP status; 429
carries ``Retry-After: 0.1``), ``rate_limit_attempts`` / ``refuse_attempts``
(handshake attempts, counted from 1, answered 429 / 503), ``capacity`` (most
sessions open at once; past it 429 ``at_capacity``), ``slow_final_s``,
``no_final`` (True: never send a final), ``drop_after_s`` (abort the first
connection's TCP once it has received that many seconds of audio),
``session_limits`` (the ``limits`` in session.begin; ``max_session_secs`` is
enforced with ``session.draining``, the last finals, ``session.end`` and a
close), ``drain_after_s`` (send ``session.draining`` on the first connection
that many seconds after it opens, and end it ``drain_grace_s`` later),
``fatal_error`` (an error event, e.g. ``{"code": "insufficient_balance"}``,
sent with ``is_fatal`` instead of the first final, then the socket closes),
``partial_every_s`` and ``silent`` (never send session.begin).
``handshakes`` records every handshake's path, query and headers; ``conns``
every session, with what it received.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from urllib.parse import parse_qsl, urlparse

import numpy as np
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

PATH = "/v1/audio/transcriptions/stream"
ALLOWED = {
    "language_code",
    "model",
    "encoding",
    "sample_rate",
    "endpointing",
    "stream_type",
    "mode",
    "return_timestamps",
    "threshold",
    "silence_duration_ms",
    "prefix_padding_ms",
    "min_speech_duration_ms",
    "token",
}
VAD_FRAME_SECS = 0.02
VAD_RMS = 300.0


def ulaw2lin(data: bytes) -> np.ndarray:
    u = ~np.frombuffer(data, dtype=np.uint8).astype(np.int32) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(sign, -magnitude, magnitude).astype(np.int16)


def alaw2lin(data: bytes) -> np.ndarray:
    a = np.frombuffer(data, dtype=np.uint8).astype(np.int32) ^ 0x55
    sign = a & 0x80
    exponent = (a >> 4) & 0x07
    mantissa = a & 0x0F
    magnitude = np.where(exponent == 0, (mantissa << 4) + 8, ((mantissa << 4) + 0x108) << (exponent - 1))
    return np.where(sign, magnitude, -magnitude).astype(np.int16)


def decode(data: bytes, encoding: str) -> np.ndarray:
    if encoding == "mulaw":
        return ulaw2lin(data)
    if encoding == "alaw":
        return alaw2lin(data)
    return np.frombuffer(data, dtype="<i2")


def dominant_hz(pcm: np.ndarray, rate: int) -> int:
    """The loudest frequency in ``pcm``, to the nearest 10 Hz (0 for silence)."""
    if len(pcm) < 64 or float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2))) < VAD_RMS / 4:
        return 0
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64)))
    hz = np.fft.rfftfreq(len(pcm), 1 / rate)[int(np.argmax(spectrum[1:])) + 1]
    return int(round(hz / 10) * 10)


class _Utterance:
    def __init__(self, idx):
        self.idx = idx
        self.audio = bytearray()  # as received (wire encoding)
        self.partials = 0
        self.ended_at: float | None = None


class _Conn:
    def __init__(self, ws, number, query):
        self.ws = ws
        self.number = number
        self.headers = dict(ws.request.headers)
        self.query = query
        self.encoding = query.get("encoding", "linear16")
        self.rate = int(query.get("sample_rate", "16000"))
        self.endpointing = query.get("endpointing", "vad")
        self.language = query.get("language_code", "hi-IN")
        self.width = 2 if self.encoding == "linear16" else 1
        self.received: list[dict] = []  # every event (audio_input with "bytes" instead of "audio")
        self.audio = bytearray()  # every audio byte received
        self.utterances: list[_Utterance] = []
        self.open: _Utterance | None = None
        self.finals: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.opened_at = time.monotonic()
        self.closing = False
        self.dropped = False
        # Server VAD state.
        self.vad_buffer = bytearray()
        self.silent_secs = 0.0

    def seconds(self, nbytes: int) -> float:
        return nbytes / (self.rate * self.width)

    @property
    def audio_secs(self) -> float:
        return self.seconds(len(self.audio))

    def events(self, kind=None) -> list[dict]:
        return [e for e in self.received if kind is None or e.get("event") == kind]


class FakeMiraiSTT:
    def __init__(
        self,
        *,
        auth=None,
        refuse_status=None,
        rate_limit_attempts=(),
        refuse_attempts=(),
        capacity=None,
        slow_final_s=0.0,
        no_final=False,
        drop_after_s=None,
        session_limits=None,
        drain_after_s=None,
        drain_grace_s=0.5,
        fatal_error=None,
        partial_every_s=0.12,
        silent=False,
    ):
        self.auth = auth
        self.refuse_status = refuse_status
        self.rate_limit_attempts = set(rate_limit_attempts)
        self.refuse_attempts = set(refuse_attempts)
        self.capacity = capacity
        self.slow_final_s = slow_final_s
        self.no_final = no_final
        self.drop_after_s = drop_after_s
        self.session_limits = session_limits
        self.drain_after_s = drain_after_s
        self.drain_grace_s = drain_grace_s
        self.fatal_error = fatal_error
        self.partial_every_s = partial_every_s
        self.silent = silent
        self.attempts = 0
        self.handshakes: list[dict] = []
        self.conns: list[_Conn] = []
        self.active = 0
        self.finals_sent: list[tuple[int, str]] = []  # (conn, text)

    # ---- views for assertions ----

    def texts(self) -> list[str]:
        return [text for _, text in self.finals_sent]

    # ---- server ----

    @asynccontextmanager
    async def serve(self):
        async def process_request(connection, request):
            self.attempts += 1
            query = dict(parse_qsl(urlparse(request.path).query))
            self.handshakes.append({"path": request.path, "query": query, "headers": dict(request.headers)})
            if self.auth is not None and not self.auth(request.headers):
                return _refusal(401, "unauthorized", "invalid API key")
            if self.refuse_status:
                return _refusal(
                    self.refuse_status, "refused", "refused", retry_after=self.refuse_status == 429
                )
            if self.attempts in self.rate_limit_attempts:
                return _refusal(429, "rate_limited", "request rate limit exceeded", retry_after=True)
            if self.attempts in self.refuse_attempts:
                return _refusal(503, "unavailable", "restarting")
            unknown = sorted(set(query) - ALLOWED)
            if unknown:
                return _refusal(400, "invalid_config", f"unknown parameter {unknown[0]!r}")
            if query.get("sample_rate", "16000") not in ("8000", "16000"):
                return _refusal(400, "invalid_config", "sample_rate must be 8000 or 16000")
            if self.capacity is not None and self.active >= self.capacity:
                return _refusal(429, "at_capacity", "STT is at capacity", retry_after=True)
            return None

        async with serve(self._handler, "127.0.0.1", 0, process_request=process_request) as server:
            port = server.sockets[0].getsockname()[1]
            yield f"ws://127.0.0.1:{port}{PATH}"

    async def _send(self, conn: _Conn, event: dict):
        conn.sent.append(event)
        await conn.ws.send(json.dumps(event, ensure_ascii=False))

    async def _handler(self, ws):
        query = dict(parse_qsl(urlparse(ws.request.path).query))
        conn = _Conn(ws, len(self.conns) + 1, query)
        self.conns.append(conn)
        self.active += 1
        tasks = []
        try:
            if self.silent:
                await ws.wait_closed()
                return
            limits = self.session_limits or {"max_session_secs": 3600, "idle_timeout_secs": 120}
            await self._send(
                conn,
                {
                    "event": "session.begin",
                    "request_id": f"req_{conn.number}",
                    "session_id": f"sttws_{conn.number}",
                    "config": dict(query),
                    "capabilities": {"partials": True},
                    "limits": limits,
                },
            )
            tasks.append(asyncio.create_task(self._finalizer(conn)))
            if self.session_limits and self.session_limits.get("max_session_secs"):
                tasks.append(
                    asyncio.create_task(self._drain_later(conn, self.session_limits["max_session_secs"]))
                )
            if self.drain_after_s is not None and conn.number == 1:
                tasks.append(
                    asyncio.create_task(self._drain_later(conn, self.drain_after_s, self.drain_grace_s))
                )
            async for raw in ws:
                event = json.loads(raw)
                await self._on_message(conn, event)
        except Exception:
            pass
        finally:
            self.active -= 1
            for task in tasks:
                task.cancel()

    async def _on_message(self, conn: _Conn, event: dict):
        if conn.dropped:
            return
        kind = event.get("event")
        if kind == "audio_input":
            data = base64.b64decode(event.get("audio") or "")
            conn.received.append({"event": "audio_input", "bytes": len(data)})
            await self._on_audio(conn, data)
            return
        conn.received.append(event)
        if kind == "speech_start":
            if conn.endpointing == "manual" and conn.open is None:
                self._open(conn)
        elif kind == "speech_end":
            if conn.endpointing == "manual" and conn.open is not None:
                self._close(conn)
        elif kind == "flush":
            if conn.open is not None:
                self._close(conn)
        elif kind == "config.update":
            fields = {k: v for k, v in event.items() if k != "event"}
            conn.language = fields.get("language_code", conn.language)
            await self._send(conn, {"event": "config.updated", "config": fields})
        elif kind == "ping":
            await self._send(conn, {"event": "pong"})
        elif kind == "end":
            await self._finish(conn, "client_end")

    async def _on_audio(self, conn: _Conn, data: bytes):
        conn.audio += data
        if conn.endpointing == "vad":
            await self._server_vad(conn, data)
        elif conn.open is not None:
            conn.open.audio += data
            await self._maybe_partial(conn)
        if self.drop_after_s and conn.number == 1 and conn.audio_secs >= self.drop_after_s:
            conn.dropped = True
            conn.ws.transport.abort()  # no close frame: a network drop

    async def _server_vad(self, conn: _Conn, data: bytes):
        conn.vad_buffer += data
        frame = int(conn.rate * VAD_FRAME_SECS) * conn.width
        silence_ms = float(conn.query.get("silence_duration_ms", 300))
        while len(conn.vad_buffer) >= frame:
            chunk = bytes(conn.vad_buffer[:frame])
            del conn.vad_buffer[:frame]
            pcm = decode(chunk, conn.encoding).astype(np.float64)
            loud = float(np.sqrt(np.mean(pcm**2))) > VAD_RMS
            if conn.open is None:
                if loud:
                    self._open(conn)
                    conn.open.audio += chunk
                    conn.silent_secs = 0.0
                    await self._send(conn, {"event": "vad.speech_start"})
                continue
            conn.open.audio += chunk
            conn.silent_secs = 0.0 if loud else conn.silent_secs + VAD_FRAME_SECS
            await self._maybe_partial(conn)
            if conn.silent_secs * 1000 >= silence_ms:
                await self._send(conn, {"event": "vad.speech_end"})
                self._close(conn)

    def _open(self, conn: _Conn):
        conn.open = _Utterance(len(conn.utterances))
        conn.utterances.append(conn.open)

    def _close(self, conn: _Conn):
        utt, conn.open = conn.open, None
        utt.ended_at = time.monotonic()
        conn.finals.put_nowait(utt)

    def text_for(self, conn: _Conn, audio: bytes) -> str:
        pcm = decode(bytes(audio), conn.encoding)
        hz = dominant_hz(pcm, conn.rate)
        if not hz:
            return ""
        return f"{hz} hertz, {len(pcm) / conn.rate:.1f} seconds"

    async def _maybe_partial(self, conn: _Conn):
        utt = conn.open
        if utt is None or not self.partial_every_s:
            return
        due = int(conn.seconds(len(utt.audio)) / self.partial_every_s)
        if due > utt.partials:
            utt.partials = due
            text = self.text_for(conn, utt.audio)
            if text:
                await self._send(
                    conn, {"event": "transcript.partial", "utterance_idx": utt.idx, "text": text}
                )

    async def _finalizer(self, conn: _Conn):
        """One final per utterance, in order, slow_final_s after it ended."""
        while True:
            utt = await conn.finals.get()
            try:
                wait = utt.ended_at + self.slow_final_s - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                if self.fatal_error is not None:
                    error = {"is_fatal": True, "message": "", **self.fatal_error}
                    await self._send(conn, {"event": "error", "error": error})
                    await conn.ws.close(1008, error.get("code", "error"))
                    return
                if self.no_final:
                    continue
                text = self.text_for(conn, utt.audio)
                self.finals_sent.append((conn.number, text))
                await self._send(
                    conn,
                    {
                        "event": "transcript.final",
                        "utterance_idx": utt.idx,
                        "text": text,
                        "language_code": conn.language,
                    },
                )
            finally:
                conn.finals.task_done()

    async def _drain_later(self, conn: _Conn, after: float, grace: float = 0.0):
        await asyncio.sleep(after)
        await self._send(conn, {"event": "session.draining", "reason": "max_duration"})
        if grace:
            await asyncio.sleep(grace)
        await self._finish(conn, "max_duration")

    async def _finish(self, conn: _Conn, reason: str):
        if conn.closing:
            return
        conn.closing = True
        if conn.open is not None:
            self._close(conn)
        try:
            await asyncio.wait_for(conn.finals.join(), 5.0)
        except TimeoutError:
            pass
        secs = conn.audio_secs
        await self._send(
            conn,
            {
                "event": "session.end",
                "request_id": f"req_{conn.number}",
                "reason": reason,
                "total_duration_s": round(time.monotonic() - conn.opened_at, 3),
                "total_utterances": len(conn.utterances),
                "audio_duration_s": round(secs, 3),
                "audio_seconds_billed": int(np.ceil(secs)),
                "cost_paise": 1,
            },
        )
        await conn.ws.close()


def _refusal(status: int, code: str, message: str, *, retry_after: bool = False) -> Response:
    body = json.dumps({"error": {"code": code, "message": message}}).encode()
    headers = Headers([("Content-Type", "application/json"), ("Content-Length", str(len(body)))])
    if retry_after:
        headers["Retry-After"] = "0.1"
    return Response(status, "Refused", headers, body)
