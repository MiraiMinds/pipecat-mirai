"""A fake of Mirai's streaming TTS socket (``/v1/audio/speech/stream``), for tests.

It implements the protocol from the server's side: session.ready on connect,
session.update, text (cut into sentences at ``. ? ! । ॥ …`` + whitespace, or a
newline, or a flush), flush (context.done once the context's sentences so far
are spoken), cancel (context.cancelled; nothing for that context after it) and
close. One sentence plays at a time: audio.start, binary frames, audio.done
(or error). The next sentence is synthesised while the current one streams.

Knobs: ``seconds`` (audio per sentence: a number, a dict from sentence to
seconds, or a callable), ``reads`` (sizes of the binary frames, cycled),
``pace`` (pause after each frame: 0 bursts as fast as the socket allows),
``first_gap`` (pause after a sentence's first frame),
``first_byte`` (synthesis time before a sentence's first frame; overlapped with
the sentence before it), ``honours_rate`` (False: always 48 kHz),
``capacity`` (sentence -> retry_after_secs: that sentence is refused once),
``always_refuse`` (sentence -> retry_after_secs: refused every time),
``fail`` (sentence -> error code, no retry_after), ``billing`` (sentence ->
chars_billed), ``cancel_delay`` (how long the playing sentence keeps streaming
after a cancel arrives), ``drop_after_bytes`` (abort the first connection's TCP
after sending that much audio), ``base64`` (send audio.chunk events instead of
binary frames) and ``close_after`` (send session.closed and
close the first connection that many seconds after it opens), ``refuse_status``
(answer every handshake with that HTTP status) and ``refuse_attempts``
(handshake attempts, counted from 1, answered 503).
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from contextlib import asynccontextmanager

import numpy as np
from websockets.asyncio.server import serve
from websockets.http11 import Response

SR = 48000
RATES = (8000, 16000, 24000, 48000)
SENTENCE_END = re.compile(r"[.?!।॥…][\"')\]]*\s|\n")


def tone(seconds: float, rate: int = SR, hz: float = 220.0) -> bytes:
    t = np.arange(int(round(seconds * rate))) / rate
    return (np.sin(2 * np.pi * hz * t) * 8000).astype("<i2").tobytes()


class _Ctx:
    def __init__(self, cid):
        self.id = cid
        self.buf = ""
        self.cut = 0  # sentences cut
        self.finished = 0  # sentences spoken or failed
        self.marks: list[int] = []


class _Conn:
    def __init__(self, ws, number):
        self.ws = ws
        self.number = number
        self.headers = dict(ws.request.headers)
        self.received: list[dict] = []
        self.rate = SR
        self.voice = "neha"
        self.contexts: dict[str, _Ctx] = {}
        self.queue: list[tuple[_Ctx, str, float]] = []  # (context, sentence, queued at)
        self.wake = asyncio.Event()
        self.playing: _Ctx | None = None
        self.cancel_at: dict[str, float] = {}
        self.audio_bytes = 0
        self.closing = False
        self.events: list[tuple[float, str, dict]] = []  # what the server sent (not audio)
        self.binary: list[tuple[float, int]] = []  # (when, bytes)


class FakeMiraiWS:
    def __init__(
        self,
        *,
        seconds=0.5,
        reads=(3200,),
        pace=0.0,
        first_gap=0.0,
        first_byte=0.0,
        honours_rate=True,
        capacity=None,
        always_refuse=None,
        fail=None,
        billing=None,
        cancel_delay=0.0,
        drop_after_bytes=None,
        close_after=None,
        refuse_status=None,
        refuse_attempts=(),
        base64=False,
    ):
        self.seconds = seconds
        self.reads = reads
        self.pace = pace
        self.first_gap = first_gap
        self.first_byte = first_byte
        self.honours_rate = honours_rate
        self.capacity = dict(capacity or {})
        self.always_refuse = dict(always_refuse or {})
        self.fail = dict(fail or {})
        self.billing = billing or (lambda s: len(s))
        self.cancel_delay = cancel_delay
        self.drop_after_bytes = drop_after_bytes
        self.close_after = close_after
        self.refuse_status = refuse_status
        self.refuse_attempts = set(refuse_attempts)  # handshake attempts (1-based) answered 503
        self.attempts = 0
        self.base64 = base64
        self.conns: list[_Conn] = []
        self.spoken: list[tuple[int, str, str]] = []  # (conn, context, sentence) fully sent
        self._requests = 0

    # ---- views for assertions ----

    def messages(self, kind=None, conn=None):
        out = []
        for c in self.conns:
            if conn is not None and c.number != conn:
                continue
            out += [m for m in c.received if kind is None or m.get("type") == kind]
        return out

    def seconds_for(self, sentence):
        s = self.seconds
        if callable(s):
            return s(sentence)
        if isinstance(s, dict):
            return s[sentence]
        return s

    # ---- server ----

    @asynccontextmanager
    async def serve(self):
        async def process_request(connection, request):
            self.attempts += 1
            if self.refuse_status:
                body = json.dumps({"error": {"code": "unauthorized", "message": "invalid API key"}})
                return Response(self.refuse_status, "Refused", _headers(len(body)), body.encode())
            if self.attempts in self.refuse_attempts:
                body = json.dumps({"error": {"code": "unavailable", "message": "restarting"}})
                return Response(503, "Unavailable", _headers(len(body)), body.encode())
            return None

        async with serve(self._handler, "127.0.0.1", 0, process_request=process_request) as server:
            port = server.sockets[0].getsockname()[1]
            yield f"ws://127.0.0.1:{port}/v1/audio/speech/stream"

    async def _send(self, conn, event):
        conn.events.append((time.monotonic(), event["type"], event))
        await conn.ws.send(json.dumps(event, ensure_ascii=False))

    async def _handler(self, ws):
        conn = _Conn(ws, len(self.conns) + 1)
        self.conns.append(conn)
        await self._send(
            conn,
            {
                "type": "session.ready",
                "session_id": f"ttsws_{conn.number}",
                "model": "mira-tts",
                "voice": conn.voice,
                "response_format": "pcm",
                "sample_rate": conn.rate,
                "encoding": "pcm_s16le",
                "audio_transport": "binary",
                "limits": {"idle_timeout_secs": 120, "max_session_secs": 3600},
            },
        )
        speaker = asyncio.create_task(self._speaker(conn))
        closer = (
            asyncio.create_task(self._close_later(conn)) if self.close_after and conn.number == 1 else None
        )
        try:
            async for raw in ws:
                msg = json.loads(raw)
                conn.received.append(msg)
                await self._on_message(conn, msg)
        except Exception:
            pass
        finally:
            speaker.cancel()
            if closer:
                closer.cancel()

    async def _close_later(self, conn):
        await asyncio.sleep(self.close_after)
        await self._send(conn, {"type": "session.closed", "reason": "max_duration"})
        await conn.ws.close(1000, "max_duration")

    async def _on_message(self, conn, msg):
        kind = msg.get("type")
        if kind == "session.update":
            if "sample_rate" in msg:
                if msg["sample_rate"] not in RATES:
                    await self._send(
                        conn, {"type": "error", "code": "invalid_request", "message": "bad rate"}
                    )
                    return
                conn.rate = msg["sample_rate"]
            elif "response_format" in msg:
                conn.rate = SR
            conn.voice = msg.get("voice", conn.voice)
            await self._send(
                conn,
                {
                    "type": "session.updated",
                    "session_id": f"ttsws_{conn.number}",
                    "voice": conn.voice,
                    "response_format": "pcm",
                    "sample_rate": conn.rate,
                    "encoding": "pcm_s16le",
                },
            )
        elif kind == "text":
            ctx = conn.contexts.get(msg["context_id"])
            if ctx is None:
                ctx = conn.contexts[msg["context_id"]] = _Ctx(msg["context_id"])
            ctx.buf += msg.get("text", "")
            while m := SENTENCE_END.search(ctx.buf):
                self._enqueue(conn, ctx, ctx.buf[: m.end()])
                ctx.buf = ctx.buf[m.end() :]
            if msg.get("flush"):
                await self._flush(conn, ctx)
        elif kind == "flush":
            ctx = conn.contexts.get(msg["context_id"])
            if ctx is None:
                await self._send(conn, {"type": "context.done", "context_id": msg["context_id"]})
            else:
                await self._flush(conn, ctx)
        elif kind == "cancel":
            cid = msg["context_id"]
            conn.queue = [q for q in conn.queue if q[0].id != cid]
            conn.contexts.pop(cid, None)
            if conn.playing is not None and conn.playing.id == cid:
                conn.cancel_at[cid] = time.monotonic() + self.cancel_delay
            else:
                await self._send(conn, {"type": "context.cancelled", "context_id": cid})
        elif kind == "close":
            conn.closing = True
            for ctx in list(conn.contexts.values()):
                await self._flush(conn, ctx)
            conn.wake.set()

    def _enqueue(self, conn, ctx, text):
        text = text.strip()
        if not text:
            return
        ctx.cut += 1
        conn.queue.append((ctx, text, time.monotonic()))
        conn.wake.set()

    async def _flush(self, conn, ctx):
        self._enqueue(conn, ctx, ctx.buf)
        ctx.buf = ""
        ctx.marks.append(ctx.cut)
        await self._settle(conn, ctx)

    async def _settle(self, conn, ctx):
        while ctx.marks and ctx.finished >= ctx.marks[0]:
            ctx.marks.pop(0)
            await self._send(conn, {"type": "context.done", "context_id": ctx.id})
        if ctx.finished == ctx.cut and not ctx.buf and not ctx.marks:
            if conn.contexts.get(ctx.id) is ctx:
                del conn.contexts[ctx.id]

    async def _speaker(self, conn):
        previous_start = 0.0
        while True:
            if not conn.queue:
                if conn.closing:
                    await self._send(conn, {"type": "session.closed", "reason": "client_close"})
                    await conn.ws.close()
                    return
                conn.wake.clear()
                await conn.wake.wait()
                continue
            ctx, text, queued_at = conn.queue.pop(0)
            conn.playing = ctx
            # Pipelined: synthesis of this sentence started when the one
            # before it started playing (or when it was queued, if later).
            ready = max(queued_at, previous_start) + self.first_byte
            if (delay := ready - time.monotonic()) > 0:
                await asyncio.sleep(delay)
            previous_start = time.monotonic()
            try:
                await self._speak(conn, ctx, text)
            finally:
                conn.playing = None

    async def _speak(self, conn, ctx, text):
        self._requests += 1
        rid = f"tts_{self._requests}"
        if ctx.id in conn.cancel_at:  # cancelled before it started
            del conn.cancel_at[ctx.id]
            await self._send(conn, {"type": "context.cancelled", "context_id": ctx.id})
            return
        refuse = self.always_refuse.get(text) or self.capacity.pop(text, None)
        if refuse is not None or text in self.fail:
            event = {"type": "error", "context_id": ctx.id, "request_id": rid}
            if refuse is not None:
                event |= {"code": "at_capacity", "message": "TTS is at capacity", "retry_after_secs": refuse}
            else:
                event |= {"code": self.fail[text], "message": "the speech node failed"}
            await self._send(conn, event)
            ctx.finished += 1
            await self._settle(conn, ctx)
            return
        rate = conn.rate if self.honours_rate else SR
        await self._send(
            conn,
            {
                "type": "audio.start",
                "context_id": ctx.id,
                "request_id": rid,
                "text": text,
                "response_format": "pcm",
                "sample_rate": rate,
                "encoding": "pcm_s16le",
            },
        )
        audio = tone(self.seconds_for(text), rate)
        pos, i = 0, 0
        while pos < len(audio):
            at = conn.cancel_at.get(ctx.id)
            if at is not None and time.monotonic() >= at:
                del conn.cancel_at[ctx.id]
                await self._send(conn, {"type": "context.cancelled", "context_id": ctx.id})
                return
            n = self.reads[i % len(self.reads)]
            piece = audio[pos : pos + n]
            conn.binary.append((time.monotonic(), len(piece)))
            if self.base64:
                chunk = {"type": "audio.chunk", "context_id": ctx.id, "request_id": rid, "data": b64(piece)}
                await conn.ws.send(json.dumps(chunk))
            else:
                await conn.ws.send(piece)
            conn.audio_bytes += len(piece)
            pos += n
            i += 1
            if self.drop_after_bytes and conn.number == 1 and conn.audio_bytes >= self.drop_after_bytes:
                conn.ws.transport.abort()  # no close frame: a network drop
                await asyncio.sleep(3600)
            await asyncio.sleep(self.first_gap if i == 1 and self.first_gap else self.pace)
        at = conn.cancel_at.pop(ctx.id, None)
        if at is not None:
            await self._send(conn, {"type": "context.cancelled", "context_id": ctx.id})
            return
        self.spoken.append((conn.number, ctx.id, text))
        await self._send(
            conn,
            {
                "type": "audio.done",
                "context_id": ctx.id,
                "request_id": rid,
                "chars_billed": self.billing(text),
                "cost_paise": 1,
                "bytes": len(audio),
                "first_byte_ms": 100,
            },
        )
        ctx.finished += 1
        await self._settle(conn, ctx)


def _headers(length):
    from websockets.datastructures import Headers

    return Headers([("Content-Type", "application/json"), ("Content-Length", str(length))])


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()
