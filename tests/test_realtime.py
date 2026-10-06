import asyncio
import base64
import json
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from pipecat.frames.frames import (
    ErrorFrame,
    InputAudioRawFrame,
    LLMContextFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.realtime import events
from pipecat.tests.utils import SleepFrame, run_test
from websockets.asyncio.server import serve

from pipecat_mirai import (
    PRODUCTION_REALTIME_URL,
    SANDBOX_REALTIME_URL,
    MiraiRealtimeLLMService,
    MiraiTurnMetrics,
)


def query(service):
    parts = urlsplit(service.base_url)
    return parts, {k: v[0] for k, v in parse_qs(parts.query).items()}


def test_url_carries_session_parameters():
    svc = MiraiRealtimeLLMService(
        api_key="sk_test",
        base_url=PRODUCTION_REALTIME_URL,
        agent_id="agt_123",
        variables={"name": "राहुल", "due": 2500},
        metadata={"crm_id": "c-9"},
        webhook_url="https://example.test/hook",
        max_duration_secs=600,
    )
    parts, q = query(svc)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == PRODUCTION_REALTIME_URL
    assert q["model"] == "mira-realtime"
    assert q["mirai_events"] == "1"
    assert q["agent_id"] == "agt_123"
    assert json.loads(q["variables"]) == {"name": "राहुल", "due": 2500}
    assert json.loads(q["metadata"]) == {"crm_id": "c-9"}
    assert q["webhook_url"] == "https://example.test/hook"
    assert q["max_duration_secs"] == "600"
    assert "?" not in parts.query  # never Pipecat's "?model=" glued onto another value


def test_defaults_to_sandbox_and_neha():
    svc = MiraiRealtimeLLMService(api_key="sk_test")
    parts, q = query(svc)
    assert f"{parts.scheme}://{parts.netloc}{parts.path}" == SANDBOX_REALTIME_URL
    assert set(q) == {"model", "mirai_events"}
    assert svc._settings.session_properties.audio.output.voice == "neha"


def test_agent_voice_is_kept_when_no_voice_given():
    svc = MiraiRealtimeLLMService(api_key="sk_test", agent_id="agt_1")
    assert svc._settings.session_properties.audio.output is None


@pytest.mark.parametrize(
    "kwargs",
    [{"max_duration_secs": 10}, {"max_duration_secs": 4000}, {"mirai": {"temprature": 0.3}}],
)
def test_bad_settings_fail_before_connecting(kwargs):
    with pytest.raises(ValueError):
        MiraiRealtimeLLMService(api_key="sk_test", **kwargs)


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("MIRAI_API_KEY", raising=False)
    monkeypatch.delenv("MIRA_API_KEY", raising=False)
    with pytest.raises(ValueError):
        MiraiRealtimeLLMService()


async def test_session_update_carries_the_mirai_block():
    svc = MiraiRealtimeLLMService(
        api_key="sk_test", voice="shruti", language="hi", mirai={"temperature": 0.3, "tool_timeout_ms": 15000}
    )
    sent = []

    async def capture(message):
        sent.append(message)

    svc._ws_send = capture
    await svc.send_client_event(events.SessionUpdateEvent(session=svc._settings.session_properties))
    session = sent[0]["session"]
    assert session["mirai"] == {
        "temperature": 0.3,
        "tool_timeout_ms": 15000,
        "language": "hi",
        "events": True,
    }
    assert session["audio"]["output"]["voice"] == "shruti"


async def test_caller_audio_is_resampled_to_24k():
    svc = MiraiRealtimeLLMService(api_key="sk_test")
    sent = []

    async def capture(event):
        sent.append(event)

    svc.send_client_event = capture
    tone = (np.sin(np.arange(800) * 2 * np.pi * 300 / 8000) * 8000).astype("<i2").tobytes()  # 100 ms at 8 kHz
    for _ in range(10):
        await svc._send_user_audio(InputAudioRawFrame(audio=tone, sample_rate=8000, num_channels=1))
    samples = sum(len(base64.b64decode(e.audio)) for e in sent) // 2
    assert abs(samples - 24000) < 24000 * 0.05  # one second of 8 kHz in, ~one second of 24 kHz out


class FakeSocket:
    def __init__(self, messages):
        self.messages = messages

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for m in self.messages:
            yield m


async def test_mirai_events_are_handled_and_never_reach_pipecat():
    from pipecat_mirai.realtime import _MiraiEventTap

    svc = MiraiRealtimeLLMService(api_key="sk_test")
    turns, applied = [], []

    @svc.event_handler("on_turn_metrics")
    async def on_turn(service, metrics):
        turns.append(metrics)

    @svc.event_handler("on_session_applied")
    async def on_applied(service, session, changed, ignored):
        applied.append((session, changed, ignored))

    msgs = [
        {"type": "mirai.session.applied", "session": {"voice": "neha"}, "changed": ["*"], "ignored": []},
        {
            "type": "mirai.turn.metrics",
            "response_id": "resp_1",
            "v2v_ms": 742.6,
            "stt_ms": 210.0,
            "llm_ms": 95.0,
            "tts_ms": None,
            "providers": {"stt": "sarvam_realtime", "llm": "mira", "tts": "mira"},
            "fallback": True,
            "failovers": [],
        },
        {"type": "response.done", "response": {"id": "resp_1", "metadata": {"mirai_v2v_ms": "742.6"}}},
        {
            "type": "response.done",
            "response": {"id": "resp_2", "metadata": {"mirai_v2v_ms": "900", "mirai_fallback": "false"}},
        },
        {"type": "mirai.something.new"},
    ]
    tap = _MiraiEventTap(FakeSocket([json.dumps(m) for m in msgs] + ["not json"]), svc)
    passed = [m async for m in tap]
    await asyncio.sleep(0.05)  # event handlers run as tasks

    assert all(not m.startswith('{"type": "mirai.') for m in passed)
    assert len(passed) == 3  # the two response.done events and the non-JSON frame
    assert [t.response_id for t in turns] == ["resp_1", "resp_2"]  # resp_1 reported once, from Mirai's event
    assert turns[0] == MiraiTurnMetrics(
        response_id="resp_1",
        v2v_ms=742.6,
        stt_ms=210.0,
        llm_ms=95.0,
        tts_ms=None,
        providers={"stt": "sarvam_realtime", "llm": "mira", "tts": "mira"},
        fallback=True,
    )
    assert turns[1].v2v_ms == 900 and turns[1].fallback is False
    assert applied == [({"voice": "neha"}, ["*"], [])]


# --- end to end against a fake Realtime server -------------------------------------


def _usage():
    details = {"cached_tokens": 0, "text_tokens": 0, "audio_tokens": 0}
    return {
        "total_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "input_token_details": details,
        "output_token_details": details,
    }


async def test_full_session_against_a_fake_mirai_server():
    seen = {"auth": None, "query": None, "updates": [], "creates": 0}
    reply = (np.sin(np.arange(4800) * 2 * np.pi * 220 / 24000) * 8000).astype("<i2").tobytes()

    async def handler(ws):
        seen["auth"] = ws.request.headers.get("Authorization")
        seen["query"] = ws.request.path
        n = iter(range(1, 1000))

        async def send(**evt):
            evt.setdefault("event_id", f"evt_{next(n)}")
            await ws.send(json.dumps(evt))

        await send(type="session.created", session={})
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "session.update":
                seen["updates"].append(msg["session"])
                await send(type="session.updated", session={})
                await send(
                    type="mirai.session.applied", session={"voice": "shruti"}, changed=["*"], ignored=[]
                )
            elif msg["type"] == "response.create":
                seen["creates"] += 1
                response = {
                    "id": "resp_1",
                    "object": "realtime.response",
                    "status": "in_progress",
                    "status_details": None,
                    "output": [],
                }
                await send(type="response.created", response=response)
                await send(
                    type="response.output_audio.delta",
                    response_id="resp_1",
                    item_id="item_1",
                    output_index=0,
                    content_index=0,
                    delta=base64.b64encode(reply).decode(),
                )
                await send(
                    type="mirai.turn.metrics",
                    response_id="resp_1",
                    v2v_ms=650.0,
                    stt_ms=7.0,
                    llm_ms=45.0,
                    tts_ms=98.0,
                    providers={"stt": "mira", "llm": "mira", "tts": "mira"},
                    fallback=False,
                    failovers=[],
                )
                done = dict(response, status="completed", usage=_usage(), metadata={"mirai_v2v_ms": "650.0"})
                await send(type="response.done", response=done)

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        svc = MiraiRealtimeLLMService(
            api_key="sk_test", base_url=f"ws://127.0.0.1:{port}/v2/realtime", voice="shruti", language="hi"
        )
        turns = []

        @svc.event_handler("on_turn_metrics")
        async def on_turn(service, metrics):
            turns.append(metrics)

        down, up = await run_test(
            svc,
            frames_to_send=[
                LLMContextFrame(LLMContext([{"role": "user", "content": "नमस्ते बोलिए"}])),
                SleepFrame(sleep=1.5),
            ],
            pipeline_params=PipelineParams(audio_out_sample_rate=24000),
            start_timeout=10.0,
        )

    assert seen["auth"] == "Bearer sk_test"
    assert "mirai_events=1" in seen["query"] and "model=mira-realtime" in seen["query"]
    assert seen["updates"][0]["mirai"]["language"] == "hi"
    assert seen["updates"][0]["audio"]["output"]["voice"] == "shruti"
    assert seen["creates"] == 1
    audio = b"".join(f.audio for f in down if isinstance(f, TTSAudioRawFrame))
    assert audio == reply  # the session kept reading after both mirai.* events
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert [t.v2v_ms for t in turns] == [650.0]
