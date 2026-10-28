"""Instant voices over the REST API: what CustomVoices sends, and how Mirai's answers come back."""

import json
import wave
from io import BytesIO

import httpx
import pytest
from fourier_frequency import CustomVoice, CustomVoices, FrequencyError
from fourier_frequency.voices import voice_language

KEY = "sk_test"
BASE = "https://mirai.test/v1"
VOICE = {
    "id": "cv_01m608ea5ewtee7gaz9dvzfe2z",
    "object": "custom_voice",
    "kind": "instant",
    "name": "Asha",
    "language": "hi",
    "accent": None,
    "gender": "female",
    "status": "ready",
    "created_at": "2026-10-28T03:21:51Z",
    "updated_at": "2026-10-28T03:21:51Z",
    "clip": {"filename": "asha.wav", "bytes": 960044, "seconds": 30.0},
    "consent": {"name": "Asha Mehta", "relationship": "employee", "attested_at": "2026-10-28T03:21:43Z"},
}


def wav_bytes(seconds: float = 1.0, rate: int = 16000) -> bytes:
    out = BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(bytes(int(seconds * rate) * 2))
    return out.getvalue()


class Recorder:
    """An httpx transport that records requests and answers from a script."""

    def __init__(self, *answers: httpx.Response):
        self.answers = list(answers)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        request.read()
        self.requests.append(request)
        return self.answers.pop(0)


def client(*answers: httpx.Response) -> tuple[CustomVoices, Recorder]:
    recorder = Recorder(*answers)
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder))
    return CustomVoices(api_key=KEY, base_url=BASE, http_client=http), recorder


def refusal(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(
        status, json={"error": {"code": code, "message": message}}, headers={"x-request-id": "req_9"}
    )


async def test_clone_sends_the_recording_and_the_consent(tmp_path):
    clip = tmp_path / "asha.wav"
    clip.write_bytes(wav_bytes(12))
    voices, seen = client(httpx.Response(201, json=VOICE))
    voice = await voices.clone(
        clip,
        name="Asha",
        language="hi-IN",
        consent_name="Asha Mehta",
        consent_relationship="Employee",
        consent_attested=True,
        gender="female",
    )
    assert voice == CustomVoice.from_api(VOICE)
    assert voice.id.startswith("cv_") and voice.clip_seconds == 30.0 and voice.ready
    assert voice.consent_relationship == "employee" and voice.raw["object"] == "custom_voice"
    (request,) = seen.requests
    assert request.method == "POST" and str(request.url) == "https://mirai.test/v2/custom-voices"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    body = request.content
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="clip"; filename="asha.wav"' in body and b"Content-Type: audio/wav" in body
    for field, value in (
        ("name", "Asha"),
        ("language", "hi"),
        ("consent_name", "Asha Mehta"),
        ("consent_relationship", "employee"),
        ("consent_attested", "true"),
        ("gender", "female"),
    ):
        assert f'name="{field}"\r\n\r\n{value}\r\n'.encode() in body
    assert b'name="accent"' not in body


async def test_clone_takes_bytes_with_a_file_name():
    voices, seen = client(httpx.Response(201, json=VOICE))
    await voices.clone(
        wav_bytes(),
        filename="take-2.mp3",
        name="Asha",
        language="en",
        consent_name="Asha",
        consent_relationship="self",
        consent_attested=True,
        accent="indian",
    )
    body = seen.requests[0].content
    assert b'filename="take-2.mp3"' in body and b"Content-Type: audio/mpeg" in body
    assert b'name="accent"\r\n\r\nindian\r\n' in body


@pytest.mark.parametrize(
    "change, message",
    [
        ({"consent_attested": False}, "consent_attested must be True"),
        ({"consent_attested": "true"}, "consent_attested must be True"),
        ({"consent_relationship": "friend"}, "consent_relationship must be one of"),
        ({"consent_name": " "}, "consent_name"),
        ({"language": "fr"}, "language must be one of"),
        ({"gender": "other"}, "gender must be"),
        ({"filename": None}, "pass filename="),
        ({"filename": "clip.m4a"}, "the recording must be one of"),
    ],
)
async def test_clone_checks_what_it_can_before_sending(change, message):
    voices, seen = client()
    args = dict(
        filename="clip.wav",
        name="Asha",
        language="hi",
        consent_name="Asha",
        consent_relationship="self",
        consent_attested=True,
    )
    args.update(change)
    with pytest.raises(ValueError, match=message):
        await voices.clone(wav_bytes(), **args)
    assert not seen.requests


async def test_a_rejected_recording_is_a_permanent_invalid_request():
    said = "This recording is 2.0 seconds long. Upload at least 10 seconds of the speaker talking."
    voices, _ = client(refusal(422, "clip_rejected", said))
    with pytest.raises(FrequencyError) as refused:
        await voices.clone(
            wav_bytes(2),
            filename="short.wav",
            name="Asha",
            language="hi",
            consent_name="Asha",
            consent_relationship="self",
            consent_attested=True,
        )
    error = refused.value
    assert (error.status, error.code, error.category, error.permanent) == (
        422,
        "clip_rejected",
        "invalid_request",
        True,
    )
    assert error.request_id == "req_9"
    assert str(error) == f"Mirai voices returned HTTP 422 clip_rejected: {said} [request req_9]"


@pytest.mark.parametrize(
    "status, code, category, permanent",
    [
        (409, "limit_reached", "quota", True),
        (409, "voice_in_use", "invalid_request", True),
        (503, "instant_voices_unavailable", "server", False),
        (502, "upstream_error", "server", False),
        (429, "at_capacity", "rate_limit", False),
        (401, "unauthorized", "authentication", True),
        (404, "not_found", "invalid_request", True),
    ],
)
async def test_refusals_keep_mirais_code(status, code, category, permanent):
    voices, _ = client(refusal(status, code, "no"))
    with pytest.raises(FrequencyError) as refused:
        await voices.delete(VOICE["id"])
    error = refused.value
    assert (error.status, error.code, error.category, error.permanent) == (status, code, category, permanent)


async def test_list_get_rename_delete_and_preview():
    sample = wav_bytes(1.5)
    voices, seen = client(
        httpx.Response(200, json={"object": "list", "data": [VOICE]}),
        httpx.Response(200, json=VOICE),
        httpx.Response(200, json={**VOICE, "name": "Asha 2"}),
        httpx.Response(204),
        httpx.Response(200, content=sample, headers={"content-type": "audio/wav"}),
    )
    async with voices:
        (listed,) = await voices.list()
        got = await voices.get(VOICE["id"])
        renamed = await voices.rename(VOICE["id"], name="Asha 2")
        await voices.delete(VOICE["id"])
        audio = await voices.preview(VOICE["id"], "नमस्ते")
    assert listed == got and renamed.name == "Asha 2" and audio == sample
    voice_url = f"https://mirai.test/v2/custom-voices/{VOICE['id']}"
    assert [(r.method, str(r.url)) for r in seen.requests] == [
        ("GET", "https://mirai.test/v2/custom-voices"),
        ("GET", voice_url),
        ("PATCH", voice_url),
        ("DELETE", voice_url),
        ("POST", voice_url + "/preview"),
    ]
    assert json.loads(seen.requests[2].content) == {"name": "Asha 2"}
    assert json.loads(seen.requests[4].content) == {"text": "नमस्ते"}


async def test_a_voice_id_must_be_a_custom_voice():
    voices, seen = client()
    with pytest.raises(ValueError, match="starts with 'cv_'"):
        await voices.get("shruti")
    with pytest.raises(ValueError, match="new name"):
        await voices.rename(VOICE["id"])
    assert not seen.requests


async def test_an_unreachable_api_is_a_connectivity_error():
    def down(request):
        raise httpx.ConnectError("refused", request=request)

    http = httpx.AsyncClient(transport=httpx.MockTransport(down))
    voices = CustomVoices(api_key=KEY, base_url=BASE, http_client=http)
    with pytest.raises(FrequencyError) as refused:
        await voices.list()
    assert refused.value.category == "connectivity" and not refused.value.permanent


def test_languages_are_base_codes():
    assert (
        voice_language("hi-IN") == "hi" and voice_language("EN_au") == "en" and voice_language("pa") == "pa"
    )
    with pytest.raises(ValueError):
        voice_language("ja")


def test_the_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MIRAI_API_KEY", "sk_env")
    voices = CustomVoices(base_url="https://mirai.test/v1/")
    assert voices._headers == {"Authorization": "Bearer sk_env"}
    assert voices._url == "https://mirai.test/v2/custom-voices"
