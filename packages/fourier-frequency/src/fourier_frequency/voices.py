#
# Copyright (c) 2026, Sona Labs Pvt Ltd
#
# SPDX-License-Identifier: BSD-2-Clause
#

"""Your own voices: instant voices made from a short recording of one person.

Mirai makes an instant voice from 10 to 60 seconds of one consenting speaker,
in about a minute. Its id (``cv_...``) then works wherever a built-in voice
does: ``TTSStream(voice="cv_...")``, ``HttpTTS(voice="cv_...")``, and an
agent's ``voice.voice_id`` on phone and web calls::

    from fourier_frequency import CustomVoices, TTSStream

    async with CustomVoices() as voices:
        voice = await voices.clone(
            "asha.wav",
            name="Asha",
            language="hi",
            consent_name="Asha Mehta",
            consent_relationship="employee",
            consent_attested=True,
        )
    async with TTSStream(voice=voice.id) as tts:
        ...

Only a recording of a person who agreed to a synthetic version of their voice
being made may be used; Mirai keeps the recording as the consent record. A
workspace holds up to 10 voices. Everything here is one request to Mirai's
API (``/v2/custom-voices``) with the workspace's API key; a refusal raises
:class:`~fourier_frequency.FrequencyError` with Mirai's code (``clip_rejected``,
``limit_reached``, ``voice_in_use``, ...) and its message, which says what to
change.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import httpx

from fourier_frequency import errors
from fourier_frequency.errors import FrequencyError
from fourier_frequency.tts import CUSTOM_VOICE_PREFIX, DEFAULT_BASE_URL, api_key_from_env, is_custom_voice

# The languages an instant voice can be made in (the voice's base language code).
CUSTOM_VOICE_LANGUAGES = ("en", "hi", "gu", "ta", "te", "bn", "kn", "ml", "mr", "pa")
# How the person attesting consent knows the speaker.
CONSENT_RELATIONSHIPS = ("self", "employee", "contractor", "customer", "other")
GENDERS = ("female", "male")
# The recording: at most this many bytes, in one of these containers (by file extension).
MAX_CLIP_BYTES = 16 * 1024 * 1024
CLIP_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".webm": "audio/webm",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
}
# Making a voice takes about a minute; everything else is quick.
CLONE_TIMEOUT_SECS = 180.0
TIMEOUT_SECS = 30.0

# Mirai's codes for this API, as error categories (beyond the HTTP status's own).
_CODE_CATEGORIES = {
    "limit_reached": errors.QUOTA,
    "instant_voices_unavailable": errors.SERVER,
    "upstream_error": errors.SERVER,
    "at_capacity": errors.RATE_LIMIT,
    "rate_limited": errors.RATE_LIMIT,
}
# Worth trying again unchanged, a little later.
_TRANSIENT_CODES = {"instant_voices_unavailable", "upstream_error", "at_capacity", "rate_limited"}


@dataclass(frozen=True)
class CustomVoice:
    """One of the workspace's own voices, as Mirai describes it.

    ``id`` is what ``voice=`` takes. ``clip_seconds`` is the recording's
    length when Mirai could measure it (a WAV). ``raw`` is Mirai's whole
    object, for fields added after this version.
    """

    id: str
    name: str
    language: str
    kind: str = "instant"
    status: str = "ready"
    accent: str | None = None
    gender: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    clip_filename: str | None = None
    clip_bytes: int | None = None
    clip_seconds: float | None = None
    consent_name: str | None = None
    consent_relationship: str | None = None
    consent_attested_at: str | None = None
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_api(cls, body: dict) -> CustomVoice:
        clip = body.get("clip") or {}
        consent = body.get("consent") or {}
        return cls(
            id=body["id"],
            name=body.get("name") or "",
            language=body.get("language") or "",
            kind=body.get("kind") or "instant",
            status=body.get("status") or "ready",
            accent=body.get("accent"),
            gender=body.get("gender"),
            created_at=body.get("created_at"),
            updated_at=body.get("updated_at"),
            clip_filename=clip.get("filename"),
            clip_bytes=clip.get("bytes"),
            clip_seconds=clip.get("seconds"),
            consent_name=consent.get("name"),
            consent_relationship=consent.get("relationship"),
            consent_attested_at=consent.get("attested_at"),
            raw=dict(body),
        )

    @property
    def ready(self) -> bool:
        return self.status == "ready"


Clip = str | os.PathLike | bytes | bytearray | IO[bytes]


class CustomVoices:
    """The workspace's own voices: make one from a recording, list, rename, preview, delete.

    Args:
        api_key: Mirai API key. Defaults to ``MIRAI_API_KEY`` (or
            ``MIRA_API_KEY``), as :class:`~fourier_frequency.TTSStream` does.
        base_url: The API base URL including ``/v1`` (the same value
            ``HttpTTS`` and ``TTSStream`` take); the voices live at
            ``<host>/v2/custom-voices``.
        http_client: An ``httpx.AsyncClient`` of your own; by default one is
            made and closed with this object.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
    ):
        key = api_key_from_env(api_key, "CustomVoices")
        self._headers = {"Authorization": f"Bearer {key}"}
        self._url = re.sub(r"/v1/?$", "", base_url.rstrip("/")) + "/v2/custom-voices"
        self._own_client = http_client is None
        self._client = http_client or httpx.AsyncClient(timeout=TIMEOUT_SECS)

    async def __aenter__(self) -> CustomVoices:
        return self

    async def __aexit__(self, *exc_info):
        await self.aclose()

    async def aclose(self):
        if self._own_client:
            await self._client.aclose()

    async def clone(
        self,
        clip: Clip,
        *,
        name: str,
        language: str,
        consent_name: str,
        consent_relationship: str,
        consent_attested: bool,
        accent: str | None = None,
        gender: str | None = None,
        filename: str | None = None,
    ) -> CustomVoice:
        """Make an instant voice from ``clip`` and return it, ready to speak.

        Args:
            clip: The recording: a path, the file's bytes or a binary file.
                10 to 60 seconds of one person speaking naturally, with no
                music, background noise or echo (30 to 60 seconds keeps an
                accent best). WAV, MP3, FLAC, OGG, OGA, WEBM, MPEG or MPGA, at
                most 16 MB.
            name: What to call the voice (1 to 60 characters).
            language: The language it will speak, as a base code: one of
                :data:`CUSTOM_VOICE_LANGUAGES` (``"en-AU"`` is ``"en"``).
            consent_name: The name of the person speaking in the recording.
            consent_relationship: How you know them: one of
                :data:`CONSENT_RELATIONSHIPS`.
            consent_attested: ``True`` only if that person agreed to a
                synthetic version of their voice being made.
            accent: Optional, up to 40 characters (``"australian"``, ``"indian"``).
            gender: ``"female"`` or ``"male"``, optional: it picks the
                built-in voice a phone call falls back to.
            filename: The file name sent with bytes or a file object; its
                extension says the container (a path's own name is used).

        Raises :class:`FrequencyError`: ``clip_rejected`` (422, the message
        says why: too short, too long, unreadable, not one clear speaker),
        ``file_too_large`` (413), ``unsupported_file_type`` (415),
        ``limit_reached`` (409, the workspace has 10 voices),
        ``instant_voices_unavailable`` (503) or ``upstream_error`` (502, try
        again in a few minutes). ``ValueError`` for what can be checked here.
        """
        if consent_attested is not True:
            raise ValueError(
                "consent_attested must be True: only a recording of a person who agreed to a "
                "synthetic version of their voice being made may be used"
            )
        if not str(consent_name or "").strip():
            raise ValueError("consent_name is the name of the person speaking in the recording")
        relationship = str(consent_relationship or "").strip().lower()
        if relationship not in CONSENT_RELATIONSHIPS:
            raise ValueError(
                f"consent_relationship must be one of {', '.join(CONSENT_RELATIONSHIPS)}; "
                f"got {consent_relationship!r}"
            )
        base = voice_language(language)
        if gender is not None and str(gender).strip().lower() not in GENDERS:
            raise ValueError(f"gender must be 'female', 'male' or None; got {gender!r}")
        if not str(name or "").strip():
            raise ValueError("name must be 1 to 60 characters")
        data, filename = _read_clip(clip, filename)
        fields = {
            "name": name,
            "language": base,
            "consent_name": consent_name,
            "consent_relationship": relationship,
            "consent_attested": "true",
        }
        if accent:
            fields["accent"] = accent
        if gender:
            fields["gender"] = str(gender).strip().lower()
        content_type = CLIP_TYPES[Path(filename).suffix.lower()]
        response = await self._request(
            "POST",
            self._url,
            data=fields,
            files={"clip": (filename, data, content_type)},
            timeout=CLONE_TIMEOUT_SECS,
        )
        return CustomVoice.from_api(response.json())

    async def list(self) -> list[CustomVoice]:
        """The workspace's voices, oldest first."""
        response = await self._request("GET", self._url)
        return [CustomVoice.from_api(item) for item in response.json().get("data") or []]

    async def get(self, voice_id: str) -> CustomVoice:
        """One voice by id; ``not_found`` (404) for an unknown or deleted one."""
        response = await self._request("GET", self._voice_url(voice_id))
        return CustomVoice.from_api(response.json())

    async def rename(
        self, voice_id: str, *, name: str | None = None, gender: str | None = None
    ) -> CustomVoice:
        """Change a voice's name and/or gender (``gender=""`` clears it)."""
        body: dict[str, str] = {}
        if name is not None:
            body["name"] = name
        if gender is not None:
            if gender not in ("", *GENDERS):
                raise ValueError(f"gender must be 'female', 'male' or ''; got {gender!r}")
            body["gender"] = gender
        if not body:
            raise ValueError("give a new name, a gender, or both")
        response = await self._request("PATCH", self._voice_url(voice_id), json=body)
        return CustomVoice.from_api(response.json())

    async def delete(self, voice_id: str) -> None:
        """Delete a voice and its recording.

        ``voice_in_use`` (409) while an agent speaks with it: the error's
        message names the agents; move them to another voice first.
        """
        await self._request("DELETE", self._voice_url(voice_id))

    async def preview(self, voice_id: str, text: str | None = None) -> bytes:
        """A short sample of the voice as WAV bytes: ``text`` (up to 300 characters), or Mirai's own line."""
        body = {"text": text} if text is not None else {}
        response = await self._request("POST", self._voice_url(voice_id) + "/preview", json=body)
        return response.content

    def _voice_url(self, voice_id: str) -> str:
        if not is_custom_voice(voice_id):
            raise ValueError(f"a custom voice id starts with {CUSTOM_VOICE_PREFIX!r}; got {voice_id!r}")
        return f"{self._url}/{voice_id}"

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, url, headers=self._headers, **kwargs)
        except httpx.HTTPError as exc:
            raise FrequencyError(
                f"Mirai voices: could not reach {url}: {exc!r}", category=errors.CONNECTIVITY
            ) from exc
        if response.status_code >= 400:
            raise _refusal(response)
        return response


def voice_language(language: str) -> str:
    """``"hi-IN"`` -> ``"hi"``; ``ValueError`` for a language an instant voice can't be made in."""
    base = str(language or "").strip().lower().replace("_", "-").split("-", 1)[0]
    if base not in CUSTOM_VOICE_LANGUAGES:
        raise ValueError(f"language must be one of {', '.join(CUSTOM_VOICE_LANGUAGES)}; got {language!r}")
    return base


def _read_clip(clip: Clip, filename: str | None) -> tuple[bytes, str]:
    """The recording's bytes and a file name whose extension names its container."""
    if isinstance(clip, str | os.PathLike):
        path = Path(clip)
        if not path.is_file():
            raise ValueError(f"no recording at {path}")
        if path.stat().st_size > MAX_CLIP_BYTES:
            raise ValueError(f"{path} is larger than 16 MB; 10 to 60 seconds of speech is enough")
        data, filename = path.read_bytes(), filename or path.name
    elif isinstance(clip, bytes | bytearray):
        data = bytes(clip)
    else:
        data = clip.read()
        filename = filename or Path(getattr(clip, "name", "") or "").name or None
    if not filename:
        raise ValueError(
            "pass filename= (with its extension, such as clip.wav) for a recording given as bytes"
        )
    if Path(filename).suffix.lower() not in CLIP_TYPES:
        raise ValueError(f"the recording must be one of {', '.join(sorted(CLIP_TYPES))}; got {filename!r}")
    if not data:
        raise ValueError("the recording is empty")
    if len(data) > MAX_CLIP_BYTES:
        raise ValueError("the recording is larger than 16 MB; 10 to 60 seconds of speech is enough")
    return data, filename


def _refusal(response: httpx.Response) -> FrequencyError:
    """Mirai's refusal as a :class:`FrequencyError`: its code, its message, the request id."""
    code, message = "", None
    try:
        error = response.json().get("error") or {}
        if isinstance(error, dict):
            code = error.get("code") if isinstance(error.get("code"), str) else ""
            message = error.get("message") if isinstance(error.get("message"), str) else None
    except (ValueError, AttributeError):
        pass
    status = response.status_code
    request_id = response.headers.get("x-request-id")
    category = _CODE_CATEGORIES.get(code) or _status_category(status)
    said = f"HTTP {status}" + (f" {code}" if code else "") + (f": {message}" if message else "")
    rid = f" [request {request_id}]" if request_id else ""
    return FrequencyError(
        f"Mirai voices returned {said}{rid}",
        category=category,
        code=code or None,
        status=status,
        request_id=request_id,
        permanent=not (code in _TRANSIENT_CODES or status >= 500 or status == 429),
    )


def _status_category(status: int) -> str:
    if status == 401:
        return errors.AUTHENTICATION
    if status == 402:
        return errors.QUOTA
    if status == 403:
        return errors.AUTHORIZATION
    if status == 429:
        return errors.RATE_LIMIT
    if status >= 500:
        return errors.SERVER
    return errors.INVALID_REQUEST
