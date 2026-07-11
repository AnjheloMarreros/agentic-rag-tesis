from __future__ import annotations

import json
import mimetypes
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "").strip()
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3").strip()
DEEPGRAM_LANGUAGE = os.getenv("DEEPGRAM_LANGUAGE", "es").strip()
DEEPGRAM_SMART_FORMAT = os.getenv("DEEPGRAM_SMART_FORMAT", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEEPGRAM_PUNCTUATE = os.getenv("DEEPGRAM_PUNCTUATE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEEPGRAM_TIMEOUT_SECONDS = int(os.getenv("DEEPGRAM_TIMEOUT_SECONDS", "120"))
DEEPGRAM_DEBUG = os.getenv("DEEPGRAM_DEBUG", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

DEEPGRAM_ENDPOINT = "https://api.deepgram.com/v1/listen"


def _guess_mime_type(filename: str | None) -> str:
    if not filename:
        return "audio/m4a"
    mime, _ = mimetypes.guess_type(filename)
    return mime or "audio/m4a"


def _extract_transcript(payload: dict[str, Any]) -> str:
    results = payload.get("results")
    if not isinstance(results, dict):
        return ""

    channels = results.get("channels")
    if not isinstance(channels, list) or not channels:
        return ""

    first_channel = channels[0]
    if not isinstance(first_channel, dict):
        return ""

    alternatives = first_channel.get("alternatives")
    if not isinstance(alternatives, list) or not alternatives:
        return ""

    first_alt = alternatives[0]
    if not isinstance(first_alt, dict):
        return ""

    transcript = first_alt.get("transcript", "")
    return str(transcript).strip()


def transcribir_audio_deepgram(
    audio_bytes: bytes,
    content_type: str | None = None,
    filename: str | None = None,
) -> str:
    if not DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY no está configurado.")

    if not audio_bytes:
        raise ValueError("El archivo de audio está vacío.")

    query = urlencode(
        {
            "model": DEEPGRAM_MODEL,
            "language": DEEPGRAM_LANGUAGE,
            "smart_format": "true" if DEEPGRAM_SMART_FORMAT else "false",
            "punctuate": "true" if DEEPGRAM_PUNCTUATE else "false",
        }
    )

    url = f"{DEEPGRAM_ENDPOINT}?{query}"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": content_type or _guess_mime_type(filename),
        "Accept": "application/json",
    }

    if DEEPGRAM_DEBUG:
        print(
            f"[DEEPGRAM] filename={filename} content_type={headers['Content-Type']} "
            f"bytes={len(audio_bytes)} model={DEEPGRAM_MODEL} language={DEEPGRAM_LANGUAGE}"
        )

    request = Request(url, data=audio_bytes, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=DEEPGRAM_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="ignore")
        raise RuntimeError(
            f"Deepgram devolvió HTTP {error.code}: {detail or error.reason}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"No se pudo conectar con Deepgram: {error.reason}") from error

    payload = json.loads(raw)
    transcript = _extract_transcript(payload)

    if DEEPGRAM_DEBUG:
        print(f"[DEEPGRAM] transcript={transcript!r}")

    if not transcript:
        raise RuntimeError("Deepgram no devolvió una transcripción válida.")

    return transcript