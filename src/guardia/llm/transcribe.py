"""Server-side speech-to-text for the dictation fallback.

The browser uses the Web Speech API when it has it; otherwise it records continuously, cuts the audio at
pauses and posts each piece here (and the whole recording once more when the user stops, for a pass with
full context). Two backends, in order:

1. ``remote``: an audio-capable chat model through the OpenAI-compatible client (role ``transcribe``;
   default ``google/gemini-2.5-flash`` on OpenRouter, same key as the other roles). The audio is re-encoded
   to mono 16 kHz MP3 by ffmpeg and sent inline; the model answers ``{"text": ...}`` under a literal
   transcription prompt with temperature 0.
2. ``whisper``: faster-whisper on the CPU (``uv sync --group audio``; ``WHISPER_MODEL``, default ``small``),
   used when no remote model is configured or when the remote call fails.

``ffmpeg`` must be on the PATH in both cases (it decodes the browser's webm/opus; PyAV is avoided because its
API drifts between releases).
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from guardia.llm import client as llm

log = logging.getLogger("guardia.transcribe")
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
SAMPLE_RATE = 16000
MP3_BITRATE = "48k"
TRANSCRIBE_SCHEMA = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False}
SYSTEM_PROMPT = (
    "Você transcreve áudio de atendimentos a mulheres em situação de violência, falado em português do Brasil. "
    "Devolva exatamente o que foi dito, com pontuação normal: não resuma, não corrija o conteúdo, não comente, não traduza, "
    "não acrescente nada que não tenha sido falado. Se não houver fala inteligível, devolva texto vazio. "
    "Vocabulário frequente: Delegacia da Mulher, medida protetiva, Lei Maria da Penha, boletim de ocorrência, CRAS, CREAS, "
    "Conselho Tutelar, Defensoria Pública, companheiro, ex-companheiro, agressor, ameaça, faca, arma de fogo."
)


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def remote_configured() -> bool:
    return llm.endpoint("transcribe").configured


def backend() -> str | None:
    """Which backend will serve the next request: ``remote``, ``whisper`` or ``None``."""
    if not has_ffmpeg():
        return None
    if remote_configured():
        return "remote"
    return "whisper" if whisper_available() else None


def available() -> bool:
    return backend() is not None


def describe() -> dict:
    b = backend()
    ep = llm.endpoint("transcribe")
    return {"backend": b, "model": ep.model if b == "remote" else (MODEL_NAME if b == "whisper" else None),
            "whisper_fallback": whisper_available() and has_ffmpeg()}


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


def warm() -> None:
    """Load the local model (and run a silent decode) so the first real chunk is fast. Only when it is the backend."""
    if backend() != "whisper":
        return
    try:
        _model().transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), language="pt", beam_size=1)
    except Exception as e:  # noqa: BLE001
        log.warning("não aqueceu o modelo de voz: %s", e)


def _decode(path: Path) -> np.ndarray:
    """Any container/codec → mono float32 @16 kHz via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    return np.frombuffer(out, dtype=np.float32)


def _encode_mp3(audio: np.ndarray) -> bytes:
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-i", "-", "-codec:a", "libmp3lame", "-b:a", MP3_BITRATE, "-f", "mp3", "-"]
    return subprocess.run(cmd, input=audio.astype(np.float32).tobytes(), capture_output=True, check=True, timeout=120).stdout


def _remote(audio: np.ndarray) -> dict | None:
    """Transcribe through the audio-capable chat model; ``None`` when it fails (caller falls back)."""
    seconds = audio.size / SAMPLE_RATE
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Transcreva este áudio."},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(_encode_mp3(audio)).decode("ascii"), "format": "mp3"}},
        ]},
    ]
    res = llm.chat(messages, role="transcribe", schema=TRANSCRIBE_SCHEMA, schema_name="transcricao", temperature=0.0,
                   max_tokens=max(400, int(seconds * 12)), allow_fallback=False)
    if res.backend == "error" or not isinstance(res.parsed, dict):
        log.warning("transcrição remota falhou: %s", res.error)
        return None
    text = str(res.parsed.get("text") or "").strip().strip('"')
    return {"text": text, "language": "pt", "duration": round(seconds, 1), "backend": "remote", "model": res.model, "latency_ms": res.latency_ms}


def _whisper(audio: np.ndarray) -> dict:
    segments, info = _model().transcribe(audio, language="pt", vad_filter=True, beam_size=1)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {"text": text, "language": info.language, "duration": round(float(info.duration or audio.size / SAMPLE_RATE), 1),
            "backend": "whisper", "model": MODEL_NAME}


def transcribe_bytes(data: bytes, suffix: str = ".webm", skip_s: float = 0.0) -> dict:
    """Return {"text", "language", "duration", "backend", "model"}. Raises if no backend can run.

    ``skip_s`` drops the first seconds of the decoded audio: the browser prepends the webm initialisation
    segment (which also carries the first ~0.5 s of sound) to every flushed chunk, so later chunks would
    otherwise repeat the opening words.
    """
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = Path(f.name)
    try:
        audio = _decode(path)
    finally:
        path.unlink(missing_ok=True)
    if skip_s > 0:
        audio = audio[int(skip_s * SAMPLE_RATE):]
    if audio.size < SAMPLE_RATE // 4:  # under 250 ms: nothing to transcribe
        return {"text": "", "language": "pt", "duration": round(audio.size / SAMPLE_RATE, 2), "backend": "none", "model": None}
    if remote_configured():
        out = _remote(audio)
        if out is not None:
            return out
    if whisper_available():
        return _whisper(audio)
    raise RuntimeError("nenhum backend de transcrição disponível (modelo remoto falhou e faster-whisper não está instalado)")
