"""Server-side speech-to-text fallback (faster-whisper, CPU).

The browser uses the Web Speech API when it has it; otherwise it records short chunks and posts
them here. Optional dependency: ``uv sync --group audio``; ``ffmpeg`` must be on the PATH (it decodes
the browser's webm/opus chunks; PyAV is avoided because its API drifts between releases). The model
(``small``, int8) is downloaded on first use into the Hugging Face cache.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
SAMPLE_RATE = 16000


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return shutil.which("ffmpeg") is not None


@lru_cache(maxsize=1)
def _model():
    from faster_whisper import WhisperModel

    return WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")


def _decode(path: Path) -> np.ndarray:
    """Any container/codec → mono float32 @16 kHz via ffmpeg."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-"]
    out = subprocess.run(cmd, capture_output=True, check=True, timeout=120).stdout
    return np.frombuffer(out, dtype=np.float32)


def transcribe_bytes(data: bytes, suffix: str = ".webm") -> dict:
    """Return {"text": str, "language": str, "duration": float}. Raises if the dependencies are missing."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = Path(f.name)
    try:
        audio = _decode(path)
        if audio.size < SAMPLE_RATE // 4:  # under 250 ms: nothing to transcribe
            return {"text": "", "language": "pt", "duration": round(audio.size / SAMPLE_RATE, 2)}
        segments, info = _model().transcribe(audio, language="pt", vad_filter=True, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return {"text": text, "language": info.language, "duration": round(float(info.duration or audio.size / SAMPLE_RATE), 1)}
    finally:
        path.unlink(missing_ok=True)
