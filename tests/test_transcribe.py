"""Transcription backends: remote model first, whisper as reserve, ffmpeg for decoding."""

import shutil
import subprocess

import numpy as np
import pytest

from guardia.llm import client as llm
from guardia.llm import transcribe as t

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg ausente")


def _wav(seconds: float = 1.0) -> bytes:
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(int(seconds * t.SAMPLE_RATE)) / t.SAMPLE_RATE)).astype(np.float32)
    cmd = ["ffmpeg", "-v", "error", "-f", "f32le", "-ac", "1", "-ar", str(t.SAMPLE_RATE), "-i", "-", "-f", "wav", "-"]
    return subprocess.run(cmd, input=tone.tobytes(), capture_output=True, check=True).stdout


def test_backend_prefers_remote_when_configured(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setattr(t, "has_ffmpeg", lambda: True)
    assert t.backend() == "remote"
    assert t.describe()["model"] == "google/gemini-2.5-flash"


def test_backend_none_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(t, "has_ffmpeg", lambda: False)
    assert t.backend() is None and not t.available()


def test_transcribe_role_defaults_to_gemini_on_openrouter(monkeypatch):
    for k in ("TRANSCRIBE_BASE_URL", "TRANSCRIBE_MODEL", "TRANSCRIBE_API_KEY", "FAST_BASE_URL", "FAST_API_KEY", "JUDGE_BASE_URL", "JUDGE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FAST_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    ep = llm.endpoint("transcribe")
    assert ep.configured and ep.model == "google/gemini-2.5-flash" and ep.api_key == "k"


@needs_ffmpeg
def test_remote_path_sends_mp3_and_parses_text(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "google/gemini-2.5-flash")
    seen = {}

    def fake_chat(messages, **kw):
        seen["audio"] = messages[1]["content"][1]["input_audio"]
        seen["kw"] = kw
        return llm.LLMResult(content='{"text": "ela pediu a medida protetiva"}', backend="transcribe", model="m", latency_ms=5, parsed={"text": "ela pediu a medida protetiva"})

    monkeypatch.setattr(llm, "chat", fake_chat)
    out = t.transcribe_bytes(_wav(1.0), suffix=".wav")
    assert out["backend"] == "remote" and out["text"] == "ela pediu a medida protetiva"
    assert seen["audio"]["format"] == "mp3" and len(seen["audio"]["data"]) > 100
    assert seen["kw"]["role"] == "transcribe" and seen["kw"]["temperature"] == 0.0


@needs_ffmpeg
def test_remote_failure_falls_back_or_raises(monkeypatch):
    monkeypatch.setenv("TRANSCRIBE_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("TRANSCRIBE_MODEL", "google/gemini-2.5-flash")
    monkeypatch.setattr(llm, "chat", lambda *a, **k: llm.LLMResult(content="", backend="error", model="m", latency_ms=1, error="boom"))
    monkeypatch.setattr(t, "whisper_available", lambda: False)
    with pytest.raises(RuntimeError):
        t.transcribe_bytes(_wav(1.0), suffix=".wav")


@needs_ffmpeg
def test_short_audio_is_skipped_without_calling_any_backend(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: pytest.fail("não deveria chamar o modelo"))
    out = t.transcribe_bytes(_wav(0.1), suffix=".wav")
    assert out["text"] == "" and out["backend"] == "none"
