"""Health of every dependency, for the header indicator, the status page and the CLI.

Probing the LLM endpoints takes seconds (remote pod, possibly booting), so the web app never probes on
request: a background thread refreshes a cache every ``REFRESH_S`` seconds and pages render from it
instantly, showing how old the check is. ``full_status()`` stays synchronous for the CLI.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from guardia.llm import transcribe
from guardia.llm.client import probe
from guardia.ml.predict import DEFAULT_PATH

ROOT = Path(__file__).resolve().parents[3]
REFRESH_S = 20
PROBE_TIMEOUT_S = 4.0


def model_status() -> dict[str, Any]:
    card = ROOT / "models" / "model_card.json"
    if not DEFAULT_PATH.exists():
        return {"ok": False, "note": "modelo não treinado (uv run guardia train)"}
    info = json.loads(card.read_text(encoding="utf-8")) if card.exists() else {}
    return {"ok": True, "path": str(DEFAULT_PATH.relative_to(ROOT)), **info}


def rag_status() -> dict[str, Any]:
    try:
        from guardia.rag.retriever import corpus_summary, vectorstore

        n = vectorstore()._collection.count()
        return {"ok": n > 0, "chunks": n, "documents": corpus_summary()}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "note": f"{type(e).__name__}: {str(e)[:160]}"}


def speech_status() -> dict[str, Any]:
    ok = transcribe.available()
    return {"ok": True, "fallback": ok, "note": "faster-whisper no servidor (fallback do navegador)" if ok else "só Web Speech API do navegador"}


def full_status(include_rag: bool = True) -> dict[str, Any]:
    out = {"generator": probe("generator", timeout=PROBE_TIMEOUT_S), "judge": probe("judge", timeout=PROBE_TIMEOUT_S),
           "model": model_status(), "speech": speech_status(), "checked_at": time.time()}
    if include_rag:
        out["rag"] = rag_status()
    return out


class StatusCache:
    """Refreshed by a daemon thread; ``snapshot()`` never blocks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"generator": {"ok": False, "backend": "checking"}, "judge": {"ok": False, "backend": "checking"},
                                      "model": model_status(), "speech": speech_status(), "rag": {"ok": False, "note": "verificando…"},
                                      "checked_at": None, "pending": True}
        self._thread: threading.Thread | None = None
        self._wake = threading.Event()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="guardia-status")
        self._thread.start()

    def _loop(self) -> None:
        while True:
            self.refresh()
            self._wake.wait(REFRESH_S)
            self._wake.clear()

    def refresh(self) -> dict[str, Any]:
        data = full_status(include_rag=True)
        data["pending"] = False
        with self._lock:
            self._data = data
        return data

    def request_refresh(self) -> None:
        self._wake.set()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            d = dict(self._data)
        d["age_s"] = int(time.time() - d["checked_at"]) if d.get("checked_at") else None
        d["llm_ready"] = bool(d["generator"].get("ok")) and bool(d["judge"].get("ok"))
        d["overall"] = "checking" if d.get("pending") else ("ok" if d["llm_ready"] and d["model"].get("ok") and d["rag"].get("ok")
                                                           else ("degraded" if d["model"].get("ok") else "down"))
        return d


CACHE = StatusCache()
