"""Health of every dependency, for the status page and the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from guardia.llm.client import probe
from guardia.ml.predict import DEFAULT_PATH

ROOT = Path(__file__).resolve().parents[3]


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


def full_status(include_rag: bool = True) -> dict[str, Any]:
    out = {"generator": probe("generator"), "judge": probe("judge"), "model": model_status()}
    if include_rag:
        out["rag"] = rag_status()
    return out
