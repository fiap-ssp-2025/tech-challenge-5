"""F2 — turn the free-text report into the form fields (JSON constrained by schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guardia.data.features import CATEGORICAL, ESCOLARIDADE_LABELS, FORM_DEFAULTS, UF_OPTIONS
from guardia.interview import TOPIC_IDS
from guardia.llm.client import chat
from guardia.llm.prompts import (
    BINARY_FIELDS,
    STRUCTURE_SCHEMA,
    STRUCTURE_SCHEMA_SPARSE,
    STRUCTURE_SYSTEM,
    build_structure_prompt,
    build_structure_prompt_sparse,
)


@dataclass
class Structured:
    case: dict[str, Any]
    signals: dict[str, Any]
    summary: str
    uncertain: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)  # interview topics the report touched (spec 003 U6)
    backend: str = "error"
    model: str = ""
    latency_ms: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.backend != "error"


def _normalise(parsed: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    case: dict[str, Any] = {}
    for key in FORM_DEFAULTS:
        if key in ("ESCOLARIDADE", "IDADE_ANOS", "SG_UF") or key in CATEGORICAL:
            continue
        case[key] = 1 if parsed.get(key) else 0
    for key, (_, mapping) in CATEGORICAL.items():
        val = str(parsed.get(key) or "IGNORADO").upper()
        case[key] = val if val in mapping.values() else "IGNORADO"
    uf = str(parsed.get("SG_UF") or "IGNORADO").upper()
    case["SG_UF"] = uf if uf in UF_OPTIONS else "IGNORADO"
    esc = str(parsed.get("ESCOLARIDADE") or "").upper()
    case["ESCOLARIDADE"] = esc if esc in ESCOLARIDADE_LABELS.values() else None
    idade = parsed.get("IDADE_ANOS")
    case["IDADE_ANOS"] = int(idade) if isinstance(idade, (int, float)) and 0 <= idade <= 120 else None
    signals = {
        "arma_mencionada": bool(parsed.get("arma_mencionada")),
        "ameaca_de_morte": bool(parsed.get("ameaca_de_morte")),
        "filhos_envolvidos": bool(parsed.get("filhos_envolvidos")),
        "local_seguro": parsed.get("local_seguro"),
    }
    return case, signals


def _expand_sparse(parsed: dict[str, Any]) -> dict[str, Any]:
    """Sparse output ("marcados": [...]) → the dense shape `_normalise` expects."""
    marked = {str(x) for x in (parsed.get("marcados") or [])}
    dense = {k: (k in marked) for k in BINARY_FIELDS}
    dense.update({k: v for k, v in parsed.items() if k != "marcados"})
    return dense


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _quoted_topics(parsed: dict[str, Any], text: str) -> list[str]:
    """Topics whose quote really is in the report (spec 003 U6): the model's claim alone is not enough."""
    body, out, raw = _norm(text), [], parsed.get("topicos_abordados") or []
    if isinstance(raw, dict):  # json_object fallback sometimes answers {"id": "trecho"}
        raw = [f"{k}: {v}" for k, v in raw.items()]
    for item in raw:
        tid, _, quote = str(item).partition(":")
        tid, quote = tid.strip(), _norm(quote.strip(" \"'“”"))
        if tid in TOPIC_IDS and tid not in out and len(quote) >= 3 and quote in body:
            out.append(tid)
    return out


def structure_report(text: str, *, fast: bool = False) -> Structured:
    """`fast=True` → live extraction while the user types: the FAST_* endpoint (or the generator) with the sparse
    schema, which is ~4x shorter to generate. The full schema is used by the explicit "Reextrair" button."""
    schema, prompt = (STRUCTURE_SCHEMA_SPARSE, build_structure_prompt_sparse(text)) if fast else (STRUCTURE_SCHEMA, build_structure_prompt(text))
    msgs = [{"role": "system", "content": STRUCTURE_SYSTEM}, {"role": "user", "content": prompt}]
    res = chat(msgs, role="fast" if fast else "generator", schema=schema, schema_name="ficha", temperature=0.0, max_tokens=600 if fast else 900,
               allow_fallback=not fast)
    if fast and res.backend == "error":  # no fast endpoint configured → use the generator
        res = chat(msgs, role="generator", schema=schema, schema_name="ficha", temperature=0.0, max_tokens=600)
    if res.backend == "error" or not res.parsed:
        return Structured(case={}, signals={}, summary="", backend="error", model=res.model, error=res.error or "sem JSON")
    case, signals = _normalise(_expand_sparse(res.parsed) if fast else res.parsed)
    return Structured(
        case=case, signals=signals, summary=str(res.parsed.get("resumo_objetivo") or ""),
        uncertain=[str(x) for x in (res.parsed.get("campos_incertos") or [])][:10],
        topics=_quoted_topics(res.parsed, text),
        backend=res.backend, model=res.model, latency_ms=res.latency_ms,
    )
