"""F2 — turn the free-text report into the form fields (JSON constrained by schema)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guardia.data.features import CATEGORICAL, ESCOLARIDADE_LABELS, FORM_DEFAULTS, UF_OPTIONS
from guardia.llm.client import chat
from guardia.llm.prompts import STRUCTURE_SCHEMA, STRUCTURE_SYSTEM, build_structure_prompt


@dataclass
class Structured:
    case: dict[str, Any]
    signals: dict[str, Any]
    summary: str
    uncertain: list[str] = field(default_factory=list)
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


def structure_report(text: str, *, fast: bool = False) -> Structured:
    """`fast=True` uses the small model (FAST_* / judge server) for live extraction while the user types."""
    res = chat(
        [{"role": "system", "content": STRUCTURE_SYSTEM}, {"role": "user", "content": build_structure_prompt(text)}],
        role="fast" if fast else "generator", schema=STRUCTURE_SCHEMA, schema_name="ficha", temperature=0.0, max_tokens=900,
        allow_fallback=not fast,
    )
    if fast and res.backend == "error":  # no fast endpoint configured → use the generator
        res = chat(
            [{"role": "system", "content": STRUCTURE_SYSTEM}, {"role": "user", "content": build_structure_prompt(text)}],
            role="generator", schema=STRUCTURE_SCHEMA, schema_name="ficha", temperature=0.0, max_tokens=900,
        )
    if res.backend == "error" or not res.parsed:
        return Structured(case={}, signals={}, summary="", backend="error", model=res.model, error=res.error or "sem JSON")
    case, signals = _normalise(res.parsed)
    return Structured(
        case=case, signals=signals, summary=str(res.parsed.get("resumo_objetivo") or ""),
        uncertain=[str(x) for x in (res.parsed.get("campos_incertos") or [])][:10],
        backend=res.backend, model=res.model, latency_ms=res.latency_ms,
    )
