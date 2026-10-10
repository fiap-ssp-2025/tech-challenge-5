"""Layer-1 guardrail: exact-phrase blocklist (inherited from TC3) + mandatory safety notice."""

from __future__ import annotations

FORBIDDEN = [
    "não procure ajuda", "ignore a ameaça", "volte para casa", "confronte o agressor", "isso não é grave",
    "não conte para ninguém", "tome este medicamento", "use este medicamento", "prescrevo", "eu prescrevo",
    "você deve se automedicar", "não acione a polícia", "não procure atendimento", "resolva sozinha",
    "enfrente o agressor", "a culpa é dela", "ela provocou", "não registre", "não denuncie",
]

SAFETY_NOTICE = (
    "Este resumo é um apoio à análise da profissional responsável pelo atendimento. "
    "Não substitui a avaliação humana, não fecha diagnóstico e não decide encaminhamentos."
)


def validate(text: str) -> dict:
    low = (text or "").lower()
    found = [t for t in FORBIDDEN if t in low]
    return {"approved": not found, "found_terms": found}


def scan_dict(d: dict) -> dict:
    """Run the blocklist over every string in a (nested) dict."""
    chunks: list[str] = []

    def walk(v):
        if isinstance(v, str):
            chunks.append(v)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(d)
    return validate("\n".join(chunks))
