"""F3 — question answering over the protocol corpus, with memory kept on the server (SQLite)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from guardia.audit.store import DB_PATH, _connect
from guardia.guardrails.blocklist import SAFETY_NOTICE, validate
from guardia.llm.client import chat
from guardia.rag.retriever import search

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    sources TEXT,
    backend TEXT,
    latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_chat_conv ON chat_turns(conversation_id, id);
"""

SYSTEM = (
    "Você responde perguntas de profissionais que atendem mulheres em situação de violência, usando SOMENTE os "
    "trechos de protocolos e leis fornecidos. Cite a fonte pelo título entre colchetes ao usar um trecho. "
    "Se os trechos não respondem, diga isso claramente e sugira o que consultar. Não decida pelo profissional, "
    "não prescreva, não faça diagnóstico. Português do Brasil, resposta objetiva em até 8 frases."
)

ANSWER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "resposta": {"type": "string"},
        "fontes_citadas": {"type": "array", "items": {"type": "string"}},
        "cobertura": {"type": "string", "enum": ["total", "parcial", "nenhuma"]},
    },
    "required": ["resposta", "fontes_citadas", "cobertura"],
}


@dataclass
class Turn:
    question: str
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    cited: list[str] = field(default_factory=list)
    coverage: str = "nenhuma"
    backend: str = "error"
    model: str = ""
    latency_ms: int = 0
    error: str | None = None
    blocked: bool = False


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]


def _init() -> None:
    with _connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)


def history(conversation_id: str, limit: int = 6) -> list[dict[str, str]]:
    _init()
    with _connect(DB_PATH) as conn:
        rows = conn.execute("SELECT question, answer FROM chat_turns WHERE conversation_id=? ORDER BY id DESC LIMIT ?",
                            (conversation_id, limit)).fetchall()
    msgs: list[dict[str, str]] = []
    for r in reversed(rows):
        msgs.append({"role": "user", "content": r["question"]})
        if r["answer"]:
            msgs.append({"role": "assistant", "content": r["answer"]})
    return msgs


def turns(conversation_id: str) -> list[dict[str, Any]]:
    _init()
    with _connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM chat_turns WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["sources"] = json.loads(d["sources"] or "[]")
        except ValueError:
            d["sources"] = []
        out.append(d)
    return out


def ask(conversation_id: str, question: str) -> Turn:
    hits = search(question, k=6)
    ctx = "\n\n".join(f"[{h.title}]\n{h.text}" for h in hits) or "(nenhum trecho recuperado)"
    msgs = [{"role": "system", "content": SYSTEM}]
    msgs += history(conversation_id)
    msgs.append({"role": "user", "content": f"Trechos recuperados:\n{ctx}\n\nPergunta: {question}\n\nResponda em JSON."})
    res = chat(msgs, role="generator", schema=ANSWER_SCHEMA, schema_name="resposta_rag", temperature=0.1, max_tokens=700)
    sources = [{"file": h.source, "title": h.title, "text": h.text, "score": round(h.score, 3)} for h in hits]
    if res.backend == "error" or not res.parsed:
        turn = Turn(question=question, answer="", sources=sources, backend="error", model=res.model, error=res.error)
    else:
        answer = str(res.parsed.get("resposta") or "").strip()
        blocked = not validate(answer)["approved"]
        turn = Turn(question=question, answer=("[bloqueado pela lista de termos proibidos]" if blocked else answer) + "\n\n" + SAFETY_NOTICE,
                    sources=sources, cited=[str(x) for x in res.parsed.get("fontes_citadas") or []],
                    coverage=str(res.parsed.get("cobertura") or "nenhuma"), backend=res.backend, model=res.model,
                    latency_ms=res.latency_ms, blocked=blocked)
    _init()
    with _connect(DB_PATH) as conn:
        conn.execute(
            "INSERT INTO chat_turns (conversation_id, ts, question, answer, sources, backend, latency_ms) VALUES (?,?,?,?,?,?,?)",
            (conversation_id, datetime.now(UTC).isoformat(timespec="seconds"), question, turn.answer,
             json.dumps([{"file": s["file"], "title": s["title"]} for s in sources], ensure_ascii=False), turn.backend, turn.latency_ms),
        )
    return turn
