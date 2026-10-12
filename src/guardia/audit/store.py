"""Audit persistence in SQLite (stdlib only).

LGPD: the free-text report is never stored. Only its SHA-256 hash and length are kept, which
allow counting and de-duplication without re-identification. Everything the model produced
(risk, probability, SHAP factors, sources, judge verdict) is stored so an analysis can be
reconstructed and explained later.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DB_PATH = ROOT / os.getenv("GUARDIA_DB_PATH", "data/generated/audit.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    input_hash TEXT,
    input_length INTEGER,
    sensitive_case INTEGER NOT NULL DEFAULT 1,
    risk_level TEXT,
    risk_factors TEXT,
    ml_probability REAL,
    ml_threshold REAL,
    ml_suggest_referral INTEGER,
    ml_model TEXT,
    ml_top_features TEXT,
    case_fields TEXT,
    used_rag INTEGER NOT NULL DEFAULT 0,
    sources TEXT,
    recommended_actions TEXT,
    llm_backend TEXT,
    llm_model TEXT,
    llm_latency_ms INTEGER,
    llm_tokens INTEGER,
    blocklist_passed INTEGER NOT NULL DEFAULT 1,
    blocked_terms TEXT,
    judge_backend TEXT,
    judge_approved INTEGER,
    judge_checks TEXT,
    final_status TEXT,
    response_json TEXT,
    response_length INTEGER
);
CREATE INDEX IF NOT EXISTS idx_analyses_ts ON analyses(ts);
CREATE INDEX IF NOT EXISTS idx_analyses_risk ON analyses(risk_level);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


_MIGRATIONS = {"steps": "TEXT", "total_ms": "INTEGER", "relato_resumo": "TEXT"}


def init(db_path: Path = DB_PATH) -> None:
    with _connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(analyses)").fetchall()}
        for col, typ in _MIGRATIONS.items():
            if col not in cols:
                conn.execute(f"ALTER TABLE analyses ADD COLUMN {col} {typ}")


def update_steps(analysis_id: int, steps: list[dict[str, Any]], total_ms: int, db_path: Path = DB_PATH) -> None:
    """Attach the per-node timeline (from the job runner) to an existing audit row."""
    init(db_path)
    with _connect(db_path) as conn:
        conn.execute("UPDATE analyses SET steps = ?, total_ms = ? WHERE id = ?", (_j(steps), total_ms, analysis_id))


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, default=str)


def persist(state: dict[str, Any], db_path: Path = DB_PATH) -> int:
    init(db_path)
    audit = state.get("audit_log") or {}
    user_input = state.get("user_input") or ""
    ml = state.get("ml") or {}
    judge = audit.get("llm_judge") or {}
    row = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "input_hash": hashlib.sha256(user_input.encode("utf-8")).hexdigest() if user_input else None,
        "input_length": len(user_input),
        "sensitive_case": 1,
        "risk_level": state.get("risk_level"),
        "risk_factors": _j(state.get("risk_factors") or []),
        "ml_probability": ml.get("probability"),
        "ml_threshold": ml.get("threshold"),
        "ml_suggest_referral": None if ml.get("suggest_referral") is None else int(bool(ml.get("suggest_referral"))),
        "ml_model": ml.get("model_name"),
        "ml_top_features": _j(ml.get("top_features") or []),
        "case_fields": _j(state.get("case") or {}),
        "used_rag": 1 if state.get("protocol_context") else 0,
        "sources": _j(state.get("protocol_sources") or []),
        "recommended_actions": _j(state.get("recommended_actions") or []),
        "llm_backend": audit.get("llm_backend"),
        "llm_model": audit.get("llm_model"),
        "llm_latency_ms": audit.get("llm_latency_ms"),
        "llm_tokens": audit.get("llm_tokens"),
        "blocklist_passed": 0 if audit.get("safety_validation") == "blocked" else 1,
        "blocked_terms": _j(audit.get("blocked_terms") or []),
        "judge_backend": judge.get("backend"),
        "judge_approved": None if judge.get("approved") is None else int(bool(judge.get("approved"))),
        "judge_checks": _j(judge.get("checks") or {}),
        "final_status": audit.get("safety_validation"),
        "response_json": _j(state.get("response") or {}),
        "relato_resumo": ((state.get("response") or {}).get("resumo") or "")[:400],
        "response_length": len(_j(state.get("response") or {})),
    }
    cols = ",".join(row)
    with _connect(db_path) as conn:
        cur = conn.execute(f"INSERT INTO analyses ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        return int(cur.lastrowid or -1)


def _hydrate(r: sqlite3.Row) -> dict[str, Any]:
    d = dict(r)
    for k in ("risk_factors", "ml_top_features", "case_fields", "sources", "recommended_actions", "blocked_terms", "judge_checks", "response_json", "steps"):
        try:
            d[k] = json.loads(d[k]) if d.get(k) else None
        except (TypeError, ValueError):
            pass
    return d


def fetch_recent(limit: int = 20, offset: int = 0, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init(db_path)
    with _connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM analyses ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
    return [_hydrate(r) for r in rows]


def fetch_one(analysis_id: int, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    init(db_path)
    with _connect(db_path) as conn:
        r = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
    return _hydrate(r) if r else None


def stats(db_path: Path = DB_PATH) -> dict[str, Any]:
    init(db_path)
    with _connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        by_risk = dict(conn.execute("SELECT COALESCE(risk_level,'?'), COUNT(*) FROM analyses GROUP BY 1").fetchall())
        suggested = conn.execute("SELECT COUNT(*) FROM analyses WHERE ml_suggest_referral = 1").fetchone()[0]
        blocked = conn.execute("SELECT COUNT(*) FROM analyses WHERE final_status LIKE 'blocked%'").fetchone()[0]
        judged = conn.execute("SELECT COUNT(*) FROM analyses WHERE judge_approved IS NOT NULL").fetchone()[0]
        judge_ok = conn.execute("SELECT COUNT(*) FROM analyses WHERE judge_approved = 1").fetchone()[0]
        avg_lat = conn.execute("SELECT AVG(llm_latency_ms) FROM analyses WHERE llm_latency_ms IS NOT NULL").fetchone()[0]
        by_backend = dict(conn.execute("SELECT COALESCE(llm_backend,'?'), COUNT(*) FROM analyses GROUP BY 1").fetchall())
    return {
        "total": total, "by_risk": by_risk, "suggested_referral": suggested, "blocked": blocked,
        "judged": judged, "judge_approved": judge_ok, "avg_llm_latency_ms": round(avg_lat or 0), "by_backend": by_backend,
    }
