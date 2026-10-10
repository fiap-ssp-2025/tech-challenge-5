"""Background analyses with real per-node progress.

The web form starts a job; the browser polls its progress (htmx, every 2 s) and receives the
result fragment when the graph finishes. Each LangGraph node is timed as it completes and a
short digest of what it produced is kept, both for the live progress view and for the history
timeline (persisted next to the audit row).
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from guardia.audit import store as audit
from guardia.graph.flow import NODES, build_graph

STEP_LABELS: dict[str, str] = {
    "assess_risk": "Avaliar risco por regras",
    "run_model": "Rodar o modelo e explicar (SHAP)",
    "retrieve_protocols": "Consultar protocolos (RAG)",
    "define_actions": "Definir ações e encaminhamentos",
    "build_explanation": "Montar a explicação",
    "prepare_audit": "Preparar o registro",
    "generate_note": "Escrever a nota (LLM)",
    "judge": "Validar com o juiz independente",
    "persist": "Gravar a auditoria",
}
STEP_ORDER = [name for name, _ in NODES]

_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_GRAPH = None


@dataclass
class Step:
    node: str
    label: str
    status: str = "pending"      # pending | running | done | error
    started: float | None = None
    duration_ms: int | None = None
    digest: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"node": self.node, "label": self.label, "status": self.status, "duration_ms": self.duration_ms, "digest": self.digest}


@dataclass
class Job:
    id: str
    user_input: str
    case: dict[str, Any]
    signals: dict[str, Any]
    created: float = field(default_factory=time.time)
    status: str = "running"      # running | done | error
    steps: list[Step] = field(default_factory=lambda: [Step(n, STEP_LABELS.get(n, n)) for n in STEP_ORDER])
    state: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    finished: float | None = None

    @property
    def elapsed_s(self) -> float:
        return (self.finished or time.time()) - self.created

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "done")


def _digest(node: str, update: dict[str, Any], state: dict[str, Any]) -> str:
    if node == "assess_risk":
        return f"risco {update.get('risk_level')} · {len(update.get('risk_factors') or [])} fator(es)"
    if node == "run_model":
        ml = update.get("ml") or {}
        if not ml:
            return f"modelo não rodou ({update.get('ml_error') or 'sem dados'})"
        return f"{ml.get('probability', 0):.0%} · {'sugere encaminhar' if ml.get('suggest_referral') else 'sem sugestão'} · {ml.get('model_name')}"
    if node == "retrieve_protocols":
        srcs = update.get("protocol_sources") or []
        return f"{len(update.get('protocol_hits') or [])} trechos de {len(srcs)} documento(s): " + ", ".join(s["title"] for s in srcs[:3])
    if node == "define_actions":
        return f"{len(update.get('recommended_actions') or [])} ações · {len(update.get('referrals') or [])} encaminhamentos"
    if node == "build_explanation":
        return str(update.get("explanation") or "")[:160]
    if node == "prepare_audit":
        return "hash do relato, risco e fontes preparados"
    if node == "generate_note":
        log = update.get("audit_log") or {}
        return f"{log.get('llm_backend')} · {log.get('llm_model') or ''} · {log.get('llm_latency_ms') or 0} ms · {log.get('safety_validation')}"
    if node == "judge":
        j = (update.get("audit_log") or {}).get("llm_judge") or {}
        if j.get("approved") is None:
            return f"não avaliado ({j.get('backend') or j.get('reason') or 'n/d'})"
        return ("aprovou" if j.get("approved") else "reprovou") + f" · confiança {j.get('confidence', 0):.0%}" + (f" · {len(j.get('overrides') or [])} revisão(ões)" if j.get("overrides") else "")
    if node == "persist":
        return f"registro #{update.get('audit_id')}" if update.get("audit_id") else "não gravado"
    return ""


def _run(job: Job) -> None:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    state: dict[str, Any] = {"user_input": job.user_input, "case": job.case, "signals": job.signals}
    idx = 0
    try:
        job.steps[0].status, job.steps[0].started = "running", time.time()
        for update in _GRAPH.stream(state, stream_mode="updates"):
            for node, delta in update.items():
                state.update(delta or {})
                step = next((s for s in job.steps if s.node == node), None)
                if step is None:
                    continue
                now = time.time()
                step.status = "done"
                step.duration_ms = int((now - (step.started or now)) * 1000)
                step.digest = _digest(node, delta or {}, state)
                idx = STEP_ORDER.index(node) + 1
                if idx < len(job.steps):
                    job.steps[idx].status, job.steps[idx].started = "running", now
        job.state = state
        job.status = "done"
    except Exception as e:  # noqa: BLE001
        job.error = f"{type(e).__name__}: {str(e)[:300]}"
        job.status = "error"
        for s in job.steps:
            if s.status == "running":
                s.status = "error"
    finally:
        job.finished = time.time()
        aid = job.state.get("audit_id") if job.state else None
        if aid:
            try:
                audit.update_steps(int(aid), [s.as_dict() for s in job.steps], int(job.elapsed_s * 1000))
            except Exception as e:  # noqa: BLE001
                logging.getLogger("guardia.jobs").warning("não gravou a linha do tempo da análise %s: %s", aid, e)


def start(user_input: str, case: dict[str, Any], signals: dict[str, Any]) -> Job:
    job = Job(id=uuid.uuid4().hex[:10], user_input=user_input, case=case, signals=signals)
    with _LOCK:
        _JOBS[job.id] = job
        # keep the registry small: drop finished jobs older than an hour
        cutoff = time.time() - 3600
        for k in [k for k, j in _JOBS.items() if j.finished and j.finished < cutoff]:
            _JOBS.pop(k, None)
    threading.Thread(target=_run, args=(job,), daemon=True, name=f"guardia-job-{job.id}").start()
    return job


def get(job_id: str) -> Job | None:
    return _JOBS.get(job_id)
