"""F1 — the LangGraph that turns a report + form into a note of support.

assess_risk → run_model → retrieve_protocols → define_actions → build_explanation
→ prepare_audit → generate_note → judge → persist → END

Every node reads and writes the same ``CaseState``. The graph never raises on an LLM failure:
``response`` carries an error note and the audit still records what happened.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from guardia.audit.store import persist as persist_audit_record
from guardia.guardrails.blocklist import SAFETY_NOTICE, scan_dict
from guardia.llm.client import chat
from guardia.llm.judge import block_message, judge_note
from guardia.llm.prompts import RESPONSE_SCHEMA, SYSTEM_PROMPT, build_response_prompt
from guardia.ml.predict import predict_case
from guardia.rag.retriever import search


class CaseState(TypedDict, total=False):
    user_input: str
    case: dict[str, Any]            # form fields (feature names) checked by the professional
    signals: dict[str, Any]         # extra risk flags (arma_mencionada, ameaca_de_morte, filhos_envolvidos, local_seguro)
    risk_level: str
    risk_factors: list[str]
    ml: dict[str, Any]
    ml_error: str
    protocol_hits: list[dict[str, Any]]
    protocol_context: str
    protocol_sources: list[dict[str, str]]
    recommended_actions: list[str]
    referrals: list[str]
    explanation: str
    audit_log: dict[str, Any]
    response: dict[str, Any]
    llm_metadata: dict[str, Any]
    audit_id: int


RISK_ORDER = ["baixo", "moderado", "alto", "critico"]

_TEXT_RULES: list[tuple[tuple[str, ...], str, str]] = [
    (("ameaça de morte", "ameaçou de morte", "vai me matar", "me matar"), "Ameaça de morte relatada", "critico"),
    (("arma de fogo", "revólver", "pistola", "espingarda"), "Arma de fogo mencionada", "critico"),
    (("faca", "facão", "arma branca", "canivete", "tesoura"), "Arma branca mencionada", "critico"),
    (("não consigo sair", "não me deixa sair", "trancada", "cárcere"), "Indício de cárcere ou restrição de liberdade", "critico"),
    (("estrangul", "enforc", "sufoc"), "Estrangulamento ou sufocação relatados", "critico"),
    (("me bateu", "apanhei", "agrediu", "socos", "tapas", "empurr", "chutou"), "Violência física relatada", "alto"),
    (("tenho medo", "com medo", "pavor"), "Medo declarado", "alto"),
    (("persegu", "stalking", "segue", "vigia"), "Perseguição relatada", "alto"),
    (("controla", "controle", "proíbe", "não deixa", "isolamento", "isolada"), "Controle coercitivo", "alto"),
    (("grávida", "gestante", "gravidez"), "Gestação: maior vulnerabilidade", "alto"),
    (("estupr", "forçou", "sem consentimento", "abuso sexual"), "Violência sexual relatada", "alto"),
    (("filho", "filha", "crianças", "criança"), "Crianças no contexto", "moderado"),
    (("humilha", "xinga", "ofende", "ameaça"), "Violência psicológica relatada", "moderado"),
    (("bebe", "bêbado", "alcool", "álcool", "droga"), "Uso de álcool/drogas pelo autor", "moderado"),
    (("dinheiro", "não deixa trabalhar", "controla o dinheiro", "cartão"), "Violência patrimonial relatada", "moderado"),
]


def assess_risk(state: CaseState) -> dict[str, Any]:
    text = (state.get("user_input") or "").lower()
    case = state.get("case") or {}
    signals = state.get("signals") or {}
    factors: list[str] = []
    level = "baixo"

    def bump(to: str) -> None:
        nonlocal level
        if RISK_ORDER.index(to) > RISK_ORDER.index(level):
            level = to

    for terms, label, lvl in _TEXT_RULES:
        if any(t in text for t in terms):
            factors.append(label); bump(lvl)
    if case.get("AG_FOGO") or case.get("AG_CORTE") or signals.get("arma_mencionada"):
        factors.append("Arma registrada no formulário"); bump("critico")
    if signals.get("ameaca_de_morte"):
        factors.append("Ameaça de morte registrada"); bump("critico")
    if case.get("AG_ENFOR"):
        factors.append("Enforcamento/sufocação registrado"); bump("critico")
    if case.get("VIOL_SEXU"):
        factors.append("Violência sexual registrada"); bump("alto")
    if case.get("VIOL_FISIC"):
        factors.append("Violência física registrada"); bump("alto")
    if str(case.get("OUT_VEZES", "")).upper() == "SIM":
        factors.append("Violência de repetição"); bump("moderado")
    if case.get("REL_PARCEIRO_INTIMO"):
        factors.append("Autor é parceiro íntimo ou ex"); bump("moderado")
    if str(case.get("CS_GESTANT", "")).upper() in ("1_TRIMESTRE", "2_TRIMESTRE", "3_TRIMESTRE", "IDADE_GEST_IGNORADA"):
        factors.append("Gestante"); bump("alto")
    if signals.get("filhos_envolvidos"):
        factors.append("Filhos envolvidos"); bump("moderado")
    if signals.get("local_seguro") is False:
        factors.append("Sem local seguro disponível"); bump("alto")
    seen: list[str] = []
    for f in factors:
        if f not in seen:
            seen.append(f)
    return {"risk_level": level, "risk_factors": seen}


def run_model(state: CaseState) -> dict[str, Any]:
    case = state.get("case") or {}
    if not case:
        return {"ml": {}, "ml_error": "sem dados estruturados"}
    try:
        pred = predict_case(case)
        return {"ml": pred.as_dict(), "ml_error": ""}
    except Exception as e:  # noqa: BLE001
        return {"ml": {}, "ml_error": f"{type(e).__name__}: {str(e)[:200]}"}


def retrieve_protocols(state: CaseState) -> dict[str, Any]:
    text = state.get("user_input") or ""
    query = text + " " + " ".join(state.get("risk_factors") or [])
    try:
        hits = search(query, k=5)
    except Exception as e:  # noqa: BLE001
        return {"protocol_hits": [], "protocol_context": "", "protocol_sources": [], "audit_log": {"rag_error": str(e)[:200]}}
    sources: list[dict[str, str]] = []
    for h in hits:
        if h.source not in [s["file"] for s in sources]:
            sources.append({"file": h.source, "title": h.title})
    return {
        "protocol_hits": [{"text": h.text, "source": h.source, "title": h.title, "score": round(h.score, 3)} for h in hits],
        "protocol_context": "\n\n".join(h.text for h in hits),
        "protocol_sources": sources,
    }


def define_actions(state: CaseState) -> dict[str, Any]:
    level = state.get("risk_level", "baixo")
    ml = state.get("ml") or {}
    actions = [
        "Acolhimento humanizado e escuta qualificada, sem julgamento.",
        "Preservar sigilo e restringir o acesso ao registro (dado sensível, LGPD).",
        "Avaliar risco imediato à integridade física e combinar um plano de segurança.",
    ]
    referrals: list[str] = []
    if level == "critico":
        actions.insert(0, "Risco iminente: acionar Polícia Militar (190) e orientar registro na DEAM agora.")
        referrals += ["Polícia Militar (190)", "DEAM / Delegacia da Mulher — medida protetiva de urgência (Lei 11.340, art. 22)"]
    if level in ("critico", "alto"):
        actions.append("Orientar e apoiar o pedido de medida protetiva de urgência.")
        actions.append("Avaliar abrigamento ou local seguro (Casa da Mulher Brasileira, casa-abrigo).")
        referrals += ["Casa da Mulher Brasileira / centro de referência da mulher", "CREAS — acompanhamento psicossocial"]
    if ml.get("suggest_referral"):
        actions.append(f"Modelo sugere encaminhamento à rede de proteção (probabilidade {ml.get('probability', 0):.0%}): discutir com a equipe.")
        referrals += ["Defensoria Pública / Ministério Público (medidas e orientação jurídica)"]
    if level in ("baixo", "moderado"):
        actions.append("Registrar o caso, oferecer os canais (Ligue 180) e agendar retorno em até 7 dias.")
        referrals += ["Ligue 180 — Central de Atendimento à Mulher"]
    case = state.get("case") or {}
    if case.get("VIOL_SEXU"):
        actions.append("Violência sexual: encaminhar à saúde em até 72h (profilaxias, norma técnica MS/MJ).")
        referrals.append("Serviço de saúde de referência para violência sexual")
    seen: list[str] = []
    for r in referrals:
        if r not in seen:
            seen.append(r)
    return {"recommended_actions": actions, "referrals": seen}


def build_explanation(state: CaseState) -> dict[str, Any]:
    ml = state.get("ml") or {}
    n_src = len(state.get("protocol_sources") or [])
    parts = [f"Risco por regras: {state.get('risk_level', 'n/d')} ({len(state.get('risk_factors') or [])} fatores)."]
    if ml:
        parts.append(f"Modelo {ml.get('model_name')}: probabilidade {ml.get('probability', 0):.0%} de perfil encaminhado "
                     f"(limiar {ml.get('threshold', 0):.0%}).")
    parts.append(f"{n_src} fonte(s) de protocolo recuperada(s).")
    return {"explanation": " ".join(parts)}


def prepare_audit(state: CaseState) -> dict[str, Any]:
    log = dict(state.get("audit_log") or {})
    log.update({"timestamp": datetime.now(UTC).isoformat(timespec="seconds"), "sensitive_case": True,
                "risk_level": state.get("risk_level"), "used_rag": bool(state.get("protocol_context")),
                "safety_validation": "pending"})
    return {"audit_log": log}


def _tidy_note(note: dict[str, Any], state: CaseState) -> dict[str, Any]:
    """Normalise what small models get wrong: bracketed/duplicated citations, terse model reading, missing notice."""
    titles = {src["title"] for src in (state.get("protocol_sources") or [])}
    cited: list[str] = []
    for raw in note.get("fontes_citadas") or []:
        t = str(raw).strip().strip("[]").strip()
        match = next((x for x in titles if x.lower() == t.lower() or t.lower() in x.lower() or x.lower() in t.lower()), None)
        t = match or t
        if t and t not in cited:
            cited.append(t)
    note["fontes_citadas"] = cited
    ml = state.get("ml") or {}
    if ml and len(str(note.get("leitura_do_modelo") or "")) < 40:
        top = ", ".join(f"{t['label'].lower()} ({t['direction']})" for t in (ml.get("top_features") or [])[:3])
        note["leitura_do_modelo"] = (
            f"O modelo estima {ml.get('probability', 0):.0%} de probabilidade de este perfil ter sido encaminhado à rede de proteção "
            f"em 2024 (limiar {ml.get('threshold', 0):.0%}). Pesaram mais: {top or 'sem fatores destacados'}. "
            "É uma leitura da prática registrada, não uma decisão.")
    note["aviso"] = note.get("aviso") or SAFETY_NOTICE
    for key in ("fatores_de_atencao", "acoes_sugeridas", "encaminhamentos", "perguntas_pendentes"):
        seen: list[str] = []
        for item in note.get(key) or []:
            item = str(item).strip()
            if item and item not in seen:
                seen.append(item)
        note[key] = seen
    return note


def generate_note(state: CaseState) -> dict[str, Any]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": build_response_prompt(dict(state))}]
    res = chat(messages, role="generator", schema=RESPONSE_SCHEMA, schema_name="nota_de_apoio", temperature=0.2, max_tokens=1000)
    log = dict(state.get("audit_log") or {})
    log.update({"llm_backend": res.backend, "llm_model": res.model, "llm_latency_ms": res.latency_ms, "llm_tokens": res.tokens})
    if res.backend == "error" or not res.parsed:
        log["safety_validation"] = "llm_error"
        note = {"nivel_risco": state.get("risk_level", "indeterminado"), "resumo": "", "erro": res.error or "LLM sem resposta",
                "acoes_sugeridas": state.get("recommended_actions") or [], "encaminhamentos": state.get("referrals") or [],
                "fontes_citadas": [], "aviso": SAFETY_NOTICE}
        return {"response": note, "audit_log": log, "llm_metadata": {"backend": "error", "error": res.error}}
    note = _tidy_note(res.parsed, state)
    scan = scan_dict(note)
    log["blocked_terms"] = scan["found_terms"]
    if not scan["approved"]:
        log["safety_validation"] = "blocked"
        note = {"nivel_risco": note.get("nivel_risco"), "resumo": "", "bloqueio": "A nota foi bloqueada pela lista de termos proibidos. "
                "Revise o caso com a equipe.", "termos": scan["found_terms"], "acoes_sugeridas": state.get("recommended_actions") or [],
                "encaminhamentos": state.get("referrals") or [], "fontes_citadas": [], "aviso": SAFETY_NOTICE}
    else:
        log["safety_validation"] = "approved"
    return {"response": note, "audit_log": log,
            "llm_metadata": {"backend": res.backend, "model": res.model, "latency_ms": res.latency_ms, "tokens": res.tokens,
                             "response_format": res.meta.get("response_format")}}


def judge(state: CaseState) -> dict[str, Any]:
    log = dict(state.get("audit_log") or {})
    note = state.get("response") or {}
    if log.get("safety_validation") in ("blocked", "llm_error"):
        log["llm_judge"] = {"backend": "skipped", "approved": None, "reason": log.get("safety_validation")}
        return {"audit_log": log}
    verdict = judge_note(note, user_input=state.get("user_input") or "", risk_level=state.get("risk_level") or "")
    log["llm_judge"] = verdict.as_dict()
    if verdict.backend == "judge" and not verdict.approved:
        log["safety_validation"] = "blocked_by_judge"
        note = {"nivel_risco": note.get("nivel_risco"), "resumo": "", "bloqueio": block_message(verdict),
                "acoes_sugeridas": state.get("recommended_actions") or [], "encaminhamentos": state.get("referrals") or [],
                "fontes_citadas": [], "aviso": SAFETY_NOTICE}
        return {"response": note, "audit_log": log}
    return {"audit_log": log}


def persist(state: CaseState) -> dict[str, Any]:
    log = dict(state.get("audit_log") or {})
    try:
        rid = persist_audit_record(dict(state))
        log["persisted_id"] = rid
        return {"audit_log": log, "audit_id": rid}
    except Exception as e:  # noqa: BLE001
        log["persist_error"] = str(e)[:200]
        return {"audit_log": log}


NODES = [
    ("assess_risk", assess_risk), ("run_model", run_model), ("retrieve_protocols", retrieve_protocols),
    ("define_actions", define_actions), ("build_explanation", build_explanation), ("prepare_audit", prepare_audit),
    ("generate_note", generate_note), ("judge", judge), ("persist", persist),
]


def build_graph():
    g = StateGraph(CaseState)
    for name, fn in NODES:
        g.add_node(name, fn)
    g.set_entry_point(NODES[0][0])
    for (a, _), (b, _) in itertools.pairwise(NODES):
        g.add_edge(a, b)
    g.add_edge(NODES[-1][0], END)
    return g.compile()


_GRAPH = None


def run_case(user_input: str, case: dict[str, Any] | None = None, signals: dict[str, Any] | None = None) -> dict[str, Any]:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH.invoke({"user_input": user_input, "case": case or {}, "signals": signals or {}})
