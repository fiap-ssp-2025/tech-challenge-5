"""Guardiã AI — server-rendered web app (FastAPI + Jinja2 + htmx 4).

Every UI route answers the full page on a direct visit and only the fragment when called by htmx
(``HX-Request: true``). Form errors come back as 422 with the form re-rendered. The server is the
only source of truth; the browser keeps no business state.

Triagem (F1/F2): the report box extracts the record live (debounced) with the fast model; the analysis
runs as a background job whose real per-node progress the page polls until the note is ready.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from guardia.audit import store as audit
from guardia.graph.flow import run_case
from guardia.guardrails.blocklist import SAFETY_NOTICE
from guardia.interview import build_guide, parse_ids
from guardia.llm import transcribe
from guardia.llm.structurer import structure_report
from guardia.rag import chat as rag_chat
from guardia.web import jobs
from guardia.web.forms import FORM, SIGNAL_FIELDS, case_facts, case_summary, label, parse_case_form
from guardia.web.status import CACHE, model_status

log = logging.getLogger("guardia.web")
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

@asynccontextmanager
async def _lifespan(_: FastAPI):
    CACHE.start()  # background probes; pages never wait for the LLM endpoints
    if transcribe.backend() == "whisper":  # warm the local speech model so the first dictated chunk is not slow
        threading.Thread(target=transcribe.warm, daemon=True, name="guardia-whisper-warm").start()
    yield


app = FastAPI(title="Guardiã AI", docs_url="/api/docs", redoc_url=None, lifespan=_lifespan)

app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
app.mount("/figures", StaticFiles(directory=str(ROOT / "docs" / "figures"), check_dir=False), name="figures")
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.filters["opt"] = label
def asset_version() -> int:
    """Newest mtime among static files → cache-busting query string (browsers held stale CSS otherwise)."""
    return int(max((p.stat().st_mtime for p in (HERE / "static").glob("*")), default=0))


templates.env.globals.update(FORM=FORM, SIGNAL_FIELDS=SIGNAL_FIELDS, case_summary=case_summary, asset_version=asset_version,
                             SAFETY_NOTICE=SAFETY_NOTICE,
                             json_dumps=lambda v: json.dumps(v, ensure_ascii=False, indent=2, default=str))

NAV = [("/", "Triagem", "triagem"), ("/historico", "Histórico", "historico")]  # Protocolos: painel lateral em todas as páginas
MIN_RELATO = 20


def is_htmx(request: Request) -> bool:
    """Fragment requests only: boosted navigation targets the whole body (``HX-Request-Type: full``)."""
    return request.headers.get("HX-Request") == "true" and request.headers.get("HX-Request-Type") != "full"


def render(request: Request, page: str, partial: str | None, ctx: dict[str, Any], status: int = 200) -> HTMLResponse:
    """Full page for direct access, fragment for htmx. Both share ``ctx``."""
    base = {"request": request, "nav": NAV, "active": ctx.get("active", ""), "model": model_status()}
    name = partial if (partial and is_htmx(request)) else page
    resp = templates.TemplateResponse(request, name, {**base, **ctx}, status_code=status)
    resp.headers["Vary"] = "HX-Request"
    return resp


def _ficha_ctx(relato: str = "", case: dict | None = None, signals: dict | None = None, errors: dict | None = None,
               structured: dict | None = None, edited: bool = False, guide: dict | None = None) -> dict[str, Any]:
    case = case or {}
    detected = [text for _, name, text in FORM.checks if case.get(name)]
    detected += [text for name, text in SIGNAL_FIELDS if (signals or {}).get(name)]
    return {"active": "triagem", "relato": relato, "case": case, "signals": signals or {}, "errors": errors or {},
            "structured": structured, "detected": detected, "ficha_editada": edited, "speech_fallback": transcribe.available(),
            "guide": guide or build_guide(case, signals, set(), set())}


def _form_signals(form: dict[str, Any]) -> dict[str, bool]:
    return {k: bool(form.get(k)) for k, _ in SIGNAL_FIELDS}


def _form_guide(form: dict[str, Any], mentioned: set[str] | None = None, case: dict | None = None,
                signals: dict | None = None) -> dict[str, Any]:
    """Interview guide from the posted form: hand-edited record + topics seen by the LLM + questions marked as asked."""
    if case is None:
        case, signals = parse_case_form(form).case, _form_signals(form)
    return build_guide(case, signals, parse_ids(form.get("guia_mencionados")) | (mentioned or set()), parse_ids(form.get("guia_feitos")))


# ── F1 · Triagem ───────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def triagem(request: Request):
    return render(request, "pages/triagem.html", None, _ficha_ctx())


@app.post("/guia", response_class=HTMLResponse)
async def guia(request: Request):
    """Interview guide recomputed from the form (record edited by hand, or a question marked/unmarked as asked)."""
    form = dict(await request.form())
    done = parse_ids(form.get("guia_feitos")) ^ parse_ids(form.get("feito"))  # clicking toggles
    form["guia_feitos"] = ",".join(done)
    return templates.TemplateResponse(request, "partials/guia.html", {"request": request, "guide": _form_guide(form)})


@app.post("/estruturar", response_class=HTMLResponse)
async def estruturar(request: Request):
    """F2: fill the record from the free-text report. ``live=1`` → fast model, debounced from the textbox.
    Returns the record fragment; a 422 when the text is too short."""
    form = dict(await request.form())
    relato = str(form.get("relato") or "").strip()
    live = str(form.get("live") or "") == "1"
    edited = str(form.get("ficha_editada") or "") == "1"
    if live and edited and len(relato) >= MIN_RELATO:
        # the professional already adjusted the record by hand: do not overwrite it, only refresh the interview guide
        st = structure_report(relato, fast=True)
        if not st.ok:
            return HTMLResponse("", status_code=204)
        guide = _form_guide(form, set(st.topics))
        return templates.TemplateResponse(request, "partials/guia.html", {"request": request, "guide": guide, "oob": True})
    if len(relato) < MIN_RELATO:
        if live:
            return HTMLResponse("", status_code=204)
        ctx = _ficha_ctx(relato, errors={"relato": "Escreva o relato (mínimo 20 caracteres) antes de extrair os campos."})
        return render(request, "pages/triagem.html", "partials/ficha.html", ctx, status=422)
    t0 = time.monotonic()
    st = structure_report(relato, fast=live)
    if not st.ok:
        ctx = _ficha_ctx(relato, errors={"estruturar": f"A LLM não respondeu ({st.error}). Preencha os campos manualmente."},
                         guide=_form_guide(form))
        return render(request, "pages/triagem.html", "partials/ficha_live.html", ctx)
    signals = {"arma_mencionada": st.signals.get("arma_mencionada"), "ameaca_de_morte": st.signals.get("ameaca_de_morte"),
               "filhos_envolvidos": st.signals.get("filhos_envolvidos"), "sem_local_seguro": st.signals.get("local_seguro") is False}
    structured = {"summary": st.summary, "uncertain": st.uncertain, "model": st.model, "backend": st.backend,
                  "latency_ms": int((time.monotonic() - t0) * 1000), "live": live}
    guide = _form_guide(form, set(st.topics), st.case, signals)
    return render(request, "pages/triagem.html", "partials/ficha_live.html", _ficha_ctx(relato, st.case, signals, structured=structured, guide=guide))


@app.post("/analisar", response_class=HTMLResponse)
async def analisar(request: Request):
    """Validate, start the background job and return the progress fragment (polls /analisar/{id})."""
    form = dict(await request.form())
    parsed = parse_case_form(form)
    if not parsed.ok:
        signals = {k: bool(form.get(k)) for k, _ in SIGNAL_FIELDS}
        ctx = _ficha_ctx(parsed.user_input, parsed.case, signals, errors=parsed.errors, edited=True)
        resp = render(request, "pages/triagem.html", "partials/ficha.html", ctx, status=422)
        resp.headers["HX-Retarget"] = "#ficha"
        resp.headers["HX-Reswap"] = "outerHTML"
        return resp
    job = jobs.start(parsed.user_input, parsed.case, parsed.signals)
    return render(request, "pages/triagem.html", "partials/progress.html", {"active": "triagem", "job": job, "relato": parsed.user_input})


@app.get("/analisar/{job_id}", response_class=HTMLResponse)
def analisar_progresso(request: Request, job_id: str):
    job = jobs.get(job_id)
    if job is None:
        return render(request, "pages/triagem.html", "partials/error.html", {"active": "triagem", "message": "Análise não encontrada (o servidor reiniciou?). Envie de novo."}, status=404)
    if job.status == "error":
        return render(request, "pages/triagem.html", "partials/error.html", {"active": "triagem", "message": f"Falha na análise: {job.error}"}, status=500)
    if job.status == "done":
        return render(request, "pages/triagem.html", "partials/result.html", {"active": "triagem", "s": job.state, "relato": job.user_input,
                                                                             "steps": [s.as_dict() for s in job.steps], "elapsed_s": job.elapsed_s})
    return render(request, "pages/triagem.html", "partials/progress.html", {"active": "triagem", "job": job, "relato": job.user_input})


@app.post("/transcrever")
async def transcrever(audio: UploadFile, skip_ms: int = Form(0)):
    """Speech-to-text fallback for browsers without the Web Speech API (audio model via OpenRouter, whisper as reserve).
    ``skip_ms`` trims the audio carried by the webm header that the browser prepends to later chunks."""
    if not transcribe.available():
        return JSONResponse({"error": "transcrição no servidor indisponível (configure TRANSCRIBE_*/FAST_* ou instale com `uv sync --group audio`)"}, status_code=503)
    data = await audio.read()
    if len(data) < 2000:
        return JSONResponse({"text": ""})
    suffix = ".webm" if "webm" in (audio.content_type or "") else (".ogg" if "ogg" in (audio.content_type or "") else ".wav")
    try:
        return JSONResponse(transcribe.transcribe_bytes(data, suffix=suffix, skip_s=max(0, min(skip_ms, 2000)) / 1000))
    except Exception as e:
        log.exception("transcription failed")
        return JSONResponse({"error": f"{type(e).__name__}: {str(e)[:200]}"}, status_code=500)


# ── F3 · Protocolos (chat com memória no servidor) ─────────────────────────────


CHAT_COOKIE = {"max_age": 60 * 60 * 24 * 7, "samesite": "lax"}


def _chat(request: Request, partial: str, ctx: dict[str, Any], cid: str, status: int = 200) -> HTMLResponse:
    """Chat fragments for the side panel; a direct visit gets the triage page with the panel already open."""
    page_ctx = {**_ficha_ctx(), "chat_open": True} if not is_htmx(request) else {}
    if not is_htmx(request) or partial.endswith("chat_thread.html"):
        page_ctx["turns"] = rag_chat.turns(cid)
    resp = render(request, "pages/triagem.html", partial, {**page_ctx, **ctx}, status=status)
    resp.set_cookie("guardia_chat", cid, **CHAT_COOKIE)
    return resp


@app.get("/protocolos", response_class=HTMLResponse)
def protocolos(request: Request):
    """The conversation thread (loaded when the panel first opens)."""
    return _chat(request, "partials/chat_thread.html", {}, request.cookies.get("guardia_chat") or rag_chat.new_conversation_id())


def _chat_case(form: dict[str, Any]) -> list[str]:
    """Context for the protocol chat (spec 004): the record fields of the triage on screen and, when it was already
    analysed, risk, model suggestion and referrals from the stored note. Never the report text."""
    if str(form.get("usar_caso") or "") != "1":
        return []
    facts = case_facts(parse_case_form(form).case, _form_signals(form))
    aid = str(form.get("analise_id") or "")
    row = audit.fetch_one(int(aid)) if aid.isdigit() else None
    if row:
        facts.append(f"Última análise: risco {row.get('risk_level') or 'indeterminado'}")
        if row.get("ml_suggest_referral"):
            facts.append("Modelo sugere encaminhar à rede de proteção")
        facts += [f"Encaminhamento sugerido: {e}" for e in ((row.get("response_json") or {}).get("encaminhamentos") or [])[:3]]
    return facts


@app.post("/protocolos/perguntar", response_class=HTMLResponse)
async def perguntar(request: Request):
    form = dict(await request.form())
    cid = request.cookies.get("guardia_chat") or str(form.get("cid") or rag_chat.new_conversation_id())
    q = str(form.get("pergunta") or "").strip()
    if len(q) < 5:
        return _chat(request, "partials/chat_error.html", {"message": "Escreva uma pergunta com pelo menos 5 caracteres."}, cid, status=422)
    try:
        turn = rag_chat.ask(cid, q, case=_chat_case(form))
    except Exception as e:
        log.exception("chat failed")
        return _chat(request, "partials/chat_error.html", {"message": f"Falha: {type(e).__name__}: {e}"}, cid, status=500)
    return _chat(request, "partials/chat_turn.html", {"t": turn}, cid)


@app.post("/protocolos/nova", response_class=HTMLResponse)
def nova_conversa(request: Request):
    """Recomeçar: a new conversation id; the panel gets the empty thread back."""
    cid = rag_chat.new_conversation_id()
    if not is_htmx(request):
        resp = RedirectResponse("/protocolos", status_code=303)
        resp.set_cookie("guardia_chat", cid, **CHAT_COOKIE)
        return resp
    return _chat(request, "partials/chat_thread.html", {}, cid)


# ── F4 · Histórico (linha do tempo dos atendimentos) ───────────────────────────


@app.get("/historico", response_class=HTMLResponse)
def historico(request: Request, page: int = 1):
    per = 12
    rows = audit.fetch_recent(limit=per + 1, offset=(page - 1) * per)
    has_more = len(rows) > per
    ctx = {"active": "historico", "rows": rows[:per], "page": page, "has_more": has_more, "stats": audit.stats()}
    return render(request, "pages/historico.html", "partials/historico_cards.html", ctx)


@app.get("/historico/{analysis_id}", response_class=HTMLResponse)
def historico_detalhe(request: Request, analysis_id: int):
    row = audit.fetch_one(analysis_id)
    if not row:
        return render(request, "pages/historico_detalhe.html", None, {"active": "historico", "row": None, "analysis_id": analysis_id}, status=404)
    steps = row.get("steps") or []
    return render(request, "pages/historico_detalhe.html", None, {"active": "historico", "row": row, "steps": steps,
                                                                 "total_ms": row.get("total_ms") or sum((s.get("duration_ms") or 0) for s in steps)})


@app.get("/auditoria")
def auditoria_redirect():
    return RedirectResponse("/historico", status_code=301)


@app.get("/auditoria/{analysis_id}")
def auditoria_detalhe_redirect(analysis_id: int):
    return RedirectResponse(f"/historico/{analysis_id}", status_code=301)


# ── Status (só o indicador do cabeçalho; sem página própria) ───────────────────


@app.get("/status/pill", response_class=HTMLResponse)
def status_pill(request: Request):
    return templates.TemplateResponse(request, "partials/status_pill.html", {"request": request, "st": CACHE.snapshot()})


@app.get("/api/health")
def health():
    return JSONResponse({"ok": True, "model": model_status().get("ok", False)})


@app.get("/api/status")
def api_status():
    st = CACHE.snapshot()
    st["speech_fallback"] = transcribe.available()
    return JSONResponse(json.loads(json.dumps(st, default=str)))


@app.post("/api/analisar")
async def api_analisar(request: Request):
    """JSON API for integrations and tests (same pipeline as the UI, synchronous)."""
    body = await request.json()
    text = str(body.get("relato") or "")
    if len(text) < MIN_RELATO:
        return JSONResponse({"error": "relato muito curto"}, status_code=422)
    state = run_case(text, body.get("case") or {}, body.get("signals") or {})
    keep = {k: state.get(k) for k in ("risk_level", "risk_factors", "ml", "protocol_sources", "recommended_actions", "referrals",
                                      "response", "llm_metadata", "audit_id")}
    return JSONResponse(json.loads(json.dumps(keep, default=str)))
