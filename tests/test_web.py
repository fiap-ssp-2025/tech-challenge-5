"""Web routes without any LLM (the endpoints are unset → graceful error notes)."""

import os
import re
import time

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GUARDIA_DB_PATH", "data/generated/test_audit.db")
os.environ["LLM_BASE_URL"] = ""
os.environ["JUDGE_BASE_URL"] = ""
os.environ["OPENAI_API_KEY"] = ""

from guardia.web.app import app

client = TestClient(app)
RELATO = "Ela conta que o companheiro a ameaçou com uma faca ontem à noite. Não é a primeira vez e há dois filhos pequenos."


@pytest.mark.parametrize("path", ["/", "/protocolos", "/historico", "/status/pill", "/api/health"])
def test_pages_render(path):
    r = client.get(path)
    assert r.status_code == 200


def test_auditoria_redirects_to_historico():
    r = client.get("/auditoria", follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "/historico"


def test_fragment_vs_full_page():
    full = client.get("/historico").text
    frag = client.get("/historico", headers={"HX-Request": "true"}).text
    assert "<html" in full and "<html" not in frag


def test_form_validation_returns_422_with_record():
    r = client.post("/analisar", data={"relato": "curto"}, headers={"HX-Request": "true"})
    assert r.status_code == 422
    assert 'id="ficha"' in r.text and r.headers.get("HX-Retarget") == "#ficha"


def test_live_extraction_is_silent_when_text_is_short_and_keeps_an_edited_record():
    assert client.post("/estruturar", data={"relato": "curto", "live": "1"}, headers={"HX-Request": "true"}).status_code == 204
    # record edited by hand: never overwritten; at most the interview guide comes back out of band
    r = client.post("/estruturar", data={"relato": RELATO, "live": "1", "ficha_editada": "1"}, headers={"HX-Request": "true"})
    assert r.status_code in (200, 204) and 'id="ficha"' not in r.text
    if r.status_code == 200:
        assert 'id="guia"' in r.text and 'hx-swap-oob="true"' in r.text


def test_triage_shows_first_interview_questions():
    html = client.get("/").text
    assert 'id="guia"' in html and "Para começar" in html and "Quem fez isso com você?" in html


def test_guide_follows_record_and_toggles_asked_questions():
    form = {"relato": RELATO, "REL_PARCEIRO_INTIMO": "on", "ameaca_de_morte": "on", "guia_mencionados": "tipo", "guia_feitos": ""}
    r = client.post("/guia", data=form, headers={"HX-Request": "true"})
    assert r.status_code == 200 and "Falta perguntar" in r.text
    assert "Quem fez isso com você?" not in r.text and "Já ameaçou matar você?" not in r.text  # covered by the record
    assert "Tem acesso a arma ou faca? Já usou?" in r.text and "3 de 11" in r.text
    r = client.post("/guia", data={**form, "feito": "arma"}, headers={"HX-Request": "true"})
    assert 'name="guia_feitos" value="arma"' in r.text and "4 de 11" in r.text
    r = client.post("/guia", data={**form, "guia_feitos": "arma", "feito": "arma"}, headers={"HX-Request": "true"})
    assert 'name="guia_feitos" value=""' in r.text  # clicking again unmarks


def test_protocol_panel_fragment_vs_page_and_restart():
    frag = client.get("/protocolos", headers={"HX-Request": "true"}).text
    assert "<html" not in frag and "chat-empty" in frag
    page = client.get("/protocolos").text
    assert "<html" in page and '<dialog id="chat-drawer"' in page and "open>" in page.split('<dialog id="chat-drawer"')[1][:200]
    boosted = client.get("/historico", headers={"HX-Request": "true", "HX-Request-Type": "full"}).text
    assert "<html" in boosted and 'id="chat-fab"' in boosted
    r = client.post("/protocolos/nova", headers={"HX-Request": "true"})
    assert r.status_code == 200 and "chat-empty" in r.text and "guardia_chat" in r.headers.get("set-cookie", "")
    assert client.post("/protocolos/perguntar", data={"pergunta": "oi"}, headers={"HX-Request": "true"}).status_code == 422


def _poll(job_id: str, timeout: float = 60.0) -> str:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/analisar/{job_id}", headers={"HX-Request": "true"})
        assert r.status_code == 200, r.text[:200]
        if "Atendimento #" in r.text or "Não foi possível" in r.text:
            return r.text
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_full_analysis_without_llm_runs_as_job_and_records_history():
    r = client.post("/analisar", data={"relato": RELATO, "AG_CORTE": "on", "REL_PARCEIRO_INTIMO": "on", "OUT_VEZES": "SIM",
                                       "SG_UF": "DF", "IDADE_ANOS": "29", "arma_mencionada": "on"}, headers={"HX-Request": "true"})
    assert r.status_code == 200 and "Analisando o caso" in r.text
    job_id = re.search(r'hx-get="/analisar/([a-f0-9]+)"', r.text).group(1)
    html = _poll(job_id)
    assert "risco critico" in html and "Atendimento #" in html and "Etapas (9)" in html
    aid = re.search(r"Atendimento #(\d+)", html).group(1)
    detail = client.get(f"/historico/{aid}").text
    assert "etapa por etapa" in detail and "Escrever a nota (LLM)" in detail
    cards = client.get("/historico", headers={"HX-Request": "true"}).text
    assert f"Atendimento #{aid}" in cards


def test_api_analisar_json():
    r = client.post("/api/analisar", json={"relato": RELATO, "case": {"AG_CORTE": 1, "REL_PARCEIRO_INTIMO": 1, "OUT_VEZES": "SIM"}})
    assert r.status_code == 200
    body = r.json()
    assert body["risk_level"] == "critico" and body["ml"]["probability"] >= 0
    assert body["protocol_sources"]


def test_transcrever_without_audio_payload():
    r = client.post("/transcrever", files={"audio": ("a.webm", b"x" * 10, "audio/webm")})
    assert r.status_code in (200, 503)


def test_protocol_chat_sends_record_facts_not_the_report():
    from guardia.rag import chat as rag_chat

    seen = {}

    def fake_ask(cid, question, case=None):
        seen["case"] = case
        return rag_chat.Turn(question=question, answer="ok", backend="test", case_facts=case or [])

    original, rag_chat.ask = rag_chat.ask, fake_ask
    try:
        form = {"pergunta": "Que medidas cabem neste caso?", "usar_caso": "1", "VIOL_SEXU": "on", "REL_PARCEIRO_INTIMO": "on",
                "ameaca_de_morte": "on", "SG_UF": "DF", "relato": RELATO}
        r = client.post("/protocolos/perguntar", data=form, headers={"HX-Request": "true"})
        assert r.status_code == 200 and "Com o caso: Violência sexual, Parceiro íntimo, Ameaça de morte" in r.text
        assert "UF: DF" in seen["case"] and not any("faca" in f for f in seen["case"])  # structured fields only
        r = client.post("/protocolos/perguntar", data={**form, "usar_caso": ""}, headers={"HX-Request": "true"})
        assert seen["case"] == [] and "Com o caso" not in r.text
    finally:
        rag_chat.ask = original
