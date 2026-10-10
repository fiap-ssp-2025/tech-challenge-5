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


@pytest.mark.parametrize("path", ["/", "/protocolos", "/historico", "/modelo", "/status", "/api/health"])
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


def test_live_extraction_is_silent_when_text_is_short_or_record_edited():
    assert client.post("/estruturar", data={"relato": "curto", "live": "1"}, headers={"HX-Request": "true"}).status_code == 204
    assert client.post("/estruturar", data={"relato": RELATO, "live": "1", "ficha_editada": "1"}, headers={"HX-Request": "true"}).status_code == 204


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
