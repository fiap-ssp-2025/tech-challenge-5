from guardia.graph.flow import assess_risk, build_explanation, define_actions
from guardia.guardrails.blocklist import scan_dict, validate
from guardia.llm.judge import _post_process


def test_assess_risk_escalates_on_weapon_and_repetition():
    out = assess_risk({"user_input": "Ele me ameaçou com uma faca. Tenho medo.", "case": {"OUT_VEZES": "SIM"}, "signals": {}})
    assert out["risk_level"] == "critico"
    assert "Arma branca mencionada" in out["risk_factors"] and "Violência de repetição" in out["risk_factors"]


def test_assess_risk_low_when_nothing_flagged():
    out = assess_risk({"user_input": "Veio pedir orientação sobre documentos.", "case": {}, "signals": {}})
    assert out["risk_level"] == "baixo" and out["risk_factors"] == []


def test_define_actions_mentions_network_for_critical():
    out = define_actions({"risk_level": "critico", "ml": {"suggest_referral": True, "probability": 0.8}, "case": {}})
    joined = " ".join(out["recommended_actions"] + out["referrals"]).lower()
    assert "190" in joined and "deam" in joined and "defensoria" in joined


def test_blocklist_scans_nested_note():
    assert validate("Oriente a buscar a DEAM.")["approved"]
    bad = scan_dict({"resumo": "ok", "acoes": ["Volte para casa e resolva sozinha"]})
    assert not bad["approved"] and "volte para casa" in bad["found_terms"]


def test_judge_post_process_requires_literal_quote():
    checks = {
        "evita_prescricao": {"pass": False, "reason": "prescreve", "quote": "Tome 20mg de fluoxetina"},
        "evita_diagnostico": {"pass": False, "reason": "não afirma diagnóstico", "quote": ""},
        "preserva_privacidade": {"pass": False, "reason": "cita nome", "quote": "trecho inexistente"},
    }
    note = '{"resumo": "Tome 20mg de fluoxetina"}'
    overrides = _post_process(checks, note)
    assert checks["evita_prescricao"]["pass"] is False           # real quote with posology stays failed
    assert checks["evita_diagnostico"]["pass"] is True           # negation in reason → pass
    assert checks["preserva_privacidade"]["pass"] is True        # quote not in note → pass
    assert len(overrides) == 2


def test_explanation_mentions_model():
    out = build_explanation({"risk_level": "alto", "risk_factors": ["x"], "ml": {"model_name": "hgb", "probability": 0.7, "threshold": 0.3}, "protocol_sources": [{"file": "a", "title": "A"}]})
    assert "hgb" in out["explanation"] and "70%" in out["explanation"]


def test_judge_does_not_block_plain_referral_as_decision():
    checks = {"nao_decide_sozinha": {"pass": False, "reason": "afirma que está decidido", "quote": "Encaminhar para Casa da Mulher Brasileira ou CREAS"},
              "evita_prescricao": {"pass": True, "reason": "", "quote": ""}}
    note = '{"acoes": ["Encaminhar para Casa da Mulher Brasileira ou CREAS"]}'
    _post_process(checks, note)
    assert checks["nao_decide_sozinha"]["pass"] is True
    checks2 = {"nao_decide_sozinha": {"pass": False, "reason": "decide", "quote": "O encaminhamento está decidido e deve ser feito pelo sistema"}}
    _post_process(checks2, '{"resumo": "O encaminhamento está decidido e deve ser feito pelo sistema"}')
    assert checks2["nao_decide_sozinha"]["pass"] is False
