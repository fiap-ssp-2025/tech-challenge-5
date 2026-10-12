"""Interview guide: coverage by record, by report mention and by hand."""

from guardia.interview import SHOW_NEXT, TOPIC_IDS, build_guide, parse_ids


def test_empty_case_starts_with_the_first_topics():
    g = build_guide({}, {}, set(), set())
    assert not g["started"] and g["covered"] == 0 and g["total"] == len(TOPIC_IDS)
    assert [t["id"] for t in g["next"]] == TOPIC_IDS[:SHOW_NEXT]


def test_record_mention_and_hand_mark_all_cover_a_topic():
    g = build_guide({"VIOL_FISIC": 1, "OUT_VEZES": "SIM"}, {"arma_mencionada": True}, {"local_seguro"}, {"autor"})
    state = {t["id"]: t["state"] for t in g["topics"]}
    assert state["tipo"] == state["recorrencia"] == state["arma"] == "ficha"
    assert state["local_seguro"] == "relato" and state["autor"] == "feito" and state["ameaca"] == "pendente"
    assert g["next"][0]["id"] == "ameaca" and g["covered"] == 5


def test_ignored_values_do_not_count_and_unknown_ids_are_dropped():
    g = build_guide({"OUT_VEZES": "IGNORADO", "IDADE_ANOS": None}, {}, set(), set())
    assert {t["id"]: t["state"] for t in g["topics"]}["recorrencia"] == "pendente"
    assert parse_ids("arma,xyz,,idade") == {"arma", "idade"}


def test_case_facts_are_short_and_skip_unknowns():
    from guardia.web.forms import case_facts

    facts = case_facts({"VIOL_FISIC": 1, "REL_PARCEIRO_INTIMO": 1, "OUT_VEZES": "SIM", "CS_GESTANT": "NAO_SE_APLICA", "SG_UF": "IGNORADO",
                        "IDADE_ANOS": 34}, {"arma_mencionada": True})
    assert facts == ["Violência física", "Parceiro íntimo", "Arma mencionada no relato", "Já ocorreu outras vezes: Sim", "Idade: 34 anos"]


def test_chat_table_gains_case_context_column(tmp_path, monkeypatch):
    import sqlite3

    from guardia.rag import chat as rag_chat

    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:  # schema before the case context column
        conn.execute("CREATE TABLE chat_turns (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, ts TEXT NOT NULL, "
                     "question TEXT NOT NULL, answer TEXT, sources TEXT, backend TEXT, latency_ms INTEGER)")
        conn.execute("INSERT INTO chat_turns (conversation_id, ts, question) VALUES ('c1', 'now', 'pergunta antiga')")
    monkeypatch.setattr(rag_chat, "DB_PATH", db)
    turns = rag_chat.turns("c1")
    assert turns[0]["case_facts"] == [] and "case_context" in turns[0]


def test_topics_need_a_literal_quote_from_the_report():
    from guardia.llm.structurer import _quoted_topics

    text = "Ela disse que NÃO tem filhos e que ele   ameaçou matá-la."
    parsed = {"topicos_abordados": ["filhos: não tem filhos", 'ameaca: "ele ameaçou matá-la"', "arma: ele tem uma faca",
                                    "gestacao:", "idade", "xyz: não tem filhos", "filhos: não tem filhos"]}
    assert _quoted_topics(parsed, text) == ["filhos", "ameaca"]
    assert _quoted_topics({"topicos_abordados": {"filhos": "não tem filhos", "arma": "faca"}}, text) == ["filhos"]


def test_not_applicable_does_not_answer_a_topic():
    g = build_guide({"CS_GESTANT": "NAO_SE_APLICA"}, {}, set(), set())
    assert {t["id"]: t["state"] for t in g["topics"]}["gestacao"] == "pendente"
