import re
from pathlib import Path

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from guardia.rag import retriever

needs_model = pytest.mark.skipif(
    not (Path(HF_HUB_CACHE) / ("models--" + retriever.EMBEDDING_MODEL.replace("/", "--"))).exists(),
    reason="modelo de embeddings não baixado",
)


def test_normalize_joins_pdf_hyphenation_and_wrapped_lines():
    raw = "Recomenda-se que\no atendimento seja realizado até 72 horas da agres\xad\nsão.\n20\nAtendimento Integral"
    assert retriever._normalize(raw) == "Recomenda-se que o atendimento seja realizado até 72 horas da agressão.\n\nAtendimento Integral"


def test_tokens_split_digits_and_drop_accents():
    assert retriever._tokens("Até 72h da agressão") == ["ate", "72", "h", "agress"]


def test_forms_and_table_of_contents_are_noise():
    assert retriever._is_noise("Profilaxia HIV Não ( )\nSim ( )\nHepatite B Não ( )")
    assert retriever._is_noise("Introdução ....... 12\nMarcos Legais ....... 13\nAtendimento ....... 24")
    assert not retriever._is_noise("Realizar profilaxia das DST/AIDS e Hepatite B nas primeiras 72 horas;")


@needs_model
@pytest.mark.parametrize("query", [
    "Violência sexual; Ameaça de morte; Parceiro íntimo\nO que os protocolos indicam para este caso?",
    "O que fazer nas primeiras horas após violência sexual?",
])
def test_sexual_violence_queries_reach_the_72_hours_passage(query):
    hits = retriever.search(query, k=6)
    assert any(h.source == "ms_violencia_sexual.txt" and re.search(r"72 ?h", h.text) for h in hits)


@needs_model
def test_short_documents_are_not_drowned_by_the_long_norm():
    hits = retriever.search("Quais medidas protetivas podem ser aplicadas ao agressor?", k=6)
    assert hits[0].source == "medidas_protetivas.txt"
    assert sum(h.source == "ms_violencia_sexual.txt" for h in hits) <= 2
