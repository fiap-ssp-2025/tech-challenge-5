"""Interview guide: the few questions still open while the report is being told (spec 003, U5–U6).

Topics follow the Formulário Nacional de Avaliação de Risco, ordered by risk priority. A topic is covered when the
record already has the field (filled by the LLM or by hand), when the live extraction says the report touched it
(even with a "no"), or when the professional marked it as asked.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from guardia.data.features import REL_GROUPS, VIOL


def _informed(case: dict[str, Any], name: str) -> bool:
    # NAO_SE_APLICA is what the extraction picks by default (e.g. pregnancy for an adult): it answers nothing
    return case.get(name) not in (None, "", "IGNORADO", "NAO_SE_APLICA")


@dataclass(frozen=True)
class Topic:
    id: str
    label: str
    question: str  # written to be read aloud to the woman
    hint: str      # what the extraction model should look for
    in_record: Callable[[dict[str, Any], dict[str, Any]], bool]


TOPICS: list[Topic] = [
    Topic("autor", "Autor", "Quem fez isso com você?", "quem agrediu e a relação com ela",
          lambda c, s: any(c.get(k) for k in REL_GROUPS)),
    Topic("tipo", "O que aconteceu", "O que aconteceu exatamente?", "o tipo de violência sofrida",
          lambda c, s: any(c.get(k) for k in VIOL)),
    Topic("ameaca", "Ameaça de morte", "Já ameaçou matar você?", "se houve ou não ameaça de morte",
          lambda c, s: bool(s.get("ameaca_de_morte"))),
    Topic("arma", "Arma", "Tem acesso a arma ou faca? Já usou?", "se há ou não arma de fogo ou branca",
          lambda c, s: bool(s.get("arma_mencionada") or c.get("AG_FOGO") or c.get("AG_CORTE"))),
    Topic("recorrencia", "Recorrência", "Já aconteceu antes? Está piorando?", "se já ocorreu outras vezes ou está se agravando",
          lambda c, s: _informed(c, "OUT_VEZES")),
    Topic("local_seguro", "Local seguro", "Você tem um lugar seguro para ficar hoje?", "se ela tem ou não onde ficar em segurança",
          lambda c, s: bool(s.get("sem_local_seguro"))),
    Topic("filhos", "Filhos", "Tem filhos ou dependentes com você?", "se há ou não filhos ou dependentes",
          lambda c, s: bool(s.get("filhos_envolvidos"))),
    Topic("gestacao", "Gestação", "Você está grávida?", "se ela está ou não grávida",
          lambda c, s: _informed(c, "CS_GESTANT")),
    Topic("alcool", "Álcool ou drogas", "A pessoa que agrediu bebe ou usa drogas?", "uso de álcool ou drogas pelo autor",
          lambda c, s: _informed(c, "AUTOR_ALCO")),
    Topic("medida_protetiva", "Medida protetiva", "Você já tem ou já pediu medida protetiva?", "medida protetiva atual ou pedida antes",
          lambda c, s: False),
    Topic("idade", "Idade", "Quantos anos você tem?", "a idade ou faixa etária dela",
          lambda c, s: _informed(c, "IDADE_ANOS") or _informed(c, "CICL_VID")),
]
TOPIC_IDS = [t.id for t in TOPICS]
SHOW_NEXT = 3


def parse_ids(raw: Any) -> set[str]:
    """Comma-separated ids from a hidden form field → known topic ids."""
    return {x for x in str(raw or "").split(",") if x in TOPIC_IDS}


def build_guide(case: dict[str, Any] | None, signals: dict[str, Any] | None, mentioned: set[str], done: set[str]) -> dict[str, Any]:
    """`signals` uses the web form names (``sem_local_seguro``). Returns the context of ``partials/guia.html``."""
    case, signals = case or {}, signals or {}
    items = []
    for t in TOPICS:
        state = "ficha" if t.in_record(case, signals) else "relato" if t.id in mentioned else "feito" if t.id in done else "pendente"
        items.append({"id": t.id, "label": t.label, "question": t.question, "state": state})
    pending = [i for i in items if i["state"] == "pendente"]
    return {"topics": items, "next": pending[:SHOW_NEXT], "covered": len(items) - len(pending), "total": len(items),
            "started": len(pending) < len(items), "mentioned": ",".join(sorted(mentioned)), "done": ",".join(sorted(done))}
