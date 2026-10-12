"""Form parsing + validation for the case form (server is the single source of truth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guardia.data.features import AG, CATEGORICAL, ESCOLARIDADE_LABELS, REL_GROUPS, UF_OPTIONS, VIOL

OPTION_LABELS: dict[str, str] = {
    "SIM": "Sim", "NAO": "Não", "IGNORADO": "Não informado", "MASCULINO": "Masculino", "FEMININO": "Feminino", "AMBOS": "Ambos",
    "UM": "Um", "DOIS_OU_MAIS": "Dois ou mais", "BRANCA": "Branca", "PRETA": "Preta", "AMARELA": "Amarela", "PARDA": "Parda",
    "INDIGENA": "Indígena", "CRIANCA": "Criança", "ADOLESCENTE": "Adolescente", "JOVEM": "Jovem", "ADULTA": "Adulta", "IDOSA": "Idosa",
    "SOLTEIRA": "Solteira", "CASADA_UNIAO": "Casada / união estável", "VIUVA": "Viúva", "SEPARADA": "Separada",
    "NAO_SE_APLICA": "Não se aplica", "1_TRIMESTRE": "1º trimestre", "2_TRIMESTRE": "2º trimestre", "3_TRIMESTRE": "3º trimestre",
    "IDADE_GEST_IGNORADA": "Gestante, idade gestacional ignorada", "HETEROSSEXUAL": "Heterossexual", "HOMOSSEXUAL": "Homossexual",
    "BISSEXUAL": "Bissexual", "TRAVESTI": "Travesti", "MULHER_TRANS": "Mulher trans", "HOMEM_TRANS": "Homem trans",
    "RESIDENCIA": "Residência", "HABITACAO_COLETIVA": "Habitação coletiva", "ESCOLA": "Escola", "LOCAL_PRATICA_ESPORTIVA": "Local de prática esportiva",
    "BAR_OU_SIMILAR": "Bar ou similar", "VIA_PUBLICA": "Via pública", "COMERCIO_SERVICOS": "Comércio / serviços",
    "INDUSTRIAS_CONSTRUCAO": "Indústria / construção", "OUTRO": "Outro", "SEXISMO": "Sexismo", "LGBTFOBIA": "LGBTfobia",
    "RACISMO": "Racismo", "INTOLERANCIA_RELIGIOSA": "Intolerância religiosa", "XENOFOBIA": "Xenofobia",
    "CONFLITO_GERACIONAL": "Conflito geracional", "SITUACAO_DE_RUA": "Situação de rua", "DEFICIENCIA": "Deficiência", "OUTROS": "Outros",
    "FUNDAMENTAL_INCOMPLETO": "Fundamental incompleto", "ENSINO_MEDIO_INCOMPLETO": "Médio incompleto",
    "ENSINO_MEDIO_COMPLETO": "Médio completo", "ENSINO_SUPERIOR": "Superior",
}


def label(opt: str) -> str:
    if opt in UF_OPTIONS:  # "UM" (one person involved) is not a state
        return opt
    return OPTION_LABELS.get(opt, opt.replace("_", " ").title())


@dataclass
class FormSpec:
    checks: list[tuple[str, str, str]]           # (group, name, label)
    selects: list[tuple[str, str, list[tuple[str, str]]]]  # (name, label, [(value, label)])

    @staticmethod
    def build() -> FormSpec:
        checks = [("Tipo de violência", n, l) for n, l in VIOL.items()]
        checks += [("Meio de agressão", n, l) for n, l in AG.items()]
        checks += [("Relação com o provável autor", n, l) for n, (l, _) in REL_GROUPS.items()]
        selects = [(n, l, [(v, label(v)) for v in list(m.values()) + ["IGNORADO"]]) for n, (l, m) in CATEGORICAL.items()]
        selects.append(("SG_UF", "UF de residência", [(u, u) for u in UF_OPTIONS] + [("IGNORADO", "Não informada")]))
        selects.append(("ESCOLARIDADE", "Escolaridade", [(v, label(v)) for v in ESCOLARIDADE_LABELS.values()] + [("", "Não informada")]))
        return FormSpec(checks=checks, selects=selects)


FORM = FormSpec.build()
SIGNAL_FIELDS = [("arma_mencionada", "Arma mencionada no relato"), ("ameaca_de_morte", "Ameaça de morte"),
                 ("filhos_envolvidos", "Filhos ou dependentes envolvidos"), ("sem_local_seguro", "Sem local seguro disponível")]


@dataclass
class ParsedForm:
    user_input: str = ""
    case: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_case_form(form: dict[str, Any]) -> ParsedForm:
    """`form` is a flat mapping (Starlette FormData → dict). Checkboxes arrive only when checked."""
    out = ParsedForm(user_input=str(form.get("relato") or "").strip())
    if len(out.user_input) < 20:
        out.errors["relato"] = "Descreva o relato com pelo menos 20 caracteres."
    if len(out.user_input) > 6000:
        out.errors["relato"] = "O relato deve ter no máximo 6000 caracteres."
    case: dict[str, Any] = {}
    for _, name, _ in FORM.checks:
        case[name] = 1 if form.get(name) else 0
    for name, _, options in FORM.selects:
        val = str(form.get(name) or "").upper()
        allowed = {v for v, _ in options}
        if name == "ESCOLARIDADE":
            case[name] = val if val in allowed and val else None
        else:
            case[name] = val if val in allowed else "IGNORADO"
    idade_raw = str(form.get("IDADE_ANOS") or "").strip()
    if idade_raw:
        try:
            idade = int(idade_raw)
            if not 0 <= idade <= 120:
                raise ValueError
            case["IDADE_ANOS"] = idade
        except ValueError:
            out.errors["IDADE_ANOS"] = "Idade deve ser um número entre 0 e 120."
    else:
        case["IDADE_ANOS"] = None
    out.case = case
    out.signals = {
        "arma_mencionada": bool(form.get("arma_mencionada")),
        "ameaca_de_morte": bool(form.get("ameaca_de_morte")),
        "filhos_envolvidos": bool(form.get("filhos_envolvidos")),
        "local_seguro": False if form.get("sem_local_seguro") else None,
    }
    return out


def case_summary(case: dict[str, Any]) -> list[tuple[str, str]]:
    """Human-readable (label, value) pairs of the informative fields, for the audit view."""
    names = {**VIOL, **AG, **{k: v[0] for k, v in REL_GROUPS.items()}, **{k: v[0] for k, v in CATEGORICAL.items()},
             "SG_UF": "UF", "ESCOLARIDADE": "Escolaridade", "IDADE_ANOS": "Idade"}
    pairs = []
    for k, v in (case or {}).items():
        if v in (None, "", 0, "IGNORADO", False):
            continue
        pairs.append((names.get(k, k), "sim" if v in (1, True) else label(str(v))))
    return pairs


# Short names for the protocol chat context; the SINAN labels are written as form questions.
FACT_NAMES = {"OUT_VEZES": "Já ocorreu outras vezes", "AUTOR_ALCO": "Autor com suspeita de uso de álcool", "AUTOR_SEXO": "Sexo do autor",
              "NUM_ENVOLV": "Envolvidos", "DEF_TRANS": "Deficiência ou transtorno", "SG_UF": "UF", "IDADE_ANOS": "Idade"}


def case_facts(case: dict[str, Any], signals: dict[str, Any] | None = None) -> list[str]:
    """The record as short readable facts ("Violência física", "Gestante: 2º trimestre") for the protocol chat.
    Structured fields only: the free-text report never leaves the triage."""
    facts = [text for _, name, text in FORM.checks if (case or {}).get(name)]
    facts = [f.split(" (")[0] for f in facts]
    facts += [text for name, text in SIGNAL_FIELDS if (signals or {}).get(name)]
    for name, text, _ in FORM.selects:
        v = (case or {}).get(name)
        if v not in (None, "", "IGNORADO", "NAO_SE_APLICA"):
            facts.append(f"{FACT_NAMES.get(name, text)}: {label(str(v))}")
    if (case or {}).get("IDADE_ANOS") not in (None, ""):
        facts.append(f"Idade: {case['IDADE_ANOS']} anos")
    return facts
