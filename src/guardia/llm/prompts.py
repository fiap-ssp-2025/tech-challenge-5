"""Prompts and JSON schemas for the generator (response + report structuring)."""

from __future__ import annotations

from typing import Any

from guardia.data.features import AG, CATEGORICAL, ESCOLARIDADE_LABELS, REL_GROUPS, UF_OPTIONS, VIOL
from guardia.interview import TOPICS

SYSTEM_PROMPT = (
    "Você é a Guardiã AI, um sistema de apoio a profissionais que atendem mulheres em situação de violência "
    "(assistência social, enfermagem, psicologia, delegacias e centros de referência). Você fala com a PROFISSIONAL, "
    "nunca com a mulher atendida.\n\n"
    "Regras invioláveis:\n"
    "- Você NÃO decide encaminhamentos, NÃO fecha diagnóstico, NÃO prescreve medicamentos e NÃO julga a mulher.\n"
    "- A decisão é sempre da profissional. Use linguagem técnica, acolhedora e sem culpabilização.\n"
    "- Baseie-se nos trechos de protocolo fornecidos e cite a fonte pelo título quando usar um trecho. "
    "Se os trechos não cobrirem um ponto, diga que não há protocolo recuperado para ele.\n"
    "- Em risco iminente (arma, ameaça de morte, cárcere), oriente acionar a Polícia Militar (190) e a DEAM imediatamente.\n"
    "- Quando faltar informação, aponte o que precisa ser perguntado em vez de dar recomendação definitiva.\n"
    "- Não reproduza nomes, endereços ou documentos da mulher ou do agressor.\n"
    "- Responda sempre em português do Brasil e sempre no formato JSON pedido."
)

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "nivel_risco": {"type": "string", "enum": ["baixo", "moderado", "alto", "critico"]},
        "resumo": {"type": "string", "description": "3 a 5 frases para a profissional: o que o relato traz e o que pesa."},
        "fatores_de_atencao": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "leitura_do_modelo": {"type": "string", "description": "1 a 3 frases explicando a probabilidade e os fatores SHAP em linguagem simples."},
        "acoes_sugeridas": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 6},
        "encaminhamentos": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "perguntas_pendentes": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "fontes_citadas": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "aviso": {"type": "string"},
    },
    "required": ["nivel_risco", "resumo", "fatores_de_atencao", "leitura_do_modelo", "acoes_sugeridas",
                 "encaminhamentos", "perguntas_pendentes", "fontes_citadas", "aviso"],
}


def _fmt_list(items: list[str] | None, empty: str = "- (nenhum)") -> str:
    return "\n".join(f"- {i}" for i in items) if items else empty


def build_response_prompt(state: dict[str, Any]) -> str:
    ml = state.get("ml") or {}
    case = state.get("case") or {}
    case_lines = []
    for k, v in case.items():
        if v in (None, "", 0, False, "IGNORADO"):
            continue
        case_lines.append(f"- {k}: {v}")
    shap_lines = [
        f"- {t['label']} = {t['value']} ({t['direction']} a probabilidade, SHAP {t['shap']:+.3f})"
        for t in (ml.get("top_features") or [])
    ]
    sources = state.get("protocol_hits") or []
    ctx = "\n\n".join(f"[{h['title']}]\n{h['text'][:550]}" for h in sources) or "(nenhum trecho recuperado)"
    ml_block = (
        f"- Probabilidade calibrada de perfil encaminhado à rede de proteção em 2024: {ml.get('probability', 0):.0%} "
        f"(limiar operacional {ml.get('threshold', 0):.0%}; sugestão do modelo: "
        f"{'encaminhar' if ml.get('suggest_referral') else 'sem sugestão automática'})\n"
        f"- Modelo: {ml.get('model_name', 'n/d')}\n"
        f"- Fatores de maior peso neste caso:\n{_fmt_list(shap_lines)}"
    ) if ml else "- Modelo de encaminhamento não disponível para este caso."
    return f"""Relato recebido pela profissional (texto livre):
\"\"\"{state.get('user_input', '').strip()}\"\"\"

Dados estruturados do atendimento (formulário conferido pela profissional):
{_fmt_list(case_lines, '- (sem dados estruturados)')}

Avaliação por regras:
- Nível de risco: {state.get('risk_level', 'indeterminado')}
- Fatores: {', '.join(state.get('risk_factors') or []) or 'nenhum fator explícito'}

Modelo de machine learning (apoio, não decisão):
{ml_block}

Ações pré-definidas pela engine de regras:
{_fmt_list(state.get('recommended_actions'))}

Encaminhamentos pré-definidos:
{_fmt_list(state.get('referrals'))}

Trechos de protocolos e leis recuperados (cite pelo título entre colchetes quando usar):
{ctx}

Tarefa: produza a nota de apoio em JSON, concisa (resumo com 3 a 4 frases; listas com até 5 itens). "fontes_citadas" deve conter apenas títulos que você usou de fato.
"nivel_risco" deve ser coerente com a avaliação por regras, podendo subir se o relato justificar, nunca descer sem justificar em "resumo".
"aviso" deve lembrar que a decisão é da profissional."""


# ── F2: structure the free-text report into the form fields ───────────────────

# Interview topics the report already answered, even negatively (feeds the live interview guide, spec 003 U6).
# Each topic carries a literal quote from the report ("id: trecho"); the server drops topics whose quote is not in
# the text (the fast model otherwise lists every topic). Flat strings: Gemini Flash Lite loops on nested objects.
TOPICS_PROPERTY: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
TOPICS_LEGEND = "; ".join(f"{t.id} = {t.hint}" for t in TOPICS)
TOPICS_RULE = ("- topicos_abordados: só os tópicos de entrevista que o relato responde explicitamente, mesmo que a resposta seja "
               "negativa (\"não tem arma\" conta como arma), no formato \"id: trecho\", onde trecho é cópia literal e curta do relato "
               "que o responde (ex.: \"filhos: não tem filhos\"). Sem trecho literal, não inclua o tópico; na dúvida, deixe de fora. "
               f"Tópicos: {TOPICS_LEGEND}")

STRUCTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **{name: {"type": "boolean"} for name in VIOL},
        **{name: {"type": "boolean"} for name in AG},
        **{name: {"type": "boolean"} for name in REL_GROUPS},
        **{name: {"type": "string", "enum": list(mapping.values()) + ["IGNORADO"]} for name, (_, mapping) in CATEGORICAL.items()},
        "SG_UF": {"type": "string", "enum": list(UF_OPTIONS) + ["IGNORADO"]},
        "ESCOLARIDADE": {"type": "string", "enum": list(ESCOLARIDADE_LABELS.values()) + ["IGNORADO"]},
        "IDADE_ANOS": {"type": ["integer", "null"], "minimum": 0, "maximum": 120},
        "arma_mencionada": {"type": "boolean"},
        "ameaca_de_morte": {"type": "boolean"},
        "filhos_envolvidos": {"type": "boolean"},
        "local_seguro": {"type": ["boolean", "null"]},
        "resumo_objetivo": {"type": "string", "description": "2 frases, sem nomes próprios."},
        "campos_incertos": {"type": "array", "items": {"type": "string"}},
        "topicos_abordados": TOPICS_PROPERTY,
    },
    "required": list(VIOL) + list(AG) + list(REL_GROUPS) + list(CATEGORICAL) + [
        "SG_UF", "ESCOLARIDADE", "IDADE_ANOS", "arma_mencionada", "ameaca_de_morte", "filhos_envolvidos",
        "local_seguro", "resumo_objetivo", "campos_incertos", "topicos_abordados",
    ],
}

STRUCTURE_SYSTEM = (
    "Você extrai, de um relato de atendimento a uma mulher em situação de violência, os campos da ficha de "
    "notificação do SINAN. Marque true apenas o que o relato afirma ou implica claramente; na dúvida, false ou "
    "IGNORADO. Nunca invente idade, UF ou escolaridade: use null/IGNORADO quando o relato não diz. "
    "Liste em campos_incertos os campos em que você ficou em dúvida, para a profissional conferir. "
    "Não reproduza nomes próprios no resumo. Responda só com o JSON."
)


def build_structure_prompt(text: str) -> str:
    legend = "\n".join(
        [f"- {k}: {v}" for k, v in VIOL.items()]
        + [f"- {k}: {v}" for k, v in AG.items()]
        + [f"- {k}: {label}" for k, (label, _) in REL_GROUPS.items()]
        + [f"- {k}: {label} (opções: {', '.join(m.values())}, IGNORADO)" for k, (label, m) in CATEGORICAL.items()]
    )
    return f"""Legenda dos campos:
{legend}
- SG_UF: sigla da UF de residência, ou IGNORADO
- ESCOLARIDADE: {', '.join(ESCOLARIDADE_LABELS.values())} ou IGNORADO
- IDADE_ANOS: inteiro ou null
- arma_mencionada / ameaca_de_morte / filhos_envolvidos / local_seguro: sinais de risco do relato
{TOPICS_RULE}

Relato:
\"\"\"{text.strip()}\"\"\"

Extraia os campos em JSON."""


# Sparse variant used live (while typing): list only the binary fields that apply instead of 25 booleans.
BINARY_FIELDS = list(VIOL) + list(AG) + list(REL_GROUPS)
STRUCTURE_SCHEMA_SPARSE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "marcados": {"type": "array", "items": {"type": "string", "enum": BINARY_FIELDS}, "description": "Campos binários presentes no relato."},
        **{name: {"type": "string", "enum": list(mapping.values()) + ["IGNORADO"]} for name, (_, mapping) in CATEGORICAL.items()},
        "SG_UF": {"type": "string", "enum": list(UF_OPTIONS) + ["IGNORADO"]},
        "ESCOLARIDADE": {"type": "string", "enum": list(ESCOLARIDADE_LABELS.values()) + ["IGNORADO"]},
        "IDADE_ANOS": {"type": ["integer", "null"], "minimum": 0, "maximum": 120},
        "arma_mencionada": {"type": "boolean"},
        "ameaca_de_morte": {"type": "boolean"},
        "filhos_envolvidos": {"type": "boolean"},
        "local_seguro": {"type": ["boolean", "null"]},
        "resumo_objetivo": {"type": "string"},
        "campos_incertos": {"type": "array", "items": {"type": "string"}},
        "topicos_abordados": TOPICS_PROPERTY,
    },
    "required": ["marcados"] + list(CATEGORICAL) + ["SG_UF", "ESCOLARIDADE", "IDADE_ANOS", "arma_mencionada", "ameaca_de_morte",
                                                     "filhos_envolvidos", "local_seguro", "resumo_objetivo", "campos_incertos",
                                                     "topicos_abordados"],
}


def build_structure_prompt_sparse(text: str) -> str:
    legend = "\n".join(
        [f"- {k}: {v}" for k, v in VIOL.items()] + [f"- {k}: {v}" for k, v in AG.items()] + [f"- {k}: {label}" for k, (label, _) in REL_GROUPS.items()]
    )
    cats = "\n".join(f"- {k}: {label} (opções: {', '.join(m.values())}, IGNORADO)" for k, (label, m) in CATEGORICAL.items())
    return f"""Campos binários (coloque em "marcados" só os que o relato afirma ou implica claramente):
{legend}

Campos de contexto (uma opção cada; IGNORADO quando o relato não diz):
{cats}
- SG_UF: sigla da UF de residência ou IGNORADO · ESCOLARIDADE: {', '.join(ESCOLARIDADE_LABELS.values())} ou IGNORADO · IDADE_ANOS: inteiro ou null
- arma_mencionada / ameaca_de_morte / filhos_envolvidos / local_seguro: sinais de risco do relato
- resumo_objetivo: 1 frase, sem nomes próprios · campos_incertos: nomes dos campos em dúvida
{TOPICS_RULE}

Relato:
\"\"\"{text.strip()}\"\"\"

Responda só com o JSON."""
