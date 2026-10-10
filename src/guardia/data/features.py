"""Feature engineering for the SINAN 2024 violence notifications.

Builds the training table for the referral model (``ENCAMINHA_PROTECAO``) from the raw
DATASUS parquet and exposes the same transformation for a single case coming from the
web form, so that training and inference never drift.

Target definition (spec §3.2): the notification registered a referral to the women's
protection network — DELEG_MULH, ATEND_MULH, MPU or DEFEN_PUBL == "1" (Sim).
Rows where none of the four fields was answered (all blank / "9" ignorado) are dropped.

``OUT_VEZES`` ("ocorreu outras vezes?") is a *feature* here, not the target (see
spec §3.1 for why the previous phases got this wrong).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parents[2]
RAW_DIR = ROOT / "data" / "raw" / "VIOLBR24.parquet"
PROCESSED_PATH = ROOT / "data" / "processed" / "sinan_encaminhamento.parquet"

TARGET = "ENCAMINHA_PROTECAO"
PROTECTION_COLS = ["DELEG_MULH", "ATEND_MULH", "MPU", "DEFEN_PUBL"]
GROUP_COL = "CS_RACA"  # fairness slices

IGN = "IGNORADO"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str  # "binary" | "categorical" | "ordinal" | "numeric"
    label: str
    options: tuple[str, ...] = ()
    help: str = ""
    group: str = ""


# ── Catalogue ───────────────────────────────────────────────────────────────────
# Every model input, with the Portuguese label used in the form and the report.

VIOL = {
    "VIOL_FISIC": "Violência física",
    "VIOL_PSICO": "Violência psicológica / moral",
    "VIOL_SEXU": "Violência sexual",
    "VIOL_TORT": "Tortura",
    "VIOL_FINAN": "Violência financeira / patrimonial",
    "VIOL_NEGLI": "Negligência / abandono",
    "VIOL_INFAN": "Trabalho infantil",
    "VIOL_LEGAL": "Intervenção legal",
    "VIOL_TRAF": "Tráfico de pessoas",
    "VIOL_OUTR": "Outro tipo de violência",
}
AG = {
    "AG_AMEACA": "Ameaça",
    "AG_FORCA": "Força corporal / espancamento",
    "AG_OBJETO": "Objeto contundente",
    "AG_CORTE": "Objeto cortante / perfurante",
    "AG_QUENTE": "Substância / objeto quente",
    "AG_ENFOR": "Enforcamento / sufocação",
    "AG_ENVEN": "Envenenamento / intoxicação",
    "AG_FOGO": "Arma de fogo",
    "AG_OUTROS": "Outro meio de agressão",
}
REL_GROUPS = {
    "REL_PARCEIRO_INTIMO": ("Parceiro íntimo (cônjuge, ex, namorado)", ["REL_CONJ", "REL_EXCON", "REL_NAMO", "REL_EXNAM"]),
    "REL_FAMILIAR": ("Familiar (pai, mãe, padrasto, madrasta, irmão, filho)", ["REL_PAI", "REL_MAE", "REL_PAD", "REL_MAD", "REL_IRMAO", "REL_FILHO"]),
    "REL_CONHECIDO": ("Conhecido, cuidador ou desconhecido", ["REL_CONHEC", "REL_CUIDA", "REL_DESCO"]),
    "REL_INSTITUCIONAL": ("Relação institucional (patrão, trabalho, polícia, agente da lei)", ["REL_PATRAO", "REL_TRAB", "REL_INST", "REL_POL", "REL_PROPRI"]),
    "REL_OUTROS_FLAG": ("Outra relação", ["REL_OUTROS"]),
}

SIM_NAO = ("SIM", "NAO", IGN)

CATEGORICAL: dict[str, tuple[str, dict[str, str]]] = {
    "OUT_VEZES": ("A violência já ocorreu outras vezes?", {"1": "SIM", "2": "NAO"}),
    "AUTOR_ALCO": ("Suspeita de uso de álcool pelo autor", {"1": "SIM", "2": "NAO"}),
    "AUTOR_SEXO": ("Sexo do provável autor", {"1": "MASCULINO", "2": "FEMININO", "3": "AMBOS"}),
    "NUM_ENVOLV": ("Número de envolvidos", {"1": "UM", "2": "DOIS_OU_MAIS"}),
    "CS_RACA": ("Raça/cor", {"1": "BRANCA", "2": "PRETA", "3": "AMARELA", "4": "PARDA", "5": "INDIGENA"}),
    "CICL_VID": ("Ciclo de vida", {"1": "CRIANCA", "2": "ADOLESCENTE", "3": "JOVEM", "4": "ADULTA", "5": "IDOSA"}),
    "SIT_CONJUG": ("Situação conjugal", {"1": "SOLTEIRA", "2": "CASADA_UNIAO", "3": "VIUVA", "4": "SEPARADA", "8": "NAO_SE_APLICA"}),
    "CS_GESTANT": ("Gestante", {"1": "1_TRIMESTRE", "2": "2_TRIMESTRE", "3": "3_TRIMESTRE", "4": "IDADE_GEST_IGNORADA", "5": "NAO", "6": "NAO_SE_APLICA"}),
    "ORIENT_SEX": ("Orientação sexual", {"1": "HETEROSSEXUAL", "2": "HOMOSSEXUAL", "3": "BISSEXUAL", "8": "NAO_SE_APLICA"}),
    "IDENT_GEN": ("Identidade de gênero", {"1": "TRAVESTI", "2": "MULHER_TRANS", "3": "HOMEM_TRANS", "8": "NAO_SE_APLICA"}),
    "DEF_TRANS": ("Possui deficiência ou transtorno", {"1": "SIM", "2": "NAO", "8": "NAO_SE_APLICA"}),
    "LOCAL_OCOR": (
        "Local da ocorrência",
        {
            "01": "RESIDENCIA", "02": "HABITACAO_COLETIVA", "03": "ESCOLA", "04": "LOCAL_PRATICA_ESPORTIVA",
            "05": "BAR_OU_SIMILAR", "06": "VIA_PUBLICA", "07": "COMERCIO_SERVICOS", "08": "INDUSTRIAS_CONSTRUCAO",
            "09": "OUTRO",
        },
    ),
    "VIOL_MOTIV": (
        "Violência motivada por",
        {
            "01": "SEXISMO", "02": "LGBTFOBIA", "03": "RACISMO", "04": "INTOLERANCIA_RELIGIOSA", "05": "XENOFOBIA",
            "06": "CONFLITO_GERACIONAL", "07": "SITUACAO_DE_RUA", "08": "DEFICIENCIA", "09": "OUTROS", "88": "NAO_SE_APLICA",
        },
    ),
}

ESCOLARIDADE_LABELS = {
    0: "FUNDAMENTAL_INCOMPLETO",
    1: "ENSINO_MEDIO_INCOMPLETO",
    2: "ENSINO_MEDIO_COMPLETO",
    3: "ENSINO_SUPERIOR",
}

_UF_MAP: dict[str, str] = json.loads((PACKAGE_DIR / "dicionario_estados_ibge.json").read_text(encoding="utf-8"))
UF_OPTIONS = tuple(sorted(set(_UF_MAP.values())))


def feature_catalogue() -> list[Feature]:
    feats: list[Feature] = []
    for name, label in VIOL.items():
        feats.append(Feature(name, "binary", label, group="Tipo de violência"))
    for name, label in AG.items():
        feats.append(Feature(name, "binary", label, group="Meio de agressão"))
    for name, (label, _) in REL_GROUPS.items():
        feats.append(Feature(name, "binary", label, group="Relação com o provável autor"))
    for name, (label, mapping) in CATEGORICAL.items():
        feats.append(Feature(name, "categorical", label, tuple(mapping.values()) + (IGN,), group="Contexto"))
    feats.append(Feature("SG_UF", "categorical", "UF de residência", UF_OPTIONS + (IGN,), group="Contexto"))
    feats.append(Feature("ESCOLARIDADE_ORD", "ordinal", "Escolaridade", tuple(ESCOLARIDADE_LABELS.values()), group="Perfil"))
    feats.append(Feature("ESCOLARIDADE_IGN", "binary", "Escolaridade ignorada", group="Perfil"))
    feats.append(Feature("IDADE_ANOS", "numeric", "Idade (anos)", group="Perfil"))
    feats.append(Feature("IDADE_IGN", "binary", "Idade ignorada", group="Perfil"))
    return feats


FEATURES: list[Feature] = feature_catalogue()
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
BINARY_COLS = [f.name for f in FEATURES if f.kind == "binary"]
CATEGORICAL_COLS = [f.name for f in FEATURES if f.kind == "categorical"]
NUMERIC_COLS = [f.name for f in FEATURES if f.kind in ("ordinal", "numeric")]


# ── Raw → table ───────────────────────────────────────────────────────────────


def _clean(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.strip()


def _binary(s: pd.Series) -> pd.Series:
    return (_clean(s) == "1").astype("int8")


def _categorical(s: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return _clean(s).map(mapping).fillna(IGN).astype("string")


def _age_years(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """NU_IDADE_N encodes unit in the first digit: 4xxx = years, 3xxx = months, 2xxx = days, 1xxx = hours."""
    code = pd.to_numeric(_clean(s), errors="coerce")
    years = pd.Series(np.nan, index=s.index, dtype="float64")
    years[code >= 4000] = code[code >= 4000] - 4000
    years[(code >= 1000) & (code < 4000)] = 0.0
    ign = years.isna().astype("int8")
    return years, ign


def _escolaridade(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    code = pd.to_numeric(_clean(s), errors="coerce")
    ord_ = pd.Series(np.nan, index=s.index, dtype="float64")
    ord_[code.isin([43, 1, 2, 3])] = 0
    ord_[code.isin([4, 5])] = 1
    ord_[code == 6] = 2
    ord_[code.isin([7, 8])] = 3
    ign = ord_.isna().astype("int8")
    return ord_, ign


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    parts = sorted(raw_dir.glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"Nenhum parquet em {raw_dir}")
    return pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)


def build_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Women, non self-inflicted, with the target defined → one row per notification."""
    df = df_raw[(_clean(df_raw["CS_SEXO"]) == "F") & (_clean(df_raw["LES_AUTOP"]) == "2")].copy()

    prot = pd.DataFrame({c: _clean(df[c]) for c in PROTECTION_COLS})
    any_yes = (prot == "1").any(axis=1)
    any_no = (prot == "2").any(axis=1)
    keep = any_yes | any_no
    df = df[keep].copy()
    target = any_yes[keep].astype("int8")

    out = pd.DataFrame(index=df.index)
    for name in VIOL:
        out[name] = _binary(df[name])
    for name in AG:
        out[name] = _binary(df[name])
    for name, (_, cols) in REL_GROUPS.items():
        out[name] = pd.concat([_binary(df[c]) for c in cols], axis=1).max(axis=1).astype("int8")
    for name, (_, mapping) in CATEGORICAL.items():
        out[name] = _categorical(df[name], mapping)
    out["SG_UF"] = _clean(df["SG_UF"]).map(_UF_MAP).fillna(IGN).astype("string")
    out["ESCOLARIDADE_ORD"], out["ESCOLARIDADE_IGN"] = _escolaridade(df["CS_ESCOL_N"])
    out["IDADE_ANOS"], out["IDADE_IGN"] = _age_years(df["NU_IDADE_N"])
    out[TARGET] = target.values

    # Context columns kept for analysis only (never fed to the model)
    out["_DELEG_MULH"] = _clean(df["DELEG_MULH"])
    out["_ATEND_MULH"] = _clean(df["ATEND_MULH"])
    out["_MPU"] = _clean(df["MPU"])
    out["_DEFEN_PUBL"] = _clean(df["DEFEN_PUBL"])
    out["_DT_NOTIFIC"] = _clean(df["DT_NOTIFIC"])
    return out.reset_index(drop=True)


def build_and_save(raw_dir: Path = RAW_DIR, out_path: Path = PROCESSED_PATH) -> pd.DataFrame:
    table = build_table(load_raw(raw_dir))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)
    return table


# ── Single case (web form / synthetic) → model row ─────────────────────────────

FORM_DEFAULTS: dict[str, Any] = {
    **{name: 0 for name in VIOL},
    **{name: 0 for name in AG},
    **{name: 0 for name in REL_GROUPS},
    **{name: IGN for name in CATEGORICAL},
    "SG_UF": IGN,
    "ESCOLARIDADE": None,
    "IDADE_ANOS": None,
}


def row_from_case(case: dict[str, Any]) -> pd.DataFrame:
    """Build a one-row DataFrame with exactly the model columns from a form-like dict.

    Accepted keys: every binary/categorical feature name (binaries as truthy values,
    categoricals as option strings), ``ESCOLARIDADE`` (option string or None) and
    ``IDADE_ANOS`` (number or None). Unknown keys are ignored.
    """
    data = {**FORM_DEFAULTS, **{k: v for k, v in case.items() if k in FORM_DEFAULTS}}
    row: dict[str, Any] = {}
    for name in BINARY_COLS:
        if name in ("ESCOLARIDADE_IGN", "IDADE_IGN"):
            continue
        row[name] = 1 if _truthy(data.get(name)) else 0
    for name in CATEGORICAL_COLS:
        opts = {f.name: f.options for f in FEATURES}[name]
        val = str(data.get(name) or IGN).upper()
        row[name] = val if val in opts else IGN
    esc = data.get("ESCOLARIDADE")
    esc_ord = {v: k for k, v in ESCOLARIDADE_LABELS.items()}.get(str(esc).upper()) if esc else None
    row["ESCOLARIDADE_ORD"] = float(esc_ord) if esc_ord is not None else np.nan
    row["ESCOLARIDADE_IGN"] = 0 if esc_ord is not None else 1
    idade = data.get("IDADE_ANOS")
    try:
        idade_f = float(idade) if idade not in (None, "") else np.nan
    except (TypeError, ValueError):
        idade_f = np.nan
    row["IDADE_ANOS"] = idade_f
    row["IDADE_IGN"] = 0 if not np.isnan(idade_f) else 1
    df = pd.DataFrame([row])[FEATURE_NAMES]
    for c in CATEGORICAL_COLS:
        df[c] = df[c].astype("string")
    return df


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "sim", "on", "yes", "x")
    return bool(v)


def catalogue_json() -> list[dict[str, Any]]:
    return [
        {"name": f.name, "kind": f.kind, "label": f.label, "options": list(f.options), "group": f.group}
        for f in FEATURES
    ]


if __name__ == "__main__":  # pragma: no cover
    t = build_and_save()
    print(t.shape, t[TARGET].mean())
