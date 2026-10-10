import numpy as np
import pandas as pd

from guardia.data.features import FEATURE_NAMES, PROTECTION_COLS, TARGET, build_table, row_from_case


def _raw(n=6):
    base = {c: [""] * n for c in [
        "CS_SEXO", "LES_AUTOP", "DELEG_MULH", "ATEND_MULH", "MPU", "DEFEN_PUBL", "SG_UF", "CS_ESCOL_N", "NU_IDADE_N",
        "DT_NOTIFIC", "OUT_VEZES", "AUTOR_ALCO", "AUTOR_SEXO", "NUM_ENVOLV", "CS_RACA", "CICL_VID", "SIT_CONJUG", "CS_GESTANT",
        "ORIENT_SEX", "IDENT_GEN", "DEF_TRANS", "LOCAL_OCOR", "VIOL_MOTIV",
    ]}
    for c in ["VIOL_FISIC", "VIOL_PSICO", "VIOL_SEXU", "VIOL_TORT", "VIOL_FINAN", "VIOL_NEGLI", "VIOL_INFAN", "VIOL_LEGAL", "VIOL_TRAF", "VIOL_OUTR",
              "AG_AMEACA", "AG_FORCA", "AG_OBJETO", "AG_CORTE", "AG_QUENTE", "AG_ENFOR", "AG_ENVEN", "AG_FOGO", "AG_OUTROS",
              "REL_CONJ", "REL_EXCON", "REL_NAMO", "REL_EXNAM", "REL_PAI", "REL_MAE", "REL_PAD", "REL_MAD", "REL_IRMAO", "REL_FILHO",
              "REL_CONHEC", "REL_CUIDA", "REL_DESCO", "REL_PATRAO", "REL_TRAB", "REL_INST", "REL_POL", "REL_PROPRI", "REL_OUTROS"]:
        base[c] = ["2"] * n
    df = pd.DataFrame(base)
    df["CS_SEXO"] = ["F", "F", "F", "M", "F", "F"]
    df["LES_AUTOP"] = ["2", "2", "2", "2", "1", "2"]
    df["DELEG_MULH"] = ["1", "2", " ", "1", "2", "9"]   # row0 yes, row1 no, row2 unanswered, row3 male, row4 self-inflicted, row5 ignored
    df["ATEND_MULH"] = ["2", "2", " ", "2", "2", "9"]
    df["MPU"] = ["2", "2", "9", "2", "2", "9"]
    df["DEFEN_PUBL"] = ["2", "2", " ", "2", "2", " "]
    df["SG_UF"] = ["53"] * n
    df["NU_IDADE_N"] = ["4029", "4017", "4040", "4030", "4022", "3011"]
    df["CS_ESCOL_N"] = ["06", "02", "", "07", "05", "09"]
    df["OUT_VEZES"] = ["1", "2", "9", "1", "1", ""]
    df["REL_CONJ"] = ["1", "2", "2", "2", "2", "2"]
    df["VIOL_FISIC"] = ["1", "1", "2", "2", "2", "2"]
    return df


def test_build_table_target_and_filters():
    t = build_table(_raw())
    # rows kept: 0 (yes), 1 (no); dropped: 2 (unanswered), 3 (male), 4 (self-inflicted), 5 (all ignored)
    assert len(t) == 2
    assert list(t[TARGET]) == [1, 0]
    assert set(FEATURE_NAMES) <= set(t.columns)
    assert t.loc[0, "REL_PARCEIRO_INTIMO"] == 1 and t.loc[1, "REL_PARCEIRO_INTIMO"] == 0
    assert t.loc[0, "OUT_VEZES"] == "SIM" and t.loc[1, "OUT_VEZES"] == "NAO"
    assert t.loc[0, "IDADE_ANOS"] == 29 and t.loc[0, "ESCOLARIDADE_ORD"] == 2
    assert t.loc[0, "SG_UF"] == "DF"
    for c in PROTECTION_COLS:
        assert c not in FEATURE_NAMES  # the target's ingredients never leak into the features


def test_row_from_case_matches_feature_schema():
    row = row_from_case({"VIOL_FISIC": "on", "OUT_VEZES": "sim", "SG_UF": "df", "IDADE_ANOS": "31", "ESCOLARIDADE": "ENSINO_SUPERIOR", "bogus": 1})
    assert list(row.columns) == FEATURE_NAMES
    assert row.loc[0, "VIOL_FISIC"] == 1 and row.loc[0, "OUT_VEZES"] == "SIM" and row.loc[0, "SG_UF"] == "DF"
    assert row.loc[0, "IDADE_ANOS"] == 31 and row.loc[0, "IDADE_IGN"] == 0
    assert row.loc[0, "ESCOLARIDADE_ORD"] == 3
    empty = row_from_case({})
    assert np.isnan(empty.loc[0, "IDADE_ANOS"]) and empty.loc[0, "IDADE_IGN"] == 1
    assert empty.loc[0, "CS_RACA"] == "IGNORADO"
