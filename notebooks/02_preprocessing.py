# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 02 · Pré-processamento do SINAN 2024 e definição do alvo
#
# Guardiã AI · Tech Challenge 5 · grupo fiap-ssp-2025
#
# Este notebook reproduz, com as funções do pacote `guardia`, a construção da tabela de treino a partir dos
# microdados brutos do SINAN (violência interpessoal/autoprovocada, 2024, DATASUS). A extração DBC → parquet
# vem do Tech Challenge 1 (`notebooks/01_extract_sinan_data.ipynb` daquele repositório).
#
# **O que muda em relação às fases anteriores**
# - O alvo deixa de ser `OUT_VEZES` ("ocorreu outras vezes?", campo retrospectivo da mesma ficha) e passa a ser
#   `ENCAMINHA_PROTECAO`: a ficha registrou encaminhamento para delegacia da mulher, centro de atendimento à mulher,
#   Ministério Público ou Defensoria. `OUT_VEZES` vira feature.
# - Ausentes viram a categoria `IGNORADO` em vez de `dropna()`: treinamos com ~281 mil linhas, não 57 mil.
# - Codificação nominal com `OneHotEncoder` dentro de um `Pipeline` (ver notebook 03), não `LabelEncoder`.

# %%
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from guardia.data.features import (
    FEATURE_NAMES, FEATURES, PROTECTION_COLS, TARGET, build_table, load_raw,
)

# %%
df_raw = load_raw()
print(f"bruto: {len(df_raw):,} notificações · {df_raw.shape[1]} colunas")

# %% [markdown]
# ## Filtros e alvo
# Mulheres (`CS_SEXO == F`) com lesão **não** autoprovocada (`LES_AUTOP == 2`). O alvo só existe quando ao menos
# um dos quatro campos de encaminhamento foi respondido (Sim ou Não).

# %%
mulheres = df_raw[(df_raw["CS_SEXO"].str.strip() == "F") & (df_raw["LES_AUTOP"].str.strip() == "2")]
print(f"mulheres, não autoprovocada: {len(mulheres):,}")
for c in PROTECTION_COLS:
    vc = mulheres[c].fillna("").str.strip().replace("", "<vazio>").value_counts()
    print(f"{c:<11} {dict(vc.head(4))}")

# %%
table = build_table(df_raw)
print(f"tabela: {len(table):,} linhas · {len(FEATURE_NAMES)} features · prevalência do alvo {table[TARGET].mean():.3f}")

# %% [markdown]
# ## Catálogo de features
# Binárias (tipos de violência, meios de agressão, relação com o autor), nominais (`IGNORADO` explícito),
# ordinal (escolaridade) e numérica (idade), cada uma com o rótulo usado no formulário e no relatório.

# %%
pd.DataFrame([{"feature": f.name, "tipo": f.kind, "grupo": f.group, "rótulo": f.label, "opções": ", ".join(f.options[:6])} for f in FEATURES])

# %% [markdown]
# ## Distribuições que importam

# %%
resumo = pd.DataFrame({
    "prevalência do alvo por OUT_VEZES": table.groupby("OUT_VEZES")[TARGET].mean().round(3),
    "n": table.groupby("OUT_VEZES").size(),
})
resumo

# %%
table.groupby("REL_PARCEIRO_INTIMO")[TARGET].agg(["mean", "size"]).round(3)

# %%
table.groupby("CS_RACA")[TARGET].agg(["mean", "size"]).round(3).sort_values("size", ascending=False)

# %%
print("idade ausente:", f"{table['IDADE_IGN'].mean():.1%}", "· escolaridade ausente:", f"{table['ESCOLARIDADE_IGN'].mean():.1%}")

# %% [markdown]
# ## Salvar

# %%
out = ROOT / "data" / "processed" / "sinan_encaminhamento.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
table.to_parquet(out, index=False)
print("salvo em", out.relative_to(ROOT))
