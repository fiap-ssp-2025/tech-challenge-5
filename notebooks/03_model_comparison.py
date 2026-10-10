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
# # 03 · Comparação de modelos, calibração, limiar e explicabilidade
#
# Guardiã AI · Tech Challenge 5 · grupo fiap-ssp-2025
#
# Treina e compara **Regressão Logística**, **Random Forest** (com busca por algoritmo genético, herança do
# Tech Challenge 2) e **HistGradientBoosting** para o alvo `ENCAMINHA_PROTECAO`. O protocolo é o mesmo do script
# `guardia.ml.train` (que gera o artefato implantado): 70/30 estratificado, validação de 15% dentro do treino para
# limiar, calibração e fitness do GA; o teste é tocado uma única vez.
#
# **Por que não acurácia.** A prevalência é ~28%: "nunca sugerir" acerta 72% das vezes e não ajuda ninguém.
# A métrica prioritária é o **recall da classe encaminhada** (deixar de sugerir um encaminhamento custa mais que
# sugerir a mais), acompanhada de precisão, F1, PR-AUC e calibração. O limiar operacional maximiza F2 na validação.

# %%
import json
import sys
from pathlib import Path

ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from IPython.display import Image, display

# %% [markdown]
# ## Rodar o treino completo
# Equivale a `make train GA=1`. Leva alguns minutos (GA com 6 indivíduos × 4 gerações).
# Se o artefato já existir, pule para a próxima seção.

# %%
RUN_TRAINING = False  # mude para True para retreinar aqui
if RUN_TRAINING:
    import sys as _s
    from guardia.ml import train
    _s.argv = ["train", "--ga"]
    train.main()

# %% [markdown]
# ## Resultados

# %%
metrics = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
print("dataset:", metrics["dataset"])
print("selecionado:", metrics["selected_model"], "·", metrics["artifact"])

# %%
rows = []
for name, m in metrics["models"].items():
    if name.startswith("_"):
        continue
    t, a = m["test"], m["test"]["at_threshold"]
    rows.append({"modelo": name, "PR-AUC": t["pr_auc"], "ROC-AUC": t["roc_auc"], "recall@thr": a["recall"], "precisão@thr": a["precision"],
                 "F1@thr": a["f1"], "especificidade": a["specificity"], "limiar": a["threshold"], "treino (s)": m["train_seconds"]})
pd.DataFrame(rows).set_index("modelo").round(3)

# %% [markdown]
# Baseline de classe majoritária e PR-AUC aleatória, para leitura honesta:

# %%
print(f"acurácia da classe majoritária: {metrics['dataset']['accuracy_majority']:.3f} · PR-AUC aleatória (= prevalência): {metrics['dataset']['pr_auc_random']:.3f}")

# %% [markdown]
# ## Busca genética (Random Forest)

# %%
if "_ga" in metrics["models"]:
    ga = metrics["models"]["_ga"]
    print("melhores hiperparâmetros:", ga["best_params"])
    pd.DataFrame(ga["history"]).groupby("generation")[["fitness", "pr_auc", "recall"]].max().round(4)

# %% [markdown]
# ## Curvas precisão-recall, calibração e SHAP global

# %%
for f in ("pr_curves.png", "calibration.png", "shap_global.png"):
    display(Image(filename=str(ROOT / "docs" / "figures" / f)))

# %% [markdown]
# ## Calibração e ponto de operação do modelo implantado

# %%
sel = metrics["models"][metrics["selected_model"]]
cal = sel["calibration"]
print(f"Brier antes {cal['brier_before']:.4f} → depois {cal['brier_after']:.4f} · limiar calibrado {cal['threshold_calibrated']:.3f}")
pd.Series(cal["test_at_threshold_calibrated"]).round(3)

# %% [markdown]
# ## Equidade: recall por raça/cor

# %%
pd.DataFrame({g: v for g, v in sel["fairness_by_race"].items() if not g.startswith("_")}).T.round(3)

# %% [markdown]
# ## Importância global (SHAP) e coeficientes da regressão logística

# %%
pd.DataFrame(sel["shap_top20"]).round(4).head(15)

# %%
pd.DataFrame(metrics["models"]["logistic_regression"]["coefficient_contributions_top20"]).round(4).head(15)

# %% [markdown]
# ## Impacto dos erros
# - **Falso negativo** (modelo não sugere, ficha registrou encaminhamento): a profissional pode deixar de considerar a rede de
#   proteção num caso com perfil de risco. É o erro mais caro; por isso o limiar privilegia recall.
# - **Falso positivo** (modelo sugere, ficha não registrou): custo de uma conversa a mais com a equipe. Aceitável.
# - O modelo reproduz a prática de 2024, inclusive sub-encaminhamento regional. Nunca deve ser usado isoladamente.
