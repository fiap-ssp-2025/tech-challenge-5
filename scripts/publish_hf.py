#!/usr/bin/env python3
"""Publish the trained referral model + metrics + figures to the group's Hugging Face org.

Repo: fiap-ssp-2025/tc5-guardia-encaminhamento (private by default). Files:
  encaminhamento.joblib, metrics.json, model_card.json, figures/*.png, README.md (model card)

Usage: uv run python scripts/publish_hf.py [--public] [--repo-id org/name]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def model_card_md(card: dict) -> str:
    t = card.get("test", {})
    return f"""---
license: apache-2.0
language: pt
tags: [tabular-classification, scikit-learn, sinan, violencia-contra-a-mulher, fiap]
library_name: scikit-learn
---

# Guardiã AI — modelo de encaminhamento à rede de proteção

**Tarefa.** {card.get('task')}

**Dados.** {card.get('data')} · treino {card.get('rows_train'):,} · teste {card.get('rows_test'):,} · prevalência {card.get('prevalence')}

**Modelo implantado.** `{card.get('model')}` dentro de um `Pipeline` scikit-learn (OneHot para nominais, imputação mediana para numéricas),
calibrado por regressão isotônica e com limiar {card.get('threshold')} escolhido por F2 na validação.

| métrica (teste, no limiar) | valor |
|---|---|
| recall (classe encaminhada) | {t.get('recall')} |
| precisão | {t.get('precision')} |
| F1 | {t.get('f1')} |
| especificidade | {t.get('specificity')} |
| PR-AUC | {card.get('pr_auc')} |
| ROC-AUC | {card.get('roc_auc')} |
| Brier após calibração | {card.get('brier_after_calibration')} |
| gap de recall por raça/cor | {card.get('recall_gap_by_race')} |

## Limites
{chr(10).join('- ' + l for l in card.get('limits', []))}

## Uso
```python
import joblib
art = joblib.load("encaminhamento.joblib")
row = ...  # DataFrame com as colunas de art["feature_names"] (ver guardia.data.features.row_from_case)
p = art["calibrated"].predict_proba(row)[:, 1]
```
Treinado em {card.get('trained_at')} pelo grupo fiap-ssp-2025 (Tech Challenge 5, FIAP Pós IA para Devs).
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", default=f"{os.getenv('HF_ORG', 'fiap-ssp-2025')}/tc5-guardia-encaminhamento")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN ausente")
    models = ROOT / "models"
    card = json.loads((models / "model_card.json").read_text(encoding="utf-8"))
    (models / "README.md").write_text(model_card_md(card), encoding="utf-8")
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=str(models), path_in_repo=".", allow_patterns=["*.joblib", "*.json", "README.md"],
                      commit_message=f"Guardiã AI · {card.get('model')} · treinado em {card.get('trained_at')}")
    figs = ROOT / "docs" / "figures"
    if figs.exists():
        api.upload_folder(repo_id=args.repo_id, folder_path=str(figs), path_in_repo="figures", allow_patterns=["*.png"],
                          commit_message="figuras: PR, calibração, SHAP")
    print(f"publicado em https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
