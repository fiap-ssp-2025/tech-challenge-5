#!/usr/bin/env python3
"""Rewrite the metrics block of README.md from models/metrics.json (run after training)."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "relatorio_tecnico.md"
START, END = "<!-- METRICS:START -->", "<!-- METRICS:END -->"


def main() -> None:
    m = json.loads((ROOT / "models" / "metrics.json").read_text(encoding="utf-8"))
    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    d = m["dataset"]
    lines = [
        f"Conjunto: {d['rows']:,} linhas · treino {d['train']:,} · validação {d['validation']:,} · teste {d['test']:,} · "
        f"prevalência {d['prevalence']:.3f} · acurácia da classe majoritária {d['accuracy_majority']:.3f}.".replace(",", "."),
        "",
        "| Modelo | PR-AUC | ROC-AUC | Recall@thr | Precisão@thr | F1@thr | Especificidade | Limiar |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in m["models"].items():
        if name.startswith("_"):
            continue
        t, a = r["test"], r["test"]["at_threshold"]
        tag = " **(implantado)**" if name == m["selected_model"] else ""
        lines.append(f"| {name}{tag} | {t['pr_auc']:.3f} | {t['roc_auc']:.3f} | {a['recall']:.3f} | {a['precision']:.3f} | {a['f1']:.3f} | {a['specificity']:.3f} | {a['threshold']:.3f} |")
    sel = m["models"][m["selected_model"]]
    cal = sel["calibration"]["test_at_threshold_calibrated"]
    lines += [
        "",
        (f"Modelo implantado: **{m['selected_model']}** calibrado (isotônica): Brier {sel['calibration']['brier_before']:.4f} → "
        f"{sel['calibration']['brier_after']:.4f}; no limiar calibrado {cal['threshold']:.3f}: recall {cal['recall']:.3f}, precisão {cal['precision']:.3f}, "
        f"F1 {cal['f1']:.3f}. Gap de recall por raça/cor: {sel['fairness_by_race']['_summary']['recall_gap']:.3f}. "
        f"Artefato: {m['artifact']['size_mb']} MB. Figuras em `docs/figures/` (PR, calibração, SHAP)."),
    ]
    if "_ga" in m["models"]:
        lines.append(f"Algoritmo genético (RF): melhores hiperparâmetros {m['models']['_ga']['best_params']}.")
    top = ", ".join(f"{t['feature']}" for t in sel["shap_top20"][:6])
    lines.append(f"Top features (média |SHAP|): {top}.")
    block = "\n".join(lines)
    for path in (README, REPORT):
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new = re.sub(re.escape(START) + ".*?" + re.escape(END), f"{START}\n{block}\n{END}", text, flags=re.DOTALL)
        path.write_text(new, encoding="utf-8")
    print("metrics block updated in README and report:", card.get("model"))


if __name__ == "__main__":
    main()
