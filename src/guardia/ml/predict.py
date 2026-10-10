"""Load the deployed referral model and score one case with per-case SHAP attributions."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from guardia.data.features import FEATURES, row_from_case

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = ROOT / os.getenv("GUARDIA_MODEL_PATH", "models/encaminhamento.joblib")

_LABELS = {f.name: f.label for f in FEATURES}


@dataclass
class Prediction:
    probability: float
    threshold: float
    suggest_referral: bool
    model_name: str
    top_features: list[dict[str, Any]] = field(default_factory=list)
    raw_probability: float | None = None

    @property
    def probability_pct(self) -> int:
        return round(self.probability * 100)

    def as_dict(self) -> dict[str, Any]:
        return {
            "probability": round(self.probability, 4),
            "raw_probability": round(self.raw_probability, 4) if self.raw_probability is not None else None,
            "threshold": round(self.threshold, 4),
            "suggest_referral": self.suggest_referral,
            "model_name": self.model_name,
            "top_features": self.top_features,
        }


@lru_cache(maxsize=1)
def load_artifact(path: str | None = None) -> dict[str, Any]:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {p}. Rode `uv run python -m guardia.ml.train`.")
    return joblib.load(p)


def _pretty(feature_out: str, value: float) -> tuple[str, str]:
    """'CS_RACA_PARDA' → ('Raça/cor', 'PARDA'); 'VIOL_FISIC' → ('Violência física', 'sim')."""
    for name in sorted(_LABELS, key=len, reverse=True):
        if feature_out == name:
            label = _LABELS[name]
            if name in ("IDADE_ANOS", "ESCOLARIDADE_ORD"):
                return label, f"{value:g}"
            return label, "sim" if value >= 0.5 else "não"
        if feature_out.startswith(name + "_"):
            return _LABELS[name], feature_out[len(name) + 1 :].replace("_", " ").lower()
    return feature_out, f"{value:g}"


def explain_row(artifact: dict[str, Any], row: pd.DataFrame, k: int = 6) -> list[dict[str, Any]]:
    pipe = artifact["pipeline"]
    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(row)
    names = artifact["feature_names_out"]
    if hasattr(model, "coef_"):
        contrib = (Xt * model.coef_[0])[0]
    else:
        import shap

        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xt)
        if isinstance(vals, list):
            contrib = vals[1][0]
        elif np.ndim(vals) == 3:
            contrib = vals[0, :, 1]
        else:
            contrib = vals[0]
    order = np.argsort(np.abs(contrib))[::-1]
    out = []
    for i in order:
        if abs(contrib[i]) < 1e-6:
            continue
        label, val = _pretty(names[i], float(Xt[0, i]))
        out.append({"feature": names[i], "label": label, "value": val, "shap": round(float(contrib[i]), 4),
                    "direction": "aumenta" if contrib[i] > 0 else "reduz"})
        if len(out) >= k:
            break
    return out


def predict_case(case: dict[str, Any], *, artifact: dict[str, Any] | None = None, k: int = 6) -> Prediction:
    art = artifact or load_artifact()
    row = row_from_case(case)
    raw = float(art["pipeline"].predict_proba(row)[:, 1][0])
    cal = float(art["calibrated"].predict_proba(row)[:, 1][0]) if art.get("calibrated") is not None else raw
    thr = float(art["threshold"])
    return Prediction(
        probability=cal, raw_probability=raw, threshold=thr, suggest_referral=cal >= thr,
        model_name=art["name"], top_features=explain_row(art, row, k=k),
    )
