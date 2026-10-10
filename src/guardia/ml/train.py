"""Train and compare the referral models (spec §3.4).

Models: Logistic Regression (interpretable baseline), Random Forest (optionally tuned by the
genetic algorithm inherited from TC2) and HistGradientBoosting. All share one preprocessing
pipeline (one-hot for nominal columns, passthrough for binaries/ordinal/numeric with median
imputation), fitted on the training split only.

Protocol:
  train 70% / test 30%, stratified, seed 42
  inside train: 85/15 split → the 15% is the validation slice used for threshold choice,
  calibration fitting and GA fitness, so the test split is touched exactly once.

Outputs (models/):
  encaminhamento.joblib   deployed artifact (pipeline + calibrated model + threshold + metadata)
  metrics.json            every model, every metric, calibration bins, fairness slices, SHAP top-20
  model_card.json         short human-readable summary used by the web app and the report
docs/figures/*.png        PR curves, calibration curve, SHAP bar chart

Run:  uv run python -m guardia.ml.train [--ga] [--quick]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from guardia.data.features import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    FEATURE_NAMES,
    GROUP_COL,
    NUMERIC_COLS,
    PROCESSED_PATH,
    TARGET,
    catalogue_json,
)

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "docs" / "figures"
SEED = 42
BETA = 2.0  # recall weighs 4x precision when choosing the operating point

GA_SPACE = {
    "n_estimators": [150, 250, 350],
    "max_depth": [8, 12, 16, 20],
    "min_samples_leaf": [5, 10, 20, 40],
    "max_features": ["sqrt", 0.3, 0.5],
}


# ── Preprocessing ───────────────────────────────────────────────────────────────


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32), CATEGORICAL_COLS),
            ("num", SimpleImputer(strategy="median"), NUMERIC_COLS),
            ("bin", "passthrough", BINARY_COLS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


def model_zoo(quick: bool) -> dict[str, Any]:
    n_rf = 120 if quick else 250
    return {
        "logistic_regression": LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5),
        "random_forest": RandomForestClassifier(
            n_estimators=n_rf, max_depth=14, min_samples_leaf=10, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200 if quick else 400, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=0.5, class_weight="balanced", random_state=SEED, early_stopping=True,
            validation_fraction=0.1,
        ),
    }


# ── Metrics ─────────────────────────────────────────────────────────────────────


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=BETA, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def threshold_free(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
    }


def best_threshold(y_val: np.ndarray, proba_val: np.ndarray) -> float:
    """Operating point: maximise F-beta (beta=2) on the validation slice."""
    p, r, t = precision_recall_curve(y_val, proba_val)
    fb = (1 + BETA**2) * p[:-1] * r[:-1] / np.maximum(BETA**2 * p[:-1] + r[:-1], 1e-9)
    return float(t[int(np.argmax(fb))])


def fairness_slices(y_true: pd.Series, pred: np.ndarray, groups: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for g in sorted(groups.unique()):
        m = groups == g
        if m.sum() < 200:
            continue
        yt, yp = y_true[m].to_numpy(), pred[m.to_numpy()]
        _tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        out[str(g)] = {
            "n": int(m.sum()),
            "prevalence": float(yt.mean()),
            "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
            "selection_rate": float(yp.mean()),
        }
    recalls = [v["recall"] for v in out.values()]
    out["_summary"] = {"recall_gap": float(max(recalls) - min(recalls)) if recalls else 0.0}
    return out


# ── Genetic algorithm (TC2 heritage, trimmed) ───────────────────────────────────


def _ga_fitness(m: dict[str, float]) -> float:
    return 0.5 * m["pr_auc"] + 0.3 * m["recall"] + 0.2 * m["f1"]


def genetic_search(
    X_tr, y_tr, X_val, y_val, *, population: int = 6, generations: int = 4, mutation: float = 0.3, elitism: int = 2
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rng = random.Random(SEED)
    pop = [{k: rng.choice(v) for k, v in GA_SPACE.items()} for _ in range(population)]
    history: list[dict[str, Any]] = []
    best, best_fit = None, -1.0
    for gen in range(generations):
        scored = []
        for ind in pop:
            pipe = make_pipeline(RandomForestClassifier(**ind, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_val)[:, 1]
            m = {**threshold_free(y_val, proba), **metrics_at(y_val, proba, best_threshold(y_val, proba))}
            fit = _ga_fitness(m)
            scored.append((fit, ind, m))
            history.append({"generation": gen, "params": ind, "fitness": fit, "pr_auc": m["pr_auc"], "recall": m["recall"]})
            print(f"    GA gen {gen + 1} {ind} → fitness {fit:.4f} (PR-AUC {m['pr_auc']:.4f})", flush=True)
        scored.sort(key=lambda s: s[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit, best = scored[0][0], deepcopy(scored[0][1])
        elite = [deepcopy(s[1]) for s in scored[:elitism]]
        while len(elite) < population:
            p1, p2 = rng.sample(scored[: max(3, elitism + 1)], 2)
            child = {k: rng.choice([p1[1][k], p2[1][k]]) for k in GA_SPACE}
            for k, opts in GA_SPACE.items():
                if rng.random() < mutation:
                    child[k] = rng.choice(opts)
            elite.append(child)
        pop = elite
    return best or pop[0], history


# ── SHAP ───────────────────────────────────────────────────────────────────────


def shap_global(pipe: Pipeline, X_sample: pd.DataFrame) -> list[dict[str, float]]:
    import shap

    prep, model = pipe.named_steps["prep"], pipe.named_steps["model"]
    Xt = prep.transform(X_sample)
    names = list(prep.get_feature_names_out())
    if isinstance(model, LogisticRegression):
        contrib = Xt * model.coef_[0]
    else:
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(Xt)
        contrib = vals[1] if isinstance(vals, list) else (vals[..., 1] if vals.ndim == 3 else vals)
    mean_abs = np.abs(contrib).mean(axis=0)
    mean_signed = contrib.mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    return [{"feature": names[i], "mean_abs_shap": float(mean_abs[i]), "mean_shap": float(mean_signed[i])} for i in order]


# ── Figures ─────────────────────────────────────────────────────────────────────


def plot_pr(curves: dict[str, tuple[np.ndarray, np.ndarray, float]], prevalence: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (p, r, ap) in curves.items():
        ax.plot(r, p, label=f"{name} (PR-AUC {ap:.3f})")
    ax.axhline(prevalence, ls="--", c="grey", lw=1, label=f"base rate {prevalence:.2f}")
    ax.set_xlabel("Recall (classe encaminhada)"); ax.set_ylabel("Precisão"); ax.set_title("Curvas precisão-recall · teste")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pr_curves.png", dpi=150); plt.close(fig)


def plot_calibration(bins_before, bins_after, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", c="grey", lw=1, label="perfeita")
    ax.plot(bins_before["mean_pred"], bins_before["frac_pos"], "o-", label="antes (modelo cru)")
    ax.plot(bins_after["mean_pred"], bins_after["frac_pos"], "s-", label="depois (isotônica)")
    ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Fração observada"); ax.set_title(f"Calibração · {name} · teste")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / "calibration.png", dpi=150); plt.close(fig)


def plot_shap(top: list[dict[str, float]], name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    feats = [t["feature"] for t in top][::-1]
    vals = [t["mean_abs_shap"] for t in top][::-1]
    colors = ["#a2353f" if t["mean_shap"] > 0 else "#0f6e6e" for t in top][::-1]
    ax.barh(feats, vals, color=colors)
    ax.set_xlabel("média |SHAP| (amostra de teste)"); ax.set_title(f"Importância global · {name}")
    fig.tight_layout(); fig.savefig(FIG_DIR / "shap_global.png", dpi=150); plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────────


def _calib_bins(y, proba, n=10) -> dict[str, list[float]]:
    frac, mean = calibration_curve(y, proba, n_bins=n, strategy="quantile")
    return {"frac_pos": [float(x) for x in frac], "mean_pred": [float(x) for x in mean]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ga", action="store_true", help="run the genetic search for the Random Forest")
    ap.add_argument("--quick", action="store_true", help="smaller models / fewer trees (smoke run)")
    ap.add_argument("--sample", type=int, default=0, help="subsample rows (debug)")
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    df = pd.read_parquet(PROCESSED_PATH)
    if args.sample:
        df = df.sample(args.sample, random_state=SEED)
    X, y = df[FEATURE_NAMES], df[TARGET].astype(int)
    groups = df[GROUP_COL]
    print(f"dataset {len(df):,} rows · prevalence {y.mean():.3f} · {len(FEATURE_NAMES)} features")

    X_trv, X_te, y_trv, y_te, _g_trv, g_te = train_test_split(X, y, groups, test_size=0.30, stratify=y, random_state=SEED)
    X_tr, X_val, y_tr, y_val = train_test_split(X_trv, y_trv, test_size=0.15, stratify=y_trv, random_state=SEED)
    prevalence = float(y_te.mean())
    majority_baseline = {"accuracy_majority": float(max(prevalence, 1 - prevalence)), "pr_auc_random": prevalence}

    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}
    pr_curves: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    zoo = model_zoo(args.quick)

    if args.ga:
        print("── GA search (Random Forest)")
        best_params, history = genetic_search(X_tr, y_tr, X_val, y_val, population=4 if args.quick else 6, generations=2 if args.quick else 4)
        zoo["random_forest_ga"] = RandomForestClassifier(**best_params, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
        results["_ga"] = {"best_params": best_params, "history": history}

    for name, est in zoo.items():
        t1 = time.time()
        pipe = make_pipeline(est)
        pipe.fit(X_tr, y_tr)
        p_val = pipe.predict_proba(X_val)[:, 1]
        thr = best_threshold(y_val.to_numpy(), p_val)
        p_te = pipe.predict_proba(X_te)[:, 1]
        res = {
            "train_seconds": round(time.time() - t1, 1),
            "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in est.get_params().items()},
            "validation": {**threshold_free(y_val.to_numpy(), p_val), **metrics_at(y_val.to_numpy(), p_val, thr)},
            "test": {**threshold_free(y_te.to_numpy(), p_te), "at_0.5": metrics_at(y_te.to_numpy(), p_te, 0.5), "at_threshold": metrics_at(y_te.to_numpy(), p_te, thr)},
        }
        p, r, _ = precision_recall_curve(y_te, p_te)
        pr_curves[name] = (p, r, res["test"]["pr_auc"])
        results[name] = res
        fitted[name] = pipe
        print(f"{name:<24} PR-AUC {res['test']['pr_auc']:.4f} ROC-AUC {res['test']['roc_auc']:.4f} "
              f"recall@thr {res['test']['at_threshold']['recall']:.3f} precision@thr {res['test']['at_threshold']['precision']:.3f} "
              f"thr {thr:.3f} ({res['train_seconds']}s)", flush=True)

    # Deployed model: best validation PR-AUC (test is never used to choose)
    # Deployed model: best validation PR-AUC, but a heavier forest only wins if it beats the boosting model by a clear
    # margin (>0.01 PR-AUC): HistGB is ~1 MB and explains in milliseconds, a deep forest can be 100 MB and slow to explain.
    candidates = [n for n in fitted]
    best_name = max(candidates, key=lambda n: results[n]["validation"]["pr_auc"])
    if (best_name.startswith("random_forest") and "hist_gradient_boosting" in fitted
            and results[best_name]["validation"]["pr_auc"] - results["hist_gradient_boosting"]["validation"]["pr_auc"] <= 0.01):
        best_name = "hist_gradient_boosting"
    best_pipe = fitted[best_name]
    print(f"── selected: {best_name}")

    # Calibration (isotonic, fitted on the validation slice only)
    calibrated = CalibratedClassifierCV(estimator=FrozenEstimator(best_pipe), method="isotonic")
    calibrated.fit(X_val, y_val)
    p_te_raw = best_pipe.predict_proba(X_te)[:, 1]
    p_te_cal = calibrated.predict_proba(X_te)[:, 1]
    thr_cal = best_threshold(y_val.to_numpy(), calibrated.predict_proba(X_val)[:, 1])
    cal = {
        "method": "isotonic (validation slice)",
        "brier_before": float(brier_score_loss(y_te, p_te_raw)),
        "brier_after": float(brier_score_loss(y_te, p_te_cal)),
        "bins_before": _calib_bins(y_te, p_te_raw),
        "bins_after": _calib_bins(y_te, p_te_cal),
        "threshold_calibrated": thr_cal,
        "test_at_threshold_calibrated": metrics_at(y_te.to_numpy(), p_te_cal, thr_cal),
    }
    results[best_name]["calibration"] = cal

    pred_te = (p_te_cal >= thr_cal).astype(int)
    results[best_name]["fairness_by_race"] = fairness_slices(y_te.reset_index(drop=True), pred_te, g_te.reset_index(drop=True))

    sample = X_te.sample(min(1500, len(X_te)), random_state=SEED)
    top = shap_global(best_pipe, sample)
    results[best_name]["shap_top20"] = top
    lr_top = shap_global(fitted["logistic_regression"], sample) if "logistic_regression" in fitted else []
    results["logistic_regression"]["coefficient_contributions_top20"] = lr_top

    plot_pr(pr_curves, prevalence)
    plot_calibration(cal["bins_before"], cal["bins_after"], best_name)
    plot_shap(top, best_name)

    artifact = {
        "name": best_name,
        "pipeline": best_pipe,
        "calibrated": calibrated,
        "threshold": thr_cal,
        "feature_names": FEATURE_NAMES,
        "feature_names_out": list(best_pipe.named_steps["prep"].get_feature_names_out()),
        "catalogue": catalogue_json(),
        "target": TARGET,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_metrics": results[best_name]["test"],
        "calibration": {k: v for k, v in cal.items() if not k.startswith("bins")},
        "shap_top20": top,
    }
    out_path = MODELS_DIR / "encaminhamento.joblib"
    joblib.dump(artifact, out_path, compress=3)
    size_mb = out_path.stat().st_size / 1e6

    summary = {
        "dataset": {"rows": len(df), "train": len(X_tr), "validation": len(X_val), "test": len(X_te),
                    "prevalence": float(y.mean()), "features": len(FEATURE_NAMES), **majority_baseline},
        "selected_model": best_name,
        "artifact": {"path": str(out_path.relative_to(ROOT)), "size_mb": round(size_mb, 1)},
        "beta_for_threshold": BETA,
        "models": results,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (MODELS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    card = {
        "model": best_name,
        "task": "Sugerir encaminhamento à rede de proteção (DELEG_MULH/ATEND_MULH/MPU/DEFEN_PUBL) a partir da ficha SINAN",
        "data": "SINAN 2024 · mulheres · violência não autoprovocada · DATASUS (público, anonimizado)",
        "rows_train": len(X_tr), "rows_test": len(X_te), "prevalence": round(float(y.mean()), 4),
        "threshold": round(thr_cal, 4),
        "test": {k: round(v, 4) for k, v in cal["test_at_threshold_calibrated"].items() if isinstance(v, float)},
        "roc_auc": round(results[best_name]["test"]["roc_auc"], 4),
        "pr_auc": round(results[best_name]["test"]["pr_auc"], 4),
        "brier_after_calibration": round(cal["brier_after"], 4),
        "recall_gap_by_race": round(results[best_name]["fairness_by_race"]["_summary"]["recall_gap"], 4),
        "limits": [
            "Aprende a prática registrada pelos serviços em 2024, inclusive sub-encaminhamento; não é um padrão-ouro.",
            "Transversal: não prevê desfecho futuro. Apoia a decisão da profissional, não a substitui.",
            "Campos ignorados são tratados como categoria própria; não há imputação de causa.",
        ],
        "artifact_size_mb": round(size_mb, 1),
        "trained_at": artifact["trained_at"],
    }
    (MODELS_DIR / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False))
    print(f"saved {out_path} ({size_mb:.1f} MB) · metrics.json · model_card.json · figures in {FIG_DIR} · {summary['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
