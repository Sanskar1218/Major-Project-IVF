"""Evaluation helpers: metrics, curves, importance and A-vs-B comparison.

Every number is computed from the predictions passed in. Metrics that are
mathematically undefined (e.g. ROC-AUC with a one-class test set, precision with
no predicted positives) are returned as ``None`` with an explanatory note rather
than being replaced by a made-up value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score, roc_curve


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    n = len(y)
    notes: list[str] = []

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    specificity = tn / (tn + fp) if (tn + fp) > 0 else None
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None
    if precision is None:
        notes.append("Precision is undefined: no case was predicted positive at this threshold.")
    if recall is None:
        notes.append("Recall is undefined: the test set has no positive cases.")

    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, p))
    else:
        auc = None
        notes.append("ROC-AUC is undefined: the test set contains only one class.")

    return {
        "n_test": n, "n_pos": int(y.sum()), "n_neg": int(n - y.sum()), "threshold": float(threshold),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": (tp + tn) / n if n else None,
        "precision": precision, "recall": recall, "specificity": specificity, "f1": f1,
        "roc_auc": auc, "brier": float(brier_score_loss(y, p)) if n else None,
        "notes": notes,
    }


def roc_curve_points(y_true, proba):
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, proba)
    return fpr, tpr, float(roc_auc_score(y, proba))


def calibration_points(y_true, proba, n_bins: int | None = None):
    """Quantile-binned reliability curve, or None if the test set is too small."""
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        return None
    bins = n_bins or min(10, len(y) // 20)
    if bins < 3:
        return None
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=bins, strategy="quantile")
    return mean_pred, frac_pos


def metrics_table(bundles: dict, threshold: float = 0.5) -> pd.DataFrame:
    """One row per (experiment, model), metrics recomputed from the stored test predictions."""
    rows = []
    for b in bundles.values():
        m = compute_metrics(b["y_test"], b["proba_test"], threshold)
        rows.append({
            "Experiment": b["experiment_label"],
            "Model": b["model_name"],
            "Features": len(b["features"]),
            "ROC-AUC": m["roc_auc"],
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "Brier": m["brier"],
            "CV ROC-AUC (train only)": b.get("cv_auc_mean"),
        })
    return pd.DataFrame(rows)


def feature_owners(pipeline) -> list[str] | None:
    """Original feature name for each column the preprocessor outputs (for aggregation)."""
    pre = pipeline.named_steps["preprocess"]
    owners: list[str] = []
    for name, trans, cols in pre.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if name == "num":
            owners.extend(cols)
        elif name == "cat":
            ohe = trans.named_steps["onehot"]
            for col, cats in zip(cols, ohe.categories_):
                owners.extend([col] * len(cats))
    return owners


def tree_importance_table(pipeline, original_features) -> pd.DataFrame | None:
    """Impurity-based importance of tree ensembles, summed back to the original columns.
    Biased toward high-cardinality / continuous features; use alongside permutation importance."""
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None
    owners = feature_owners(pipeline)
    imp = np.asarray(model.feature_importances_)
    if owners is None or len(owners) != len(imp):
        return None
    agg = {f: 0.0 for f in original_features}
    for o, v in zip(owners, imp):
        agg[o] += float(v)
    return (pd.DataFrame({"feature": list(agg), "importance": list(agg.values())})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def permutation_importance_table(pipeline, X_test, y_test, seed: int, n_repeats: int = 10) -> pd.DataFrame | None:
    """Drop in test ROC-AUC when one original column is shuffled. Works for any model.
    Reflects what *this fitted model* relies on, not causal effects."""
    if len(np.unique(y_test)) < 2:
        return None
    r = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                               n_repeats=n_repeats, random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": list(X_test.columns),
                          "importance_mean": r.importances_mean,
                          "importance_std": r.importances_std})
            .sort_values("importance_mean", ascending=False).reset_index(drop=True))


def bootstrap_auc_difference(y_true, p_a, p_b, n_boot: int = 500, seed: int = 42) -> dict | None:
    """Paired bootstrap of AUC(B) - AUC(A) over the *same* test rows.

    The interval reflects test-set sampling noise only (not retraining variability,
    not a different hospital), so it is a minimum bar for claiming an improvement.
    """
    y = np.asarray(y_true).astype(int)
    p_a, p_b = np.asarray(p_a, float), np.asarray(p_b, float)
    if len(np.unique(y)) < 2:
        return None
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(roc_auc_score(yy, p_b[idx]) - roc_auc_score(yy, p_a[idx]))
    if len(diffs) < 50:
        return None
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    auc_a, auc_b = float(roc_auc_score(y, p_a)), float(roc_auc_score(y, p_b))
    return {"auc_a": auc_a, "auc_b": auc_b, "delta": auc_b - auc_a,
            "ci_low": float(lo), "ci_high": float(hi), "n_boot_valid": len(diffs),
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def interpret_comparison(comp: dict, is_demo: bool) -> str:
    """Plain-language reading of an A-vs-B comparison that never overclaims."""
    if comp is None or comp.get("delta") is None:
        return "The comparison could not be computed (test set too small or single-class)."
    d, lo, hi = comp["delta"], comp["ci_low"], comp["ci_high"]
    core = (f"Combined minus clinical-only ROC-AUC = {d:+.3f} (95% bootstrap interval {lo:+.3f} to {hi:+.3f}).")
    if is_demo:
        return (core + " This is SYNTHETIC data in which the male/embryo variables were deliberately given "
                "predictive weight, so any gain is built in and is not evidence about real patients.")
    if comp["ci_excludes_zero"] and d > 0:
        return (core + " On this single test split the interval excludes zero, which is consistent with a gain from the extra "
                "feature groups. It is one dataset and one split; it still needs external validation before any claim.")
    if comp["ci_excludes_zero"] and d < 0:
        return core + " The combined model scored lower on this split; adding features did not help here."
    return core + " The interval includes zero, so this experiment does not demonstrate a difference between the two feature sets."
