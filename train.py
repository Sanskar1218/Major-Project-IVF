"""Training pipeline: Experiment A (clinical-only) vs Experiment B (combined features).

Design rules enforced here
--------------------------
* One train/test split of the RAW table is made first; all imputers, scalers and
  encoders live inside sklearn Pipelines and are fitted on training rows only.
* Both experiments use exactly the same train and test rows.
* Cross-validation (model-selection signal) runs on the training rows only. The
  test set is touched once, for the final report.
* If patient IDs repeat, the split is grouped so a patient is never on both sides.
* Evaluation is refused (not faked) when the data are too small or one-class.
* Saved models are verified against a SHA-256 manifest before loading, and the app
  offers no way to load an uploaded model file.

CLI:  python train.py --demo
      python train.py --csv data/my.csv --target live_birth --positive 1 --task "Live birth" --id-col patient_id
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_utils as du
import evaluate as ev

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]
EXPERIMENT_LABELS = {"A": "A: Clinical-only", "B": "B: Combined features"}


class TrainingError(Exception):
    """Raised when training/evaluation cannot be done reliably; message is user-facing."""


class ModelIntegrityError(Exception):
    """Raised when saved artefacts are missing, altered or from another scikit-learn version."""


def _make_model(name: str, seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1)
    if name == "Gradient Boosting":
        # scikit-learn's histogram boosting: lightweight, handles small data, no extra install.
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed)
    raise ValueError(f"Unknown model: {name}")


def prepare_X(df: pd.DataFrame, numeric_cols, categorical_cols) -> pd.DataFrame:
    """Type-normalise the raw columns. No statistics are learned here."""
    X = pd.DataFrame(index=df.index)
    for c in numeric_cols:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in categorical_cols:
        X[c] = df[c].astype(object).map(lambda v: v if pd.isna(v) else str(v))
    return X


def build_pipeline(model_name: str, numeric_cols, categorical_cols, seed: int = du.SEED) -> Pipeline:
    """Impute -> (scale for LR) / one-hot -> classifier, all fitted on training data only."""
    parts = []
    if numeric_cols:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_name == "Logistic Regression":  # scaling matters for LR, not for trees
            steps.append(("scaler", StandardScaler()))
        parts.append(("num", Pipeline(steps), list(numeric_cols)))
    if categorical_cols:
        parts.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), list(categorical_cols)))
    if not parts:
        raise TrainingError("No features selected.")
    return Pipeline([("preprocess", ColumnTransformer(parts, remainder="drop")),
                     ("model", _make_model(model_name, seed))])


def make_split(y: np.ndarray, groups=None, test_size: float = 0.25, seed: int = du.SEED):
    """Stratified split of row positions; grouped by patient when IDs repeat."""
    idx = np.arange(len(y))
    if groups is None:
        return train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    n_splits = max(2, int(round(1 / test_size)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr, te = next(iter(sgkf.split(idx, y, groups)))
    if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
        raise TrainingError("A patient-grouped split left one class missing from the train or test set. "
                            "More data (or fewer large patient groups) is needed.")
    return tr, te


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def experiment_features(mapping: dict, exp: str, extra_groups, df_cols, exclude=()) -> list[str]:
    groups = set(du.CLINICAL_GROUPS) | (set(extra_groups) if exp == "B" else set())
    return [c for c, g in mapping.items() if g in groups and c in df_cols and c not in exclude]


def run_experiments(
    df: pd.DataFrame,
    target_col: str,
    positive_value,
    mapping: dict,
    task_label: str,
    is_demo: bool = False,
    id_col: str | None = None,
    experiments=("A", "B"),
    extra_groups=du.EXTRA_GROUPS,
    model_names=None,
    test_size: float = 0.25,
    seed: int = du.SEED,
    cv_folds: int = 5,
    n_boot: int = 500,
    perm_repeats: int = 10,
    progress_cb=None,
) -> dict:
    """Train and evaluate every requested (experiment, model) pair. Returns {'meta', 'bundles'}."""
    model_names = list(model_names or MODEL_NAMES)
    if target_col not in df.columns:
        raise TrainingError(f"Target column '{target_col}' not found.")
    warnings: list[str] = []

    # --- target ---------------------------------------------------------------
    y_all, keep = du.binarize_target(df[target_col], positive_value)
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.append(f"{n_dropped} row(s) with a missing target were dropped.")
    d = df.loc[keep].reset_index(drop=True)

    ok, errs, warns = du.check_evaluability(y_all, test_size)
    if not ok:
        raise TrainingError("Reliable evaluation is not possible with this data. " + " ".join(errs))
    warnings += warns

    # --- split (raw table, before any preprocessing) ----------------------------
    groups = du.groups_if_repeated(d, id_col)
    tr_idx, te_idx = make_split(y_all, groups, test_size, seed)
    y_tr, y_te = y_all[tr_idx], y_all[te_idx]
    d_tr, d_te = d.iloc[tr_idx], d.iloc[te_idx]
    if groups is not None:
        assert not set(groups[tr_idx]) & set(groups[te_idx]), "patient appears in both train and test"

    # --- experiments ---------------------------------------------------------------
    exclude = {target_col} | ({id_col} if id_col else set())
    feats = {e: experiment_features(mapping, e, extra_groups, d.columns, exclude) for e in experiments}
    runnable = {}
    for e, f in feats.items():
        f = [c for c in f if d_tr[c].notna().any()]  # drop columns that are entirely missing in training
        if not f:
            warnings.append(f"Experiment {e} skipped: no usable features in its groups.")
        elif e == "B" and "A" in feats and set(f) == set(feats["A"]):
            warnings.append("Experiment B skipped: it would use exactly the same features as A "
                            "(no male-factor or embryonic columns were mapped/selected).")
        else:
            runnable[e] = f
    if not runnable:
        raise TrainingError("No experiment can run: assign columns to feature groups first.")

    # CV folds for model selection: training rows only.
    n_splits = min(int(cv_folds), int(np.bincount(y_tr).min()))
    cv_splits = None
    if n_splits >= 2:
        if groups is not None:
            cv_splits = list(StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
                             .split(np.zeros(len(y_tr)), y_tr, groups[tr_idx]))
        else:
            cv_splits = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed)
                             .split(np.zeros(len(y_tr)), y_tr))

    bundles: dict = {}
    total = len(runnable) * len(model_names)
    done = 0
    for e, f in runnable.items():
        num, cat = du.detect_feature_types(d, f)
        X_tr, X_te = prepare_X(d_tr, num, cat), prepare_X(d_te, num, cat)
        ref = {c: float(X_tr[c].median()) for c in num}
        ref.update({c: X_tr[c].mode(dropna=True).iloc[0] for c in cat})
        ranges = {c: (float(X_tr[c].min()), float(X_tr[c].max())) for c in num}
        cats = {c: sorted(X_tr[c].dropna().unique().tolist()) for c in cat}

        for mname in model_names:
            if progress_cb:
                progress_cb(done / total, f"Experiment {e} · {mname}")
            pipe = build_pipeline(mname, num, cat, seed)
            cv_mean = cv_std = None
            if cv_splits:
                try:
                    sc = cross_val_score(pipe, X_tr, y_tr, cv=cv_splits, scoring="roc_auc")
                    cv_mean, cv_std = float(np.mean(sc)), float(np.std(sc))
                except Exception as exc:  # report, don't hide
                    warnings.append(f"Cross-validation failed for {mname} (exp {e}): {exc}")
            pipe.fit(X_tr, y_tr)
            proba = pipe.predict_proba(X_te)[:, 1]

            bid = f"{e}__{_slug(mname)}"
            bundles[bid] = {
                "id": bid, "experiment": e, "experiment_label": EXPERIMENT_LABELS[e], "model_name": mname,
                "pipeline": pipe, "features": f, "numeric": num, "categorical": cat,
                "reference": ref, "ranges": ranges, "categories": cats,
                "cv_auc_mean": cv_mean, "cv_auc_std": cv_std, "cv_folds": n_splits if cv_splits else None,
                "y_test": y_te, "proba_test": proba,
                "permutation_importance": ev.permutation_importance_table(pipe, X_te, y_te, seed, perm_repeats),
                "tree_importance": ev.tree_importance_table(pipe, f),
                "task_label": task_label, "is_demo": bool(is_demo),
                "sklearn_version": sklearn.__version__,
            }
            done += 1

    comparisons = []
    if "A" in runnable and "B" in runnable:
        for mname in model_names:
            a, b = bundles[f"A__{_slug(mname)}"], bundles[f"B__{_slug(mname)}"]
            c = ev.bootstrap_auc_difference(y_te, a["proba_test"], b["proba_test"], n_boot, seed)
            comparisons.append({"model": mname, **(c or {"delta": None})})
    if progress_cb:
        progress_cb(1.0, "Done")

    meta = {
        "task_label": task_label, "is_demo": bool(is_demo), "target_col": target_col,
        "positive_value": str(positive_value), "n_rows": int(len(d)), "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "class_counts": {"train": {"negative": int((y_tr == 0).sum()), "positive": int((y_tr == 1).sum())},
                         "test": {"negative": int((y_te == 0).sum()), "positive": int((y_te == 1).sum())}},
        "seed": seed, "test_size": test_size,
        "split_type": "patient-grouped stratified" if groups is not None else "stratified random",
        "features": runnable, "warnings": warnings, "comparisons": comparisons,
        "created_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "sklearn_version": sklearn.__version__,
    }
    return {"meta": meta, "bundles": bundles}


# --------------------------------------------------------------------------- #
# Prediction helpers (use the exact fitted pipeline)
# --------------------------------------------------------------------------- #
def _missing(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and not v.strip())


def validate_inputs(bundle: dict, values: dict) -> tuple[list[str], list[str]]:
    """Return (errors, warnings). Any error blocks prediction."""
    errors, warns = [], []
    for c in bundle["features"]:
        v = values.get(c)
        if _missing(v):
            errors.append(f"'{c}' is required.")
            continue
        if c in bundle["numeric"]:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                errors.append(f"'{c}' must be a number.")
                continue
            if not np.isfinite(fv):
                errors.append(f"'{c}' must be a finite number.")
                continue
            lo, hi = bundle["ranges"][c]
            span = max(hi - lo, 1e-9)
            if fv < lo - 0.25 * span or fv > hi + 0.25 * span:
                warns.append(f"'{c}'={fv:g} is well outside the training range [{lo:g}, {hi:g}]; "
                             "the estimate is an extrapolation.")
        elif str(v) not in bundle["categories"][c]:
            errors.append(f"'{c}' must be one of {bundle['categories'][c]}.")
    return errors, warns


def _row_frame(bundle: dict, values: dict) -> pd.DataFrame:
    row = {c: (float(values[c]) if c in bundle["numeric"] else str(values[c])) for c in bundle["features"]}
    return pd.DataFrame([row], columns=bundle["features"])


def predict_single(bundle: dict, values: dict, threshold: float = 0.5) -> dict:
    errors, warns = validate_inputs(bundle, values)
    if errors:
        raise ValueError(" ".join(errors))
    row = _row_frame(bundle, values)
    p = float(bundle["pipeline"].predict_proba(row)[0, 1])
    return {"probability": p, "predicted_positive": p >= threshold, "threshold": threshold,
            "row": row, "warnings": warns}


def local_explanation(bundle: dict, row: pd.DataFrame) -> pd.DataFrame:
    """Occlusion-style explanation: how much does the estimate change if one input is
    replaced by the typical (training median/mode) value? Not SHAP, not causal, and it
    ignores interactions; it only describes this model's behaviour for this input."""
    pipe = bundle["pipeline"]
    base = float(pipe.predict_proba(row)[0, 1])
    out = []
    for c in bundle["features"]:
        alt = row.copy()
        alt[c] = bundle["reference"][c]
        out.append({"feature": c, "value": row[c].iloc[0], "typical value": bundle["reference"][c],
                    "effect on estimate": base - float(pipe.predict_proba(alt)[0, 1])})
    t = pd.DataFrame(out)
    return t.reindex(t["effect on estimate"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Saving / loading with integrity checks
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def save_run(results: dict, models_dir: Path | str = MODELS_DIR) -> Path:
    """Write bundles + run metadata and a SHA-256 manifest. Replaces any previous run."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = models_dir / "manifest.json"
    if manifest_path.exists():  # remove only files an earlier run created
        try:
            for name in json.loads(manifest_path.read_text()).get("files", {}):
                if Path(name).name == name:
                    (models_dir / name).unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
    files = {}
    for bid, b in results["bundles"].items():
        p = models_dir / f"{bid}.joblib"
        joblib.dump(b, p)
        files[p.name] = _sha256(p)
    meta_p = models_dir / "run_meta.json"
    meta_p.write_text(json.dumps(_jsonable(results["meta"]), indent=2))
    files[meta_p.name] = _sha256(meta_p)
    manifest_path.write_text(json.dumps({"format": 1, "sklearn_version": sklearn.__version__, "files": files}, indent=2))
    return models_dir


def load_run(models_dir: Path | str = MODELS_DIR) -> dict:
    """Load a previously saved run, after verifying every file against the manifest.

    Only files named in the manifest are read. The hash check catches corruption and
    accidental edits; the real safeguard is that this app never accepts a model upload.
    """
    models_dir = Path(models_dir)
    mp = models_dir / "manifest.json"
    if not mp.exists():
        raise FileNotFoundError("No saved models found. Train first.")
    try:
        manifest = json.loads(mp.read_text())
    except ValueError as exc:
        raise ModelIntegrityError("The model manifest is unreadable. Retrain to regenerate it.") from exc
    if manifest.get("sklearn_version") != sklearn.__version__:
        raise ModelIntegrityError(f"Saved models were built with scikit-learn {manifest.get('sklearn_version')}, "
                                  f"but {sklearn.__version__} is installed. Retrain the models.")
    bundles = {}
    for name, digest in manifest["files"].items():
        if Path(name).name != name:
            raise ModelIntegrityError(f"Unsafe file name in manifest: {name}")
        p = models_dir / name
        if not p.exists():
            raise ModelIntegrityError(f"Missing file: {name}. Retrain to regenerate the models.")
        if _sha256(p) != digest:
            raise ModelIntegrityError(f"{name} does not match its recorded checksum (modified or corrupted).")
    for name in manifest["files"]:
        if name.endswith(".joblib"):
            b = joblib.load(models_dir / name)
            bundles[b["id"]] = b
    meta = json.loads((models_dir / "run_meta.json").read_text())
    return {"meta": meta, "bundles": bundles}


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def _cli() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate the IVF prototype from the command line.")
    ap.add_argument("--demo", action="store_true", help="use synthetic demonstration data")
    ap.add_argument("--csv", help="path to a real CSV")
    ap.add_argument("--target", help="target column (required with --csv)")
    ap.add_argument("--positive", help="value of the target that counts as the positive outcome")
    ap.add_argument("--task", help="what the target represents, e.g. 'Live birth' (required with --csv)")
    ap.add_argument("--id-col", help="optional patient-ID column (enables grouped splitting)")
    ap.add_argument("--mapping", help="optional JSON file {column: group}; groups: "
                                      + ", ".join(du.GROUP_LABELS))
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=du.SEED)
    ap.add_argument("--test-size", type=float, default=0.25)
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()

    if a.demo == bool(a.csv):
        ap.error("choose exactly one of --demo or --csv")
    if a.demo:
        print(du.DEMO_WARNING)
        df = du.generate_synthetic_data(a.rows, a.seed)
        target, positive, task, id_col, mapping, is_demo = (
            du.DEMO_TARGET, 1, du.DEMO_TASK_LABEL, du.DEMO_ID_COL, du.DEMO_MAPPING, True)
    else:
        if not (a.target and a.positive is not None and a.task):
            ap.error("--csv requires --target, --positive and --task (the outcome is never guessed)")
        df, notes = du.load_csv(a.csv)
        for n in notes:
            print("note:", n)
        target, positive, task, id_col, is_demo = a.target, a.positive, a.task, a.id_col, False
        if a.mapping:
            mapping = json.loads(Path(a.mapping).read_text())
        else:
            mapping = {c: g for c, (g, why) in du.suggest_mapping(df, target).items()}
            print("Auto-suggested mapping (override with --mapping):")
            for c, g in mapping.items():
                print(f"  {c:35s} {g}")
    res = run_experiments(df, target, positive, mapping, task, is_demo, id_col, test_size=a.test_size, seed=a.seed)
    print(f"\nTask: {task} | rows={res['meta']['n_rows']} | split={res['meta']['split_type']}")
    for w in res["meta"]["warnings"]:
        print("warning:", w)
    print(ev.metrics_table(res["bundles"]).round(3).to_string(index=False))
    for c in res["meta"]["comparisons"]:
        print(f"\n[{c['model']}]", ev.interpret_comparison(c, is_demo))
    if not a.no_save:
        print("\nSaved to", save_run(res))


if __name__ == "__main__":
    _cli()
