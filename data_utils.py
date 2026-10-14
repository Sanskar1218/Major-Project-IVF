"""Data utilities for the IVF outcome-prediction prototype.

Contents
--------
* Synthetic demonstration-data generator (transparent, seeded).
* Safe CSV loading with friendly errors.
* Schema inspection: dtypes, missing values, constant columns, identifiers.
* Heuristic column-to-feature-group suggestions (always user-overridable).
* Target inspection / binarisation (the user must choose the target explicitly).
* Data-quality and leakage warnings.
* A check for whether reliable evaluation is possible at all.

Nothing in here fits a model or touches a test set.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 42
MAX_UPLOAD_MB = 25

DEMO_WARNING = (
    "DEMO MODE — All records and outcomes are synthetic. Results are for software "
    "demonstration only and do not represent real clinical evidence."
)
DEMO_TARGET = "synthetic_outcome"
DEMO_TASK_LABEL = "Synthetic demonstration outcome"
DEMO_ID_COL = "patient_id"

# Feature groups a column can be assigned to.
GROUP_LABELS = {
    "female_clinical": "Female clinical",
    "other_clinical": "Other pre-treatment clinical",
    "male_factor": "Male factor",
    "embryonic": "Embryonic",
    "exclude": "Exclude (ID / post-treatment / unused)",
}
LABEL_TO_GROUP = {v: k for k, v in GROUP_LABELS.items()}
CLINICAL_GROUPS = ("female_clinical", "other_clinical")  # used by Experiment A
EXTRA_GROUPS = ("male_factor", "embryonic")  # added in Experiment B

TASK_OPTIONS = ["Implantation", "Clinical pregnancy", "Live birth", "Other (describe)"]


class DataError(Exception):
    """Raised for problems with user-supplied data that we can explain."""


# --------------------------------------------------------------------------- #
# Synthetic demonstration data
# --------------------------------------------------------------------------- #
# The outcome is drawn from a logistic model whose coefficients are listed here
# and shown in the app. Because embryo/male features are given real weight in
# this generator, a combined model *will* beat a clinical-only model on this
# data -- by construction, not as evidence about real patients.
SYNTH_SPEC = {
    "intercept": -1.0,
    "age_per_year_from_35": -0.12,
    "bmi_per_unit_from_23 (absolute)": -0.03,
    "log_amh": 0.15,
    "per_previous_attempt": -0.12,
    "log_sperm_concentration_over_45": 0.20,
    "motility_per_10_points_over_50": 0.10,
    "morphology_per_point_over_5": 0.06,
    "stage_effect": {"cleavage": 0.0, "morula": 0.2, "blastocyst": 0.7},
    "quality_per_point_over_6": 0.35,
    "per_extra_embryo_transferred": 0.20,
    "unobserved_noise_sd": 0.8,
}

DEMO_MAPPING = {
    "patient_id": "exclude",
    "maternal_age": "female_clinical",
    "bmi": "female_clinical",
    "amh_ng_ml": "female_clinical",
    "previous_ivf_attempts": "female_clinical",
    "sperm_concentration_m_per_ml": "male_factor",
    "sperm_motility_pct": "male_factor",
    "sperm_morphology_score": "male_factor",
    "embryo_stage": "embryonic",
    "embryo_quality_score": "embryonic",
    "embryos_transferred": "embryonic",
}


def generate_synthetic_data(n_rows: int = 1500, seed: int = SEED, missing_rate: float = 0.03) -> pd.DataFrame:
    """Create a synthetic IVF-like table. ALL values are invented."""
    s = SYNTH_SPEC
    rng = np.random.default_rng(seed)
    n = int(n_rows)

    # Female factors
    age = np.clip(rng.normal(35.0, 4.5, n), 21, 46).round()
    bmi = np.clip(rng.normal(25.5, 4.5, n), 16, 45).round(1)
    amh = np.clip(rng.lognormal(0.7 - 0.07 * (age - 35), 0.7), 0.05, 15).round(2)  # declines with age
    prev = np.clip(rng.poisson(0.9, n), 0, 6)

    # Male factors
    conc = np.clip(rng.lognormal(np.log(45), 0.6, n), 1, 200).round(1)
    motility = np.clip(rng.normal(50, 15, n), 0, 95).round()
    morph = np.clip(rng.normal(5, 2.5, n), 0, 15).round(1)

    # Embryonic factors
    stage = rng.choice(["cleavage", "morula", "blastocyst"], size=n, p=[0.25, 0.10, 0.65])
    quality = np.clip(rng.normal(6, 1.8, n), 1, 10).round(1)
    n_transferred = rng.choice([1, 2, 3], size=n, p=[0.55, 0.38, 0.07])

    stage_eff = pd.Series(stage).map(s["stage_effect"]).to_numpy(dtype=float)
    logit = (
        s["intercept"]
        + s["age_per_year_from_35"] * (age - 35)
        + s["bmi_per_unit_from_23 (absolute)"] * np.abs(bmi - 23)
        + s["log_amh"] * np.log(amh)
        + s["per_previous_attempt"] * prev
        + s["log_sperm_concentration_over_45"] * np.log(conc / 45)
        + s["motility_per_10_points_over_50"] * (motility - 50) / 10
        + s["morphology_per_point_over_5"] * (morph - 5)
        + stage_eff
        + s["quality_per_point_over_6"] * (quality - 6)
        + s["per_extra_embryo_transferred"] * (n_transferred - 1)
        + rng.normal(0, s["unobserved_noise_sd"], n)
    )
    outcome = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame(
        {
            "patient_id": [f"SYN-{i + 1:05d}" for i in range(n)],
            "maternal_age": age,
            "bmi": bmi,
            "amh_ng_ml": amh,
            "previous_ivf_attempts": prev,
            "sperm_concentration_m_per_ml": conc,
            "sperm_motility_pct": motility,
            "sperm_morphology_score": morph,
            "embryo_stage": stage,
            "embryo_quality_score": quality,
            "embryos_transferred": n_transferred,
            DEMO_TARGET: outcome,
        }
    )
    # Random missingness (MCAR) in a few measured columns, so imputation is exercised.
    for col in ["bmi", "amh_ng_ml", "sperm_concentration_m_per_ml", "sperm_motility_pct",
                "sperm_morphology_score", "embryo_quality_score"]:
        df.loc[rng.random(n) < missing_rate, col] = np.nan
    return df


# --------------------------------------------------------------------------- #
# CSV loading
# --------------------------------------------------------------------------- #
def load_csv(source, max_mb: float = MAX_UPLOAD_MB):
    """Read a CSV from a path, bytes or an uploaded-file object.

    Returns ``(DataFrame, notes)``. Raises :class:`DataError` with a readable
    message for empty, oversized, unparsable or implausibly tiny files.
    """
    if isinstance(source, (str, Path)):
        try:
            raw = Path(source).read_bytes()
        except OSError as exc:
            raise DataError(f"Could not read the file: {exc}") from exc
    elif isinstance(source, (bytes, bytearray)):
        raw = bytes(source)
    elif hasattr(source, "getvalue"):
        raw = source.getvalue()
    elif hasattr(source, "read"):
        raw = source.read()
    else:
        raise DataError("Unsupported input: expected a file path, bytes or an uploaded file.")

    if not raw or not raw.strip():
        raise DataError("The file is empty.")
    if len(raw) > max_mb * 1024 * 1024:
        raise DataError(f"The file is larger than the {max_mb:.0f} MB limit for this prototype.")

    df, last_err = None, None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            if df.shape[1] == 1 and re.search(r"[;\t|]", str(df.columns[0])):
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")  # sniff delimiter
            break
        except UnicodeDecodeError as exc:
            last_err = exc
        except pd.errors.EmptyDataError as exc:
            raise DataError("The file has no columns or rows to read.") from exc
        except (pd.errors.ParserError, ValueError) as exc:
            raise DataError(f"The file could not be parsed as CSV: {exc}") from exc
    if df is None:
        raise DataError(f"The file's text encoding could not be read: {last_err}")

    notes: list[str] = []
    # Tidy column names: strip whitespace, fill blanks, make unique.
    seen: dict[str, int] = {}
    names = []
    for i, c in enumerate(df.columns):
        name = str(c).strip() or f"unnamed_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    df.columns = names

    n0, c0 = df.shape
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all").reset_index(drop=True)
    if df.shape != (n0, c0):
        notes.append(f"Removed {n0 - df.shape[0]} fully empty row(s) and {c0 - df.shape[1]} fully empty column(s).")

    if df.shape[0] < 10:
        raise DataError(f"Only {df.shape[0]} usable row(s) found; at least 10 are needed to inspect a dataset.")
    if df.shape[1] < 2:
        raise DataError("Only one column found. Check the delimiter, and that the file has a target and features.")

    # Columns that are numbers stored as text -> numeric (reported, never silent).
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        orig_n, conv_n = df[c].notna().sum(), conv.notna().sum()
        if orig_n and conv_n >= 0.95 * orig_n:
            lost = int(orig_n - conv_n)
            df[c] = conv
            notes.append(
                f"Column '{c}' was read as text but is mostly numeric; converted to numbers"
                + (f" ({lost} unparseable value(s) set to missing)." if lost else ".")
            )
    return df, notes


# --------------------------------------------------------------------------- #
# Schema inspection
# --------------------------------------------------------------------------- #
def is_numeric_col(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def detect_feature_types(df: pd.DataFrame, cols) -> tuple[list[str], list[str]]:
    """Split columns into (numeric, categorical) using dtypes only (no value statistics)."""
    numeric = [c for c in cols if is_numeric_col(df[c])]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical


def describe_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        nun = df[c].nunique(dropna=True)
        kind = "constant" if nun <= 1 else ("numeric" if is_numeric_col(df[c]) else "categorical")
        rows.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "kind": kind,
            "unique values": int(nun),
            "missing %": round(100 * df[c].isna().mean(), 1),
        })
    return pd.DataFrame(rows)


def _tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t for t in re.split(r"[^A-Za-z0-9]+", s.lower()) if t]


def _hit(tokens: list[str], keywords) -> bool:
    """Token-level keyword match: short keywords must match a whole token
    (so 'age' does not match 'stage'); longer ones match as a prefix."""
    for kw in keywords:
        if len(kw) <= 3:
            if kw in tokens:
                return True
        elif any(t.startswith(kw) for t in tokens):
            return True
    return False


_ID_TOKENS = {"id", "mrn", "uuid", "guid", "pid", "identifier"}
_LEAKAGE_KW = ["pregnan", "birth", "implant", "hcg", "outcome", "gestat", "fetal", "foetal",
               "heartbeat", "miscarr", "success", "born", "deliver", "live"]
_MALE_KW = ["sperm", "semen", "motil", "morpholog", "concentr", "male", "paternal"]
_EMBRYO_KW = ["embryo", "blast", "stage", "grade", "transfer", "oocyte", "egg", "fertili", "cleav", "pgt", "morula"]
_FEMALE_KW = ["age", "bmi", "amh", "fsh", "lh", "estradiol", "prior", "previous", "attempt", "infertil",
              "parity", "gravid", "antral", "afc", "endometri", "maternal", "female"]


def looks_like_identifier(name: str) -> bool:
    t = _tokens(name)
    if _ID_TOKENS & set(t):
        return True
    if ("patient" in t or "couple" in t) and {"number", "no", "code", "ref"} & set(t):
        return True
    return t in (["patient"], ["name"])


_LEAKAGE_SUBSTR = ["pregnan", "birth", "implant", "hcg", "outcome", "gestat", "fetal", "foetal",
                   "heartbeat", "miscarr", "success"]


def leakage_name_flag(name: str) -> bool:
    """True if a column *name* suggests a post-treatment result (possible leakage).
    Also matches on the squashed name so 'hCG', 'BetaHCG' or 'LiveBirth' are caught
    even when camelCase splitting would break the keyword apart."""
    squashed = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return any(k in squashed for k in _LEAKAGE_SUBSTR) or _hit(_tokens(name), _LEAKAGE_KW)


def find_identifier_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        uniq_ratio = df[c].nunique(dropna=True) / max(len(df), 1)
        if looks_like_identifier(c) or (not is_numeric_col(df[c]) and uniq_ratio >= 0.9 and len(df) >= 20):
            out.append(c)
    return out


def suggest_mapping(df: pd.DataFrame, target_col: str | None = None) -> dict[str, tuple[str, str]]:
    """Heuristic ``column -> (group, reason)``. Unknown columns default to 'exclude'
    so nothing is silently included; the user assigns them explicitly."""
    out: dict[str, tuple[str, str]] = {}
    for c in df.columns:
        if c == target_col:
            continue
        t = _tokens(c)
        if looks_like_identifier(c):
            out[c] = ("exclude", "looks like an identifier")
        elif leakage_name_flag(c):
            out[c] = ("exclude", "name suggests a post-treatment result (possible leakage)")
        elif df[c].nunique(dropna=True) <= 1:
            out[c] = ("exclude", "constant column")
        elif _hit(t, _MALE_KW):
            out[c] = ("male_factor", "name suggests a male-factor variable")
        elif _hit(t, _EMBRYO_KW):
            out[c] = ("embryonic", "name suggests an embryo/oocyte variable")
        elif _hit(t, _FEMALE_KW):
            out[c] = ("female_clinical", "name suggests a female clinical variable")
        else:
            out[c] = ("exclude", "unrecognised — assign a group manually if it is a pre-treatment variable")
    return out


# --------------------------------------------------------------------------- #
# Target handling
# --------------------------------------------------------------------------- #
def inspect_target(series: pd.Series) -> dict:
    vc = series.dropna().value_counts()
    return {
        "classes": list(vc.index),
        "counts": {str(k): int(v) for k, v in vc.items()},
        "n_classes": int(len(vc)),
        "n_missing": int(series.isna().sum()),
        "usable": len(vc) == 2,
    }


def binarize_target(series: pd.Series, positive_value) -> tuple[np.ndarray, np.ndarray]:
    """Return (y, keep_mask). Rows with a missing target are dropped by the mask.
    ``positive_value`` is the class the user declared as the positive outcome."""
    keep = series.notna().to_numpy()
    y = (series.astype(str) == str(positive_value)).to_numpy()[keep].astype(int)
    return y, keep


# --------------------------------------------------------------------------- #
# Data-quality, identifier and leakage checks
# --------------------------------------------------------------------------- #
def check_identifiers(df: pd.DataFrame, id_col: str | None) -> dict:
    res = {"duplicate_rows": int(df.duplicated().sum()), "repeated_ids": 0, "n_ids": None, "messages": []}
    if res["duplicate_rows"]:
        res["messages"].append(f"{res['duplicate_rows']} fully duplicated row(s) found.")
    if id_col and id_col in df.columns:
        ids = df[id_col].dropna()
        res["n_ids"] = int(ids.nunique())
        res["repeated_ids"] = int(len(ids) - ids.nunique())
        if res["repeated_ids"]:
            res["messages"].append(
                f"'{id_col}' repeats: {len(ids)} rows but only {res['n_ids']} distinct IDs (repeated cycles per patient). "
                "The train/test split will be grouped by this ID so no patient appears on both sides."
            )
    return res


def groups_if_repeated(df: pd.DataFrame, id_col: str | None):
    """Group labels for a grouped split, or None if there is nothing to group."""
    if not id_col or id_col not in df.columns:
        return None
    ids = df[id_col].fillna("__missing__").astype(str)
    if ids.nunique() == len(ids):
        return None  # every row is its own patient: nothing to group
    return ids.to_numpy()


def check_feature_quality(df: pd.DataFrame, feature_cols) -> list[tuple[str, str]]:
    """Return a list of (severity, message); severity is 'error' | 'warning'."""
    out = []
    for c in feature_cols:
        s = df[c]
        if s.nunique(dropna=True) <= 1:
            out.append(("warning", f"'{c}' is constant (or entirely missing) and carries no information."))
        miss = s.isna().mean()
        if miss > 0.5:
            out.append(("warning", f"'{c}' is {miss:.0%} missing; imputed values will dominate it."))
        elif miss > 0.2:
            out.append(("warning", f"'{c}' is {miss:.0%} missing."))
        if not is_numeric_col(s) and s.nunique(dropna=True) > max(30, 0.3 * len(df)):
            out.append(("warning", f"'{c}' has very many distinct text values ({s.nunique()}); it may be an identifier."))
        if looks_like_identifier(c):
            out.append(("error", f"'{c}' looks like an identifier and should not be a feature."))
        if leakage_name_flag(c):
            out.append(("warning", f"'{c}' may be a post-treatment variable (name resembles an outcome/result). "
                                   "Including it can leak the answer into the model."))
    return out


def scan_single_feature_predictiveness(df: pd.DataFrame, y: np.ndarray, feature_cols, auc_threshold: float = 0.97):
    """Flag any single feature that almost perfectly separates the classes — a classic
    leakage signature. Uses the whole labelled table, only to *warn*; no model is fit."""
    out = []
    y = np.asarray(y)
    for c in list(feature_cols)[:80]:
        s = df[c]
        m = s.notna().to_numpy()
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        if is_numeric_col(s):
            auc = roc_auc_score(y[m], s.to_numpy(dtype=float)[m])
            if max(auc, 1 - auc) >= auc_threshold:
                out.append(f"'{c}' alone separates the classes with AUC≈{max(auc, 1 - auc):.3f}. "
                           "This is suspiciously high; check whether it is recorded after the outcome.")
        else:
            vals = s.astype(str).to_numpy()[m]
            if len(set(vals)) > max(20, 0.2 * m.sum()):
                continue
            g = pd.DataFrame({"v": vals, "y": y[m]}).groupby("v")["y"].agg(["sum", "count"])
            acc = np.maximum(g["sum"], g["count"] - g["sum"]).sum() / g["count"].sum()
            base = max(y[m].mean(), 1 - y[m].mean())
            if acc >= 0.97 and acc - base > 0.1:
                out.append(f"'{c}' alone predicts the label with ~{acc:.0%} accuracy (baseline {base:.0%}). "
                           "Check for leakage.")
    return out


MIN_ROWS = 60
MIN_MINORITY = 20


def check_evaluability(y: np.ndarray, test_size: float = 0.25) -> tuple[bool, list[str], list[str]]:
    """Decide whether a held-out evaluation can be meaningful at all.

    The thresholds are pragmatic minimums for this prototype, NOT guarantees of
    statistical adequacy: an AUC from a few dozen positives is still very noisy.
    Returns ``(ok, errors, warnings)``.
    """
    y = np.asarray(y)
    errors, warnings = [], []
    n = len(y)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        errors.append("Only one outcome class is present, so no model can be trained or evaluated.")
        return False, errors, warnings
    minority = int(counts.min())
    if n < MIN_ROWS:
        errors.append(f"Only {n} labelled rows (need at least {MIN_ROWS}) — reliable evaluation is not possible.")
    if minority < MIN_MINORITY:
        errors.append(f"The rarer class has only {minority} rows (need at least {MIN_MINORITY}) — "
                      "metrics such as ROC-AUC would be meaningless.")
    if minority * test_size < 5:
        errors.append(f"With a {test_size:.0%} test split the test set would hold ~{minority * test_size:.1f} "
                      "rarer-class rows (need at least 5). Increase the test fraction or add data.")
    if not errors:
        if n < 300 or minority < 100:
            warnings.append(f"Small sample (n={n}, rarer class={minority}): expect wide uncertainty in every metric.")
        if minority / n < 0.1:
            warnings.append("Strong class imbalance: accuracy is misleading; look at ROC-AUC, recall and calibration.")
    return (not errors), errors, warnings
