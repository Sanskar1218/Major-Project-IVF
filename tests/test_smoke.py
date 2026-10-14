"""Smoke and validation tests. Run from the project folder:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data_utils as du  # noqa: E402
import evaluate as ev  # noqa: E402
import train as tr  # noqa: E402


@pytest.fixture(scope="module")
def demo_df():
    return du.generate_synthetic_data(1000, seed=7)


@pytest.fixture(scope="module")
def demo_results(demo_df):
    return tr.run_experiments(demo_df, du.DEMO_TARGET, 1, du.DEMO_MAPPING, du.DEMO_TASK_LABEL, is_demo=True,
                              id_col=du.DEMO_ID_COL, n_boot=100, perm_repeats=3)


# ---------------- synthetic data ----------------
def test_synthetic_is_reproducible_and_valid(demo_df):
    again = du.generate_synthetic_data(1000, seed=7)
    pd.testing.assert_frame_equal(demo_df, again)
    assert set(demo_df[du.DEMO_TARGET].unique()) == {0, 1}
    assert demo_df.isna().any().any()  # missingness present so imputation is exercised
    assert not du.generate_synthetic_data(1000, seed=8).equals(demo_df)


# ---------------- CSV loading ----------------
@pytest.mark.parametrize("raw", [b"", b"   \n", b"onlyonecolumn\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n"])
def test_load_csv_rejects_bad_files(raw):
    with pytest.raises(du.DataError):
        du.load_csv(raw)


def test_load_csv_rejects_missing_file(tmp_path):
    with pytest.raises(du.DataError):
        du.load_csv(tmp_path / "nope.csv")


def test_load_csv_handles_semicolons_and_numeric_text():
    rows = "\n".join(f"{i};{i * 1.5};{'a' if i % 2 else 'b'}" for i in range(30))
    df, notes = du.load_csv(("x;y;z\n" + rows).encode())
    assert list(df.columns) == ["x", "y", "z"] and len(df) == 30


# ---------------- mapping heuristics ----------------
def test_mapping_heuristics():
    df = pd.DataFrame({c: [1, 2, 3, 4] for c in
                       ["patient_id", "female_age", "embryo_stage", "sperm_count", "live_birth_flag", "hcg_level", "favourite_colour"]})
    m = {c: g for c, (g, _) in du.suggest_mapping(df, None).items()}
    assert m["patient_id"] == "exclude"
    assert m["female_age"] == "female_clinical"
    assert m["embryo_stage"] == "embryonic"  # 'stage' must not match 'age'
    assert m["sperm_count"] == "male_factor"
    assert m["live_birth_flag"] == "exclude" and m["hcg_level"] == "exclude"  # leakage-like names
    assert m["favourite_colour"] == "exclude"  # unknown -> excluded, never silently included


# ---------------- evaluability ----------------
def test_evaluability_refuses_small_or_one_class():
    assert not du.check_evaluability(np.zeros(500, int))[0]
    assert not du.check_evaluability(np.array([0] * 50 + [1] * 5))[0]
    assert du.check_evaluability(np.array([0] * 300 + [1] * 200))[0]


def test_training_refuses_tiny_dataset(demo_df):
    with pytest.raises(tr.TrainingError):
        tr.run_experiments(demo_df.head(40), du.DEMO_TARGET, 1, du.DEMO_MAPPING, "x")


# ---------------- training pipeline ----------------
def test_all_models_and_experiments_run(demo_results):
    b = demo_results["bundles"]
    assert len(b) == 6
    for bundle in b.values():
        m = ev.compute_metrics(bundle["y_test"], bundle["proba_test"])
        assert 0.5 < m["roc_auc"] <= 1 and 0 <= m["accuracy"] <= 1 and 0 <= m["brier"] <= 1
    # same test rows for A and B
    for name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        np.testing.assert_array_equal(b[f"A__{name}"]["y_test"], b[f"B__{name}"]["y_test"])
    assert len(b["A__random_forest"]["features"]) == 4 and len(b["B__random_forest"]["features"]) == 10
    assert len(demo_results["meta"]["comparisons"]) == 3


def test_preprocessing_fitted_on_train_only(demo_df):
    """Imputer medians must equal the TRAINING medians, not whole-data medians."""
    res = tr.run_experiments(demo_df, du.DEMO_TARGET, 1, du.DEMO_MAPPING, "x", id_col=du.DEMO_ID_COL,
                             model_names=["Logistic Regression"], experiments=("A",), n_boot=50, perm_repeats=2)
    b = res["bundles"]["A__logistic_regression"]
    y, _ = du.binarize_target(demo_df[du.DEMO_TARGET], 1)
    tr_idx, _ = tr.make_split(y, None, 0.25, du.SEED)
    X_tr = tr.prepare_X(demo_df.iloc[tr_idx], b["numeric"], b["categorical"])
    stats = b["pipeline"].named_steps["preprocess"].named_transformers_["num"].named_steps["imputer"].statistics_
    np.testing.assert_allclose(stats, X_tr[b["numeric"]].median().to_numpy())


def test_reproducible(demo_df):
    kw = dict(model_names=["Random Forest"], experiments=("B",), n_boot=50, perm_repeats=2)
    a = tr.run_experiments(demo_df, du.DEMO_TARGET, 1, du.DEMO_MAPPING, "x", **kw)["bundles"]["B__random_forest"]
    b = tr.run_experiments(demo_df, du.DEMO_TARGET, 1, du.DEMO_MAPPING, "x", **kw)["bundles"]["B__random_forest"]
    np.testing.assert_allclose(a["proba_test"], b["proba_test"])


def test_experiment_b_skipped_without_extra_features(demo_df):
    mapping = {c: ("female_clinical" if g == "female_clinical" else "exclude") for c, g in du.DEMO_MAPPING.items()}
    res = tr.run_experiments(demo_df, du.DEMO_TARGET, 1, mapping, "x", model_names=["Logistic Regression"],
                             n_boot=50, perm_repeats=2)
    assert set(b["experiment"] for b in res["bundles"].values()) == {"A"}
    assert any("Experiment B skipped" in w for w in res["meta"]["warnings"])


# ---------------- a "real-looking" CSV with a different schema ----------------
def _real_like(n=600, seed=3):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, 350, n)  # repeated patients (several cycles)
    age = rng.normal(35, 4, n).round()
    sperm = rng.lognormal(3.5, 0.5, n)
    live = rng.random(n) < 1 / (1 + np.exp(0.15 * (age - 32)))
    return pd.DataFrame({
        "PatientID": pid, "Age at treatment": age, "Sperm count": sperm,
        "Embryo grade": rng.choice(["A", "B", "C"], n), "Live Birth": np.where(live, "Yes", "No"),
        "hCG result": live * 5 + rng.normal(0, 0.1, n),  # post-treatment leakage column
    })


def test_real_like_csv_workflow_with_grouping_and_leakage():
    df = _real_like()
    # leakage is flagged by name and by single-feature predictiveness
    assert du.leakage_name_flag("hCG result")
    y, _ = du.binarize_target(df["Live Birth"], "Yes")
    assert du.scan_single_feature_predictiveness(df, y, ["hCG result"])
    mapping = {c: g for c, (g, _) in du.suggest_mapping(df, "Live Birth").items()}
    assert mapping["hCG result"] == "exclude" and mapping["PatientID"] == "exclude"
    res = tr.run_experiments(df, "Live Birth", "Yes", mapping, "Live birth", id_col="PatientID",
                             n_boot=50, perm_repeats=2)
    assert res["meta"]["split_type"] == "patient-grouped stratified"
    assert res["meta"]["task_label"] == "Live birth"
    # no patient on both sides of the split
    y_all, keep = du.binarize_target(df["Live Birth"], "Yes")
    groups = du.groups_if_repeated(df, "PatientID")
    tr_idx, te_idx = tr.make_split(y_all, groups, 0.25, du.SEED)
    assert not set(groups[tr_idx]) & set(groups[te_idx])


def test_target_must_have_two_classes():
    s = pd.Series(["a", "b", "c", "a"])
    assert not du.inspect_target(s)["usable"]


# ---------------- metrics edge cases ----------------
def test_one_class_test_set_is_handled():
    m = ev.compute_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert m["roc_auc"] is None and m["recall"] is None
    assert any("only one class" in n for n in m["notes"])
    assert ev.roc_curve_points([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert ev.bootstrap_auc_difference([0, 0, 0], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) is None


def test_no_predicted_positives_gives_undefined_precision():
    m = ev.compute_metrics([0, 1, 0, 1], [0.1, 0.2, 0.1, 0.3], threshold=0.9)
    assert m["precision"] is None and m["f1"] == 0


def test_demo_interpretation_never_claims_evidence():
    txt = ev.interpret_comparison({"delta": 0.07, "ci_low": 0.01, "ci_high": 0.12, "ci_excludes_zero": True}, True)
    assert "SYNTHETIC" in txt and "not evidence" in txt


# ---------------- prediction ----------------
def test_prediction_validation_and_estimate(demo_results):
    b = demo_results["bundles"]["B__logistic_regression"]
    vals = {c: b["reference"][c] for c in b["features"]}
    out = tr.predict_single(b, vals, 0.4)
    assert 0 <= out["probability"] <= 1 and out["threshold"] == 0.4
    assert len(tr.local_explanation(b, out["row"])) == len(b["features"])
    for bad in (None, "", float("nan"), "abc", float("inf")):
        v = dict(vals, maternal_age=bad)
        with pytest.raises(ValueError):
            tr.predict_single(b, v)
    with pytest.raises(ValueError):
        tr.predict_single(b, dict(vals, embryo_stage="not-a-stage"))
    with pytest.raises(ValueError):
        tr.predict_single(b, {k: v for k, v in vals.items() if k != "bmi"})


# ---------------- persistence ----------------
def test_save_load_roundtrip_and_tamper_detection(demo_results, tmp_path):
    tr.save_run(demo_results, tmp_path)
    loaded = tr.load_run(tmp_path)
    assert set(loaded["bundles"]) == set(demo_results["bundles"])
    b0 = next(iter(loaded["bundles"].values()))
    np.testing.assert_allclose(b0["proba_test"], demo_results["bundles"][b0["id"]]["proba_test"])
    # tamper with a model file -> refused
    victim = next(tmp_path.glob("*.joblib"))
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(tr.ModelIntegrityError):
        tr.load_run(tmp_path)


def test_load_without_manifest_is_graceful(tmp_path):
    with pytest.raises(FileNotFoundError):
        tr.load_run(tmp_path)


def test_manifest_with_path_traversal_rejected(demo_results, tmp_path):
    tr.save_run(demo_results, tmp_path)
    mp = tmp_path / "manifest.json"
    man = json.loads(mp.read_text())
    man["files"]["../evil.joblib"] = "0" * 64
    mp.write_text(json.dumps(man))
    with pytest.raises(tr.ModelIntegrityError):
        tr.load_run(tmp_path)


# ---------------- app ----------------
def test_app_pages_render_without_exceptions():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    for p in at.sidebar.radio[0].options:
        at.sidebar.radio[0].set_value(p).run()
        assert not at.exception, (p, at.exception)
