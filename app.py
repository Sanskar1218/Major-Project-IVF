"""Streamlit dashboard for the IVF outcome-prediction prototype.

Run with:  streamlit run app.py

Everything shown here is computed from the data loaded in the current session.
Nothing is sent to an external service.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data_utils as du
import evaluate as ev
import train as tr

st.set_page_config(page_title="Intelligent IVF Optimization", page_icon="🧬", layout="wide")

TITLE = "An Intelligent IVF Optimization: Next-Generation Frameworks for Successful Implantation Prediction"
DISCLAIMER = "Experimental model estimate — not a medical diagnosis or treatment recommendation."
MODE_DEMO, MODE_UPLOAD = "Demonstration (synthetic data)", "Upload a real CSV"
PAGES = ["1 · Project Overview", "2 · Dataset Explorer", "3 · Model Training",
         "4 · Results & Explainability", "5 · IVF Outcome Prediction", "6 · Research Roadmap"]

# Validated categorical slots (blue, orange, aqua) — colour follows the entity, never its rank.
EXP_COLORS = {"A": "#2a78d6", "B": "#eb6834"}
MODEL_COLORS = {"Logistic Regression": "#2a78d6", "Random Forest": "#eb6834", "Gradient Boosting": "#1baf7a"}
SEQ_BLUE = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#104281"]]
GRID = "rgba(128,128,128,0.22)"

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#0f766e 0%,#0b2545 100%);color:#fff;padding:1.6rem 1.8rem;border-radius:14px;margin-bottom:1rem}
.hero h1{font-size:1.55rem;line-height:1.3;margin:0 0 .4rem 0;color:#fff}
.hero p{margin:0;opacity:.92}
.banner{padding:.8rem 1rem;border-radius:10px;margin:.6rem 0 1rem 0;border-left:6px solid}
.banner-demo{background:rgba(245,158,11,.14);border-color:#f59e0b}
.banner-real{background:rgba(37,99,235,.12);border-color:#2563eb}
.banner-safe{background:rgba(220,38,38,.10);border-color:#dc2626}
.flow{display:flex;flex-wrap:wrap;align-items:stretch;gap:.4rem;margin:.5rem 0 1rem 0}
.step{flex:1 1 130px;min-width:130px;padding:.7rem .8rem;border-radius:10px;background:rgba(15,118,110,.09);border:1px solid rgba(15,118,110,.35);font-size:.86rem}
.step b{display:block;margin-bottom:.2rem}
.step span{opacity:.85}
.arrow{align-self:center;opacity:.55;font-size:1.2rem}
.chip{display:inline-block;padding:.15rem .6rem;border-radius:999px;font-size:.78rem;font-weight:600}
.chip-demo{background:rgba(245,158,11,.22);border:1px solid #f59e0b}
.chip-real{background:rgba(37,99,235,.20);border:1px solid #2563eb}
.chip-plan{background:rgba(128,128,128,.18);border:1px solid rgba(128,128,128,.55)}
.chip-ok{background:rgba(5,150,105,.18);border:1px solid #059669}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def f3(v) -> str:
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def banner(kind: str, html: str) -> None:
    st.markdown(f'<div class="banner banner-{kind}">{html}</div>', unsafe_allow_html=True)


def demo_banner() -> None:
    banner("demo", f"<b>{du.DEMO_WARNING}</b>")


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=45, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def plot(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", key=key)


def init_state() -> None:
    s = st.session_state
    s.setdefault("data_mode", MODE_DEMO)
    s.setdefault("demo_rows", 1500)
    s.setdefault("demo_seed", du.SEED)
    s.setdefault("configs", {})
    s.setdefault("results", None)
    s.setdefault("upload_df", None)
    s.setdefault("upload_name", None)
    s.setdefault("upload_sig", None)
    s.setdefault("upload_notes", [])
    s.setdefault("upload_error", None)
    s.setdefault("last_pred", None)


def default_demo_config() -> dict:
    return {"source": "demo", "target": du.DEMO_TARGET, "positive": 1, "task_label": du.DEMO_TASK_LABEL,
            "id_col": du.DEMO_ID_COL, "mapping": dict(du.DEMO_MAPPING)}


def get_demo_df() -> pd.DataFrame:
    s = st.session_state
    key = (int(s.demo_rows), int(s.demo_seed))
    if s.get("demo_key") != key:
        s.demo_df = du.generate_synthetic_data(key[0], key[1])
        s.demo_key = key
        s.configs["demo"] = default_demo_config()
    return s.demo_df


def active_data():
    """(DataFrame or None, 'demo' | 'upload') for the data source selected on the Dataset page."""
    if st.session_state.data_mode == MODE_DEMO:
        return get_demo_df(), "demo"
    return st.session_state.upload_df, "upload"


def bundle_label(b: dict) -> str:
    return f"{b['experiment_label']} · {b['model_name']}"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> str:
    s = st.session_state
    with st.sidebar:
        st.markdown("### 🧬 IVF Outcome Prototype")
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        if s.data_mode == MODE_DEMO:
            st.markdown('<span class="chip chip-demo">DEMO MODE · synthetic data</span>', unsafe_allow_html=True)
        else:
            nm = s.upload_name or "no file yet"
            st.markdown(f'<span class="chip chip-real">REAL-DATA MODE · {nm}</span>', unsafe_allow_html=True)
        src = "demo" if s.data_mode == MODE_DEMO else "upload"
        cfg = s.configs.get(src)
        st.caption(("✅ Configuration saved: " + cfg["task_label"]) if cfg else "⚪ No configuration saved yet")
        res = s.results
        if res:
            st.caption(f"✅ Models trained ({'synthetic demo' if res['meta']['is_demo'] else 'real data'}, "
                       f"{res['meta']['created_utc']} UTC)")
        else:
            st.caption("⚪ No trained models in this session")
        st.divider()
        st.caption("Experimental research prototype for a college demonstration. Not a medical device and not "
                   "clinically validated.")
    return page


# --------------------------------------------------------------------------- #
# Page 1 — Overview
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.markdown(f'<div class="hero"><h1>{TITLE}</h1><p>A transparent, reproducible machine-learning prototype that '
                "asks one research question: does adding male-factor and embryonic information improve IVF outcome "
                "prediction compared with clinical features alone?</p></div>", unsafe_allow_html=True)
    if st.session_state.data_mode == MODE_DEMO:
        demo_banner()
    banner("real", "<b>Synthetic vs real data.</b> Demonstration mode uses invented records generated by a documented "
                   "formula, only to show the software workflow. Performance on them says <u>nothing</u> about clinical "
                   "accuracy. Real-data mode analyses a CSV you upload; even then, results are single-dataset research "
                   "outputs without external validation.")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("The prediction problem")
        st.write(
            "IVF treatment can end at several different points, and each is a **different prediction task**: "
            "embryo *implantation*, *clinical pregnancy* (e.g. a confirmed fetal heartbeat) and *live birth*. "
            "Outcomes depend on the woman's age and ovarian reserve, sperm quality, and embryo quality. "
            "This prototype estimates the probability of **one binary outcome that you declare explicitly** and trains "
            "models on the labels actually present in the dataset. It never relabels one outcome as another."
        )
        st.write("Two feature sets are compared on exactly the same train/test rows: **Experiment A** uses clinical "
                 "(female and other pre-treatment) variables only; **Experiment B** adds male-factor and embryonic "
                 "variables.")
    with c2:
        st.subheader("What is implemented")
        st.dataframe(pd.DataFrame({
            "Approach": ["A · Clinical-only model", "B · Combined clinical + male + embryo model",
                         "C · Image-based multimodal extension"],
            "Status": ["Implemented", "Implemented", "Not implemented (roadmap only)"],
        }), hide_index=True, width="stretch")
        st.caption("Approach C needs labelled microscopy images, which this prototype does not include. The tabular "
                   "morphology/quality *scores* used here are not raw images.")

    st.subheader("Pipeline")
    steps = [("1 · Data", "Synthetic demo or your CSV"), ("2 · Checks", "Target, leakage, IDs, missingness"),
             ("3 · Split", "Stratified; grouped by patient if IDs repeat"),
             ("4 · Preprocess", "Impute, scale, encode — fitted on train only"),
             ("5 · Models", "Logistic regression, random forest, gradient boosting"),
             ("6 · Experiments", "A clinical-only vs B combined"),
             ("7 · Evaluate", "ROC, calibration, importance, bootstrap A vs B"),
             ("8 · Estimate", "Probability + configurable threshold")]
    html = '<div class="flow">'
    for i, (t, d) in enumerate(steps):
        html += f'<div class="step"><b>{t}</b><span>{d}</span></div>'
        if i < len(steps) - 1:
            html += '<div class="arrow">→</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

    st.subheader("How to use this app")
    st.markdown("1. **Dataset Explorer** – keep the synthetic demo, or upload a CSV, choose the target and map columns.\n"
                "2. **Model Training** – pick experiments and models, then train.\n"
                "3. **Results & Explainability** – compare models and A vs B; inspect plots.\n"
                "4. **IVF Outcome Prediction** – enter values for one hypothetical case.")
    banner("safe", f"<b>{DISCLAIMER}</b> This tool must not be used to decide whether anyone should undergo, avoid "
                   "or change fertility treatment.")


# --------------------------------------------------------------------------- #
# Page 2 — Dataset Explorer
# --------------------------------------------------------------------------- #
def page_dataset() -> None:
    s = st.session_state
    st.header("Dataset Explorer")
    mode = st.radio("Data source", [MODE_DEMO, MODE_UPLOAD], index=[MODE_DEMO, MODE_UPLOAD].index(s.data_mode),
                    horizontal=True, key="w_data_mode")
    s.data_mode = mode  # persisted copy (widget keys are dropped when another page is shown)

    if mode == MODE_DEMO:
        demo_banner()
        c1, c2 = st.columns(2)
        s.demo_rows = c1.number_input("Number of synthetic records", 200, 20000, int(s.demo_rows), step=100)
        s.demo_seed = c2.number_input("Random seed", 0, 10**6, int(s.demo_seed), step=1)
        with st.expander("How the synthetic outcome is generated (transparent recipe)"):
            st.write("Each record's outcome is drawn from a Bernoulli distribution whose log-odds are a weighted sum "
                     "of the variables below plus random noise. Because the male-factor and embryo variables are "
                     "**deliberately given weight**, a combined model is expected to beat a clinical-only one on this "
                     "data. That is built into the generator and is not evidence about real patients.")
            st.json(du.SYNTH_SPEC)
        df, source = get_demo_df(), "demo"
    else:
        banner("real", "<b>Real-data mode.</b> The file is processed in memory by this app and is not sent anywhere. "
                       "Only upload data you are authorised to use, and prefer de-identified records. The app does not "
                       "provide, scrape or assume access to any clinical registry data.")
        up = st.file_uploader("Upload IVF treatment records (CSV, max 25 MB)", type=["csv"])
        if up is not None:
            sig = (up.name, up.size)
            if s.upload_sig != sig:
                s.upload_sig = sig
                try:
                    s.upload_df, s.upload_notes = du.load_csv(up)
                    s.upload_name, s.upload_error = up.name, None
                    s.configs.pop("upload", None)
                    s.results = None if (s.results and not s.results["meta"]["is_demo"]) else s.results
                except du.DataError as exc:
                    s.upload_df, s.upload_name, s.upload_notes, s.upload_error = None, None, [], str(exc)
        if s.upload_error:
            st.error(f"Could not use this file: {s.upload_error}")
        if s.upload_df is None:
            if not s.upload_error:
                st.info("Upload a CSV to begin. Until then the Training page has no real dataset to use. "
                        "You can switch back to the synthetic demonstration at any time.")
            return
        for n in s.upload_notes:
            st.info(n)
        df, source = s.upload_df, "upload"

    render_inspection(df, source)


def render_inspection(df: pd.DataFrame, source: str) -> None:
    s = st.session_state
    saved = s.configs.get(source)
    n_rows, n_cols = df.shape
    m = st.columns(4)
    m[0].metric("Rows", f"{n_rows:,}")
    m[1].metric("Columns", n_cols)
    m[2].metric("Duplicate rows", int(df.duplicated().sum()))
    m[3].metric("Cells missing", f"{100 * df.isna().to_numpy().mean():.1f}%")

    t1, t2 = st.tabs(["Sample records", "Columns & missing values"])
    with t1:
        st.dataframe(df.head(25), width="stretch")
    with t2:
        st.dataframe(du.describe_columns(df), hide_index=True, width="stretch")
        miss = (df.isna().mean() * 100).sort_values(ascending=False)
        miss = miss[miss > 0]
        if len(miss):
            fig = go.Figure(go.Bar(x=miss.values[::-1], y=miss.index[::-1], orientation="h", marker_color="#2a78d6",
                                   hovertemplate="%{y}: %{x:.1f}% missing<extra></extra>"))
            fig.update_layout(title="Missing values by column (%)", xaxis_title="% missing")
            plot(style(fig, max(220, 26 * len(miss) + 90)), "miss_chart")
        else:
            st.success("No missing values.")

    ds_key = f"{source}-{df.shape[0]}-{df.shape[1]}-{abs(hash(tuple(df.columns))) % 100000}"
    cols = list(df.columns)

    # ---- Step 1: target -----------------------------------------------------------------
    st.subheader("Step 1 · Choose the target outcome")
    def_t = saved["target"] if saved else (du.DEMO_TARGET if source == "demo" else None)
    target = st.selectbox("Target column (the outcome to predict)", cols,
                          index=cols.index(def_t) if def_t in cols else None,
                          placeholder="Select the outcome column…", key=f"tgt_{ds_key}")
    if target is None:
        st.info("Select the column that holds the outcome. The app never guesses it.")
        return
    info = du.inspect_target(df[target])
    if info["n_missing"]:
        st.warning(f"{info['n_missing']} row(s) have a missing target and will be dropped from training.")
    if not info["usable"]:
        st.error(f"'{target}' has {info['n_classes']} distinct non-missing value(s): "
                 f"{list(info['counts'])[:10]}. A binary target with exactly two classes is required. "
                 "Choose another column, or prepare a two-class column first.")
        return
    classes = sorted(info["classes"], key=str)
    def_pos = None
    if saved and saved["target"] == target:
        def_pos = next((c for c in classes if str(c) == str(saved["positive"])), None)
    elif source == "demo":
        def_pos = next((c for c in classes if str(c) == "1"), None)
    positive = st.selectbox("Which value is the positive outcome?", classes, format_func=str,
                            index=classes.index(def_pos) if def_pos is not None else None,
                            placeholder="Select the positive class…", key=f"pos_{ds_key}_{target}")
    task_opts = ([du.DEMO_TASK_LABEL] if source == "demo" else []) + du.TASK_OPTIONS
    def_task = None
    if saved:
        def_task = saved["task_label"] if saved["task_label"] in task_opts else du.TASK_OPTIONS[-1]
    elif source == "demo":
        def_task = du.DEMO_TASK_LABEL
    task_sel = st.selectbox("What does this target actually represent?", task_opts,
                            index=task_opts.index(def_task) if def_task in task_opts else None,
                            placeholder="Implantation, clinical pregnancy, live birth…", key=f"task_{ds_key}")
    task_label = task_sel
    if task_sel == du.TASK_OPTIONS[-1]:
        task_label = st.text_input("Describe the outcome", value=(saved["task_label"] if saved and saved["task_label"]
                                   not in task_opts else ""), key=f"task_txt_{ds_key}").strip() or None
    st.caption("Implantation, clinical pregnancy and live birth are different prediction tasks. Models are labelled with "
               "the task you choose here and are trained on exactly the labels in this column; the app never converts "
               "one outcome into another.")
    counts = pd.Series(info["counts"])
    fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=["#eb6834" if str(k) == str(positive) else "#2a78d6"
                                                                       for k in counts.index],
                           text=counts.values, textposition="outside",
                           hovertemplate="class %{x}: %{y} rows<extra></extra>"))
    fig.update_layout(title="Target class distribution (orange = positive class)", xaxis_title=target,
                      yaxis_title="rows", showlegend=False)
    plot(style(fig, 300), f"target_dist_{ds_key}")
    if positive is None or task_label is None:
        st.info("Choose the positive class and describe what the target represents to continue.")
        return

    # ---- Step 2: identifier + mapping -----------------------------------------------------------
    st.subheader("Step 2 · Patient identifier and feature mapping")
    id_cands = [c for c in du.find_identifier_columns(df) if c != target]
    id_opts = ["(none)"] + [c for c in cols if c != target]
    def_id = (saved["id_col"] if saved and saved["target"] == target else
              (du.DEMO_ID_COL if source == "demo" else next((c for c in id_cands if du.looks_like_identifier(c)), None)))
    id_sel = st.selectbox("Patient identifier column (optional)", id_opts,
                          index=id_opts.index(def_id) if def_id in id_opts else 0, key=f"id_{ds_key}_{target}")
    id_col = None if id_sel == "(none)" else id_sel
    if id_cands and id_col is None:
        st.caption(f"Possible identifier column(s) detected: {', '.join(id_cands)}. Setting one enables a patient-grouped split "
                   "when patients have several cycles.")
    sugg = du.suggest_mapping(df, target)
    rows = []
    for c in cols:
        if c == target:
            continue
        if saved and saved["target"] == target and c in saved["mapping"]:
            g, why = saved["mapping"][c], "from saved configuration"
        elif source == "demo":
            g, why = du.DEMO_MAPPING.get(c, sugg[c][0]), "demo mapping"
        else:
            g, why = sugg[c]
        if c == id_col:
            g, why = "exclude", "selected as patient ID"
        rows.append({"column": c, "type": "numeric" if du.is_numeric_col(df[c]) else "categorical",
                     "missing %": round(100 * df[c].isna().mean(), 1), "feature group": du.GROUP_LABELS[g],
                     "suggestion basis": why})
    st.caption("Columns default to *Exclude* unless their name clearly matches a group. Review every row: assign each "
               "pre-treatment variable to a group, and keep post-treatment results out.")
    edited = st.data_editor(
        pd.DataFrame(rows), hide_index=True, width="stretch", key=f"map_{ds_key}_{target}_{id_col}",
        disabled=["column", "type", "missing %", "suggestion basis"],
        column_config={"feature group": st.column_config.SelectboxColumn(
            "feature group", options=list(du.GROUP_LABELS.values()), required=True)})
    mapping = {r["column"]: du.LABEL_TO_GROUP[r["feature group"]] for _, r in edited.iterrows()}
    if id_col:
        mapping[id_col] = "exclude"

    # ---- Step 3: checks ---------------------------------------------------------------------------
    st.subheader("Step 3 · Data-quality and leakage checks")
    y, keep = du.binarize_target(df[target], positive)
    d = df.loc[keep]
    feat_cols = [c for c, g in mapping.items() if g != "exclude"]
    if not feat_cols:
        st.warning("No feature columns are selected yet.")
    else:
        g = pd.DataFrame({"group": [du.GROUP_LABELS[x] for x in mapping.values() if x != "exclude"],
                          "col": feat_cols})
        g["numeric"] = [du.is_numeric_col(df[c]) for c in feat_cols]
        summ = g.groupby("group").agg(features=("col", "count"), numeric=("numeric", "sum")).reset_index()
        summ["categorical"] = summ["features"] - summ["numeric"]
        st.dataframe(summ, hide_index=True, width="stretch")
    issues = du.check_feature_quality(d, feat_cols) if feat_cols else []
    leak = du.scan_single_feature_predictiveness(d, y, feat_cols) if feat_cols else []
    idinfo = du.check_identifiers(df, id_col)
    ok, errs, warns = du.check_evaluability(y, 0.25)
    problems = 0
    for sev, msg in issues:
        (st.error if sev == "error" else st.warning)(msg)
        problems += 1
    for msg in leak:
        st.warning("Possible leakage — " + msg)
        problems += 1
    for msg in idinfo["messages"]:
        st.warning(msg)
        problems += 1
    for msg in errs:
        st.error(msg)
        problems += 1
    for msg in warns:
        st.warning(msg)
        problems += 1
    has_extra = any(mapping[c] in du.EXTRA_GROUPS for c in feat_cols)
    if feat_cols and not has_extra:
        st.info("No male-factor or embryonic columns are mapped, so Experiment B would equal Experiment A and will be skipped.")
    if not problems:
        st.success("No data-quality, identifier or leakage warnings. (Heuristics only; they cannot prove the "
                   "absence of leakage — check when each variable is recorded relative to the outcome.)")
    st.caption("Name-based and single-feature checks catch only obvious problems. You remain responsible for "
               "confirming that every feature is known *before* the outcome.")

    cfg = {"source": source, "target": target, "positive": positive, "task_label": task_label,
           "id_col": id_col, "mapping": mapping}
    clinical_or_any = any(mapping[c] in (*du.CLINICAL_GROUPS, *du.EXTRA_GROUPS) for c in feat_cols)
    c1, c2 = st.columns([1, 3])
    if c1.button("Save configuration", type="primary", disabled=not clinical_or_any):
        s.configs[source] = cfg
        st.rerun()
    if saved == cfg:
        c2.success("Configuration saved. Continue to **3 · Model Training**.")
    elif not clinical_or_any:
        c2.warning("Assign at least one column to a feature group to save.")
    else:
        c2.info("Unsaved changes — press **Save configuration** to use them for training.")


# --------------------------------------------------------------------------- #
# Page 3 — Training
# --------------------------------------------------------------------------- #
def page_training() -> None:
    s = st.session_state
    st.header("Model Training")
    df, source = active_data()
    cfg = s.configs.get(source)
    if df is None or cfg is None:
        st.warning("Load a dataset and save its configuration on **2 · Dataset Explorer** first.")
        _load_saved_button()
        return
    if source == "demo":
        demo_banner()
    st.write(f"**Prediction task:** {cfg['task_label']} · **target:** `{cfg['target']}` "
             f"(positive = `{cfg['positive']}`) · **rows:** {len(df):,}")

    groups_present = [g for g in du.EXTRA_GROUPS if any(v == g for v in cfg["mapping"].values())]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Experiments")
        exp_a = st.checkbox("Experiment A — Clinical-only", True)
        exp_b = st.checkbox("Experiment B — Combined features", True)
        extra = st.multiselect("Feature groups added in Experiment B", groups_present, default=groups_present,
                               format_func=lambda g: du.GROUP_LABELS[g],
                               help="Only groups that have mapped columns in your data can be selected.")
        st.caption("Tabular morphology/quality **scores** are used. Raw sperm or embryo microscopy images are not "
                   "part of this prototype.")
    with c2:
        st.subheader("Models and settings")
        models = st.multiselect("Models", tr.MODEL_NAMES, default=tr.MODEL_NAMES)
        test_size = st.slider("Test fraction", 0.15, 0.40, 0.25, 0.05)
        cv_folds = st.slider("Cross-validation folds (training rows only)", 3, 10, 5)
        seed = st.number_input("Random seed", 0, 10**6, du.SEED, step=1)

    exps = [e for e, on in (("A", exp_a), ("B", exp_b)) if on]
    idc = cfg["id_col"]
    for e in exps:
        f = tr.experiment_features(cfg["mapping"], e, extra, df.columns, {cfg["target"], idc})
        with st.expander(f"Experiment {e} features ({len(f)})"):
            st.write(", ".join(f) if f else "none")

    y, keep = du.binarize_target(df[cfg["target"]], cfg["positive"])
    ok, errs, warns = du.check_evaluability(y, test_size)
    st.subheader("Validation before training")
    if ok:
        st.success("The data are large enough to attempt a held-out evaluation (minimum thresholds only — not a guarantee "
                   "of reliable metrics).")
    for m in errs:
        st.error(m)
    for m in warns:
        st.warning(m)
    st.caption("Split: stratified random" + (", grouped by patient ID because IDs repeat"
               if du.groups_if_repeated(df.loc[keep].reset_index(drop=True), idc) is not None else "")
               + ". Imputers, scalers and encoders are fitted on training rows only; the test set is used once for the report.")

    can_train = ok and bool(exps) and bool(models)
    b1, b2 = st.columns([1, 1])
    if b1.button("Train & evaluate", type="primary", disabled=not can_train):
        bar = st.progress(0.0, text="Starting…")
        try:
            res = tr.run_experiments(
                df, cfg["target"], cfg["positive"], cfg["mapping"], cfg["task_label"], is_demo=(source == "demo"),
                id_col=idc, experiments=exps, extra_groups=extra, model_names=models, test_size=test_size,
                seed=int(seed), cv_folds=cv_folds, progress_cb=lambda f, t: bar.progress(min(f, 1.0), text=t))
        except (tr.TrainingError, du.DataError, ValueError) as exc:
            bar.empty()
            st.error(f"Training could not be completed: {exc}")
        else:
            bar.empty()
            s.results, s.last_pred = res, None
            try:
                tr.save_run(res)
            except OSError as exc:
                st.warning(f"Models were trained but could not be saved to disk ({exc}). They remain available in this session.")
            st.success("Training finished. Open **4 · Results & Explainability**.")
    with b2:
        _load_saved_button()

    res = s.results
    if res:
        st.subheader("Latest run")
        meta = res["meta"]
        if meta["is_demo"]:
            demo_banner()
        st.write(f"Task: **{meta['task_label']}** · split: {meta['split_type']} · train {meta['n_train']:,} / "
                 f"test {meta['n_test']:,} · test positives {meta['class_counts']['test']['positive']}")
        for w in meta["warnings"]:
            st.warning(w)
        st.dataframe(ev.metrics_table(res["bundles"]), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "Brier",
                                              "CV ROC-AUC (train only)"]})


def _load_saved_button() -> None:
    if st.button("Load previously saved models"):
        try:
            st.session_state.results = tr.load_run()
            st.session_state.last_pred = None
            st.success("Loaded saved models after verifying their checksums.")
            st.rerun()
        except FileNotFoundError:
            st.info("No saved models found yet. Train first.")
        except tr.ModelIntegrityError as exc:
            st.error(str(exc))


# --------------------------------------------------------------------------- #
# Page 4 — Results
# --------------------------------------------------------------------------- #
def short_reading(c: dict) -> str:
    if c.get("delta") is None:
        return "not computable"
    if c["ci_excludes_zero"]:
        return "B higher; interval excludes 0" if c["delta"] > 0 else "A higher; interval excludes 0"
    return "no demonstrable difference (interval includes 0)"


def page_results() -> None:
    st.header("Results & Explainability")
    res = st.session_state.results
    if not res:
        st.info("No results yet. Train models on **3 · Model Training** (or load saved ones).")
        return
    meta, bundles = res["meta"], res["bundles"]
    if meta["is_demo"]:
        demo_banner()
    st.write(f"**Prediction task:** {meta['task_label']} · target `{meta['target_col']}` = `{meta['positive_value']}` · "
             f"{meta['split_type']} split · train {meta['n_train']:,} / test {meta['n_test']:,} "
             f"({meta['class_counts']['test']['positive']} positive, {meta['class_counts']['test']['negative']} negative in test)")
    thr = st.slider("Decision threshold for class-based metrics", 0.05, 0.95, 0.50, 0.05, key="res_thr",
                    help="ROC-AUC and Brier score do not depend on the threshold; accuracy, precision, recall, F1 and the "
                         "confusion matrix do.")
    exps = sorted({b["experiment"] for b in bundles.values()})
    models = [m for m in tr.MODEL_NAMES if any(b["model_name"] == m for b in bundles.values())]

    t_cmp, t_roc, t_cm, t_exp, t_lim = st.tabs(
        ["Model comparison", "ROC & calibration", "Confusion matrix", "Feature importance", "Limitations"])

    with t_cmp:
        tbl = ev.metrics_table(bundles, thr)
        st.dataframe(tbl, hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "Brier",
                                              "CV ROC-AUC (train only)"]})
        st.caption("All values are computed from this run's held-out test predictions. 'CV ROC-AUC' uses training rows only "
                   "and is the signal for choosing a model; the test set is not used for selection. Blank = undefined.")
        fig = go.Figure()
        for e in exps:
            vals = []
            for m in models:
                b = bundles.get(f"{e}__{tr._slug(m)}")
                vals.append(ev.compute_metrics(b["y_test"], b["proba_test"], thr)["roc_auc"] if b else None)
            fig.add_bar(name=tr.EXPERIMENT_LABELS[e], x=models, y=vals, marker_color=EXP_COLORS[e],
                        text=[f3(v) for v in vals], textposition="outside",
                        hovertemplate="%{x}<br>ROC-AUC %{y:.3f}<extra>" + tr.EXPERIMENT_LABELS[e] + "</extra>")
        fig.update_layout(title="Test ROC-AUC by model and experiment", barmode="group", bargap=0.3,
                          yaxis=dict(range=[0, 1.08], title="ROC-AUC"))
        plot(style(fig, 360), "cmp_bar")

        if meta["comparisons"]:
            st.subheader("Clinical-only (A) vs combined (B), same test rows")
            ct = pd.DataFrame([{"Model": c["model"], "AUC A": c.get("auc_a"), "AUC B": c.get("auc_b"),
                                "Δ (B − A)": c.get("delta"), "95% CI low": c.get("ci_low"),
                                "95% CI high": c.get("ci_high"), "Reading": short_reading(c)}
                               for c in meta["comparisons"]])
            st.dataframe(ct, hide_index=True, width="stretch",
                         column_config={c: st.column_config.NumberColumn(format="%.3f")
                                        for c in ["AUC A", "AUC B", "Δ (B − A)", "95% CI low", "95% CI high"]})
            first = next((c for c in meta["comparisons"] if c.get("delta") is not None), None)
            if first:
                st.info(ev.interpret_comparison(first, meta["is_demo"]))
            st.caption("Interval = paired bootstrap over test rows (test-sampling noise only). It does not capture "
                       "retraining variability or differences between clinics.")
        else:
            st.info("Only one experiment was run, so no A-vs-B comparison is available.")

        cc = meta["class_counts"]
        fig = go.Figure()
        fig.add_bar(name="Negative", x=["Train", "Test"], y=[cc["train"]["negative"], cc["test"]["negative"]],
                    marker_color="#2a78d6")
        fig.add_bar(name="Positive", x=["Train", "Test"], y=[cc["train"]["positive"], cc["test"]["positive"]],
                    marker_color="#eb6834")
        fig.update_layout(title="Class distribution in the train and test sets", barmode="group", bargap=0.4,
                          yaxis_title="rows")
        plot(style(fig, 300), "class_dist")
        st.warning("Accuracy, ROC-AUC and feature importance alone do not establish clinical usefulness. That would need "
                   "external validation, calibration in the target population, and evidence that using the model improves "
                   "decisions or outcomes.")

    with t_roc:
        e_sel = st.radio("Experiment", exps, format_func=lambda e: tr.EXPERIMENT_LABELS[e], horizontal=True, key="roc_exp")
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(128,128,128,.7)", dash="dash"))
        fig2 = go.Figure()
        fig2.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(128,128,128,.7)", dash="dash"))
        cal_note = False
        for m in models:
            b = bundles.get(f"{e_sel}__{tr._slug(m)}")
            if not b:
                continue
            r = ev.roc_curve_points(b["y_test"], b["proba_test"])
            if r:
                fig.add_scatter(x=r[0], y=r[1], mode="lines", name=f"{m} (AUC {r[2]:.3f})",
                                line=dict(color=MODEL_COLORS[m], width=2), hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>")
            cal = ev.calibration_points(b["y_test"], b["proba_test"])
            if cal:
                fig2.add_scatter(x=cal[0], y=cal[1], mode="lines+markers", name=m,
                                 line=dict(color=MODEL_COLORS[m], width=2), marker=dict(size=8),
                                 hovertemplate="predicted %{x:.2f}<br>observed %{y:.2f}<extra></extra>")
            else:
                cal_note = True
        fig.update_layout(title="ROC curves (test set)", xaxis_title="False positive rate", yaxis_title="True positive rate")
        fig2.update_layout(title="Calibration (quantile bins, test set)", xaxis_title="Mean predicted probability",
                           yaxis_title="Observed fraction positive")
        with c1:
            plot(style(fig, 420), "roc_fig")
        with c2:
            plot(style(fig2, 420), "cal_fig")
            if cal_note:
                st.caption("Calibration omitted for a model: the test set is too small for at least 3 bins.")
        st.caption("A well-calibrated model lies near the dashed diagonal. Good AUC with poor calibration means the "
                   "probabilities themselves should not be taken at face value.")

    with t_cm:
        labels = {bundle_label(b): k for k, b in bundles.items()}
        sel = st.selectbox("Model", list(labels), key="cm_sel")
        b = bundles[labels[sel]]
        mt = ev.compute_metrics(b["y_test"], b["proba_test"], thr)
        z = [[mt["tn"], mt["fp"]], [mt["fn"], mt["tp"]]]
        fig = go.Figure(go.Heatmap(z=z, x=["Predicted negative", "Predicted positive"], y=["Actual negative", "Actual positive"],
                                   colorscale=SEQ_BLUE, showscale=False, text=z, texttemplate="%{text}",
                                   textfont=dict(size=22), xgap=3, ygap=3,
                                   hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
        fig.update_layout(title=f"Confusion matrix at threshold {thr:.2f}")
        fig.update_yaxes(autorange="reversed")
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            plot(style(fig, 360), "cm_fig")
        with cc2:
            for lab, k in [("ROC-AUC", "roc_auc"), ("Accuracy", "accuracy"), ("Precision", "precision"),
                           ("Recall / sensitivity", "recall"), ("Specificity", "specificity"), ("F1", "f1"), ("Brier", "brier")]:
                st.metric(lab, f3(mt[k]))
        for n in mt["notes"]:
            st.warning(n)

    with t_exp:
        labels = {bundle_label(b): k for k, b in bundles.items()}
        sel = st.selectbox("Model", list(labels), key="imp_sel")
        b = bundles[labels[sel]]
        pi = b["permutation_importance"]
        if pi is not None:
            top = pi.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["importance_mean"], y=top["feature"], orientation="h", marker_color="#2a78d6",
                                   error_x=dict(type="data", array=top["importance_std"], thickness=1),
                                   hovertemplate="%{y}: ΔAUC %{x:.4f}<extra></extra>"))
            fig.update_layout(title="Permutation importance: drop in test ROC-AUC when the column is shuffled",
                              xaxis_title="Mean decrease in ROC-AUC (± SD over repeats)")
            plot(style(fig, max(260, 28 * len(top) + 110)), "perm_fig")
        else:
            st.info("Permutation importance needs both classes in the test set.")
        ti = b["tree_importance"]
        if ti is not None:
            top = ti.head(15).iloc[::-1]
            fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color="#eb6834",
                                   hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
            fig.update_layout(title="Tree-ensemble impurity importance (sums to 1; training-set based)",
                              xaxis_title="Importance")
            plot(style(fig, max(260, 28 * len(top) + 110)), "tree_fig")
        else:
            st.caption("Impurity-based importance exists only for tree ensembles that expose it (here: random forest). "
                       "Permutation importance above works for every model.")
        st.caption("Importance describes what this fitted model relies on. It is not evidence of a causal effect, and "
                   "correlated features can share or hide importance.")

    with t_lim:
        st.markdown(
            "- **Single split, single dataset.** Metrics come from one held-out test set; small test sets give very wide "
            "uncertainty.\n"
            "- **No external validation.** Performance on other clinics, populations or time periods is unknown.\n"
            "- **Outcome definition matters.** A model trained on one outcome (implantation, clinical pregnancy, live birth) "
            "says nothing about the others.\n"
            "- **Leakage can inflate scores.** The checks here are heuristics; any variable recorded after the outcome "
            "invalidates the evaluation.\n"
            "- **Association, not causation.** Feature importance does not show that changing a feature changes the outcome.\n"
            "- **Calibration and utility not proven.** Even a well-calibrated model needs a decision-curve or impact "
            "analysis before any clinical use.\n"
            "- **Synthetic data.** In demo mode the outcome is generated from a formula, so scores only show that the "
            "software runs.")
        banner("safe", f"<b>{DISCLAIMER}</b>")


# --------------------------------------------------------------------------- #
# Page 5 — Prediction
# --------------------------------------------------------------------------- #
def page_prediction() -> None:
    s = st.session_state
    st.header("IVF Outcome Prediction")
    res = s.results
    if not res:
        st.info("Train models on **3 · Model Training** first.")
        return
    bundles = res["bundles"]
    if res["meta"]["is_demo"]:
        demo_banner()
    labels = {bundle_label(b): k for k, b in bundles.items()}
    best = max(bundles.values(), key=lambda b: (b["cv_auc_mean"] is not None, b["cv_auc_mean"] or 0))
    names = list(labels)
    sel = st.selectbox("Model", names, index=names.index(bundle_label(best)), key="pred_model",
                       help="Default: highest cross-validated ROC-AUC on training rows (the test set is not used to choose).")
    b = bundles[labels[sel]]
    bid = b["id"]
    st.write(f"**Prediction task:** {b['task_label']} · uses {len(b['features'])} features "
             f"({'clinical only' if b['experiment'] == 'A' else 'clinical + male-factor + embryonic'})")

    def fill_typical():
        for c in b["features"]:
            s[f"in_{bid}_{c}"] = float(b["reference"][c]) if c in b["numeric"] else b["reference"][c]

    st.button("Fill with typical training values", on_click=fill_typical,
              help="Sets every field to the training median/mode. Fields left empty block prediction.")
    values = {}
    grid = st.columns(3)
    for i, c in enumerate(b["features"]):
        with grid[i % 3]:
            if c in b["numeric"]:
                lo, hi = b["ranges"][c]
                intlike = float(lo).is_integer() and float(hi).is_integer() and float(b["reference"][c]).is_integer()
                values[c] = st.number_input(c, value=None, step=1.0 if intlike else 0.1, format="%g",
                                            placeholder=f"training range {lo:g}–{hi:g}", key=f"in_{bid}_{c}")
            else:
                values[c] = st.selectbox(c, b["categories"][c], index=None, placeholder="Select…", key=f"in_{bid}_{c}")
    thr = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05, key="pred_thr")

    if st.button("Estimate probability", type="primary"):
        errors, _ = tr.validate_inputs(b, values)
        if errors:
            s.last_pred = None
            st.error("Cannot estimate yet: " + " ".join(errors))
        else:
            out = tr.predict_single(b, values, thr)
            s.last_pred = {"bid": bid, "values": dict(values), "out": out,
                           "expl": tr.local_explanation(b, out["row"])}
    lp = s.last_pred
    if not lp or lp["bid"] != bid:
        banner("safe", f"<b>{DISCLAIMER}</b>")
        return
    if lp["values"] != values:
        st.caption("Inputs changed since this estimate. Press **Estimate probability** to update.")
    out, p = lp["out"], lp["out"]["probability"]
    for w in out["warnings"]:
        st.warning(w)
    st.subheader("Estimate")
    m1, m2 = st.columns(2)
    m1.metric(f"Estimated probability · {b['task_label']}", f"{p:.1%}")
    pos = p >= thr
    m2.metric(f"Predicted class at threshold {thr:.2f}", "Positive" if pos else "Negative")
    fig = go.Figure()
    fig.add_bar(x=[1], y=[""], orientation="h", marker_color="rgba(128,128,128,.25)", hoverinfo="skip")
    fig.add_bar(x=[p], y=[""], orientation="h", marker_color="#2a78d6", hovertemplate="estimate %{x:.1%}<extra></extra>")
    fig.add_shape(type="line", x0=thr, x1=thr, y0=-0.5, y1=0.5, line=dict(color="#eb6834", width=3))
    fig.add_annotation(x=thr, y=0.5, text=f"threshold {thr:.2f}", showarrow=False, yshift=12, font=dict(color="#eb6834"))
    fig.update_layout(barmode="overlay", showlegend=False, xaxis=dict(range=[0, 1], tickformat=".0%"),
                      yaxis=dict(visible=False), title="Estimated probability")
    plot(style(fig, 170), "prob_bar")
    banner("safe", f"<b>{DISCLAIMER}</b>")

    ex = lp["expl"].head(6)
    st.subheader("Which inputs moved this estimate most")
    fig = go.Figure(go.Bar(x=ex["effect on estimate"][::-1], y=ex["feature"][::-1], orientation="h",
                           marker_color=["#2a78d6" if v >= 0 else "#eb6834" for v in ex["effect on estimate"][::-1]],
                           hovertemplate="%{y}: %{x:+.1%}<extra></extra>"))
    fig.update_layout(title="Change in estimate vs the typical training value (blue raises, orange lowers)",
                      xaxis=dict(tickformat="+.0%", title="Effect on estimated probability"))
    plot(style(fig, max(240, 40 * len(ex) + 100)), "local_fig")
    st.caption("Occlusion-style explanation: each input is swapped for its typical training value and the change in "
               "this model's estimate is shown. It describes the model's behaviour, not causes, and ignores interactions.")
    if res["meta"]["is_demo"]:
        st.caption("This model was trained on synthetic data; the estimate has no clinical meaning.")


# --------------------------------------------------------------------------- #
# Page 6 — Roadmap
# --------------------------------------------------------------------------- #
def page_roadmap() -> None:
    st.header("Research Roadmap")
    banner("real", "<b>Everything on this page is planned future work.</b> None of it is implemented in this prototype.")
    items = [
        ("Raw sperm microscopy images and CNNs",
         "Train a small convolutional network (or fine-tune a compact pretrained one) on a labelled sperm-image set to "
         "produce morphology predictions or image embeddings. Requires real labelled images, a documented patient-level "
         "train/test split and licence review. Published sets such as HuSHeM and SMIDS exist; check their terms and "
         "suitability first. Tabular morphology scores in this prototype are not images and must not be presented as "
         "image classification."),
        ("Embryo images and time-lapse data",
         "Extract features from static embryo photographs or time-lapse sequences (e.g. cleavage timings, blastocyst "
         "expansion) with proper annotation and clinic-specific imaging protocols."),
        ("Multimodal feature fusion",
         "Combine image-derived embeddings with the clinical, male-factor and embryonic tables, e.g. late fusion of "
         "per-modality model outputs or a joint model. The planned test is the same as here: does fusion beat the "
         "clinical-only baseline on held-out *real* data, with an uncertainty interval?"),
        ("External validation across fertility clinics",
         "Freeze the model, then evaluate it unchanged at other clinics and later time periods to measure how well it "
         "generalises; report performance per site."),
        ("Probability calibration and clinical-utility evaluation",
         "Recalibrate (e.g. isotonic or Platt scaling on a separate calibration set) and assess net benefit with decision-"
         "curve analysis; ideally prospective impact studies with clinicians."),
    ]
    for title, text in items:
        with st.container(border=True):
            st.markdown(f'**{title}** <span class="chip chip-plan">Planned · not implemented</span>', unsafe_allow_html=True)
            st.write(text)
    st.subheader("Where the current prototype stands")
    st.dataframe(pd.DataFrame({
        "Capability": ["Tabular clinical-only model", "Tabular combined model", "Calibration plot (visual check)",
                       "Image models (sperm / embryo)", "Multimodal fusion", "External validation",
                       "Clinical-utility analysis"],
        "Status": ["Implemented", "Implemented", "Implemented (diagnostic only)", "Not implemented",
                   "Not implemented", "Not implemented", "Not implemented"]}), hide_index=True, width="stretch")
    st.caption("Image inference is deliberately absent: there are no image labels, no image training code and no image "
               "predictions anywhere in this application.")


# --------------------------------------------------------------------------- #
init_state()
get_demo_df()  # guarantees a demo dataset + default config exist
_page = sidebar()
{PAGES[0]: page_overview, PAGES[1]: page_dataset, PAGES[2]: page_training,
 PAGES[3]: page_results, PAGES[4]: page_prediction, PAGES[5]: page_roadmap}[_page]()
