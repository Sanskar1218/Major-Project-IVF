# An Intelligent IVF Optimization: Next-Generation Frameworks for Successful Implantation Prediction

A college-project prototype: a Streamlit dashboard plus a reproducible scikit-learn pipeline that estimates the probability of an IVF treatment outcome and tests one research question:

> Does adding **male-factor** and **embryonic** variables improve outcome prediction compared with **clinical features alone**?

> **Not a clinical tool.** This is an experimental research prototype. It has not been clinically validated, must not be used for medical decisions, and must never be used to decide whether anyone should undergo, avoid or change fertility treatment. See [Why this is not a decision-support system](#10-why-this-is-not-a-clinically-validated-decision-support-system).

## 1. What the project does

- Generates a **synthetic demonstration dataset**, or loads **your own CSV**, with a data-inspection page (shape, columns, missing values, target checks, leakage and identifier warnings, feature mapping).
- Trains three models: **Logistic Regression** (baseline), **Random Forest**, and **Gradient Boosting** (scikit-learn's `HistGradientBoostingClassifier`, so there is no XGBoost dependency).
- Runs two experiments on *exactly the same train/test rows*:
  - **Experiment A – clinical-only:** female clinical and other pre-treatment variables.
  - **Experiment B – combined:** clinical + male-factor + embryonic variables.
- Reports ROC-AUC, accuracy, precision, recall, F1, confusion matrix, Brier score, ROC curves, calibration, permutation importance, tree importance, and a paired-bootstrap interval for the A-vs-B AUC difference.
- Offers a prediction page that uses the exact fitted preprocessing pipeline, a configurable threshold, and an occlusion-style explanation of the inputs.
- Includes a roadmap page for future image/multimodal work. **That work is not implemented.**

Every displayed metric is computed from the current run's predictions; nothing is hardcoded.

## 2. Research problem and prediction target

IVF can be assessed at different endpoints: embryo **implantation**, **clinical pregnancy**, **live birth**. They are different prediction tasks. In real-data mode **you** choose the target column, which value is positive, and what the outcome represents. Models and results are labelled with that task. The app never converts one outcome into another (a live-birth column trains a live-birth model).

## 3. Dataset options

| Mode | What it is | What it proves |
|---|---|---|
| **Demonstration (default)** | Invented records from a documented formula (`data_utils.SYNTH_SPEC`, also shown in the app). Seeded, so reproducible. Carries a permanent *DEMO MODE* warning. | Only that the software works. The male/embryo variables are given real weight in the generator, so Experiment B beating A is **built in**. |
| **Upload a real CSV** | Your own records, processed in memory. Nothing is sent to any service. | Whatever the data support, subject to the limitations below. |

The project does not include, scrape or assume access to any clinical registry data (for example, patient-level HFEA records). Only use data you are authorised to use, preferably de-identified.

## 4. Install

Requires Python 3.10+ (tested on Python 3.13 with scikit-learn 1.9, pandas 3.0, Streamlit 1.65, Plotly 7.1).

```bash
python -m venv .venv

# activate it:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Windows (cmd):         .venv\Scripts\activate.bat
#   macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
```

No GPU, API key, paid service or large download is needed.

## 5. Launch

```bash
streamlit run app.py
```

Open the URL it prints (usually http://localhost:8501). In VS Code, open the `ivf_prediction` folder and run this in the integrated terminal.

**Google Colab:** the pipeline works there without the UI, `python train.py --demo`. Running the Streamlit UI in Colab needs a tunnel, which is easier to do locally.

## 6. Upload a real CSV

1. Go to **2 · Dataset Explorer** and choose **Upload a real CSV**.
2. Upload the file (comma-separated; semicolon/tab are auto-detected; max 25 MB). Invalid, empty or unparsable files give an explanatory message.
3. **Step 1:** pick the target column, the positive class, and what the target represents. Targets must have exactly two classes.
4. **Step 2:** optionally pick a patient-ID column (enables a patient-grouped split when IDs repeat), then review the feature-group table. Columns default to *Exclude* unless their name clearly matches a group; change any row. Suspected post-treatment columns (names like *hCG*, *pregnancy*, *birth*) default to excluded.
5. **Step 3:** read the warnings (missingness, constant columns, identifiers, duplicates, leakage, sample size), then **Save configuration**.
6. Go to **3 · Model Training**, choose experiments and models, and train.

If the data are too small or one class is too rare (fewer than 60 rows, fewer than 20 in the rarer class, or fewer than 5 rarer-class rows in the test set), training is refused with an explanation instead of producing meaningless metrics. These minimums are practical floors, not guarantees; an AUC from a few dozen positive cases is still very noisy.

Command-line equivalent:

```bash
python train.py --csv data/my.csv --target live_birth --positive 1 --task "Live birth" --id-col patient_id
# optional: --mapping mapping.json   ({"column": "female_clinical" | "other_clinical" | "male_factor" | "embryonic" | "exclude"})
```

## 7. Run the synthetic demonstration

The app starts in demo mode with a ready-made configuration:

1. **3 · Model Training → Train & evaluate** (about 20–40 seconds).
2. **4 · Results & Explainability** to compare models and A vs B.
3. **5 · IVF Outcome Prediction** → *Fill with typical training values* → *Estimate probability*.

Or from a terminal: `python train.py --demo`.

## 8. Interpreting the metrics

| Metric | Meaning | Caution |
|---|---|---|
| **ROC-AUC** | Chance a random positive case gets a higher score than a random negative one (0.5 = chance). Threshold-independent. | Says nothing about whether probabilities are accurate. Undefined if the test set has one class. |
| **Accuracy** | Share of correct class calls at the chosen threshold. | Misleading with imbalanced classes. |
| **Precision** | Of cases predicted positive, share truly positive. | Undefined if nothing is predicted positive. |
| **Recall (sensitivity)** | Of true positives, share found. | Trades off against precision via the threshold. |
| **F1** | Harmonic mean of precision and recall. | Ignores true negatives. |
| **Brier score** | Mean squared error of the probabilities (lower is better). | Compare with the base-rate score; it is not on an intuitive scale. |
| **Calibration plot** | Predicted vs observed rates in quantile bins; near the diagonal is good. | Needs enough test rows (≥ 3 bins of ~20). |
| **CV ROC-AUC** | Cross-validated AUC on **training rows only**; the model-selection signal. | The test set is never used to choose a model. |
| **A-vs-B Δ AUC + 95% interval** | Paired bootstrap over test rows. If the interval includes 0, the experiment does **not** demonstrate a difference. | Reflects test-sampling noise only, not retraining variability or other clinics. |
| **Permutation importance** | Drop in test AUC when a column is shuffled. | Describes the fitted model, not causation; correlated features share credit. |

Accuracy, ROC-AUC and feature importance alone do not establish clinical usefulness.

## 9. Known limitations

- One split of one dataset; no external validation; wide uncertainty on small data.
- Leakage detection is heuristic (name matching plus a single-feature near-perfect-predictor scan). You must confirm each feature is known **before** the outcome.
- Implantation, clinical pregnancy and live birth may be recorded per cycle, per transfer or per embryo; the app does not model that structure. Per-embryo labels with several embryos per cycle need careful handling by you.
- Missing values are imputed (median / most frequent) without modelling why they are missing.
- Tabular morphology and quality **scores** are used. Raw sperm or embryo images are not.
- Saved models (`models/`) are scikit-learn pickles via `joblib`. The app only loads files it wrote, after a SHA-256 manifest check and a scikit-learn version check, and offers no upload path for model files. Never load model files from sources you do not trust.
- The synthetic data's outcome comes from a formula; performance on it is not evidence of anything clinical.

## 10. Why this is not a clinically validated decision-support system

Clinical decision support requires, at minimum: a defined clinical question and population; prospective or external validation across clinics; calibrated, well-characterised probabilities; evaluation of net benefit and impact on outcomes; bias and fairness analysis; regulatory review; and clinician oversight. This prototype has none of these. Its outputs are labelled *"Experimental model estimate — not a medical diagnosis or treatment recommendation."* and it never advises for or against treatment.

## Project layout

```
ivf_prediction/
├── app.py            Streamlit dashboard (6 pages)
├── data_utils.py     synthetic data, CSV loading, schema/target/leakage checks
├── train.py          pipelines, split, experiments A/B, save/load, prediction helpers, CLI
├── evaluate.py       metrics, curves, importance, bootstrap A-vs-B
├── tests/test_smoke.py   23 tests (pipeline, leakage, persistence, edge cases, app pages)
├── requirements.txt
├── .gitignore        excludes models/, data/ and *.csv so no patient data is committed
├── models/           trained artefacts are written here
└── data/             put your CSVs here (git-ignored)
```

Run the tests: `pip install pytest` then `python -m pytest -q`.

## Optional image extension (not implemented)

A CNN/transfer-learning experiment would need a real labelled image set, a documented patient-level split, and a licence review. This repository contains no image code, labels or predictions; the Roadmap page describes the plan only.
