# tabular-ml-pipeline

Research slices: **leakage-safe sklearn tabular pipelines** with **grouped CV**, **probability calibration**, and **reliability diagrams**.

Public data only. Methodology demos — not clinical, forestry, or production accuracy claims.

---

## Slice 1 — GroupKFold + calibration (breast_cancer)

### Hypothesis

1. Fitting impute/scale on the full matrix *before* CV leaks information into fold metrics.
2. When rows share an entity id, random `KFold` overstates OOS quality; `GroupKFold` is the honest default.
3. Wrapping the train-fold estimator in sigmoid calibration should improve Brier / ECE when the base probabilities are miscalibrated — and should be measured, not assumed.

### Results (GroupKFold k=5, 40 synthetic groups)

| Model | mean accuracy | mean ROC-AUC | mean Brier | mean ECE |
| --- | ---: | ---: | ---: | ---: |
| Logistic (raw) | 0.974 | 0.994 | **0.021** | **0.027** |
| Logistic (sigmoid-calibrated) | 0.975 | 0.995 | 0.025 | 0.048 |
| HistGradientBoosting (raw) | 0.968 | 0.993 | 0.027 | 0.033 |

On this easy set, raw logistic was already well calibrated; sigmoid did not help.

```bash
python scripts/run_breast_cancer_slice.py
```

---

## Slice 2 — Reliability diagrams on harder grouped covertype

### Hypothesis

1. Reliability diagrams + ECE make miscalibration visible beyond accuracy.
2. On a harder grouped OpenML/sklearn dataset (not the easy breast_cancer toy), raw logistic/HGB will show higher ECE than on breast_cancer.
3. Isotonic/Platt calibration on train folds may cut ECE — measure and report if it helps or hurts.

### Dataset

- **Source**: `sklearn.datasets.fetch_covtype` (UCI Covertype), subsample of 5,000 rows (`random_state=0`).
- **Label**: binary `cover_type == 2` (Lodgepole Pine) vs rest.
- **Features**: 10 continuous cartographic vars + 4 wilderness indicators (14 total).
- **Groups**: soil-type id = `argmax` of the native soil one-hots. Soil columns are **not** used as features (that would leak group identity into `X`). Rare soil types in the subsample (`< 40` rows) merge into a residual group → **21 groups**.
- First run downloads/caches covertype via sklearn.

### Method

- Same leakage-safe pipelines as slice 1 (`SimpleImputer`→`StandardScaler`→`LogisticRegression`; impute→`HistGradientBoosting`).
- Outer `GroupKFold` (k=5); when calibrating, `GroupAwareCalibratedClassifier` builds inner GroupKFold splits from **train groups only**.
- Compare **raw** vs **Platt (sigmoid)** vs **isotonic**.
- Metrics: accuracy, ROC-AUC, Brier, equal-width ECE (10 bins), plus pooled OOF reliability diagrams (PNG via matplotlib, ASCII under `figures/`).

### Results (this slice, GroupKFold k=5, 5000-row covertype subsample)

| Model | mean accuracy | mean ROC-AUC | mean Brier | mean ECE | pooled ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic (raw) | 0.681 | 0.741 | 0.211 | 0.120 | 0.086 |
| Logistic (sigmoid / Platt) | 0.677 | 0.739 | 0.211 | 0.127 | 0.089 |
| Logistic (isotonic) | **0.690** | 0.741 | **0.202** | **0.093** | **0.046** |
| HistGradientBoosting (raw) | 0.753 | 0.816 | 0.171 | 0.066 | 0.017 |
| HistGradientBoosting (sigmoid) | 0.752 | 0.814 | 0.172 | 0.064 | 0.025 |

**Breast-cancer reference** (same logistic pipeline, slice 1): mean ECE **0.027**, pooled ECE **0.018**.

### What the numbers say

- Hypothesis (2) holds: covertype logistic mean ECE (~0.12) is ~4× breast_cancer (~0.027). Accuracy alone (0.68) hides how far probabilities sit from the diagonal — see `figures/reliability_logistic_raw.txt`.
- Hypothesis (3) is mixed: **isotonic helped** (Brier 0.211→0.202, mean ECE 0.120→0.093, pooled ECE 0.086→0.046). **Platt/sigmoid did not** on logistic (ECE slightly worse). HGB was already better calibrated; sigmoid was roughly neutral on mean ECE.
- Reliability diagrams make the high-confidence / low-accuracy bins visible — ECE alone is a scalar; the bins show *where* calibration fails.

```bash
python scripts/run_reliability_slice.py
```

### Assumptions and limits (slice 2)

- Covertype subsample + soil-type groups is a public research construct, not an operational stand-inventory model.
- Equal-width ECE is sensitive to binning and class imbalance; report Brier alongside it.
- Isotonic can overfit small calibration folds; GroupAware inner splits mitigate leakage but not variance.
- No nested hyperparameter search; HGB `max_iter=80` is a fixed demo setting.
- Not a clinical, credit, or production claim.

---

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_breast_cancer_slice.py
python scripts/run_reliability_slice.py
```

## Layout

```
src/tabular_ml/
  calibration.py   # ECE + reliability bins / ASCII / PNG
  pipeline.py      # logistic + HGB builders
  evaluate.py      # GroupKFold eval + group-aware calibration + compare helpers
  data.py          # breast_cancer (synthetic groups) + covertype (soil groups)
scripts/run_breast_cancer_slice.py
scripts/run_reliability_slice.py
figures/           # reliability ASCII (and PNG when script is run)
tests/
```

## Next slices (not done)

- Nested CV for `C` / tree depth
- Target / frequency encoding behind the same leakage-safe fold wall
- OpenML Adult with `native-country` as an alternate real group column
