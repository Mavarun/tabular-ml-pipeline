# tabular-ml-pipeline

Research slice: **leakage-safe sklearn tabular pipelines** with **grouped CV** and **probability calibration** diagnostics.

Public toy data only (`sklearn.datasets.load_breast_cancer`). This is a methodology demo, not a clinical or production accuracy claim.

## Today's hypothesis

1. Fitting impute/scale on the full matrix *before* CV leaks information into fold metrics.
2. When rows share an entity id, random `KFold` overstates OOS quality; `GroupKFold` is the honest default.
3. Wrapping the train-fold estimator in sigmoid calibration should improve Brier / ECE when the base probabilities are miscalibrated — and should be measured, not assumed.

## Method

- **Baseline**: `SimpleImputer(median)` → `StandardScaler` → `LogisticRegression` inside one `Pipeline` (prep never fit outside the train fold).
- **Second model**: median impute → `HistGradientBoostingClassifier` (no scale).
- **CV**: `GroupKFold` on synthetic entity groups carved from a shuffled breast-cancer matrix (demo construct for group leakage, not real patients).
- **Calibration**: `GroupAwareCalibratedClassifier` precomputes inner `GroupKFold` splits from *train groups only*, then fits `CalibratedClassifierCV` on those index pairs so outer test groups never enter calibration.
- **Metrics**: accuracy, ROC-AUC, Brier score, equal-width ECE (10 bins).

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_breast_cancer_slice.py
```

## Results (this slice, GroupKFold k=5, 40 groups)

| Model | mean accuracy | mean ROC-AUC | mean Brier | mean ECE |
| --- | ---: | ---: | ---: | ---: |
| Logistic (raw) | 0.974 | 0.994 | **0.021** | **0.027** |
| Logistic (sigmoid-calibrated) | 0.975 | 0.995 | 0.025 | 0.048 |
| HistGradientBoosting (raw) | 0.968 | 0.993 | 0.027 | 0.033 |

On this easy, nearly linear public set, **raw logistic was already well calibrated**. Sigmoid calibration did not help Brier/ECE (it slightly worsened both). That is a real negative result for hypothesis (3) *on this dataset*, not a failure of the pipeline — the machinery is there to catch when calibration helps or hurts.

## Assumptions and limits

- Synthetic groups are a teaching device. Real entity ids (patient, account, ticker day) belong in the data, not invented after the fact for production claims.
- Breast cancer is linearly separable enough that almost any leakage-safe linear model looks strong. Do not read these numbers as domain performance.
- ECE with equal-width bins is sensitive to binning and sample size; treat it as a diagnostic beside Brier.
- No target encoding / high-cardinality categoricals yet. No nested hyperparameter search. No claim of live or production accuracy.

## Layout

```
src/tabular_ml/
  calibration.py   # ECE
  pipeline.py      # logistic + HGB builders
  evaluate.py      # GroupKFold eval + group-aware calibration
  data.py          # breast_cancer + synthetic groups
scripts/run_breast_cancer_slice.py
tests/
```

## Next slices (not done)

- OpenML dataset with real group column
- Reliability diagrams committed as figures
- Nested CV for `C` / tree depth
- Target / frequency encoding behind the same leakage-safe fold wall
