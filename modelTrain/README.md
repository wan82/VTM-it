# modelTrain — Training and Evaluation (Python, Makefile + venv)

Reads `TrainingData/Training.csv`, performs **data cleaning + feature selection**, and trains
**Naive Bayes / Decision Tree / Random Forest** separately per QP (multi-class qt_max 0–4).
Models are evaluated with **GroupKFold(5, grouped by sequence)**, and the pipeline produces
**confusion matrix figures** and a **full Markdown report**.

## Pipeline

1. **Data cleaning**: ±Inf is treated as missing and dropped together with NaN (Training.csv is
   already mostly clean; this is a safety net). Labels are rounded to integer classes; duplicate rows
   are counted (not removed).
2. **Feature selection**: ① drop near-constant features; ② drop highly correlated redundant features
   (|corr| > threshold, default 0.95); ③ compute per-QP mutual information to rank discriminative power;
   ④ **wrapper forward selection**: starting from zero features, add features one at a time in MI
   order, scoring each step by the mean Weighted-F1 of 3 models × 4 QPs. The smallest feature count
   whose score reaches **peak − tolerance** is taken as the "sweet spot", and only those features are
   kept. `--no-wrapper` disables this step; `--top-k` simply takes the top K.
3. **Training + evaluation**: one set per QP (22/27/32/37), each training NB/DT/RF.
   **GroupKFold(5)** groups by `sequence` (each video goes entirely to either train or test, measuring
   generalization to unseen videos). Accuracy / Macro-F1 / Weighted-F1 are computed from out-of-fold
   predictions, and confusion matrices are plotted.
4. **Outputs**: `report.md` (full report, including the wrapper curve and sweet spot), `metrics.csv`,
   `figures/cm_<QP>_<model>.png` (12 figures), and `figures/featureWrapper.png` (feature count vs. performance).

## Usage

```bash
cd modelTrain
make            # first run creates the venv and installs sklearn etc.; trains and reports if Training.csv is found
```

Parameters can be overridden:

```bash
make CORR=0.9 TOPK=10                 # correlation threshold / keep only the top 10 features
make INPUT=/path/Training.csv OUT=/path
# Call the script directly (more options):
./run.sh --rf-estimators 200 --max-rows 20000 --folds 5
```

- Other targets: `make setup` / `make clean` / `make distclean`.
- `--max-rows` randomly subsamples rows for quick trial runs; `--rf-estimators` sets the number of
  trees in the random forest (default 100).
- Wrapper options: `--wrapper-tol` (sweet-spot tolerance, default 0.003; smaller keeps more features,
  0 takes the performance peak), `--wrapper-rf` / `--wrapper-folds` / `--wrapper-sample` (lightweight
  settings used during the search, for speed), `--no-wrapper` (disable).

Default paths (relative to the repo root, derived from the script location, so nothing needs to
change on a new machine):
- Input: `<repo root>/Data/TrainingData/Training.csv`
- Models: `<repo root>/Data/ModelResults`; reports: `<repo root>/Report/ModelTraining`

## Outputs

```
Data/ModelResults/
├── qtForest.bin                    ★ VTM-readable forest model (all 4 QPs in one file, produced directly by training)
└── models/<qp>.json                per-QP metadata: model / feature order / classes / CV metrics (for humans only)

Report/ModelTraining/
├── report.md                       full report (data overview / cleaning / feature selection / per-QP model comparison / summary / final model)
├── metrics.csv                     metrics table (QP × model: acc / CV mean ± std / macro-F1 / weighted-F1)
├── figures/cm_<QP>_<model>.png     12 confusion matrices (rows = true, columns = predicted, with counts and row-normalized %)
└── figures/featureWrapper.png      feature count vs. performance curve (sweet spot marked)
```

> `qtForest.bin` is **serialized directly** during training (no joblib, no conversion step). The
> default `RFMINLEAF=50` bounds the size of the RF/decision trees, so the model is a few MB. To embed
> it in VTM, point `QT_FOREST_MODEL` at the bin; the feature order is listed in `<qp>.json`.

## Notes

- The multi-class qt_max (0–4) is **imbalanced** (mostly 0 at high QP), so Macro-F1 is noticeably lower
  than accuracy, as expected. The report gives accuracy, Macro-F1 and Weighted-F1 together for a
  complete picture.
- **Grouping by sequence** in cross-validation is stricter than random stratification and closer to
  real-world generalization; a random stratified split can be run separately to see the optimistic upper bound.
- **Model selection is done at the algorithm level**: the three algorithms are compared by their mean
  Weighted-F1 over the 4 QPs, the best one becomes the final algorithm, and it is then trained once per
  QP and saved (so deployment uses a single algorithm).
- The `22QP/27QP/32QP/37QP` (QP-value) columns in Training.csv are **excluded from the feature set** by
  default, keeping the 18 spatial features unchanged; the matching QP-value can be added as a feature if needed.
- The pipeline is easy to extend, e.g. to binary classification (split / no split) or to a single
  model with QP as a shared feature.

## MTT Training (`trainMtt.py`, a separate model independent of QT)

QT uses `train.py`; **MTT uses its own** `trainMtt.py`, and the two do not affect each other. It reads
`Data/TrainingData/TrainingMtt.csv` (produced by `make mttFeature`) and predicts the **5-class
splitMode** of each MTT node:

- Class codes: `NO=0, BTH=1, BTV=2, TTH=3, TTV=4` (the C++ side maps these back to PartSplit).
- **4 models, one per base QP** (22/27/32/37), selected by baseQP at inference time, same as QT.
- The classes are **heavily imbalanced** (NO ≈ 64%), so **balanced sample weights** are used throughout,
  and the main metrics are **macro-F1 + per-class F1 + confusion matrix**, not accuracy (always
  predicting NO already gives 64%).
- GroupKFold(5) by sequence; algorithm-level selection only chooses between **Decision Tree / Random
  Forest** (highest mean macro-F1 over the 4 QPs). Naive Bayes is reported for reference only (the
  forest bin only stores tree models).
- Training directly exports `Data/ModelResults/mttForest.bin` (format `"QTF1"`, the same as
  `qtForest.bin`, readable directly by the C++ `QtForestPredictor`; bin traversal has been verified to
  match sklearn predict sample-for-sample).

```bash
make train-mtt          # or `make trainMtt` from the repo root
#  → Data/ModelResults/mttForest.bin + mttModels/QP<qp>.json
#  → Report/MttTraining/{report.md, metrics.csv, figures/cm_QP<qp>_<model>.png}
```

Features are the 22 columns from `mttFeature` (② directionality + ③ split-cost proxies + ④ shape/QP +
CTU-level ① context). All are used by default, and the report gives their MI ranking (no wrapper step,
since the feature set is already a small, curated one).
