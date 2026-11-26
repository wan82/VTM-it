# VTM-it — Fast CU Partitioning Research Workspace for VVC/VTM

Research on fast CU partitioning for **VTM**, the H.266/VVC reference software.
Spatial features are extracted per CTU from raw YUV, and the ground-truth QT partition depth
obtained by decoding with the official VTM serves as the label. Together they form a training set
for models that predict whether and how a CTU should be split, so the encoder can skip part of the
RDO search and reduce encoding time (ET) at a small BD-rate cost.

## Directory Layout

```
VTM-it/
├── Makefile                    Top-level driver: runs every sub-project from the repo root (no cd needed)
├── VVCSoftware_VTM-VTM-20.2/   Official VTM 20.2 reference software (+ arm64 build support and prediction hooks)
├── perSequenceCfg/             Per-sequence VTM .cfg files (authoritative resolution / bit depth / frame rate / frame count)
├── Data/                       Data hub
│   ├── binEncode202/             RA bitstreams encoded by VTM-20.2 (<seq>_RA_str<qp>.bin); decoded to get QT/MTT labels  *
│   ├── labelCtu/                 Labels: one CSV per sequence, per-CTU max QT depth (qt_max) at 4 QPs
│   ├── labelMtt/                 Labels: sampled MTT partitions (one CSV per sequence×QP, per node w/h/mttDepth/splitMode)  *
│   ├── temp/tempYuvFeature/      Intermediate: per-CTU spatial features from the collector (one CSV per sequence)  *
│   ├── temp/trainMerged/         Intermediate: per-sequence merged tables (features + labels)  *
│   ├── TrainingData/Training.csv Final training set (the only file here)  *
│   ├── Log/FastTest/             fastTest encoding logs  *
│   ├── LogOriginalEnc/           Encoding logs from the unmodified encoder (baseline)
│   └── ModelResults/             Machine artifacts: qtForest.bin + models/<QP>.json (no reports)  *
├── Report/                     All reports live here (data goes to Data/, reports go to Report/)
│   ├── ModelTraining/            Training report: report.md + metrics.csv + figures/
│   ├── Analysis/                 Log analysis report: analysis.md + bdSummary.csv
│   ├── QTReport.md               Project report: the overall pipeline
│   └── QTReport_ConfGating.md    Project report: confidence gating + bidirectional early termination, method and comparison
├── yuvFeatureExtract/          Sub-project ① (C++): YUV → features, and merges features × labels into Training.csv
├── modelTrain/                 Sub-project ② (Python): trains models, writes reports, produces qtForest.bin
├── logAnalyze/                 Sub-project ③ (Python): BD-rate / Time-Reduction log analysis
├── .venv/                      Virtual environment shared by all Python sub-projects (created on first run)  *
└── oldFashion/                 Historical archive: early embedded prototypes + post-mortems + retired Python merger  *
```

`*` = generated locally or too large, so not tracked in this repository (see `.gitignore`).
Everything marked this way can be regenerated with the `make` targets below, given the source YUVs.

> **Data and reports are kept separate**: `Data/` only holds machine artifacts such as data, logs and
> model binaries; `Report/` collects every human-readable report.
> Feature extraction and label merging are both done in C++. The two Python sub-projects
> (modelTrain / logAnalyze) **share one virtual environment, `VTM-it/.venv`**, and each installs
> only the dependencies it needs.

## Naming Conventions (file names vs. column names)

This repository deliberately keeps **two naming schemes** apart. They follow different rules and
should not be mixed up:

- **File and directory names → camelCase.** Every folder, source file, script and generated artifact
  owned by this repository uses camelCase, e.g. `yuvFeatureExtract/`, `labelCtu/`, `tempYuvFeature/`,
  `perSequenceCfg/`, `qtForest.bin`, `bdSummary.csv`, `featureWrapper.png`, `exportQtForest.py`, and the
  C++ binaries `yuvFeatureCollector` / `yuvLabelMerger`. **Exceptions**: `VVCSoftware_VTM-VTM-20.2/`
  (the official VTM name, left unchanged) and `oldFashion/` (historical archive, left unchanged).

- **CSV column names → kept exactly as in the data schema, not camelCased.** The column names of
  `Training.csv` are a data contract between training and inference: the C++ collector, the merger
  and the Python training code all match on these literal strings, so renaming them would break one
  side. The following are therefore kept as-is: `sequence, POC_ID, pos_x, pos_y`, the 4 label columns
  `22qt_max/27qt_max/32qt_max/37qt_max`, the 4 QP-value columns `22QP/27QP/32QP/37QP`, and the 18
  spatial feature columns (e.g. `Sobel_mean`, `Lap_std`).

- **When a column name appears inside a generated file name, the column name wins and is kept
  verbatim.** A few artifacts embed a column-name token in their file name; it keeps the column
  spelling rather than camelCase so it maps one-to-one to the CSV column:
  - Per-QP model metadata: `Data/ModelResults/models/22qt_max.json` (…27/32/37)
  - Confusion matrix figures: `Report/ModelTraining/figures/cm_22qt_max_RandomForest.png`, etc.
    (template: `cm_<labelColumn>_<modelName>.png`)

  > In short: "the file-name shell is camelCase, embedded column-name tokens are untouched". Renaming
  > the columns as well would require changing the C++ collector/merger, `LABEL_COLS/QP_VALUE_COLS` in
  > `train.py`, and the feature-name matching in VTM's `QtForestPredictor` at the same time — a
  > cross-language data-contract change, never a one-sided edit.

## End-to-End Data Flow

```
Official VTM 20.2 decode ──► Data/labelCtu/             (ground-truth QT depth labels, every CTU)
                                                        │
Raw YUV ─►[yuvFeatureExtract: collect]─► Data/temp/tempYuvFeature/  (spatial features of sampled CTUs)
                                                        │
                  [yuvFeatureExtract: merge] inner join on (POC_ID,pos_x,pos_y) + drop NaN
                                                        ▼
                                         Data/TrainingData/Training.csv
                                                        │
                       [modelTrain] clean → feature selection → train NB/DT/RF → GroupKFold(5)
                                                        ▼
              model bin → Data/ModelResults/ ; reports / confusion matrices → Report/ModelTraining/
```

## Quick Start

The repository root has a **top-level Makefile** that drives every sub-project without `cd`:

```bash
# From the VTM-it root:
make help        # list all commands
make extract     # extract: YUV → intermediate features (tempYuvFeature)
make merge       # merge: features × labelCtu → Training.csv (with QP-values)
make data        # = extract + merge
make train       # train → model bin → Data/ModelResults; report → Report/ModelTraining
make all         # data + train, full pipeline
make clean       # clean the sub-projects' build outputs
make distclean   # clean + remove the shared venv and training artifacts
```

End-to-end validation after embedding the models into VTM:

```bash
make fastTest    # build VTM + encode 33 frames of BQSquare (QP22/27/32/37), logs → Data/Log/FastTest
make analyze     # compare against Data/LogOriginalEnc, compute BD-rate / Time-Reduction → Report/Analysis
```

Command-line variables are **passed through automatically** to the sub-projects, for example:

```bash
make merge KEEP=no                                   # do not keep per-sequence intermediate tables
make train CORR=0.9 TOPK=8                           # tune feature-selection parameters
make fastTest YUV=/path/to/BQSquare_416x240_60.yuv   # use a specific YUV
```

Each sub-project can still be run on its own with `make` inside its directory (see each one's `README.md`).

## Embedding into the VTM Encoder (two orthogonal predictors, independently switchable)

Both trained models can be plugged directly into the official VTM 20.2 encoder and are switched on
via environment variables:

- **QT early termination** (`qtForest.bin`, `#QTPRED`): predicts the maximum QT depth at the CTU root,
  with bidirectional early termination and confidence gating.
  See [`VVCSoftware_VTM-VTM-20.2/QTPRED.md`](VVCSoftware_VTM-VTM-20.2/QTPRED.md).
- **MTT direction pruning** (`mttForest.bin`, `#MTTPRED`): at nodes where BT/TT is allowed, predicts
  the splitMode and, when confident, prunes the BT/TT splits in the opposing direction (QUAD and
  no-split are never touched). See [`VVCSoftware_VTM-VTM-20.2/MTTPRED.md`](VVCSoftware_VTM-VTM-20.2/MTTPRED.md).

```bash
export QT_FOREST_MODEL=$(pwd)/Data/ModelResults/qtForest.bin    # enable QT early termination
export MTT_FOREST_MODEL=$(pwd)/Data/ModelResults/mttForest.bin  # enable MTT pruning (either or both)
./VVCSoftware_VTM-VTM-20.2/build/.../EncoderAppStatic -c <RA cfg> -c <seq cfg> -q 32 ...
```

Both C++ inference paths match the training side **sample-for-sample / element-for-element**
(features are byte-identical, and forest traversal agrees with sklearn `predict`).
With the corresponding environment variable unset, the encoder behaves exactly like the original.

## Sub-projects

### yuvFeatureExtract (C++, CMake + Makefile, run per module)
- **Collect** (`make extract`): reads luma from raw YUV, normalizes to 10-bit, tiles 128×128 CTUs
  (dropping incomplete edge CTUs), and extracts **18 spatial features** aligned with the existing
  `128csv` format. Supports per-sequence YAML configuration, automatic exclusion of useless frames,
  and sampling by fixed count or by ratio.
- **Merge** (`make merge`): inner-joins features with the `labelCtu` labels on `(POC_ID,pos_x,pos_y)`,
  drops rows containing NaN, and writes `Data/TrainingData/Training.csv`; per-sequence intermediate
  tables go to `Data/temp/trainMerged/` (`make merge KEEP=no` discards them). Following VTM (RA, GOP=32),
  it also derives from the POC the actual slice QP under each of the 4 base QPs and adds the columns
  `22QP/27QP/32QP/37QP`.
  Training.csv columns: `sequence,POC_ID,pos_x,pos_y, 4×qt_max, 4×QP, 18×features` (30 columns).
- **MTT labels** (`make mttLabel`): decodes the RA bitstreams in `Data/binEncode202/` and exports MTT
  partition labels sampled as "random POC × random block" → `Data/labelMtt/<seq>_QP<qp>.csv`
  (per node: `w/h/qtDepth/mttDepth/splitMode`). Entropy decoding runs inside the VTM decoder
  (`#MTTDUMP` hook, zero overhead by default); orchestration and sampling live in the `mttLabel/`
  module. See [`yuvFeatureExtract/mttLabel/README.md`](yuvFeatureExtract/mttLabel/README.md).
- **MTT features** (`make mttFeature`): attaches spatial features to every block in labelMtt
  (② directionality + ③ split-cost proxies + ④ shape/QP, with ① complexity of the enclosing CTU as
  context) → `Data/TrainingData/TrainingMtt.csv`.
  See [`yuvFeatureExtract/mttFeature/README.md`](yuvFeatureExtract/mttFeature/README.md).
- In-house libraries are in `libs/` (MiniYaml / YuvGrayLib / SpatialFeatureLib / **MttSampleLib /
  MttFeatureLib**); the collector is in `collector/`, the merger in `merger/`, MTT label orchestration
  in `mttLabel/`, and MTT features in `mttFeature/`.

### modelTrain (Python, Makefile + venv)
- **QT** (`make train`, `train.py`): reads `Training.csv`, performs data cleaning and feature selection,
  and trains **Naive Bayes / Decision Tree / Random Forest** separately per QP (multi-class qt_max 0–4),
  evaluated with **GroupKFold(5, grouped by sequence)**. Outputs a full report plus 12 confusion
  matrices, and **directly produces** `qtForest.bin` during training (per-QP metadata in `models/<qp>.json`).
- **MTT** (`make trainMtt`, `trainMtt.py`): a separate model, independent of QT. Reads `TrainingMtt.csv`
  and predicts the **5-class splitMode** (NO/BTH/BTV/TTH/TTV) of each MTT node, with 4 models (one per QP),
  balanced weights and macro-F1. Directly produces `mttForest.bin` (same bin format as `qtForest.bin`,
  readable from C++). Reports are in `Report/MttTraining`.

### logAnalyze (Python, Makefile + shared venv)
- **Analysis** (`make analyze`): aligns fastTest logs with the baseline logs POC by POC and computes
  Bjøntegaard BD-rate and Time-Reduction → `Report/Analysis`.
  See [`logAnalyze/README.md`](logAnalyze/README.md).

## About the VTM Reference Software

`VVCSoftware_VTM-VTM-20.2/` is based on the official release. Two kinds of changes were made:

- **Build**: an `NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")` check added in 3 CMakeLists files so it
  builds on Apple Silicon with the scalar fallback.
- **Research hooks**, each tagged in the source and inactive unless its environment variable is set:
  `#QTPRED` (`QtForestPredictor`), `#MTTPRED` (`MttForestPredictor`) in the encoder, and `#MTTDUMP`
  (`MttLabelDumper`) in the decoder.

`perSequenceCfg/` provides the official encoding parameters for each sequence (the width/height/bit
depth/frame count in the collector config are filled in from these).

## oldFashion (historical archive, not tracked in this repository)

The two early prototypes that embedded the predictor **directly inside the encoder** are archived
locally for reference only:
- `VTM_VVC_128/`: VTM 20.0, fed partition decisions from an offline CSV lookup (fragile, crashed on large sequences).
- `VTM-GOP/`: VTM 20.2, real-time in-encoder features + Naive Bayes prediction (contained a bug that
  disabled the residual features).
- Post-mortem and fix-log documents for the fatal issues in both versions.

The current workflow is **offline and decoupled**: features and labels are generated offline,
outside the encoder, which allows clean training and validation and avoids the crashes and the
training/inference feature mismatch of the early embedded approach.
