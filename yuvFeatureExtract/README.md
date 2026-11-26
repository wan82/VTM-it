# yuvFeatureExtract — YUV Feature Collection + Label Merging (C++)

Both steps are done in C++, so downstream only needs one Python environment (modelTrain):
1. **Collect**: extract 18 **spatial features** for each 128×128 CTU from raw YUV → intermediate `tempYuvFeature`.
2. **Merge**: join the features with the QT depth labels in `labelCtu` on `(POC_ID,pos_x,pos_y)` → `Training.csv`.

The MTT-side modules live here as well: [`mttLabel/`](mttLabel/README.md) (sampled MTT partition labels)
and [`mttFeature/`](mttFeature/README.md) (block-level features for the MTT training set).

## Directory Layout

```
yuvFeatureExtract/
├── CMakeLists.txt          top level (aggregates libs / collector / merger / mttLabel / mttFeature)
├── Makefile                make entry point (wraps cmake)
├── libs/                   in-house libraries (each in its own folder)
│   ├── MiniYaml/             minimal YAML-subset parser (no external dependencies)
│   ├── SpatialFeatureLib/    spatial features: moments, Sobel/Laplacian, Levene/Kruskal-Wallis
│   ├── YuvGrayLib/           grayscale image matrix: YUV reading, bit-depth conversion, CTU tiling, useless-frame detection
│   ├── MttSampleLib/         sampling plan shared by the MTT label exporter
│   └── MttFeatureLib/        pure functions for the MTT block features
├── collector/              collection sub-project: YUV → intermediate features
│   ├── src/main.cpp
│   └── config/config.yaml    example configuration
├── merger/                 merge sub-project: features × labels → Training.csv
│   └── src/main.cpp
├── mttLabel/               MTT partition label export (see mttLabel/README.md)
└── mttFeature/             MTT block features → TrainingMtt.csv (see mttFeature/README.md)
```

## Build and Run

```bash
cd yuvFeatureExtract
make            # configure and build (Release)

make run        # ① collect: YUV → Data/temp/tempYuvFeature (uses collector/config/config.yaml)
make merge      # ② merge: features × labelCtu → Data/TrainingData/Training.csv
make data       # ① + ② in one go

# Merge option: keep the per-sequence intermediate tables or not (default yes)
make merge KEEP=no
# Custom paths:
make merge LABELS=/path FEATURES=/path TRAINING=/path/Training.csv TEMP_MERGED=/path
```

Merge output: `Data/TrainingData/` holds **only `Training.csv`** (rows containing NaN already removed);
per-sequence intermediate tables go to `Data/temp/trainMerged/` (deleted when `KEEP=no`).

The merge step also **computes the QP-value the way VTM-VVC does** and adds 4 columns `22QP,27QP,32QP,37QP`:
with the RA default GOP=32, the temporal layer TID is derived from `POC%32`, and (base QP, TID) then
gives the actual slice QP of the frame containing the CTU (consistent with the decoded PocInfo, e.g.
POC0 → 19/24/29/34). Column order: keys → 4 qt_max → 4 QP → 18 features.

Requires CMake ≥ 3.13 and a C++17 compiler.

## Processing Steps (per sequence)

1. **Read luma**: using the width/height/bit depth/chroma format from the YAML, read the luma plane
   frame by frame from the raw YUV (chroma is skipped).
2. **Normalize bit depth**: whether the source is 8-bit or 10-bit, left-shift to an internal 10-bit
   (matching the VTM default): `out = in << (10 - inputBitDepth)`.
3. **Determine frame count**: use `frames` if the YAML provides it, otherwise infer it as file size ÷ bytes per frame.
4. **Exclude useless frames (automatic detection)**:
   - blank / near-constant frames: whole-frame standard deviation < `blank_std_threshold`;
   - duplicate frames: mean absolute difference to the previous valid frame < `duplicate_mad_threshold`.
   - Detection runs on a pixel subsample (`detect_pixel_stride`) for speed. Excluded frame indices are
     printed for checking.
5. **Estimate row count**: `usable frames × complete CTUs per frame` (edge CTUs smaller than 128 are dropped).
6. **Sampling**: each sample independently picks a random frame and a random CTU position, deduplicated
   by (frame, position), until `ctu_per_yuv` is reached (default 10, can be overridden per sequence).
   `random_seed` makes it reproducible.
7. **Extract features**: compute the 18 spatial features for each sampled 128×128 block and write one row.
8. **(Optional) temporal residual features `res_*`** (`collect_residual: true`, on by default):
   - Subtract the **co-located block in the L0 reference** of the block's frame and compute
     `res_mean/res_std/res_skew/res_kurt` (4 columns, appended after the 18 spatial features).
   - **L0[0] reference POC = POC with its lowest set bit cleared** (RA GOP32 dyadic structure, verified
     against the cfg: 16→0, 24→16, 3→2, …).
   - The residual needs a reference frame, so **I-frames are excluded automatically**
     (`POC % IntraPeriod == 0`). IntraPeriod defaults to the smallest multiple of `gop_size` that is
     ≥ the frame rate, and can be overridden per sequence with `intra_period:` (it should match the encoding setup).
   - With I-frames excluded, the training set only contains inter blocks — consistent with the QT model
     **being deployed on inter slices only**.
   - ⚠️ **Consistency caveat**: offline collection currently uses the **source frame** as L0, whereas the
     encoder uses the **reconstructed reference frame** reco(L0) online. The source frame is used first
     to check whether the residual helps at all (weighted-F1); if it does and is to be embedded, L0
     should switch to reconstructed frames decoded from `binEncode202` to stay aligned.
9. **Output**: `<output_dir>/<name>.csv`, sorted by (POC, y, x).

## Output CSV Columns (21 columns, aligned with 128csv)

```
POC_ID,pos_x,pos_y,
mean,std,skew,kurt,
Sobel_mean,Sobel_std,Sobel_skew,Sobel_kurt,
Lap_mean,Lap_std,Lap_skew,Lap_kurt,
LeveneTest,KrusWTest,SobelLevene,SobelKrusW,LapLevene,LapKrusW
```

When `collect_residual` is enabled, the 4 `res_*` columns are appended after these.

- For raw YUV, `POC_ID` is the **frame index in display order**.
- Feature formulas strictly follow the project's original `w_cuProcess.cpp`: mean, std (population),
  skew = m3/σ³, kurt = m4/m2², Sobel = √(Gx²+Gy²), Levene (classic W over four quadrants),
  Kruskal-Wallis (simple ranks).
- **Laplacian** did not exist in the original `w_cuProcess`; here it uses the standard 4-neighbour 3×3
  kernel and takes the absolute response. Exact value-for-value parity with the legacy `128csv`
  generator would require matching that script's Laplacian kernel definition.

## Configuration (YAML)

Global: `output_dir`, `ctu_size`, `internal_bitdepth`, `default_ctu_per_yuv`, `default_sample_ratio`,
`random_seed`, `chroma_format`, `detect_useless_frames`, `blank_std_threshold`,
`duplicate_mad_threshold`, `detect_pixel_stride`.

Per sequence (`sequences:` list): required `path`/`width`/`height`/`bitdepth`;
optional `name`, `framerate`, `frames` (inferred from file size if absent), `chroma_format`,
`ctu_per_yuv` (fixed count), `sample_ratio` (ratio).

**Two sampling modes** (highest priority first):
1. per-sequence `sample_ratio` (> 0) → `ceil(estimated rows × ratio)`
2. per-sequence `ctu_per_yuv` → fixed count
3. global `default_sample_ratio` (> 0) → by ratio
4. global `default_ctu_per_yuv` → fixed count

Example: a global `default_sample_ratio: 0.03` samples about 3% of each YUV; to fix one sequence at
200 rows, add `ctu_per_yuv: 200` to that sequence (rule 2 takes precedence over rule 3).

See the example in `collector/config/config.yaml`.

## In-house Library API at a Glance

- `sf::computeCtuFeatures(const sf::Mat& block) -> CtuFeatures` (with `csvHeader()` / `toCsvFields()`)
- `sf::mean/stddev/skewness/kurtosis/sobel/laplacian/levene/kruskalWallis`
- `yg::YuvReader` (frame reading + bit-depth conversion), `yg::GrayFrame::extractBlock`,
  `yg::enumerateFullCtus`, `yg::detectUsableFrames`
- `myaml::parseFile` / `getInt/getDouble/getStr/getBool`
