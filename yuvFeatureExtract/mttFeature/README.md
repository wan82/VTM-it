# mttFeature — Spatial Features for MTT Label Blocks

Attaches spatial features to the MTT partition blocks sampled in `Data/labelMtt/`, producing the MTT
training set `Data/TrainingData/TrainingMtt.csv`.

## Feature Design (blocks use only ②③④; ① is CTU-level only)

| Group | Columns | Purpose |
|-------|---------|---------|
| **② Directionality** | `dirEx` (mean&#124;Gx&#124;), `dirEy` (mean&#124;Gy&#124;), `dirRatio` = (Ex−Ey)/(Ex+Ey), `rowMeanVar`, `colMeanVar` | Horizontal vs. vertical split: more vertical edges → leans BTV, more horizontal edges → leans BTH |
| **③ Split-cost proxies** | `tbDiff` (top/bottom halves), `lrDiff` (left/right halves), `ttHDiff`, `ttVDiff` (1:2:1 thirds) | Directly mimic what the encoder compares; the most discriminative group — a large difference → split in that direction |
| **④ Shape / context** | `qpValue` (derived from POC), `aspectLog2` = log2(w/h), `areaLog2`, plus `qtDepth/mttDepth` from the labels | Shape strongly constrains VVC partitioning; higher QP → fewer splits |
| **① CTU-level complexity** (context, not block-level) | `ctuMean,ctuStd,ctuLapMean,ctuLapStd,ctuSobelMean,ctuSobelStd` | How "busy" the enclosing 128×128 CTU is overall; the block itself only looks at direction / split / shape |

> Convention: block-level features **exclude** ① complexity/energy (mean, std, Laplacian, Sobel
> magnitude); those are computed once at CTU level as context. Block-level features focus on
> "which direction, and how to cut".

### ⑤ Temporal residual `res_*` (appended when `collect_residual: true`)

On the **residual block** `D = current block − co-located L0 block` (L0[0] reference POC = POC with its
lowest set bit cleared), the core ②③ quantities are recomputed, plus moments:

| Columns | Description |
|---------|-------------|
| `res_mean,res_std,res_skew,res_kurt` | Residual **energy / distribution** — larger residual → more likely to keep splitting (depth signal) |
| `res_dirEx,res_dirEy,res_dirRatio,res_rowMeanVar,res_colMeanVar` | Residual **edge orientation** — decides horizontal vs. vertical split |
| `res_tbDiff,res_lrDiff,res_ttHDiff,res_ttVDiff` | Residual **split proxies (the key group)** — which cut line the residual energy falls on (1:1 or 1:2:1) → BT vs. TT |

> Why MTT must look at the residual: BT/TT are RDO decisions on the **motion-compensated residual**,
> which cannot be seen in the source pixels (the root cause of the purely spatial MTT model's failure).
> QT gets by with residual **energy** (moments); MTT also needs the residual's **spatial distribution**
> (the `res_*Diff` split proxies).
>
> **I-frames are excluded automatically**: I-frames have no L0. Rows are dropped based on `IntraPeriod`
> (explicit `intra_period` > smallest multiple of `gop_size` ≥ frame rate > `gop_size`).
> **Source/reconstruction consistency**: offline uses the source frame as L0, while the encoder uses the
> reconstructed reco(L0) online, so there is a quantization-noise mismatch (same "route a" as for QT).

## Output

`TrainingMtt.csv` = the original labelMtt columns + the feature columns above:

```
sequence,POC_ID,pos_x,pos_y,w,h,qtDepth,mttDepth,splitMode,QP,
  qpValue,aspectLog2,areaLog2,dirEx,dirEy,dirRatio,rowMeanVar,colMeanVar,
  tbDiff,lrDiff,ttHDiff,ttVDiff,
  ctuMean,ctuStd,ctuLapMean,ctuLapStd,ctuSobelMean,ctuSobelStd
```

(followed by the `res_*` columns when the residual is enabled)

On the training side (Python), `sequence,POC_ID,pos_x,pos_y` are IDs, `splitMode` (and `mttDepth`) are
labels, and everything else is a feature.

## Workflow

For each sequence: take the YUV information (path / width / height / bit depth) from the collector's
`config.yaml` → open a `YuvReader` (normalized to 10-bit). Read `labelMtt/<seq>_QP*.csv` row by row,
crop the actual block by `(POC,x,y,w,h)` and compute ②③④, then compute ① for the enclosing 128 CTU
(cached once per CTU); frames are also cached by POC.
**Block boundaries and CTUs come from the real partitioning; source pixels come from the source YUV
(the input the encoder sees).**

## Usage

```bash
# One command from the repo root
make mttFeature                 # → Data/TrainingData/TrainingMtt.csv

# Or inside the sub-project / with explicit paths
cd yuvFeatureExtract
make mtt-feature
build/mttFeature/mttFeatureBuilder collector/config/config.yaml \
    --labels-dir /path/to/labelMtt --out /path/to/TrainingMtt.csv
```

> Prerequisite: every sequence's `path:` in the collector `config.yaml` must point to an existing
> source YUV (the same config as the collector; machine-specific).

## Reused Libraries

`YuvGrayLib` (YUV reading / 10-bit conversion), `SpatialFeatureLib` (Sobel / Laplacian / moments, used
for CTU-level ①), and the new `libs/MttFeatureLib` (pure functions for ②③④).
