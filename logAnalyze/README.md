# logAnalyze — BD-rate / Time-Reduction Analysis (Python)

Compares the logs of **fastTest (QT early termination)** against the **original encoder** and
computes the BD-rate and the encoding-time reduction (Time-Reduction).

## Method (POC-aligned)

- Parse `bits / Y-PSNR / ET (seconds)` for every POC from each log.
- fastTest only encodes POC 0..N (33 frames by default), while the original logs cover the whole
  sequence → **only POCs present on both sides are used**, so both sides are compared on the same frames.
- For each sequence and QP, aggregate over the aligned POCs:
  - `bitrate (kbps) = (Σbits / frames) × frameRate / 1000` (frameRate taken from `perSequenceCfg/<Seq>.cfg`)
  - `Y-PSNR = mean over the aligned POCs`
  - `ET = sum of ET over the aligned POCs`
- 4 QPs → **BD-rate** (original as anchor, fast as test; positive = higher bitrate = worse).
  Per-QP **Time-Reduction = (ET_fast − ET_orig) / ET_orig** (**negative = less encoding time = faster**),
  averaged over the QPs.

## Usage

```bash
cd logAnalyze
make            # reuses the shared venv at the repo root (VTM-it/.venv), runs the analysis and writes the report
```

Paths can be overridden:

```bash
make FAST=/path/Log/FastTest ORIG=/path/LogOriginalEnc OUT=/path
```

Defaults:
- fast logs: `Data/Log/FastTest` (produced by `make fastTest`)
- original logs: `Data/LogOriginalEnc`
- frameRate: `perSequenceCfg/<Seq>.cfg`
- output: `Report/Analysis/` (`analysis.md` + `bdSummary.csv`)

## Input / Output Conventions

- Logs are named `<Seq>_QP<qp>.log` (same for fastTest and original) and are paired automatically by sequence name.
- Outputs `analysis.md` (summary + per-QP detail tables) and `bdSummary.csv` (sequence, BD-rate %, TR %).

## Notes

- BD-rate uses the standard Bjøntegaard method (cubic polynomial fit of log10(rate) against PSNR,
  integrated over the common PSNR range).
- frameRate only affects the absolute kbps shown; BD-rate is insensitive to constant scaling of the
  bitrate (the result is unchanged).
- All Python sub-projects share `VTM-it/.venv`; this one only additionally needs numpy.
