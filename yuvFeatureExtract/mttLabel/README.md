# mttLabel — Sampled Export of MTT Partition Labels

Decodes the **MTT (BT/TT) partitions** from the RA bitstreams in `Data/binEncode202/` and turns them
into training labels. Only a small subset is sampled (random POC × random block), to avoid exporting
millions of blocks.

## Why Go Through the VTM Decoder

The `.bin` files are CABAC entropy-coded, and **only the VTM decoder can recover the partition tree**
(this is also how labelCtu was produced). The work is therefore split as follows:

- **Entropy decoding**: done by VTM's `DecoderApp`, with a sampling dump hook
  `source/Lib/DecoderLib/MttLabelDumper.*` (tagged `#MTTDUMP`, **off by default, zero overhead**).
- **Orchestration + sampling plan + writing to disk**: this module's `mttLabelBuilder` (C++) plus the
  library `libs/MttSampleLib`.

> Decoding the full stream is unavoidable (blocks depend on each other, so a single block cannot be
> decoded on its own), but it is fast (the bins are only a few MB). What needs to be saved is data
> volume — so the dump step writes only a fraction of the nodes.

## Output

One CSV per `(sequence, QP)`: `Data/labelMtt/<seq>_QP<qp>.csv`, with columns:

```
sequence, POC_ID, pos_x, pos_y, w, h, qtDepth, mttDepth, splitMode, QP
```

- One row = one **MTT-stage node** (a QT leaf and every BT/TT decision node below it, i.e. non-QUAD nodes).
- `splitMode ∈ {NO, BTH, BTV, TTH, TTV}`: the block's own split decision
  (NO = no further split; BTH/BTV = binary horizontal/vertical; TTH/TTV = ternary horizontal/vertical).
- Luma only; implicit forced splits at picture boundaries are skipped (they are not real decisions).
- **MTT partitions differ per QP**, so files are split by (sequence, QP) and carry a QP column, unlike
  labelCtu, which puts all 4 QPs in one row.

This is a **superset**: it supports training an "MTT depth cap" (using mttDepth) as well as a
"per-node 5-way mode" (using splitMode), so changing the prediction target later does not require
decoding again.

## Usage

```bash
# One command from the repo root (builds the DecoderApp with #MTTDUMP + this orchestrator, then decodes and samples)
make mttLabel

# Or inside this sub-project:
cd yuvFeatureExtract
make mtt-label                       # same as above
# Specify the decoder / directories manually:
build/mttLabel/mttLabelBuilder mttLabel/config/mtt.yaml \
    --decoder /path/to/DecoderAppStatic --bin-dir /path/to/binEncode202 --out-dir /path/to/labelMtt
build/mttLabel/mttLabelBuilder mttLabel/config/mtt.yaml --dry-run   # only print the commands that would run
```

## Sampling Parameters (edit `config/mtt.yaml`)

| Key | Meaning | Default |
|-----|---------|---------|
| `poc_count`   | Number of random POCs sampled per sequence (≤ 0 = all POCs) | 8 |
| `block_ratio` | Probability (0–1) that each MTT node is written out | 0.03 |
| `random_seed` | Sampling seed (reproducible results) | 42 |
| `qps`         | Base QPs to decode for each sequence | 22,27,32,37 |
| `bin_dir` / `out_dir` / `bin_suffix` | Input bitstream directory / output directory / bin file-name suffix | see yaml |

Sampling is done on the decoder side by hashing `(seed,poc,x,y,w,h)`, so it is deterministic and
independent of decoding order; all QPs of the same sequence use the same set of POCs for easy comparison.

## Decoder Environment Variables (`#MTTDUMP`, set automatically by the orchestrator)

`MTT_DUMP_OUT` (output path; unset → disabled), `MTT_DUMP_SEQ`, `MTT_DUMP_QP`,
`MTT_DUMP_POCS` (comma-separated POC list, empty = all), `MTT_DUMP_BLOCK_RATIO`, `MTT_DUMP_SEED`.
**Without `MTT_DUMP_OUT`, the decoder behaves exactly like the original.**

## Next Steps

The sampling scheme in `libs/MttSampleLib` is reused downstream: the sampled (variable-size,
rectangular) MTT blocks get YUV spatial features (including directional features) extracted and are
fed into the MTT prediction model — see [`../mttFeature/README.md`](../mttFeature/README.md).
