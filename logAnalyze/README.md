# analyze —— BD-rate / Time-Reduction 分析（Python）

对比 **fastTest（QT 早停）** 与 **原始编码** 的日志，算出 BD-rate 与编码时间下降（Time-Reduction）。

## 做法（逐 POC 对齐）

- 从每个 log 解析每个 POC 的 `bits / Y-PSNR / ET(秒)`；
- fastTest 只编了 POC 0..N（默认 33 帧），原始日志编了整段序列 → **只取两边都有的 POC 对齐**，
  保证比的是同一批帧；
- 每序列每 QP 在对齐的 POC 上聚合：
  - `bitrate(kbps) = (Σbits / 帧数) × frameRate / 1000`（frameRate 取自 `perSequenceCfg/<Seq>.cfg`）
  - `Y-PSNR = 对齐 POC 的平均`
  - `ET = 对齐 POC 的 ET 之和`
- 4 个 QP → **BD-rate**（原始为锚、fast 为测试，正值=码率升高=变差）；
  每 QP 的 **Time-Reduction = (ET快 − ET原) / ET原**（**负值=编码时间下降=更快**），取平均。

## 运行

```bash
cd analyze
make            # 复用根目录共享 venv（VTM-it/.venv），分析后出报告
```

路径可覆盖：

```bash
make FAST=/路径/Log/FastTest ORIG=/路径/LogOriginalEnc OUT=/路径
```

默认：
- fast 日志：`Data/Log/FastTest`（由 `make fastTest` 生成）
- 原始日志：`Data/LogOriginalEnc`
- frameRate：`perSequenceCfg/<Seq>.cfg`
- 输出：`Data/Log/FastTest/`（`analysis.md` + `bdSummary.csv`）

## 输入/输出约定

- 日志命名 `<Seq>_QP<qp>.log`（fastTest 与原始一致），按序列名自动配对。
- 输出 `analysis.md`（汇总 + 逐 QP 明细表）与 `bdSummary.csv`（序列, BD-rate%, TR%）。

## 备注

- BD-rate 用标准 Bjøntegaard（log10(rate) 对 PSNR 做三次多项式拟合，在公共 PSNR 区间积分）。
- frameRate 只影响绝对 kbps 的显示；BD-rate 对码率的常数倍缩放不敏感（结果不变）。
- 所有 Python 子项目共用 `VTM-it/.venv`，本项目只额外需要 numpy。
