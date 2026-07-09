# mttFeature —— 给 MTT 标签块配空域特征

把 `Data/labelMtt/` 里采样到的 MTT 划分块，配上空域特征，做成 MTT 训练集
`Data/TrainingData/TrainingMtt.csv`。

## 特征设计（块级只用 ②③④，①只在 CTU 级）

| 组 | 列 | 作用 |
|----|----|------|
| **② 方向性** | `dirEx`(mean&#124;Gx&#124;), `dirEy`(mean&#124;Gy&#124;), `dirRatio`=(Ex−Ey)/(Ex+Ey), `rowMeanVar`, `colMeanVar` | 管「横分 vs 竖分」：竖向边多→偏 BTV，横向边多→偏 BTH |
| **③ 划分代价代理** | `tbDiff`(上下半), `lrDiff`(左右半), `ttHDiff`, `ttVDiff`(1:2:1三分) | 直接模仿编码器在比的东西，判别力最强；差异大→该方向划分 |
| **④ 形状/上下文** | `qpValue`(按POC推), `aspectLog2`=log2(w/h), `areaLog2`, 以及标签里的 `qtDepth/mttDepth` | 形状强约束 VVC 划分；QP 越大越少分 |
| **① CTU 级复杂度**（上下文，非块级） | `ctuMean,ctuStd,ctuLapMean,ctuLapStd,ctuSobelMean,ctuSobelStd` | 块所在 128×128 CTU 的整体「忙不忙」；块自己只看方向/划分/形状 |

> 约定：块级**不放**①复杂度/能量（均值、std、Laplacian、Sobel 幅值），只在 CTU 级算一次作为上下文；
> 块级专注「往哪个方向、怎么切」。

### ⑤ 时域残差 `res_*`（`collect_residual: true` 时追加）

在**残差块** `D = 当前块 − L0同位块`（L0[0] 参考 POC = 去掉 POC 最低置位比特）上，重算 ②③ 的核心量再加矩：

| 列 | 说明 |
|----|------|
| `res_mean,res_std,res_skew,res_kurt` | 残差**能量/分布**——残差越大→越可能继续划分（深度信号） |
| `res_dirEx,res_dirEy,res_dirRatio,res_rowMeanVar,res_colMeanVar` | 残差**边的朝向**——决定横分还是竖分 |
| `res_tbDiff,res_lrDiff,res_ttHDiff,res_ttVDiff` | 残差**划分代理（最关键）**——残差能量落在哪条切线（1:1 还是 1:2:1）→ BT vs TT |

> 为什么 MTT 必须看残差：BT/TT 是 RDO 对**运动补偿残差**的决策，源像素里看不出来（纯空域 MTT 模型失败的根因）。
> QT 用残差**能量**（矩）就够；MTT 还要残差的**空间分布**（划分代理 `res_*Diff`）。
>
> **I 帧自动排除**：I 帧无 L0；按 `IntraPeriod`（显式 `intra_period` > 帧率推 ≥它的最小 `gop_size` 倍数 > `gop_size`）整行丢弃。
> **源/重建一致性**：离线用源帧 L0，编码器在线用重建帧 reco(L0)，存在量化噪声失配（沿用 QT 的「路线 a」）。

## 输出

`TrainingMtt.csv` = labelMtt 原列 + 上面的特征列：

```
sequence,POC_ID,pos_x,pos_y,w,h,qtDepth,mttDepth,splitMode,QP,
  qpValue,aspectLog2,areaLog2,dirEx,dirEy,dirRatio,rowMeanVar,colMeanVar,
  tbDiff,lrDiff,ttHDiff,ttVDiff,
  ctuMean,ctuStd,ctuLapMean,ctuLapStd,ctuSobelMean,ctuSobelStd
```

训练侧（Python）把 `sequence,POC_ID,pos_x,pos_y` 当 ID，`splitMode`（和 `mttDepth`）当标签，其余当特征。

## 工作流

对每个序列：从 collector 的 `config.yaml` 取 YUV 信息（path/宽高/位深）→ 开 `YuvReader`（统一转 10bit）；
逐条读 `labelMtt/<seq>_QP*.csv`，按 `(POC,x,y,w,h)` 裁出真实块算 ②③④，再算其所在 128 CTU 的 ①（每 CTU 缓存一次），
帧也按 POC 缓存。**块边界与 CTU 都来自真实划分，源像素取自源 YUV（编码器看到的输入）。**

## 用法

```bash
# 根目录一键
make mttFeature                 # → Data/TrainingData/TrainingMtt.csv

# 或在子项目里 / 手动指定
cd yuvFeatureExtract
make mtt-feature
build/mttFeature/mttFeatureBuilder collector/config/config.yaml \
    --labels-dir /路径/labelMtt --out /路径/TrainingMtt.csv
```

> 前提：collector `config.yaml` 里每个序列的 `path:` 指向真实存在的源 YUV（与采集器同一份配置，机器特定）。

## 复用的库

`YuvGrayLib`（读 YUV/转10bit）、`SpatialFeatureLib`（Sobel/Laplacian/矩，用于 CTU 级 ①）、
新增 `libs/MttFeatureLib`（②③④ 的纯函数）。
