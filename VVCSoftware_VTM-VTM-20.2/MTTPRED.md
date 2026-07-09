# MTTPRED —— 把 MTT 划分模型嵌入 VTM 做方向剪枝

继 QT 早停（`QTPRED.md`）之后，把训练好的 **MTT 划分模式模型** 也嵌进官方 VTM 20.2：
编码器在每个「能做 BT/TT 的节点」（= MTT 决策点）提取 22 个特征，预测该块的 **splitMode**
（NO/BTH/BTV/TTH/TTV），置信度够时**剪掉与预测相悖的 BT/TT 模式**，从而跳过这些方向的 RDO、省时间。
所有改动用 `#MTTPRED` 标注，可一键搜索。与 QT 早停**相互独立**、可分别开关。

## 改了哪些文件（均带 `#MTTPRED` 标注）

| 文件 | 改动 |
|------|------|
| `source/Lib/EncoderLib/MttForestPredictor.h`  | **新增**：MTT 预测器接口 |
| `source/Lib/EncoderLib/MttForestPredictor.cpp` | **新增**：模型加载 + ②③④①特征 + 森林推理 |
| `source/Lib/EncoderLib/EncCu.h`  | 加成员 `m_mttPredActive/m_mttPredCode` + CTU 上下文缓存 |
| `source/Lib/EncoderLib/EncCu.cpp` | include；MTT 节点算预测；模式循环里按预测方向剪 BT/TT |

> EncoderLib 的 CMakeLists 用 `file(GLOB *.cpp)`，新增两文件**自动参与编译**，无需改构建脚本。

## 工作原理

1. **何时预测**：在 `xCompressCU` 里，当 `partitioner.canSplit(...)` 表明该节点**有 BT/TT 候选**
   （inter slice、亮度、块不越界、模型可用）时，对当前块预测一次。这正好对应训练采样的「MTT 决策节点」分布。
2. **特征（22 个，与训练严格一致）**：取原始亮度块统一缩放到 10bit，算
   ②方向性(`dirEx/dirEy/dirRatio/rowMeanVar/colMeanVar`) + ③划分代价代理(`tbDiff/lrDiff/ttHDiff/ttVDiff`)
   + ④形状/QP(`w/h/qtDepth/mttDepth/qpValue/aspectLog2/areaLog2`) + ①该块所在 128 CTU 的复杂度上下文
   (`ctuMean/ctuStd/ctuLapMean/ctuLapStd/ctuSobelMean/ctuSobelStd`，按 CTU 缓存一次)。
   按基础 QP(22/27/32/37) 选对应森林，得到 splitMode 预测 + 置信度。
3. **置信度门控**：只有 `置信度 ≥ 阈值`（默认 0.7，`MTT_FOREST_CONF`）才启用剪枝；不确信就全搜。
4. **方向剪枝（默认 orient，保守）**：
   - 预测**竖**（BTV/TTV）→ 跳过**横**向 BT/TT（`CU_HORZ_SPLIT/CU_TRIH_SPLIT`）；
   - 预测**横**（BTH/TTH）→ 跳过**竖**向 BT/TT；
   - 预测 **NO（不分）** → 跳过**全部** BT/TT（`MTT_FOREST_PRUNE_NO=0` 可关）；
   - **永不剪 QUAD、不划分、块模式** —— QT 由 QtForestPredictor 负责，这里只动 BT/TT。
   - `MTT_FOREST_PRUNE=exact` 改为「只留预测的那一种 BTT」，更激进、更省时但风险更高。

> C++ 侧 22 个特征与训练库(`MttFeatureLib`+`SpatialFeatureLib`)**逐元素完全一致**（200 随机块最大差 0），
> 森林遍历与 sklearn `predict` **逐样本一致**（合成模型 3200/3200）。

## 用法

```bash
# 1) 训练 MTT 模型（产出 mttForest.bin）
cd /Users/w/Documents/Dev/VTM-it
make mttLabel      # 解码 binEncode202 采样 MTT 标签
make mttFeature    # 配 ②③④① 特征 → TrainingMtt.csv
make trainMtt      # → Data/ModelResults/mttForest.bin

# 2) 构建 VTM（同 QT）
cd VVCSoftware_VTM-VTM-20.2 && mkdir -p build && cd build && cmake .. && make -j EncoderApp

# 3) 编码时用环境变量指定 MTT 模型（可与 QT 模型同时开）
export MTT_FOREST_MODEL=/Users/w/Documents/Dev/VTM-it/Data/ModelResults/mttForest.bin
export MTT_FOREST_CONF=0.7          # 可选：置信度阈值（默认 0.7，越高越保守）
export MTT_FOREST_PRUNE_NO=1        # 可选：预测NO时是否跳过全部BTT（默认1）
export MTT_FOREST_PRUNE=orient      # 可选：orient(默认,剪相反方向) / exact(只留预测那种)
# 也可同时设 QT_FOREST_MODEL 一起跑（QT早停 + MTT剪枝）
./bin/EncoderAppStatic -c .../encoder_randomaccess_vtm.cfg -c .../<seq>.cfg -q 32 ...
```

启动时打印 `[MTTPRED] 已加载 4 个 QP 模型 ...`；**不设 `MTT_FOREST_MODEL` 就是原版行为**。

## 重要前提 / 注意

- **只对 RA、GOP=32、基础 QP ∈ {22,27,32,37}** 有效（QP-value 与训练条件烘焙在一起）。
- 是「按预测剪 BT/TT 方向」的有损加速：以少量 BD-rate 换 ET 下降，幅度取决于模型精度；置信度门控用于保 BD-rate。
- 与 QT 早停正交：可单开 QT、单开 MTT、或两者同开。
- 本仓库环境无法整体编译 VTM，C++ 改动已做特征/推理一致性验证与逐文件核对，请在本机 `cmake && make`
  实测一条短序列确认（可临时只设 `MTT_FOREST_MODEL` 与原版对照 BD-rate / 时间）。
