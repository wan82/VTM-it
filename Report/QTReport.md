# QTReport —— 基于随机森林的 VVC 快速 CU 划分（特征→训练→嵌入 VTM）

本报告记录从训练数据到把模型嵌入 VTM 编码器的完整工作：QP-value 的引入、特征选择（包裹式前向选择）、
模型训练与选择、最佳模型保存与导出、以及在官方 VTM 20.2 编码器里用预测层级终止 QT 的实现与验证。

---

## 1. 目标与总体思路

对每个 128×128 的 CTU，用少量**空域特征 + 该 CTU 所在帧的实际 QP（QP-value）**预测它的
**QT 最大深度 `qt_max`（0–4）**，编码时让该 CTU 在预测层级**终止 QT 划分**，从而跳过更深 QT 的
RDO 搜索，以很小的 BD-rate 代价换取编码时间（ET）下降。

端到端数据流（均由根目录 `make` 调度）：

```
官方VTM解码 ─► labelCtu/<seq>.csv         (真实 qt_max 标签, 全CTU覆盖)
YUV ─►[yuvFeatureExtract 采集]─► tempYuvFeature/<seq>.csv   (空域特征)
        [yuvFeatureExtract 合并: 内连接 + 去NaN + 加QP-value] ─► TrainingData/Training.csv
        [modelTrain: 清洗→特征筛选→训练NB/DT/RF→GroupKFold(5)→选模型→保存bin] ─► ModelResults/
        [modelTrain: export] ─► qtForest.bin ─►[嵌入 VTM EncCu] 预测并终止QT
```

---

## 2. 训练数据 Training.csv（30 列）

`make merge`（C++ 合并器）把特征与标签按 `(POC_ID, pos_x, pos_y)` 内连接、去掉含 NaN 的行，并加入
按 VTM 推导的 QP-value，得到最终训练集，列结构为：

```
sequence, POC_ID, pos_x, pos_y,
22qt_max, 27qt_max, 32qt_max, 37qt_max,        # 4 个 QP 下的真实 QT 深度（标签）
22QP, 27QP, 32QP, 37QP,                        # 4 个 QP 下的实际 slice QP（QP-value）
mean, std, skew, kurt,                          # 18 个空域特征
Sobel_mean, Sobel_std, Sobel_skew, Sobel_kurt,
Lap_mean, Lap_std, Lap_skew, Lap_kurt,
LeveneTest, KrusWTest, SobelLevene, SobelKrusW, LapLevene, LapKrusW
```

### 2.1 QP-value 的推导（完全仿照 VTM RA, 默认 GOP=32）

`QP-value = baseQP + 偏移(baseQP, TID)`，其中 `TID = f(POC % 32)` 是时域层。偏移表取自 VTM RA 各
基础 QP 的时域层 QP 偏移，与从官方码流解码出的 `PocInfo` 完全一致（例：POC0→19/24/29/34，
POC1(TID5)→29/36/41/46）。该逻辑在 C++ 合并器与 VTM 推理模块中各实现一份，保证训练/推理一致。

| TID 来源 (gop=POC%32) | TID |
|---|---|
| 0 | 0 |
| 16 | 1 |
| 8, 24 | 2 |
| 4,12,20,28 | 3 |
| 2,6,…,30（偶非4倍数） | 4 |
| 奇数 | 5 |

---

## 3. 特征选择：包裹式前向选择（找甜蜜点）

`modelTrain/train.py` 的特征筛选分四步：① 去近常量；② 去高相关冗余（|corr|>0.95，剔除 `Lap_kurt`，
18→17）；③ 互信息排名；④ **包裹式前向选择**——按互信息从高到低逐个加入特征，每步用
**3 个模型 × 4 个 QP 的平均 Weighted-F1** 评估，取**性能达到峰值−容差（默认 0.003）的最小特征数**为甜蜜点。

实测（全量、默认设置）逐步曲线在 **k=5 见顶（0.4555），之后随特征增多反而下降**，甜蜜点 = **5**：

```
k=1 +Sobel_mean 0.4342   k=2 +Sobel_std 0.4449   k=3 +Lap_std 0.4435
k=4 +std        0.4474   k=5 +Lap_mean  0.4555 ← 甜蜜点
k=6..17 单调走低（多余特征在分组验证下过拟合）
```

**最终选用 5 个空域特征**：`Sobel_mean, Sobel_std, Lap_std, std, Lap_mean`。

### 3.1 QP-value 为必选特征（不参与 wrapper）

QP-value 列被排除在 wrapper 候选之外，**每个 QP 的模型固定额外加入它自己的 QP-value**
（22qt_max→22QP，…，37qt_max→37QP）。因此每个模型最终用 **5 个空域特征 + 1 个匹配 QP-value = 6 个特征**，
实际编码时每个 CTU 只需算这 6 个量，开销很小。加入匹配 QP-value 后各 QP 的 Weighted-F1 均有提升。

---

## 4. 模型训练与选择

对 22/27/32/37 各训一套 **朴素贝叶斯 / 决策树 / 随机森林**（多分类 qt_max 0–4），用
**GroupKFold(5) 按序列分组**评估（同一视频整体只进训练或测试，反映对“没见过的视频”的泛化）。

实测折外指标（Weighted-F1）：

| QP | NaiveBayes | DecisionTree | RandomForest |
|----|-----------|--------------|--------------|
| 22qt_max | 0.3229 | 0.3255 | **0.3616** |
| 27qt_max | 0.4073 | 0.4037 | **0.4401** |
| 32qt_max | 0.5008 | 0.4793 | **0.5319** |
| 37qt_max | 0.6028 | 0.5698 | **0.6202** |

### 4.1 模型选择在“算法层面”进行

比较每种算法在 4 个 QP 上的**平均 Weighted-F1**，平均最高者作为**唯一最终算法**，再用它在每个 QP 各训
一个模型保存（保证部署只用一种算法）：

```
NaiveBayes = 0.4585   DecisionTree = 0.4446   RandomForest = 0.4884  ← 最终选用
```

QP 越大准确率越高（高 QP 多为不划分，更易）；多分类+分组验证较严格，Macro-F1 低于准确率属正常。

---

## 5. 训练时直接产出 C++ 模型（无转换步骤）

每个 QP 用最终算法在**全量数据**上重训，**训练时直接序列化**成一个 C++(VTM) 可读的森林文件
`ModelResults/qtForest.bin`（小端二进制，4 个 QP 合一），另存每 QP 的 `models/<qp>.json`
（仅供人看：特征顺序/类别/指标）。**不经 joblib、不需要任何转换脚本**。

```
"QTF1", int32 nModels;
每模型: baseQp; nFeatures + 特征名; nClasses + 类别标签; nTrees;
        每树: nNodes; 每节点(left,right,feat:int32, threshold:double); 每节点类别概率(double[nClasses])
```

C++ 端按各树遍历到叶子、累加类别概率、取 argmax —— 与 sklearn `RandomForestClassifier.predict` 一致。

> **模型规模在训练时自动限制**：默认 `--rf-min-leaf 50`（同时约束 RF 与决策树），
> `qtForest.bin` 约几 MB、推理快、精度基本不变（早期 `min_leaf=1` 的满树森林会达数百 MB，已不再使用）。

---

## 6. 嵌入 VTM（官方 20.2，全部改动带 `#QTPRED` 标注）

| 文件 | 改动 |
|------|------|
| `source/Lib/EncoderLib/QtForestPredictor.h`  | **新增**：预测器接口（单例，模型路径取自环境变量 `QT_FOREST_MODEL`） |
| `source/Lib/EncoderLib/QtForestPredictor.cpp` | **新增**：模型加载 + 空域特征/QP-value 计算 + 森林推理 |
| `source/Lib/EncoderLib/EncCu.h`  | 加成员 `m_qtPredActive / m_qtPredDepth` |
| `source/Lib/EncoderLib/EncCu.cpp` | `#include`；CTU 根算预测；`isModeSplit` 处到达预测深度跳过 QT |

EncoderLib 的 CMakeLists 用 `file(GLOB *.cpp)`，新增文件自动编译，无需改构建脚本。

**运行逻辑**：
1. **CTU 根**（`currQtDepth==0 && currMtDepth==0`、inter slice、完整 128×128、非边界、模型可用）：
   取原始亮度并缩放到 10bit（与训练一致），算 5 个空域特征 + `getQpValue(POC, baseQP)`，预测 `qt_max`，
   存入 `m_qtPredDepth`，置 `m_qtPredActive=true`。
2. **划分递归**：`isModeSplit` 处若 `m_qtPredActive && 当前是QT划分 && currQtDepth >= m_qtPredDepth`，
   **跳过该 QT 划分测试**（块模式与 BT/TT 仍照常测试，`bestCS` 一定有有效结果，不会崩）。
3. 边界 CTU / intra slice / 未设环境变量 / 模型加载失败 → **完全不介入**，行为同原版。

---

## 7. 验证

- **C++ 推理 vs sklearn `predict`**：随机特征向量逐样本对照，**800/800 完全一致**（涵盖 4 个 QP 模型，
  含非纯叶子的概率平均情形）—— 证明二进制格式、加载、森林遍历、argmax 全部正确。
- **语法检查**：`EncCu.cpp` 在 VTM 头文件环境下逐文件 `-fsyntax-only`，**零自身错误**（括号配对、
  `getOrigBuf/getBitDepth/CU_QUAD_SPLIT/getPartSplit` 等 API 用法正确）；`QtForestPredictor.cpp`
  独立编译干净。
- 受限于本环境无法整体编译 VTM，**最终需在本机 `cmake && make` 完整编译并实测一条短序列**确认
  ET 下降与 BD-rate 在可接受范围。

---

## 8. 使用步骤

```bash
# 在 VTM-it 根目录（总控 Makefile，无需 cd）
make data                      # 采集特征 + 合并标签(含QP-value) → Training.csv
make train                     # 训练 + 报告 + 直接产出 qtForest.bin（默认已限制模型规模）

# 构建 VTM 并编码（RA）
cd VVCSoftware_VTM-VTM-20.2 && mkdir -p build && cd build && cmake .. && make -j EncoderApp
export QT_FOREST_MODEL="$(git rev-parse --show-toplevel)/Data/ModelResults/qtForest.bin"
./bin/EncoderAppStatic -c encoder_randomaccess_vtm.cfg -c <seq>.cfg -q 32 ...
```

启动会打印 `[QTPRED] 已加载 4 个 QP 模型 ...`；不设 `QT_FOREST_MODEL` 即原版行为。

---

## 9. 注意事项 / 局限

- 仅对 **RA、GOP=32、基础 QP ∈ {22,27,32,37}** 有效（QP-value 偏移表的前提）；其它 QP 无对应模型自动不介入。
- 这是“预测层级强制终止 QT”的**有损加速**：BD-rate 会有少量上升换 ET 下降，幅度取决于模型精度。
- 特征是在原始亮度（缩到 10bit）上算的，与训练侧严格一致；8bit 序列会自动左移到 10bit。
- 模型选择、特征选择、QP-value 均可在 `train.py` 参数中调整（`--wrapper-tol`、`--rf-min-leaf`、
  `--corr-thr`、`--top-k` 等），无需改代码。

---

## 附：本轮新增/改动文件清单

**VTM（official 20.2，`#QTPRED` 标注）**
- 新增 `source/Lib/EncoderLib/QtForestPredictor.h`、`.cpp`
- 改 `source/Lib/EncoderLib/EncCu.h`、`EncCu.cpp`
- 新增说明 `VVCSoftware_VTM-VTM-20.2/QTPRED.md`

**modelTrain**
- 改 `train.py`：QP-value 作匹配特征、包裹式特征选择、算法层面选模型、**训练时直接产出
  `qtForest.bin`**、`--rf-min-leaf`（默认 50，限制 RF/决策树规模）
- `export_qtForest.py` 已弃用（导出并入训练，无需单独转换）

**yuvFeatureExtract**
- 改 `merger`：合并时加 4 列 QP-value

**根目录**
- 总控 `Makefile`（`extract/merge/data/train/export/all`），`.gitignore`，本报告 `QTReport.md`
