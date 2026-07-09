# modelTrain —— 训练与评估（Python，Makefile + venv）

读取 `TrainingData/Training.csv`，做**数据清洗 + 特征筛选**，按 QP 分别训练
**朴素贝叶斯 / 决策树 / 随机森林**（多分类 qt_max 0–4），用 **GroupKFold(5, 按序列分组)**
评估，生成**混淆矩阵图**与**完整 Markdown 报告**。

## 流程

1. **数据清洗**：±Inf 视作缺失并连同 NaN 删除（Training.csv 已基本干净，此处兜底）；
   标签取整为类别；统计重复行（不删）。
2. **特征筛选**：① 去近常量特征；② 去高相关冗余（|corr| > 阈值，默认 0.95）；
   ③ 算每个 QP 的互信息做判别力排名；④ **包裹式前向选择**：按互信息排名从 0 个开始逐个加特征，
   每步用 3 个模型 × 4 个 QP 的平均 Weighted-F1 评估，取**性能达到峰值−容差**的最小特征数为
   “甜蜜点”，只保留这些特征（剩下的不再加）。`--no-wrapper` 关闭、`--top-k` 直接取前 K。
3. **训练 + 评估**：对 22/27/32/37 各一套，每套训 NB/DT/RF；
   **GroupKFold(5)** 以 `sequence` 分组（同一视频整体只进训练或测试，反映对“没见过的视频”的泛化）。
   取折外预测算准确率 / Macro-F1 / Weighted-F1，并画混淆矩阵。
4. **产出**：`report.md`（完整报告，含 wrapper 曲线与甜蜜点）、`metrics.csv`、
   `figures/cm_<QP>_<model>.png`（12 张）、`figures/featureWrapper.png`（特征数-性能曲线）。

## 运行

```bash
cd modelTrain
make            # 首次自动建 venv 装 sklearn 等；找到 Training.csv 则训练并出报告
```

参数可覆盖：

```bash
make CORR=0.9 TOPK=10                 # 高相关阈值 / 只取前10个特征
make INPUT=/path/Training.csv OUT=/path
# 直接调脚本（更多参数）：
./run.sh --rf-estimators 200 --max-rows 20000 --folds 5
```

- 其它目标：`make setup` / `make clean` / `make distclean`。
- `--max-rows` 随机抽样若干行，便于快速试跑；`--rf-estimators` 控制随机森林树数（默认 100）。
- wrapper 相关：`--wrapper-tol`（甜蜜点容差，默认 0.003，越小留的特征越多；设 0 取性能峰值点）、
  `--wrapper-rf` / `--wrapper-folds` / `--wrapper-sample`（搜索时的轻量设置，加速）、`--no-wrapper`（关闭）。

默认路径（相对仓库根，自动按脚本位置推导，换机器无需改）：
- 输入：`<仓库根>/Data/TrainingData/Training.csv`
- 模型：`<仓库根>/Data/ModelResults`；报告：`<仓库根>/Report/ModelTraining`

## 输出

```
Data/ModelResults/
├── report.md                       完整报告（数据概览/清洗/特征筛选/逐QP模型对比/汇总/最终模型）
├── metrics.csv                     指标表（QP×模型：acc / cv均值±std / macro-F1 / weighted-F1）
├── figures/cm_<QP>_<model>.png     12 张混淆矩阵（行=真实，列=预测，含计数与行归一化%）
├── figures/featureWrapper.png     特征数-性能曲线（标出甜蜜点）
├── qtForest.bin                   ★ VTM 可读的森林模型（4个QP合一，训练时直接产出）
└── models/<qp>.json                每QP元数据：模型/特征顺序/类别/CV指标（仅供人看）
```

> 训练时**直接序列化**出 `qtForest.bin`（不经 joblib、无需转换）。默认 `RFMINLEAF=50` 已限制
> RF/决策树规模，模型约几 MB。嵌入 VTM 时设 `QT_FOREST_MODEL` 指向该 bin；特征顺序见 `<qp>.json`。

## 说明

- 多分类 qt_max(0–4) 类别**不均衡**（高 QP 时大量为 0），Macro-F1 会明显低于准确率，属正常；
  报告里同时给了准确率、Macro-F1、Weighted-F1 以便全面判断。
- **按序列分组**的交叉验证比随机分层更严格、更接近真实泛化；想看随机分层的乐观上界可另行对比。
- **模型选择在算法层面**：比较三种算法各自在 4 个 QP 上的平均 Weighted-F1，平均最高者作为最终算法，
  再用它在每个 QP 各训一个模型保存（保证部署只用一种算法）。
- Training.csv 里的 `22QP/27QP/32QP/37QP`（QP-value）默认**不进特征集**，以保持 18 个空域特征不变；
  想把匹配的 QP-value 当特征用，告知即可。
- 想改成二分类（分/不分）或加 QP 作统一特征，告知即可，pipeline 很容易扩展。

## MTT 训练（`trainMtt.py`，与 QT 分开的独立模型）

QT 用 `train.py`，**MTT 单独**用 `trainMtt.py`，互不影响。读 `Data/TrainingData/TrainingMtt.csv`
（由 `make mttFeature` 生成），预测每个 MTT 节点的 **splitMode 5 分类**：

- 类别码：`NO=0, BTH=1, BTV=2, TTH=3, TTV=4`（C++ 侧据此映射回 PartSplit）。
- **按基础 QP 分 4 个模型**（22/27/32/37），嵌入时按 baseQP 选，与 QT 一致。
- 类别**严重不平衡**（NO≈64%），统一用 **balanced 样本权重**，主指标看 **macro-F1 + 逐类 F1 + 混淆矩阵**，
  不看 accuracy（全猜 NO 就有 64%）。
- 按序列 GroupKFold(5)；算法级选择只在 **决策树/随机森林** 里挑（4 QP 平均 macro-F1 最高），
  朴素贝叶斯仅作对照（森林 bin 只存树模型）。
- 训练时直接导出 `Data/ModelResults/mttForest.bin`（格式 `"QTF1"`，与 `qtForest.bin` 同一套，
  C++ `QtForestPredictor` 可直接读；已验证 bin 遍历与 sklearn predict 逐样本一致）。

```bash
make train-mtt          # 或根目录 make trainMtt
#  → Data/ModelResults/mttForest.bin + mttModels/QP<qp>.json
#  → Report/MttTraining/{report.md, metrics.csv, figures/cm_QP<qp>_<model>.png}
```

特征用 `mttFeature` 那 22 列（②方向性 + ③划分代价代理 + ④形状/QP + CTU 级①上下文）；
默认全用并在报告里给出 MI 排名（不做 wrapper，特征集已是精选的小集合）。
