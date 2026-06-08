# VTM-it —— VVC/VTM 快速 CU 划分研究工作区

围绕 H.266/VVC 参考软件 **VTM** 的快速 CU 划分（fast partitioning）研究：
从原始 YUV 提取 CTU 的空域特征，配上从官方 VTM 解码得到的真实 QT 划分深度作标签，
合成训练数据，用于训练「按特征预测 CTU 是否/如何划分」的模型，从而在编码时跳过部分
RDO 搜索、以很小的 BD-rate 代价换取编码时间（ET）的下降。

## 目录结构

```
VTM-it/
├── Makefile                   总控：在根目录一站式调度两个子项目（无需 cd）
├── VVCSoftware_VTM-VTM-20.2/   官方 VTM 20.2 参考软件（已加 arm64 构建支持）
├── perSequenceCfg/           VTM 各测试序列的 .cfg（分辨率/位深/帧率/帧数等权威参数）
├── Data/                       数据中心
│   ├── binEncode202/            VTM-20.2 编出的 RA 码流(<seq>_RA_str<qp>.bin)，解码出 QT/MTT 标签
│   ├── labelCtu/                标签：每序列一个 CSV，逐 CTU 的 4 个 QP 下 QT 最大深度(qt_max)
│   ├── labelMtt/                标签：采样的 MTT 划分(每 序列×QP 一个 CSV，逐节点 w/h/mttDepth/splitMode)
│   ├── temp/tempYuvFeature/     中间件：采集器输出的逐 CTU 空域特征(每序列一个 CSV)
│   ├── temp/trainMerged/         中间件：每序列合并表（特征+标签）
│   ├── TrainingData/Training.csv 最终训练集（只此一个文件）
│   ├── Log/FastTest/             fastTest 编码日志
│   ├── LogOriginalEnc/         原版编码日志（对照基线）
│   └── ModelResults/             机器产物：qtForest.bin + models/<QP>.json（不放报告）
├── Report/                      所有报告集中地（数据归 Data，报告归 Report）
│   ├── ModelTraining/             训练报告：report.md + metrics.csv + figures/
│   ├── Analysis/                  日志分析报告：analysis.md + bdSummary.csv
│   ├── QTReport.md                项目报告：整体 pipeline
│   └── QTReport_ConfGating.md     项目报告：置信度门控 + 双向早停 方法与对比
├── yuvFeatureExtract/          子项目①(C++)：YUV→特征，并把特征×标签合并成 Training.csv
├── modelTrain/                 子项目②(Python)：训练模型 + 出报告 + 产出 qtForest.bin
├── logAnalyze/                 子项目③(Python)：BD-rate / Time-Reduction 日志分析
├── .venv/                      所有 Python 子项目共用的虚拟环境（运行后生成）
└── oldFashion/                 历史归档：早期内嵌原型 + 复盘文档 + 退役的 Python 合并器
```

> **数据与报告分离**：`Data/` 只放数据/日志/模型 bin 等机器产物；`Report/` 集中放所有人读的报告。
> 特征提取与标签合并都由 C++ 完成；两个 Python 子项目（modelTrain / logAnalyze）**共用一个
> 虚拟环境 `VTM-it/.venv`**，各自按需安装依赖。

## 命名约定（文件名 vs 列名）

本仓库刻意把**两种命名**分开，两者规则不同、不要混淆：

- **文件名 / 目录名 → 驼峰命名法（camelCase）。** 仓库自有的所有文件夹、源码、脚本、生成产物
  都用驼峰，例如 `yuvFeatureExtract/`、`labelCtu/`、`tempYuvFeature/`、`perSequenceCfg/`、
  `qtForest.bin`、`bdSummary.csv`、`featureWrapper.png`、`exportQtForest.py`，C++ 二进制
  `yuvFeatureCollector` / `yuvLabelMerger`。**例外**：`VVCSoftware_VTM-VTM-20.2/`（官方 VTM 原名，
  不改）与 `oldFashion/`（历史归档，不改）。

- **CSV 列名 → 保持数据 schema 原样，不驼峰化。** `Training.csv` 的列名是训练/推理双方约定的
  数据契约，C++ 采集器、合并器与 Python 训练侧都按这些字面量对齐，改了会两边脱节。因此保留：
  `sequence, POC_ID, pos_x, pos_y`、4 个标签列 `22qt_max/27qt_max/32qt_max/37qt_max`、
  4 个 QP-value 列 `22QP/27QP/32QP/37QP`，以及 18 个空域特征列（如 `Sobel_mean`、`Lap_std`）。

- **当列名出现在生成文件名里时，列名优先、原样保留。** 少数产物的文件名内嵌了列名 token，
  仍沿用列名写法而非驼峰，以便和 CSV 列一一对应：
  - 每 QP 模型元数据：`Data/ModelResults/models/22qt_max.json`（…27/32/37）
  - 混淆矩阵图：`Report/ModelTraining/figures/cm_22qt_max_RandomForest.png` 等
    （命名模板为 `cm_<标签列名>_<模型名>.png`）

  > 即「文件名外壳用驼峰，内嵌的列名 token 不动」。如果将来要连列名一起改，需同时改动
  > C++ 采集/合并器、`train.py` 的 `LABEL_COLS/QP_VALUE_COLS`、以及 VTM 侧
  > `QtForestPredictor` 的特征名匹配——属于跨语言的数据契约变更，不要只改单边。

## 端到端数据流

```
官方 VTM 20.2 解码  ──►  Data/labelCtu/          (真实 QT 深度标签, 全 CTU 覆盖)
                                                     │
YUV 原始序列 ─►[yuvFeatureExtract: 采集]─► Data/temp/tempYuvFeature/  (采样 CTU 的空域特征)
                                                     │
                       [yuvFeatureExtract: 合并] 按 (POC_ID,pos_x,pos_y) 内连接 + 去NaN
                                                     ▼
                                      Data/TrainingData/Training.csv
                                                     │
                              [modelTrain] 清洗→特征筛选→训练NB/DT/RF→GroupKFold(5)
                                                     ▼
                       模型bin → Data/ModelResults/ ；报告/混淆矩阵 → Report/ModelTraining/
```

## 快速开始

根目录有一个**总控 Makefile**，无需 cd 即可控制两个子项目：

```bash
# 在 VTM-it 根目录直接：
make help        # 查看所有命令
make extract     # 提取：YUV → 特征中间件(tempYuvFeature)
make merge       # 汇总：特征 × labelCtu → Training.csv（含 QP-value）
make data        # = extract + merge
make train       # 训练→模型bin→Data/ModelResults；报告→Report/ModelTraining
make all         # data + train 全流程
make clean       # 清理两个子项目的 build / venv
```

嵌入 VTM 后做端到端验证：

```bash
make fastTest    # 编译VTM + 对BQSquare编码33帧(QP22/27/32/37)，日志→Data/Log/FastTest
make analyze     # 对比 Data/LogOriginalEnc，算 BD-rate / Time-Reduction → Report/Analysis
```

命令行参数会**自动透传**给子项目，例如：

```bash
make merge KEEP=no                  # 合并后不保留每序列中间表
make train CORR=0.9 TOPK=8          # 调特征筛选参数
make fastTest YUV=/路径/BQSquare_416x240_60.yuv   # 指定 YUV
```

各子项目仍可单独进入用 `make`（细节见各自的 `README.md`）。

## 子项目说明

### yuvFeatureExtract（C++，CMake + Makefile，按模块运行）
- **采集**（`make extract`）：从裸 YUV 读灰度、统一转 10bit、切 128×128 CTU（丢边缘），提取与
  既有 `128csv` 对齐的 **18 个空域特征**；支持 YAML 逐序列配置、自动排除无用帧、按「固定条数/比例」采样。
- **合并**（`make merge`）：把特征与 `labelCtu` 标签按 `(POC_ID,pos_x,pos_y)` 内连接、去掉含 NaN 的行，
  生成 `Data/TrainingData/Training.csv`；每序列中间表放 `Data/temp/trainMerged/`
  （`make merge KEEP=no` 可不保留中间表）。还会仿照 VTM(RA, GOP=32) 由 POC 推出 4 个基础 QP
  下的实际 slice QP，加入 `22QP/27QP/32QP/37QP` 四列。
  Training.csv 列：`sequence,POC_ID,pos_x,pos_y, 4×qt_max, 4×QP, 18×特征`（共 30 列）。
- **MTT 标签**（`make mttLabel`）：解码 `Data/binEncode202/` 的 RA 码流，按「随机 POC × 随机 block」
  采样导出 MTT 划分标签 → `Data/labelMtt/<seq>_QP<qp>.csv`（逐节点 `w/h/qtDepth/mttDepth/splitMode`）。
  熵解码走 VTM 解码器（`#MTTDUMP` 钩子，默认零开销），编排/采样在 `mttLabel/` 模块。详见
  `yuvFeatureExtract/mttLabel/README.md`。
- **MTT 特征**（`make mttFeature`）：给 labelMtt 的每个块配空域特征（②方向性 + ③划分代价代理 +
  ④形状/QP，块所在 CTU 的 ①复杂度作上下文）→ `Data/TrainingData/TrainingMtt.csv`。详见
  `yuvFeatureExtract/mttFeature/README.md`。
- 自建函数库在 `libs/`（MiniYaml/YuvGrayLib/SpatialFeatureLib/**MttSampleLib/MttFeatureLib**），
  采集器在 `collector/`，合并器在 `merger/`，MTT 标签编排在 `mttLabel/`，MTT 特征在 `mttFeature/`。

### modelTrain（Python，Makefile + venv）
读 `Training.csv`，做数据清洗 + 特征筛选，按 QP 分别训练 **朴素贝叶斯/决策树/随机森林**
（多分类 qt_max 0–4），用 **GroupKFold(5, 按序列)** 评估，输出完整报告、12 张混淆矩阵，
并在训练时**直接产出** VTM 可读的森林模型 `qtForest.bin`（每 QP 元数据存 `models/<qp>.json`），可直接嵌入 VTM。

## 关于 VTM 参考软件

`VVCSoftware_VTM-VTM-20.2/` 是未改动的官方版本，只在 3 个 CMakeLists 里加了
`NOT (CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")` 判断，使其能在 Apple Silicon 上以标量回退构建。
`perSequenceCfg/` 提供各序列的官方编码参数（采集器 config 中的宽高/位深/帧数即据此填写）。

## oldFashion（历史归档）

早期把预测器**直接嵌进编码器**的两版原型已归档于此，仅作参考：
- `VTM_VVC_128/`：VTM 20.0，离线 CSV 查表喂划分结果（脆弱、大序列易崩）。
- `VTM-GOP/`：VTM 20.2，编码器内实时特征 + 朴素贝叶斯预测（含一个残差特征失效的 bug）。
- `VTM加速框架_问题清单.md` / `VTM加速框架_修复记录.md`：两版的致命问题复盘与修复记录。

当前工作流改为「**离线解耦**」：特征/标签都在编码器外离线生成，便于干净地训练与验证，
规避了早期内嵌方案的崩溃与训练/推理特征不一致问题。
