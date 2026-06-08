# labelMerge —— 标签 × 特征 合并工具（Python，Makefile 驱动）

把 `label-ctu`（QT 深度标签）和中间件 `temp-yuvFeature`（采集器输出的空域特征）
按 `(POC_ID, pos_x, pos_y)` 合并成训练数据，输出到 `TrainingData/`。

## 数据流

```
采集器(collector)  ──►  Data/temp/temp-yuvFeature/   (中间件：每序列一个特征CSV)
                                  │
Data/label-ctu/<name>.csv ──────►│  (labelMerge 按键内连接)
                                  ▼
                         Data/TrainingData/      (最终训练数据)
```

## 合并逻辑

- **输入**
  - `label-ctu/<name>.csv`：每序列一个，含 4 个 qtmax 标签
    （`POC_ID,pos_x,pos_y,22qt_max,27qt_max,32qt_max,37qt_max`）。
  - `temp-yuvFeature/<name>.csv`：每序列一个，含 18 个空域特征。
- **合并**：同名序列按 `(POC_ID,pos_x,pos_y)` **内连接**
  → 训练表 `TrainingData/<name>.csv`（列：键 + 标签 + 特征）。
- **总表**：所有序列拼起来、**去掉含 NaN 的行**，得到干净的
  `TrainingData/Training.csv`（首列加 `sequence`）—— 最终训练集。

## 用 Makefile 运行（推荐）

```bash
cd labelMerge
make            # 首次自动建 venv 装 pandas；检测到 temp-yuvFeature 有数据则合并到 TrainingData
```

- **会自动检测** `temp-yuvFeature` 里有没有特征 CSV：有就合并，没有就提示先跑采集器（`cd ../yuvFeatureExtract && make run`）。
- 其它目标：`make setup`（只建 venv）、`make clean`（删 venv）、`make distclean`（删 venv + TrainingData）。
- 路径可覆盖：`make FEATURES=/path LABELS=/path OUT=/path`。

默认路径：
- 中间件特征：`/Users/w/Documents/Dev/VTM-it/Data/temp/temp-yuvFeature`
- 标签：      `/Users/w/Documents/Dev/VTM-it/Data/label-ctu`
- 输出：      `/Users/w/Documents/Dev/VTM-it/Data/TrainingData`

> 不用 Makefile 也行：`./setup.sh` 建环境，`./run.sh [参数]` 直接跑 `merge.py`。

## 输出

```
Data/TrainingData/
├── <name>.csv     每序列训练表（键 + 4个qtmax + 18个特征，25列；保留可能的NaN）
└── Training.csv    全部序列拼合并去掉含NaN的行（首列 sequence，26列）—— 最终训练集
```

## 备注

- **内连接**：只保留「既有特征又有标签」的 CTU。特征是采样子集、label-ctu 是全覆盖，
  所以结果就是被采样到的那些 CTU 的训练样本。
- 汇总里 `未匹配特征`>0 表示采样到的 (POC,位置) 在 label-ctu 没标签；
  `缺标签`>0 表示某个 qtmax 列为空（如某序列某 QP 的解码数据不完整，会被总表的去 NaN 过滤掉）。
- 标签列直接取自 `label-ctu/<name>.csv` 里除键以外的所有列，因此换 QP 组合也无需改代码。
- 本工具只并 **label-ctu 标签 + temp-yuvFeature 特征**；若要再并入 `PocInfo` 的 QP/TID，
  可按 `POC_ID` 左连接进来。
