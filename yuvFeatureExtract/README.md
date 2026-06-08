# yuvFeatureExtract —— YUV 特征采集 + 标签合并（C++）

两件事都用 C++ 做，下游只需一个 Python 环境（modelTrain）：
1. **采集**：从原始 YUV 提取 128×128 CTU 的 18 个**空域特征** → 中间件 `tempYuvFeature`。
2. **合并**：把特征与 `labelCtu` 的 QT 深度标签按 `(POC_ID,pos_x,pos_y)` 合并 → `Training.csv`。

## 目录结构

```
yuvFeatureExtract/
├── CMakeLists.txt          顶层（聚合 libs / collector / merger）
├── Makefile                make 启动（包装 cmake）
├── libs/                   自建函数库（单独文件夹）
│   ├── MiniYaml/             极简 YAML 子集解析（无外部依赖）
│   ├── SpatialFeatureLib/    空域特征：矩、Sobel/Laplacian、Levene/Kruskal-Wallis
│   └── YuvGrayLib/           灰度图像矩阵：读YUV、位深转换、CTU分割、无用帧检测
├── collector/             采集子项目：YUV → 特征中间件
│   ├── src/main.cpp
│   └── config/config.yaml   示例配置
└── merger/                合并子项目：特征 × 标签 → Training.csv
    └── src/main.cpp
```

## 构建与运行

```bash
cd yuvFeatureExtract
make            # 配置并编译（Release）

make run        # ① 采集：YUV → Data/temp/tempYuvFeature（用 collector/config/config.yaml）
make merge      # ② 合并：特征 × labelCtu → Data/TrainingData/Training.csv
make data       # ①+② 一条龙

# 合并相关可覆盖：是否保留每序列中间表（默认 yes）
make merge KEEP=no
# 自定义路径：
make merge LABELS=/path FEATURES=/path TRAINING=/path/Training.csv TEMP_MERGED=/path
```

合并产物：`Data/TrainingData/` 下**只放 `Training.csv`**（已去掉含 NaN 的行）；
每序列中间表放到 `Data/temp/trainMerged/`（`KEEP=no` 时删除）。

合并时还会**仿照 VTM-VVC 计算 QP-value** 并加入 4 列 `22QP,27QP,32QP,37QP`：
按 RA 默认 GOP=32，由 `POC%32` 推出时域层 TID，再由 (基础QP, TID) 得到该 CTU 所在帧的实际 slice QP
（与解码出的 PocInfo 一致，如 POC0→19/24/29/34）。列顺序：键 → 4个qt_max → 4个QP → 18个特征。

需要 CMake ≥ 3.13 和支持 C++17 的编译器。

## 处理流程（每个序列）

1. **读灰度**：按 YAML 给的宽高/位深/色度格式，从裸 YUV 逐帧读取亮度分量（跳过色度）。
2. **统一位深**：无论源是 8bit 还是 10bit，都左移到内部 10bit（对齐 VTM 默认）。`out = in << (10 - inputBitDepth)`。
3. **确定帧数**：YAML 给了 `frames` 就用它，否则按文件大小 ÷ 每帧字节数推断。
4. **排除无用帧（自动检测）**：
   - 空白/近常量帧：全帧标准差 < `blank_std_threshold`；
   - 重复帧：与前一有效帧的平均绝对差 < `duplicate_mad_threshold`。
   - 检测在像素抽样（`detect_pixel_stride`）下进行以提速。被排除的帧索引会打印出来核对。
5. **预估行数**：`可用帧数 × 每帧完整CTU数`（边缘不足 128 的 CTU 丢弃）。
6. **采样**：每条样本独立地`随机选一帧 + 随机选一个CTU位置`，按 (帧, 位置) 去重，
   直到达到 `ctu_per_yuv`（默认 10，可在序列里覆盖）。`random_seed` 保证可复现。
7. **提特征**：对每个采样到的 128×128 块算 18 个空域特征，写一行。
8. **输出**：`<output_dir>/<name>.csv`，按 (POC, y, x) 排序。

## 输出 CSV 列（21 列，对齐 128csv）

```
POC_ID,pos_x,pos_y,
mean,std,skew,kurt,
Sobel_mean,Sobel_std,Sobel_skew,Sobel_kurt,
Lap_mean,Lap_std,Lap_skew,Lap_kurt,
LeveneTest,KrusWTest,SobelLevene,SobelKrusW,LapLevene,LapKrusW
```

- `POC_ID` 对裸 YUV 即**显示顺序的帧索引**。
- 特征公式严格对齐项目原有 `w_cuProcess.cpp`：mean、std（总体）、skew=m3/σ³、kurt=m4/m2²、
  Sobel=√(Gx²+Gy²)、Levene（四象限经典 W）、Kruskal-Wallis（简单秩）。
- **Laplacian** 原 `w_cuProcess` 没有，这里用标准 4-邻域 3×3 核取绝对响应。
  若要与旧 `128csv` 的生成脚本**逐值**对齐，请提供旧脚本的 Laplacian 核定义，我再调整。

## 配置项（YAML）

全局：`output_dir`、`ctu_size`、`internal_bitdepth`、`default_ctu_per_yuv`、`default_sample_ratio`、
`random_seed`、`chroma_format`、`detect_useless_frames`、`blank_std_threshold`、
`duplicate_mad_threshold`、`detect_pixel_stride`。

每个序列（`sequences:` 列表）：必填 `path`/`width`/`height`/`bitdepth`；
可选 `name`、`framerate`、`frames`（缺省按文件大小推断）、`chroma_format`、
`ctu_per_yuv`（固定条数）、`sample_ratio`（按比例）。

**采样数量两种模式**（优先级从高到低）：
1. 序列级 `sample_ratio`（>0）→ `ceil(预估可生成 × 比例)`
2. 序列级 `ctu_per_yuv` → 固定条数
3. 全局 `default_sample_ratio`（>0）→ 按比例
4. 全局 `default_ctu_per_yuv` → 固定条数

例：全局 `default_sample_ratio: 0.03` 让每个 YUV 采约 3%；
某序列想固定 200 条，则在该序列加 `ctu_per_yuv: 200`（第 2 条优先于第 3 条）。

见 `collector/config/config.yaml` 示例。

## 自建函数库 API 速览

- `sf::computeCtuFeatures(const sf::Mat& block) -> CtuFeatures`（含 `csvHeader()` / `toCsvFields()`）
- `sf::mean/stddev/skewness/kurtosis/sobel/laplacian/levene/kruskalWallis`
- `yg::YuvReader`（读帧+位深转换）、`yg::GrayFrame::extractBlock`、
  `yg::enumerateFullCtus`、`yg::detectUsableFrames`
- `myaml::parseFile` / `getInt/getDouble/getStr/getBool`
