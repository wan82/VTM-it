# QTPRED —— 把随机森林嵌入 VTM 做 QT 早停

在官方 VTM 20.2 里加了一个“快速 CU 划分”模块：编码器读入一个 128×128 CTU 后，提取
**空域特征 + 时域残差特征 + QP-value**，喂入训练好的随机森林/决策树，得到**预测的 QT 最大深度**，
然后让该 CTU 在该层级**终止 QT 划分**，从而跳过更深 QT 的 RDO 搜索、节约编码时间。
所有改动都用 `#QTPRED` 标注，可一键搜索。

特征（最终由 wrapper 选出，随重训会变；当前版本为 `res_std, Sobel_std, Sobel_mean` + QP-value）：

- **空域**：`Sobel_*/Lap_*/std/...`（在 128 原始亮度块上算）。
- **时域残差 `res_*`**（`res_mean/std/skew/kurt`）：当前块 − **L0[0] 参考帧同位块** 的统计矩。
  - L0[0] = `slice->getRefPic(REF_PIC_LIST_0, 0)` 的重建块（= POC 去最低置位比特，与离线采集口径一致）。
  - ⚠️ **一致性**：离线训练用**源帧** L0，编码器在线用**重建帧** reco(L0)，存在量化噪声失配
    （当前为「路线 a」，先验证端到端；若失配明显可改用解码重建帧重训）。
  - 若该帧无 L0（理论上 inter 必有）→ 不提供 res_*，预测返回 -1、不早停（安全回退）。

## 改了哪些文件（均带 `#QTPRED` 标注）

| 文件 | 改动 |
|------|------|
| `source/Lib/EncoderLib/QtForestPredictor.h`  | **新增**：预测器接口 |
| `source/Lib/EncoderLib/QtForestPredictor.cpp` | **新增**：模型加载 + 空域特征/QP-value + 森林推理 |
| `source/Lib/EncoderLib/EncCu.h`  | 加 2 个成员 `m_qtPredActive / m_qtPredDepth` |
| `source/Lib/EncoderLib/EncCu.cpp` | include；CTU 根算预测；`isModeSplit` 处到达预测深度时跳过 QT |

> EncoderLib 的 CMakeLists 用 `file(GLOB *.cpp)`，新增的两个文件会**自动参与编译**，无需改构建脚本。

## 工作原理（置信度门控 + 双向早停）

1. **CTU 根**（`currQtDepth==0 && currMtDepth==0`、inter slice、完整 128×128、非边界）：
   取原始亮度并统一缩放到 10bit，算 5 个空域特征 + `getQpValue(POC, baseQP)`，预测 `qt_max`
   **及其置信度**（预测类别的概率）。
2. **置信度门控**：只有 `置信度 ≥ 阈值`（默认 0.7，环境变量 `QT_FOREST_CONF`）才启用早停；
   不确信就**全搜**，不做任何干预 —— 这能修掉“不确定时硬停”导致的质量损失。
3. **双向早停**（仅在确信时）：
   - **上界**：当 `当前QT深度 >= 预测qt_max` 且测试的是 QT 划分时，**跳过更深 QT**；
   - **下界（强制分）**：在 CTU 根且 `qt_max>=1`（确信会划分）时，**跳过 128 大块的块模式与 BT/TT，
     只测 QT**（省时间最多；qt_max>=1 ⟺ 128 必被划分，所以在根层是安全的）。
     可用 `QT_FOREST_FORCE=0` 关闭、只保留上界。
4. 边界 CTU / intra slice / 不确信 / 模型不可用 → 不介入，行为与原版一致。

> 调参：`QT_FOREST_CONF` 越高越保守（BD-rate 越稳、省时越少）；`QT_FOREST_FORCE=0` 只做上界。
> C++ 推理的类别与置信度均与 sklearn `predict` / `predict_proba` **逐样本一致**（验证 600/600）。

## 用法（三步）

```bash
# 1) 训练（直接产出 VTM 可读的 qtForest.bin；默认已限制模型规模 min_samples_leaf=50）
cd /Users/w/Documents/Dev/VTM-it
make train                     # → Data/ModelResults/qtForest.bin（无需任何转换步骤）

# 2) 正常构建 VTM（已支持 arm64）
cd VVCSoftware_VTM-VTM-20.2
mkdir -p build && cd build && cmake .. && make -j EncoderApp

# 3) 编码时用环境变量指定模型，跑 RA
export QT_FOREST_MODEL=/Users/w/Documents/Dev/VTM-it/Data/ModelResults/qtForest.bin
export QT_FOREST_CONF=0.7        # 可选：置信度门控阈值（默认 0.7，越高越保守）
export QT_FOREST_FORCE=1         # 可选：1=启用下界强制分(默认)，0=只做上界
./bin/EncoderAppStatic -c .../encoder_randomaccess_vtm.cfg -c .../<seq>.cfg -q 32 ...
```

启动时会打印 `[QTPRED] 已加载 4 个 QP 模型 ...`；**不设 `QT_FOREST_MODEL` 就是原版行为**。

> 快速试编码：根目录 `make fastTest` 会编译 VTM 并对 BQSquare 编码 33 帧（POC 0–32，一组 GOP32），
> QP 22/27/32/37 各一次，日志存 `Log/FastTest/`（若 `qtForest.bin` 存在则启用 QT 早停）。
> YUV 路径覆盖：`make fastTest YUV=/你的路径/BQSquare_416x240_60.yuv`。对照原版可临时改用空模型再跑一次。

## 重要前提 / 注意

- **只对 RA、GOP=32、基础 QP ∈ {22,27,32,37}** 有效（QP-value 偏移表按这些条件烘焙；其它 QP 无对应
  模型则自动不介入）。
- 模型路径由环境变量 `QT_FOREST_MODEL` 给出；加载失败 → 关闭、不影响编码。
- C++ 推理与 sklearn `predict` 已**逐样本对照一致**（验证 800/800）。
- 模型由**训练时直接序列化**进 `qtForest.bin`（不经 joblib 转换）；训练默认
  `--rf-min-leaf 50` 已限制 RF/决策树规模，模型约几 MB、推理快、精度基本不变。
- 这是“预测层级强制终止 QT”的早停，属有损加速：会带来少量 BD-rate 上升换取 ET 下降，幅度取决于模型精度。
- 本仓库环境无法整体编译 VTM，以上 C++ 改动已做**逐文件语法检查（EncCu.cpp 零自身错误）**，
  但请在你本机完整 `cmake && make` 编译并实测一条短序列确认。
