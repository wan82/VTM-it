# mttLabel —— MTT 划分标签采样导出

从 `Data/binEncode202/` 的 RA 码流里解出 **MTT（BT/TT）划分**，做成训练标签。
只采一小部分（随机 POC × 随机 block），避免把上百万个块全导出来。

## 为什么要走 VTM 解码器

`.bin` 是 CABAC 熵编码的，**只有 VTM 解码器能还原划分树**（labelCtu 当年也是这么来的）。
所以本模块的分工是：

- **熵解码**：VTM `DecoderApp` 完成，附带一个采样 dump 钩子
  `source/Lib/DecoderLib/MttLabelDumper.*`（打 `#MTTDUMP` 标，**默认关闭、零开销**）。
- **编排 + 采样计划 + 落盘**：本模块 `mttLabelBuilder`（C++）+ 函数库 `libs/MttSampleLib`。

> 整流解码是必须的（块之间有依赖，不能只解一个块），但很快（bin 才几 MB）。
> 真正要省的是数据量——所以在 dump 那一步按比例只写一小部分。

## 输出

每个 `(序列, QP)` 一个 CSV：`Data/labelMtt/<seq>_QP<qp>.csv`，列：

```
sequence, POC_ID, pos_x, pos_y, w, h, qtDepth, mttDepth, splitMode, QP
```

- 一行 = 一个 **MTT 阶段节点**（QT 叶子及其下的每个 BT/TT 决策节点，即非 QUAD 节点）。
- `splitMode ∈ {NO, BTH, BTV, TTH, TTV}`：该块自身的划分决策
  （NO=不再分；BTH/BTV=二叉横/竖；TTH/TTV=三叉横/竖）。
- 只采亮度；跳过越过图像边界的隐式强制划分（不是真正的决策）。
- **MTT 划分逐 QP 不同**，所以按 (序列,QP) 分文件、带 QP 列，不像 labelCtu 那样 4 QP 合一行。

这是个**超集**：既能训「MTT 深度封顶」（用 mttDepth），也能训「逐节点 5 路模式」（用 splitMode），
后期换预测目标不用重解。

## 用法

```bash
# 根目录一键（自动构建带 #MTTDUMP 的 DecoderApp + 本编排器，再解码采样）
make mttLabel

# 或在本子项目里：
cd yuvFeatureExtract
make mtt-label                       # 同上
# 手动指定解码器/目录：
build/mttLabel/mttLabelBuilder mttLabel/config/mtt.yaml \
    --decoder /路径/DecoderAppStatic --bin-dir /路径/binEncode202 --out-dir /路径/labelMtt
build/mttLabel/mttLabelBuilder mttLabel/config/mtt.yaml --dry-run   # 只打印将执行的命令
```

## 采样参数（改 `config/mtt.yaml`）

| 键 | 含义 | 默认 |
|----|------|------|
| `poc_count`   | 每序列随机采的 POC 个数（≤0=全部 POC） | 8 |
| `block_ratio` | 每个 MTT 节点被写出的概率 0~1 | 0.03 |
| `random_seed` | 采样种子（结果可复现） | 42 |
| `qps`         | 每序列要解码的基础 QP | 22,27,32,37 |
| `bin_dir` / `out_dir` / `bin_suffix` | 输入码流目录 / 输出目录 / bin 命名后缀 | 见 yaml |

采样在解码器侧用 `(seed,poc,x,y,w,h)` 哈希做，确定性、与解码顺序无关；同一序列的各 QP 用同一组 POC，便于对照。

## 解码器环境变量（`#MTTDUMP`，编排器自动设好）

`MTT_DUMP_OUT`（输出路径，未设→关闭）、`MTT_DUMP_SEQ`、`MTT_DUMP_QP`、
`MTT_DUMP_POCS`（逗号 POC 列表，空=全部）、`MTT_DUMP_BLOCK_RATIO`、`MTT_DUMP_SEED`。
**不设 `MTT_DUMP_OUT` 时解码行为与原版完全一致。**

## 后续

`libs/MttSampleLib` 的采样口径会被复用：下一步针对采到的（变尺寸、矩形）MTT block
提取 YUV 空域特征（含方向性特征），喂入 MTT 预测模型。
