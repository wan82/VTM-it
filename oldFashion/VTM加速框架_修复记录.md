# VTM 加速框架 — 修复记录（CHANGELOG）

> 日期 2026-06-02 · 已在你提供的工作目录原文件上直接修改（你已备份）
> 语法验证：两个 `EncCu.cpp` + `w_cuProcess.cpp` 均通过 `g++ -std=c++17 -fsyntax-only`，零自身错误。
> 仍需在你本机用 cmake 正常重新构建确认链接与运行。

---

## 一、20.2 / 在线 NB 版（VTM-VTM-20.2）

### `source/Lib/EncoderLib/EncCu.cpp`
- **2.1 残差特征恒为0**（128/64/32 三处）：`convertPelBufToIntMatrix(cuMatrix, …)` → `convertPelBufToIntMatrix(l0Matrix, …)`。时域残差特征不再恒为 0。
- **2.3 L0 空指针**：`l0Frame = (numRefIdxL0>0) ? getRefPic(...) : nullptr`，取缓冲改为三目判空，L0 不可用时退回当前帧（残差为 0），不再裸解引用。
- **2.4 参考帧缓冲**：`l0Frame->getOrigBuf(...)` → `l0Frame->getRecoBuf(...)`，用重建缓冲，避免读到可能已释放的原始缓冲。
- **2.7 时域距离语义**：`w_time_res_L0 = POC − 参考帧个数` → `POC − 实际 L0[0] 参考帧 POC`。
- **去重标志复位**：`w_new_cu` 增加 `else { w_new_cu = false; }`，否则恒为 true、去重失效、对同一 CU 重复做重特征计算。

### `source/Lib/EncoderLib/w_cuProcess.cpp`
- **2.2 未捕获异常 → 安全回退**（共 12 处 `throw` 全部移除）：
  - `get_qp_value`：未知 gop / 非 CTC QP / 未知 tid → 一律 `return qp`（**现在 QP≠22/27/32/37、非 GOP32 结构也不会再崩**，只是该 CU 不做 QP 加权）。
  - `convertPelBufToIntMatrix`（两个重载）：尺寸无效/超界 → 返回空矩阵或 `cuSize×cuSize` 零矩阵。
  - `calculate_skewness / calculate_kurtosis`：N<2 → `return 0.0`。
  - `calculate_levene / calculate_kruskal_w`：非方阵 / 边长非偶 → `return 0.0`。
  - `calculate_levene_kruskal`：非方阵 → `return {0.0, 0.0}`。

---

## 二、20.0 / CSV 查表版（VTM-128-20.0）

### `source/Lib/EncoderLib/EncCu.cpp`
彻底重写了脆弱的查表机制：

- **1.1 + 1.2 全局顺序指针 → 映射查表**：删除全局 `int idBlock / csvID / int** csvInfo` 与 `while(!findTargetCTU)` 循环；改为：
  - 解析 CSV 到 `std::map<long long,int> csvMap`，key = `(POC<<32)|ctuId`；
  - 编码时**由 CTU 实际像素位置直接算出光栅 ctuId**（`(ly/128)*ctusPerRow + lx/128`）后查表。
  - 一举消除"同一 CTU 重复进入→指针错位→越界/死循环"，也不再有任何无边界数组访问。
- **1.3 边界 CTU**：边界 CTU 一律 `isPCTU=false`，交回 VTM 原生强制划分逻辑，加速不介入（杜绝"强制不分→非法→崩溃"）。
- **1.5 CSV 缺失/不全**：打不开或查不到该 CTU → 该 CTU 走完整搜索；`std::stoi` 逐行包 `try/catch`，非法行跳过。空表也安全。
- 顺带：`#define WAN_CHANGE 1;` 的多余分号去掉（与 `wan.h` 定义一致，消除潜在宏重定义告警）；新增 `#include <map>`。

---

## 三、本次**未**改动项（及原因）

| 项 | 版本 | 为何保留 |
|----|------|----------|
| 1.4 / 2.5 线程安全 | 两版 | 全局量改成员变量改动面大、风险高；单线程编码不受影响。若要开 WPP/帧并行再重构。 |
| 1.6 csvInfo 释放 | 20.0 | 原 `csvInfo` 已移除；新 `csvMap` 为单次分配、随进程退出回收，非每 CU 泄漏。 |
| 2.6 强制 QT 无兜底 | 20.2 | 已由 64/32 层门控 `currQtDepth>0 && currMtDepth==0`（QT 必可用）规避，属隐患非现患。 |
| 2.8 仅 QT-only / 重开销 | 20.2 | 属性能调优（开 128 层 / 换轻量特征），非缺陷；涉及实验设计，留给你决定。 |

---

## 四、⚠️ 重要：20.2 需要重新验证 / 可能需重训模型

修复 **2.1 + 2.4 + 2.7**（以及 `w_new_cu` 复位）会**真实改变喂给朴素贝叶斯的特征值**——
此前残差特征恒为 0、时域权重失真，模型实际是在"半瞎"输入下工作的。修复后：

- 若 `w_pred_model.cpp` 里的分箱/先验是在**这套带 bug 的 C++ 特征**上训练的 → 现在输入分布变了，**预测会变，需要按新特征重新训练**，否则结果不可预期。
- 若模型是在**正确特征**（如 Python 侧）上训练的 → 修复让推理与训练对齐，**应当改善**。

无论哪种，原先报告的 **ET −3.79% / BD-rate +0.874% 不再成立**，需重跑获取新数。好消息是 2.1 修好后残差特征终于真正生效，算法上限应高于旧数。

20.0 的修复仅改容错与查表机制、不改特征语义，对其"命中即跳过"的结果在 CSV 正确时不变，但现在在 A1 等大序列上不会再崩。
