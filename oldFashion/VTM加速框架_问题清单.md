# VTM 快速 CU 划分加速框架 — 问题清单

> 代码审查记录 · 生成日期 2026-06-02
> 覆盖两套实现:
> - **20.0 / CSV 查表版** — `VTM_VVC_128/VVCSoftware_VTM-128-20.0`(`WAN_CHANGE` 宏)
> - **20.2 / 在线 NB 版** — `VTM-GOP/VVCSoftware_VTM-VTM-20.2`(`w_cuProcess` / `w_pred_model`)

---

## 0. 总览

| 版本 | 机制 | 实测结果 | 稳定性现状 |
|------|------|----------|------------|
| 20.0 CSV | 离线算好划分结果写入 CSV,编码时按 `(POC, CTUid)` 查表 | ET −20.18% / BD-rate +0.956% | **结构性脆弱,大序列(A1)易崩,跑通靠对齐运气** |
| 20.2 NB | 编码器内实时提取纹理/时域特征 + 朴素贝叶斯预测 | ET −3.79% / BD-rate +0.874% | **结构性已修稳,但有 1 个隐藏正确性 bug + 一组硬编码异常地雷** |

**核心结论**
- 两者 BD-rate 接近 ⇒ 模型判断精度同档;ET 差距来自**预测开销**(离线≈0 vs 在线很重)与**作用层级**(CTU 顶层 vs 仅 64 层)。
- 20.2 把 20.0 的必崩根因(全局顺序指针、越界、边界强制不分)都修掉了;它能跑完 full-CTC 不是侥幸,而是真的更稳。
- 20.2 残留的最大隐患是一个**不崩但让模型半瞎的残差特征 bug**,以及**写死 CTC 条件(QP/GOP)的未捕获异常**。

严重程度标记:🔴 致命(崩溃) · 🟠 高 · 🟡 中 · 🔵 低/性能 · ⚪ 设计局限

---

## 1. 20.0 / CSV 查表版

主要改动文件:`source/Lib/EncoderLib/EncCu.cpp`(核心)、`CommonLib/wan.h`、`EncoderLib/EncCfg.h`、`App/EncoderApp/EncAppCfg.cpp`、`DecoderLib/DecCu.cpp`。

### 1.1 🔴 缺少"每个 CTU 只处理一次"的去重 → 全局指针错位 → 越界崩溃
**位置:** `EncCu.cpp` 128×128 区域处理段(全局 `idBlock` / `csvID`)

`idBlock`、`csvID` 是文件级全局量,逻辑假设"每个 128 CTU 恰好被访问一次、按光栅顺序计数一次"。但 `xCompressCU` 对同一 128×128 区域**会被重复进入**(dual-tree、`numRoundRdo==2` 的 mode-constraint 两轮、边界重试)。该段**没有任何"是不是新 CTU"的判断**,同一位置被进入两次 → `idBlock/csvID` 多走一步 → 此后整条序列对齐全错位 → 最终 `csvID` 冲出数组越界崩溃。

> 对照:20.2 版专门加了 `w_new_cu` + `old_cu_x/y/w/h`(注释"去除重复判断")就是修这个坑。**此问题在 20.2 已修复。**

**修复:** 引入与 20.2 相同的 `(pos, size)` 去重;或彻底放弃全局顺序指针,改用 `(POC, 位置)` 直接索引。

### 1.2 🔴 查找循环无边界 / 无终止保护
**位置:** `EncCu.cpp` `while (!findTargetCTU)`

```cpp
while (!findTargetCTU){
  if(csvInfo[csvID][0]==POC && csvInfo[csvID][1]==idBlock){ ...; csvID++; }
  else { /* 只打印,csvID 不前进 */ }
}
```
两种死法:① 匹配条件永不成立 → 固定 `csvID` 上**死循环卡死**;② `csvID` 已被前面累积推过 `csvRows` → `csvInfo[csvID]` **越界读 → 段错误**。缺 `csvID < csvRows` 保命检查。

**修复:** 加 `if (csvID >= csvRows) { /* 安全回退:按默认全搜索 */ break; }`。

### 1.3 🔴 边界 CTU + 预测=0 → 强制不划分 → 非法状态崩溃
**位置:** `EncCu.cpp` 边界分支

```cpp
// 进入前: doBlock=true, doQT=false, doBTTT=false
if(!isBoundary){ if(predSplit){doQT=true; doBlock=false;} }
else          { if(predSplit){doQT=true; doBTTT=true; doBlock=false;} }
```
边界 CTU 若 CSV 给 **0**,三个标志保持初始 `doBlock=true / doQT=false` → 把超出画面的边界 128 块当单个 CU 编码。VVC 规定边界必须划分,这是非法状态 → 断言或空 `bestCS` 后续空指针。A1 为 3840×2160,2160/128=16.875,**底部整行都是边界 CTU**,只要其中任一预测为 0 即崩。

**修复:** 边界 CTU 一律强制 `doQT=true`(忽略预测值)。

### 1.4 🟠 全局可变状态非线程安全、不复位
`idBlock`、`csvID`、`csvInfo` 为文件级全局量。多线程/WPP/帧并行下并发读写 → race → 错位/越界。多次编码或 multi-layer 重入时也不重新读表。

**修复:** 改为编码器实例成员变量,并在 slice/帧初始化时复位。

### 1.5 🟠 CSV 打不开或行数不足 → 立即越界
打开失败时仅 `cerr` 一句即继续,`csvRows=0` → 首次 `csvInfo[0]` 越界。CSV 行数少于实际编码访问的 `(POC,CTU)` 数 → 某帧后 `csvID` 冲顶崩溃。

**修复:** 打开失败/空表时直接关闭加速(回退原始全搜索),不进入查表分支。

### 1.6 🟡 `csvInfo` 读入后从不释放
`new int*[csvRows]` 分配后无 `delete[]` — 内存泄漏。单次运行影响有限,但属资源管理缺陷。

### 1.7 ⚪ 设计局限:查表只能表达 CTU 级决策
表按 `(POC, CTUid)` 索引,每个 key 仅一个决策,**天然只能停在 CTU 顶层**。划分是动态展开的递归树(QT→BT/TT),访问哪些子 CU 取决于前面 RD 结果,无法预先枚举对应 → 无法可靠地为子层查表。这是促使转向 20.2 在线方案的根本原因,非 bug。

---

## 2. 20.2 / 在线 NB 版

主要文件:`source/Lib/EncoderLib/EncCu.cpp`、`w_cuProcess.cpp`、`w_pred_model.cpp`。
当前编译配置:`W_LEVEL_128=0, W_LEVEL_64=1, W_LEVEL_32=0, W_MODEL_NB=1`(仅 64 层生效)。

### 2.1 🔴 正确性 bug:残差特征恒为 0(让模型半瞎)
**位置:** `EncCu.cpp` 三处 — **行 789(128)、853(64)、913(32)**

```cpp
PelBuf l0Matrix = l0Frame->getOrigBuf(...);                       // 取了 L0 帧
CU_MATRIX l0Matrix_int = convertPelBufToIntMatrix(cuMatrix, ...); // 却转的是 cuMatrix(当前帧)!
CU_MATRIX pooling_L0_matrix = maxPooling(l0Matrix_int);
CU_MATRIX pooling_res = subtractMatrices(pooling_matrix, pooling_L0_matrix); // 当前-当前 = 全 0
```
`l0Matrix` 取出后**根本没用**,转矩阵仍传 `cuMatrix`。于是 `pooling_res` 恒为零矩阵,所有 `res_std / res_kurt / res_levene / res_kursW` **时域残差特征全是常数**。模型实际只靠 mean/std/sobel 等空域特征判断 — 一半输入是噪声。这极可能是节约率与 BD-rate 上不去的主因之一。

**修复:** 三处的 `convertPelBufToIntMatrix(cuMatrix, ...)` 改为传入 `l0Matrix`。修复后建议同步复核/重训模型的残差特征分箱。

### 2.2 🔴 未捕获异常 → `std::terminate`(写死 CTC 条件)
**位置:** `w_cuProcess.cpp` `get_qp_value` / `convertPelBufToIntMatrix`

```cpp
else { throw std::invalid_argument("Unsupported QP value: " ...); }  // get_qp_value
```
`get_qp_value` 写死只接受 **QP ∈ {22,27,32,37}**;`gop = poc%32` 的 tid 映射写死 **RA 的 GOP32** 结构。`convertPelBufToIntMatrix` 也有多处 `throw`。而 VTM 编码主循环 `xCompressCU` **无任何 try/catch** → 一抛即崩溃。

后果:`w_qp = getBaseQP()`,**只要用非 CTC 的 QP(如 QP30)就当场崩**;换 Low-Delay 或别的 GOP 大小 → tid 语义错乱(特征被错误缩放)。整套被硬绑死在"RA + GOP32 + 四个 CTC QP"上。

**修复:** 异常路径改为"安全回退到原始全搜索"(返回标志位而非 throw),或在调用处包 try/catch 并在 catch 中关闭该 CU 的加速。

### 2.3 🟠 L0 空指针只打印不拦截
**位置:** `EncCu.cpp` 行 787 / 852 / 912 附近

```cpp
Picture* l0Frame = slice.getRefPic(REF_PIC_LIST_0, 0);
if (!l0Frame){ std::cout << "L0 frame is null"; }   // 只打印,没 return
PelBuf l0Matrix = l0Frame->getOrigBuf(...);          // 紧接着解引用 → 空指针崩溃
```
`numRefIdxL0 <= 0` 的判断同样只打印不跳过。P/B 帧通常有参考帧故平时不触发,但无护栏即裸崩。

**修复:** 判空后 `is_predicted_cu=false; doBlock=true;` 直接回退,不做预测。

### 2.4 🟡 对参考帧使用 `getOrigBuf` 语义错误且缓冲可能已释放
取参考应使用**重建缓冲**而非原始(uncompressed)缓冲;且参考帧编码完成后其原始缓冲可能已释放 → 即便修了 2.1,也可能读到已释放内存 → 垃圾/崩溃。

**修复:** 改用参考帧的重建缓冲(reconstruction buffer)。

### 2.5 🟡 全局量非线程安全
`old_cu_x/y/w/h`、`w_poc_id`、`w_qp`、`w_L0_id`、`w_time_res_L0` 均为文件级全局变量。WPP / 帧级并行下多线程并发读写 → 去重判断错乱、撕裂读 → 特征错乱甚至崩。单线程无碍。

**修复:** 收敛为编码器实例成员变量。

### 2.6 🟡 强制 QT 无回退兜底
预测=1 时 `doQT=true / doBlock=false`,跳过所有块模式与 BT/TT,只留 QT。当前靠 64/32 层门控 `currQtDepth>0 && currMtDepth==0` 隐含保证 QT 可用;一旦某配置使该节点 QT 不可用 → `bestCS` 为空 → 下游空指针。

**修复:** 加兜底:若强制后无任一有效模式产生 `bestCS`,回退测块。

### 2.7 🟡 `w_time_res_L0` 语义错误
**位置:** `EncCu.cpp`
```cpp
w_time_res_L0 = w_poc_id - slice.getNumRefIdx(REF_PIC_LIST_0);  // POC 减"参考帧个数"
```
用参考帧**数量**(如 4)而非实际 L0 参考帧的 POC,语义错误 → `scale_res_value` 的时域缩放权重失真。不崩,但削弱残差特征质量(与 2.1 叠加)。

**修复:** 用当前 POC 与实际 L0 参考帧 POC 之差。

### 2.8 🔵 性能:纯 QT 预测器 + 重特征开销
门控逻辑实为**二值 QT 决策**:预测=1 强制四叉树(省块模式+MTT,省得多);预测=0 仅跳过 QT、仍测块+BT+TT(只省 QT 一支,省得少)。若多数 CU 判为 0,毛节约本就小,又被每 CU 的 Sobel/池化/Levene/Kruskal 在线开销抵掉 → 净 3.79%。

**提升方向:**(a) 打开 `W_LEVEL_128`,恢复顶层高杠杆剪枝(inline 下不会再像 20.0 那样崩);(b) 用更轻量统计量替换 Levene/Kruskal-Wallis,或对特征做预计算以降开销。

---

## 3. 优先级汇总

| # | 版本 | 严重 | 问题 | 一句话修复 |
|---|------|------|------|------------|
| 1.1 | 20.0 | 🔴 | 无去重 → 指针错位越界 | 引入 20.2 的 `(pos,size)` 去重 |
| 1.2 | 20.0 | 🔴 | 查找循环无边界/无终止 | 加 `csvID<csvRows` 回退 |
| 1.3 | 20.0 | 🔴 | 边界 CTU 预测 0 → 强制不分 | 边界一律强制 `doQT=true` |
| 1.4 | 20.0 | 🟠 | 全局态非线程安全/不复位 | 收成员变量 + 复位 |
| 1.5 | 20.0 | 🟠 | CSV 缺失/不足 → 越界 | 失败即关闭加速回退 |
| 1.6 | 20.0 | 🟡 | csvInfo 内存泄漏 | 配 `delete[]` |
| 1.7 | 20.0 | ⚪ | 查表只能 CTU 级 | (已由转 20.2 解决) |
| **2.1** | **20.2** | **🔴** | **残差特征恒为 0** | **三处 `cuMatrix`→`l0Matrix`** |
| 2.2 | 20.2 | 🔴 | 未捕获异常 + 写死 QP/GOP | throw 改安全回退 |
| 2.3 | 20.2 | 🟠 | L0 空指针只打印不拦 | 判空即回退 |
| 2.4 | 20.2 | 🟡 | 参考帧用 origBuf | 改用重建缓冲 |
| 2.5 | 20.2 | 🟡 | 全局态非线程安全 | 收成员变量 |
| 2.6 | 20.2 | 🟡 | 强制 QT 无兜底 | 加无有效模式回退 |
| 2.7 | 20.2 | 🟡 | time_res 语义错误 | 用实际 L0 参考 POC |
| 2.8 | 20.2 | 🔵 | QT-only + 重开销 | 开 128 层 + 轻量特征 |

**建议处理顺序**
1. 若仅维持现状跑 CTC:20.2 现有结果可用,无需改动。
2. 若要把 20.2 扩成独立工作 / 提升指标:先修 **2.1(残差)**,再修 **2.2 / 2.3**(才敢跑非标准 QP/GOP),最后试 **2.8**(开 128 层)拉节约率。
3. 若仍需用 20.0(为其更高节约率):必须先修 **1.1 / 1.2 / 1.3** 三个致命项才能在 A1 等大序列稳定运行。

---

*本清单基于静态代码审查,行号以当前工作区文件为准;修复后请重跑回归确认 BD-rate 与稳定性。*
