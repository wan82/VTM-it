// VarianceTests.h —— 方差齐性 / 分布差异统计量
// 做法与 w_cuProcess.cpp 完全一致：把方阵切成 2x2 共 4 个子块（各为边长一半），
// 把 4 个子块当作 4 组样本：
//   Levene        : 经典 Levene W 统计量（基于到组均值的绝对偏差）
//                   W = (N-k)/(k-1) * Σ ni (zi_bar - z_bar)^2 / ΣΣ (zij - zi_bar)^2
//   Kruskal-Wallis: H = 12/(N(N+1)) Σ Ri^2/ni - 3(N+1)，秩为 1..N 的简单秩（不做并列校正）
// 输入需为偶数边长的方阵（128x128 CTU 满足）；非法输入返回 0。
#pragma once
#include "Matrix.h"

namespace sf {

double levene(const Mat& m);
double kruskalWallis(const Mat& m);

} // namespace sf
