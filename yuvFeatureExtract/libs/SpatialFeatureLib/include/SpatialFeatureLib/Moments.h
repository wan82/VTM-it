// Moments.h —— 一阶~四阶统计矩
// 公式严格对齐项目原有 w_cuProcess.cpp 的定义：
//   mean = Σx / N
//   std  = sqrt( E[x^2] - E[x]^2 )            （总体标准差）
//   skew = m3 / (sqrt(m2))^3                  （m2,m3 为总体中心矩，除以 N）
//   kurt = m4 / (m2^2)                        （原始峰度，非“超额峰度”）
#pragma once
#include "Matrix.h"

namespace sf {

double mean(const Mat& m);
double stddev(const Mat& m);
double skewness(const Mat& m);
double kurtosis(const Mat& m);

} // namespace sf
