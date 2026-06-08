// Matrix.h —— 二维灰度矩阵类型别名（10bit 灰度，整型存储）
#pragma once
#include <vector>

namespace sf {

// 一个 CTU 块 / 滤波结果图都用这个二维矩阵表示。
// 元素为 int（足以容纳 10bit 灰度 0..1023，以及 Sobel/Laplacian 的中间结果）。
using Mat = std::vector<std::vector<int>>;

} // namespace sf
