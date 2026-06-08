// Filters.h —— Sobel 与 Laplacian 滤波
//   Sobel : 3x3 Gx/Gy 核，输出 = sqrt(Gx^2 + Gy^2)，边界像素置 0
//           （与 w_cuProcess.cpp 的 sobelFilter 完全一致）
//   Laplacian: 标准 4-邻域 3x3 核 {0,1,0; 1,-4,1; 0,1,0}，输出取绝对值，边界置 0
//           （原 w_cuProcess 无 Laplacian，此处采用标准核；若需与旧 128csv 的
//            生成脚本逐值对齐，请提供旧脚本的核定义，我再调整。）
#pragma once
#include "Matrix.h"

namespace sf {

Mat sobel(const Mat& input);
Mat laplacian(const Mat& input);

} // namespace sf
