// CtuFeatures.h —— 把一个 128x128 CTU 块算成一条与 yuvData/128csv 对齐的特征
// 列顺序严格对齐现有 128csv 表头中的 18 个特征：
//   mean,std,skew,kurt,
//   Sobel_mean,Sobel_std,Sobel_skew,Sobel_kurt,
//   Lap_mean,Lap_std,Lap_skew,Lap_kurt,
//   LeveneTest,KrusWTest,SobelLevene,SobelKrusW,LapLevene,LapKrusW
#pragma once
#include "Matrix.h"
#include <array>
#include <string>

namespace sf {

struct CtuFeatures {
    double mean = 0, std = 0, skew = 0, kurt = 0;
    double sobelMean = 0, sobelStd = 0, sobelSkew = 0, sobelKurt = 0;
    double lapMean = 0, lapStd = 0, lapSkew = 0, lapKurt = 0;
    double levene = 0, krusW = 0;
    double sobelLevene = 0, sobelKrusW = 0;
    double lapLevene = 0, lapKrusW = 0;

    // 18 个特征对应的 CSV 表头（不含 POC_ID/pos_x/pos_y）
    static std::string csvHeader();
    // 按表头顺序拼成 CSV 字段（不含位置列）
    std::string toCsvFields() const;
};

// 对一个灰度 CTU 块计算全部 18 个空域特征
CtuFeatures computeCtuFeatures(const Mat& block);

} // namespace sf
