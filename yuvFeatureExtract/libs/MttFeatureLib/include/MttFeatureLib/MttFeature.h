// MttFeature.h —— MTT 块级特征（②方向性 + ③划分代价代理 + ④形状/QP），纯函数、无第三方依赖
//
// 设计（与用户约定）：MTT 块级只用 ②③④；①复杂度/能量在 CTU 级单独算（见 mttFeature 编排）。
//   ② 方向性：分离的水平/垂直梯度能量等，管「横分 vs 竖分」。
//   ③ 划分代价代理：把块按候选划分切开比差异，直接模仿编码器在比的东西，判别力最强。
//   ④ 形状/上下文：长宽比、面积、按 POC 推的 QP-value。
//
// 输入块为 sf::Mat（vector<vector<int>>，10bit 亮度，行优先，可为任意 w×h 矩形）。
#pragma once

#include <string>
#include <vector>

#include "SpatialFeatureLib/Matrix.h"   // sf::Mat

namespace mf {

// ② 方向性特征
struct DirFeatures {
  double ex = 0;          // 水平梯度能量 mean|Gx|（竖向边多 → 大 → 倾向竖分 BTV）
  double ey = 0;          // 垂直梯度能量 mean|Gy|（横向边多 → 大 → 倾向横分 BTH）
  double ratio = 0;       // (ex - ey) / (ex + ey)，∈[-1,1]，>0 偏竖、<0 偏横
  double rowMeanVar = 0;  // 各行均值的方差（沿竖直方向内容变化 → 倾向横分）
  double colMeanVar = 0;  // 各列均值的方差（沿水平方向内容变化 → 倾向竖分）
};

// ③ 候选划分代价代理（差异越大越倾向该方向划分；均按像素归一、尺度可比）
struct SplitProxy {
  double tbDiff = 0;   // 上半 vs 下半 的均值差异（大 → BTH 横分）
  double lrDiff = 0;   // 左半 vs 右半 的均值差异（大 → BTV 竖分）
  double ttHDiff = 0;  // 1:2:1 水平三分（上/中/下）块间差异（大 → TTH）
  double ttVDiff = 0;  // 1:2:1 垂直三分（左/中/右）块间差异（大 → TTV）
};

// ④ 形状/上下文（不依赖像素，只看尺寸与 POC/QP）
struct ShapeFeatures {
  double aspectLog2 = 0; // log2(w/h)，>0 宽扁、<0 高瘦
  double areaLog2 = 0;   // log2(w*h)
  int    qpValue = 0;    // 按 VTM RA(GOP=32) 由 POC 与基础QP 推出的实际 slice QP
};

DirFeatures   directionalFeatures(const sf::Mat& blk);
SplitProxy    splitProxyFeatures(const sf::Mat& blk);
ShapeFeatures shapeFeatures(int w, int h, int poc, int baseQp);

// VTM RA(默认 GOP=32) 的 QP-value（与 yuvFeatureExtract/merger、解码器钩子完全一致）
int getQpValue(int poc, int baseQp);

// CSV 表头与一行字段（不含 keys/label，只含本库产出的特征列，顺序固定）
std::string featureHeader();   // "qpValue,aspectLog2,areaLog2,dirEx,...,ttVDiff"
std::string featureFields(const DirFeatures& d, const SplitProxy& s, const ShapeFeatures& sh);

} // namespace mf
