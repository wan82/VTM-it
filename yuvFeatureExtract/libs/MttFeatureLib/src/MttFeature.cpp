#include "MttFeatureLib/MttFeature.h"

#include <cmath>
#include <sstream>

namespace mf {

namespace {

int rows(const sf::Mat& m) { return (int)m.size(); }
int cols(const sf::Mat& m) { return m.empty() ? 0 : (int)m[0].size(); }

// 子矩形 [y0,y1) × [x0,x1) 的像素均值
double meanRect(const sf::Mat& m, int x0, int y0, int x1, int y1) {
  if (x1 <= x0 || y1 <= y0) return 0.0;
  double s = 0.0; long long n = 0;
  for (int y = y0; y < y1; ++y)
    for (int x = x0; x < x1; ++x) { s += m[y][x]; ++n; }
  return n ? s / n : 0.0;
}

} // namespace

DirFeatures directionalFeatures(const sf::Mat& blk) {
  DirFeatures f;
  const int H = rows(blk), W = cols(blk);
  if (H == 0 || W == 0) return f;

  // 分离的 Sobel 水平/垂直梯度能量（内部像素，边界跳过）
  static const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  static const int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
  double sx = 0.0, sy = 0.0; long long n = 0;
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x) {
      int gx = 0, gy = 0;
      for (int ky = -1; ky <= 1; ++ky)
        for (int kx = -1; kx <= 1; ++kx) {
          int p = blk[y + ky][x + kx];
          gx += p * Gx[ky + 1][kx + 1];
          gy += p * Gy[ky + 1][kx + 1];
        }
      sx += std::abs(gx); sy += std::abs(gy); ++n;
    }
  if (n) { f.ex = sx / n; f.ey = sy / n; }
  const double tot = f.ex + f.ey;
  f.ratio = tot > 0.0 ? (f.ex - f.ey) / tot : 0.0;

  // 行均值的方差 / 列均值的方差
  if (H > 0) {
    double sm = 0, sm2 = 0;
    for (int y = 0; y < H; ++y) { double rm = meanRect(blk, 0, y, W, y + 1); sm += rm; sm2 += rm * rm; }
    double mu = sm / H; f.rowMeanVar = sm2 / H - mu * mu; if (f.rowMeanVar < 0) f.rowMeanVar = 0;
  }
  if (W > 0) {
    double sm = 0, sm2 = 0;
    for (int x = 0; x < W; ++x) { double cm = meanRect(blk, x, 0, x + 1, H); sm += cm; sm2 += cm * cm; }
    double mu = sm / W; f.colMeanVar = sm2 / W - mu * mu; if (f.colMeanVar < 0) f.colMeanVar = 0;
  }
  return f;
}

SplitProxy splitProxyFeatures(const sf::Mat& blk) {
  SplitProxy s;
  const int H = rows(blk), W = cols(blk);
  if (H == 0 || W == 0) return s;

  // 上下两半（→ BTH）
  if (H >= 2) {
    double top = meanRect(blk, 0, 0, W, H / 2);
    double bot = meanRect(blk, 0, H / 2, W, H);
    s.tbDiff = std::abs(top - bot);
  }
  // 左右两半（→ BTV）
  if (W >= 2) {
    double l = meanRect(blk, 0, 0, W / 2, H);
    double r = meanRect(blk, W / 2, 0, W, H);
    s.lrDiff = std::abs(l - r);
  }
  // 1:2:1 水平三分 上/中/下（→ TTH）
  if (H >= 4) {
    double a = meanRect(blk, 0, 0,        W, H / 4);
    double b = meanRect(blk, 0, H / 4,    W, H - H / 4);
    double c = meanRect(blk, 0, H - H / 4, W, H);
    s.ttHDiff = (std::abs(a - b) + std::abs(b - c) + std::abs(a - c)) / 3.0;
  }
  // 1:2:1 垂直三分 左/中/右（→ TTV）
  if (W >= 4) {
    double a = meanRect(blk, 0,         0, W / 4,     H);
    double b = meanRect(blk, W / 4,     0, W - W / 4, H);
    double c = meanRect(blk, W - W / 4, 0, W,         H);
    s.ttVDiff = (std::abs(a - b) + std::abs(b - c) + std::abs(a - c)) / 3.0;
  }
  return s;
}

// ---- VTM RA(GOP=32) QP-value（与解码器/合并器一致）----
int getQpValue(int poc, int baseQp) {
  auto gopToTid = [](int gop) -> int {
    if (gop == 0) return 0; if (gop == 16) return 1;
    if (gop == 8 || gop == 24) return 2;
    if (gop % 2 == 1) return 5; if (gop % 4 != 0) return 4; return 3;
  };
  static const int o22[6] = {-3,0,1,4,6,7}, o27[6] = {-3,1,2,5,7,9},
                   o32[6] = {-3,2,3,6,8,9}, o37[6] = {-3,3,4,6,8,9};
  const int tid = gopToTid(((poc % 32) + 32) % 32);
  const int* t = baseQp==22?o22 : baseQp==27?o27 : baseQp==32?o32 : baseQp==37?o37 : nullptr;
  const int off = (t && tid >= 0 && tid <= 5) ? t[tid] : 0;
  return baseQp + off;
}

ShapeFeatures shapeFeatures(int w, int h, int poc, int baseQp) {
  ShapeFeatures f;
  f.aspectLog2 = (w > 0 && h > 0) ? std::log2((double)w / (double)h) : 0.0;
  f.areaLog2   = (w > 0 && h > 0) ? std::log2((double)w * (double)h) : 0.0;
  f.qpValue    = getQpValue(poc, baseQp);
  return f;
}

std::string featureHeader() {
  return "qpValue,aspectLog2,areaLog2,"
         "dirEx,dirEy,dirRatio,rowMeanVar,colMeanVar,"
         "tbDiff,lrDiff,ttHDiff,ttVDiff";
}

std::string featureFields(const DirFeatures& d, const SplitProxy& s, const ShapeFeatures& sh) {
  std::ostringstream o;
  o << sh.qpValue << ',' << sh.aspectLog2 << ',' << sh.areaLog2 << ','
    << d.ex << ',' << d.ey << ',' << d.ratio << ',' << d.rowMeanVar << ',' << d.colMeanVar << ','
    << s.tbDiff << ',' << s.lrDiff << ',' << s.ttHDiff << ',' << s.ttVDiff;
  return o.str();
}

} // namespace mf
