// ============================================================================
// #MTTPRED  MttForestPredictor 实现（自定义新增文件）
// 特征公式严格对齐 yuvFeatureExtract 的 MttFeatureLib(②③④) 与 SpatialFeatureLib(①CTU)。
// 二进制模型格式见 modelTrain/trainMtt.py（小端，魔数 "QTF1"）。
// ============================================================================
#include "MttForestPredictor.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

namespace
{
using Mat = std::vector<std::vector<int>>;

int matRows(const Mat& m) { return (int)m.size(); }
int matCols(const Mat& m) { return m.empty() ? 0 : (int)m[0].size(); }

// ---- ① CTU 用：与 SpatialFeatureLib / QtForestPredictor 一致 ----
double matMean(const Mat& m)
{
  double s = 0.0; long long n = 0;
  for (auto& r : m) for (int v : r) { s += v; ++n; }
  return n ? s / n : 0.0;
}
double matStd(const Mat& m)
{
  double s = 0.0, sq = 0.0; long long n = 0;
  for (auto& r : m) for (int v : r) { s += v; sq += double(v) * v; ++n; }
  if (!n) return 0.0;
  double mu = s / n, var = sq / n - mu * mu;
  return std::sqrt(var < 0.0 ? 0.0 : var);
}
Mat sobelMag(const Mat& in)
{
  static const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  static const int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
  int H = matRows(in), W = matCols(in);
  Mat out(H, std::vector<int>(W, 0));
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x)
    {
      int sx = 0, sy = 0;
      for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx)
      { int p = in[y+ky][x+kx]; sx += p*Gx[ky+1][kx+1]; sy += p*Gy[ky+1][kx+1]; }
      out[y][x] = (int)std::lround(std::sqrt(double(sx)*sx + double(sy)*sy));
    }
  return out;
}
Mat laplacianAbs(const Mat& in)
{
  static const int K[3][3] = {{0,1,0},{1,-4,1},{0,1,0}};
  int H = matRows(in), W = matCols(in);
  Mat out(H, std::vector<int>(W, 0));
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x)
    {
      int acc = 0;
      for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx)
        acc += in[y+ky][x+kx] * K[ky+1][kx+1];
      out[y][x] = std::abs(acc);
    }
  return out;
}

// ---- 子矩形 [y0,y1)×[x0,x1) 像素均值（③用）----
double meanRect(const Mat& m, int x0, int y0, int x1, int y1)
{
  if (x1 <= x0 || y1 <= y0) return 0.0;
  double s = 0.0; long long n = 0;
  for (int y = y0; y < y1; ++y) for (int x = x0; x < x1; ++x) { s += m[y][x]; ++n; }
  return n ? s / n : 0.0;
}

// VTM RA(GOP=32) 的 QP-value（与训练/解码一致）
int gopToTid(int gop)
{
  if (gop == 0) return 0; if (gop == 16) return 1;
  if (gop == 8 || gop == 24) return 2;
  if (gop % 2 == 1) return 5; if (gop % 4 != 0) return 4; return 3;
}
int getQpValue(int poc, int baseQp)
{
  static const int o22[6]={-3,0,1,4,6,7}, o27[6]={-3,1,2,5,7,9},
                   o32[6]={-3,2,3,6,8,9}, o37[6]={-3,3,4,6,8,9};
  const int tid = gopToTid(((poc % 32) + 32) % 32);
  const int* t = baseQp==22?o22: baseQp==27?o27: baseQp==32?o32: baseQp==37?o37: nullptr;
  return baseQp + ((t && tid>=0 && tid<=5) ? t[tid] : 0);
}

// 小端读取
int32_t rdI32(std::istream& f) { int32_t v=0; f.read((char*)&v,4); return v; }
double  rdF64(std::istream& f) { double  v=0; f.read((char*)&v,8); return v; }
std::string rdStr(std::istream& f) { int32_t n=rdI32(f); std::string s(n,'\0'); if(n>0) f.read(&s[0],n); return s; }
} // namespace

// ---------------- 单例 / 加载 ----------------
MttForestPredictor& MttForestPredictor::instance()
{
  static MttForestPredictor inst;
  return inst;
}

MttForestPredictor::MttForestPredictor()
{
  const char* env = std::getenv("MTT_FOREST_MODEL");
  if (!env || !*env)
  {
    std::cerr << "[MTTPRED] 未设置 MTT_FOREST_MODEL，MTT 预测剪枝关闭（行为同原版）\n";
    return;
  }
  if (const char* c = std::getenv("MTT_FOREST_CONF"))    { try { m_confThreshold = std::stod(c); } catch (...) {} }
  if (const char* n = std::getenv("MTT_FOREST_PRUNE_NO")) { m_pruneNo = (std::string(n) != "0"); }
  if (const char* p = std::getenv("MTT_FOREST_PRUNE"))   { m_pruneExact = (std::string(p) == "exact"); }
  load(env);
}

void MttForestPredictor::load(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "[MTTPRED] 打不开模型文件: " << path << "（已关闭）\n"; return; }
  char magic[4] = {0}; f.read(magic, 4);
  if (std::string(magic, 4) != "QTF1") { std::cerr << "[MTTPRED] 模型魔数错误（已关闭）\n"; return; }

  int nModels = rdI32(f);
  for (int mi = 0; mi < nModels; ++mi)
  {
    Model md;
    md.baseQp = rdI32(f);
    int nf = rdI32(f);
    for (int i = 0; i < nf; ++i) md.featNames.push_back(rdStr(f));
    int nc = rdI32(f);
    for (int i = 0; i < nc; ++i) md.classes.push_back(rdI32(f));
    int nt = rdI32(f);
    md.trees.resize(nt);
    for (int ti = 0; ti < nt; ++ti)
    {
      Tree& tr = md.trees[ti];
      int nn = rdI32(f);
      tr.left.resize(nn); tr.right.resize(nn); tr.feat.resize(nn);
      tr.thr.resize(nn); tr.prob.assign(nn, std::vector<double>(nc, 0.0));
      for (int n = 0; n < nn; ++n)
      { tr.left[n]=rdI32(f); tr.right[n]=rdI32(f); tr.feat[n]=rdI32(f); tr.thr[n]=rdF64(f); }
      for (int n = 0; n < nn; ++n)
        for (int c = 0; c < nc; ++c) tr.prob[n][c] = rdF64(f);
    }
    m_models.push_back(std::move(md));
  }
  if (!f.good()) { std::cerr << "[MTTPRED] 模型读取不完整（已关闭）\n"; m_models.clear(); return; }
  m_enabled = true;
  std::cerr << "[MTTPRED] 已加载 " << m_models.size() << " 个 QP 模型: " << path
            << " | conf=" << m_confThreshold << " pruneNo=" << m_pruneNo
            << " mode=" << (m_pruneExact ? "exact" : "orient") << "\n";
}

const MttForestPredictor::Model* MttForestPredictor::modelForQp(int baseQp) const
{
  for (auto& m : m_models) if (m.baseQp == baseQp) return &m;
  return nullptr;
}

// ---------------- ① CTU 上下文 ----------------
void MttForestPredictor::computeCtuCtx(const Mat& ctu, double out6[6]) const
{
  out6[0] = matMean(ctu);
  out6[1] = matStd(ctu);
  Mat lap = laplacianAbs(ctu); out6[2] = matMean(lap); out6[3] = matStd(lap);
  Mat sob = sobelMag(ctu);     out6[4] = matMean(sob); out6[5] = matStd(sob);
}

// ---------------- 特征组装 ----------------
std::vector<double> MttForestPredictor::featuresFor(const Mat& blk, const double ctuCtx6[6],
                                                    int w, int h, int qtDepth, int mttDepth,
                                                    int poc, int baseQp) const
{
  const Model* md = modelForQp(baseQp);
  if (!md || blk.empty() || blk[0].empty()) return {};

  const int H = matRows(blk), W = matCols(blk);

  // ② 方向性：分离 Sobel Gx/Gy 能量 + 行/列均值方差
  static const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  static const int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
  double sx = 0.0, sy = 0.0; long long ng = 0;
  for (int y = 1; y < H - 1; ++y)
    for (int x = 1; x < W - 1; ++x)
    {
      int gx = 0, gy = 0;
      for (int ky = -1; ky <= 1; ++ky) for (int kx = -1; kx <= 1; ++kx)
      { int p = blk[y+ky][x+kx]; gx += p*Gx[ky+1][kx+1]; gy += p*Gy[ky+1][kx+1]; }
      sx += std::abs(gx); sy += std::abs(gy); ++ng;
    }
  double dirEx = ng ? sx / ng : 0.0, dirEy = ng ? sy / ng : 0.0;
  double dirRatio = (dirEx + dirEy) > 0.0 ? (dirEx - dirEy) / (dirEx + dirEy) : 0.0;
  double rowMeanVar = 0.0, colMeanVar = 0.0;
  if (H > 0) { double sm=0,sm2=0; for (int y=0;y<H;++y){ double rm=meanRect(blk,0,y,W,y+1); sm+=rm; sm2+=rm*rm; }
               double mu=sm/H; rowMeanVar=sm2/H-mu*mu; if(rowMeanVar<0)rowMeanVar=0; }
  if (W > 0) { double sm=0,sm2=0; for (int x=0;x<W;++x){ double cm=meanRect(blk,x,0,x+1,H); sm+=cm; sm2+=cm*cm; }
               double mu=sm/W; colMeanVar=sm2/W-mu*mu; if(colMeanVar<0)colMeanVar=0; }

  // ③ 划分代价代理
  double tbDiff=0,lrDiff=0,ttHDiff=0,ttVDiff=0;
  if (H >= 2) tbDiff = std::abs(meanRect(blk,0,0,W,H/2) - meanRect(blk,0,H/2,W,H));
  if (W >= 2) lrDiff = std::abs(meanRect(blk,0,0,W/2,H) - meanRect(blk,W/2,0,W,H));
  if (H >= 4) { double a=meanRect(blk,0,0,W,H/4), b=meanRect(blk,0,H/4,W,H-H/4), c=meanRect(blk,0,H-H/4,W,H);
                ttHDiff=(std::abs(a-b)+std::abs(b-c)+std::abs(a-c))/3.0; }
  if (W >= 4) { double a=meanRect(blk,0,0,W/4,H), b=meanRect(blk,W/4,0,W-W/4,H), c=meanRect(blk,W-W/4,0,W,H);
                ttVDiff=(std::abs(a-b)+std::abs(b-c)+std::abs(a-c))/3.0; }

  // ④ 形状/QP
  double aspectLog2 = (w>0&&h>0) ? std::log2((double)w/(double)h) : 0.0;
  double areaLog2   = (w>0&&h>0) ? std::log2((double)w*(double)h) : 0.0;
  int    qpValue    = getQpValue(poc, baseQp);

  std::map<std::string, double> fb;
  fb["w"]=w; fb["h"]=h; fb["qtDepth"]=qtDepth; fb["mttDepth"]=mttDepth;
  fb["qpValue"]=qpValue; fb["aspectLog2"]=aspectLog2; fb["areaLog2"]=areaLog2;
  fb["dirEx"]=dirEx; fb["dirEy"]=dirEy; fb["dirRatio"]=dirRatio;
  fb["rowMeanVar"]=rowMeanVar; fb["colMeanVar"]=colMeanVar;
  fb["tbDiff"]=tbDiff; fb["lrDiff"]=lrDiff; fb["ttHDiff"]=ttHDiff; fb["ttVDiff"]=ttVDiff;
  fb["ctuMean"]=ctuCtx6[0]; fb["ctuStd"]=ctuCtx6[1];
  fb["ctuLapMean"]=ctuCtx6[2]; fb["ctuLapStd"]=ctuCtx6[3];
  fb["ctuSobelMean"]=ctuCtx6[4]; fb["ctuSobelStd"]=ctuCtx6[5];

  std::vector<double> x;
  x.reserve(md->featNames.size());
  for (const std::string& name : md->featNames)
  {
    auto it = fb.find(name);
    if (it == fb.end()) { std::cerr << "[MTTPRED] 不支持的特征: " << name << "\n"; return {}; }
    x.push_back(it->second);
  }
  return x;
}

int MttForestPredictor::predictSplit(const Mat& blk, const double ctuCtx6[6], int w, int h,
                                     int qtDepth, int mttDepth, int poc, int baseQp,
                                     double* conf) const
{
  if (conf) *conf = 0.0;
  if (!m_enabled) return -1;
  std::vector<double> x = featuresFor(blk, ctuCtx6, w, h, qtDepth, mttDepth, poc, baseQp);
  if (x.empty()) return -1;
  return predictFromRaw(x, baseQp, conf);
}

int MttForestPredictor::predictFromRaw(const std::vector<double>& x, int baseQp, double* conf) const
{
  if (conf) *conf = 0.0;
  if (!m_enabled) return -1;
  const Model* md = modelForQp(baseQp);
  if (!md || x.size() != md->featNames.size()) return -1;

  const int nc = (int)md->classes.size();
  std::vector<double> acc(nc, 0.0);
  for (const Tree& tr : md->trees)
  {
    int node = 0;
    while (tr.left[node] != -1)
    {
      int fidx = tr.feat[node];
      node = (x[fidx] <= tr.thr[node]) ? tr.left[node] : tr.right[node];
    }
    for (int c = 0; c < nc; ++c) acc[c] += tr.prob[node][c];
  }
  int best = 0; double total = 0.0;
  for (int c = 0; c < nc; ++c) { total += acc[c]; if (acc[c] > acc[best]) best = c; }
  if (conf && total > 0.0) *conf = acc[best] / total;
  return md->classes[best];
}
