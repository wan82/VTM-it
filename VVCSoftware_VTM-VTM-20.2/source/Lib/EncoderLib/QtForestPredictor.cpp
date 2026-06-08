// ============================================================================
// #QTPRED  QtForestPredictor 实现（自定义新增文件）
// 二进制模型格式见 modelTrain/train.py 的 forest_block_bytes（小端，魔数 "QTF1"）。
// ============================================================================
#include "QtForestPredictor.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>

// ---------------- 空域特征（公式与 SpatialFeatureLib / 训练侧严格一致） ----------------
namespace
{
using Mat = std::vector<std::vector<int>>;

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
double matSkew(const Mat& m)
{
  double mu = matMean(m); long long n = 0; double m2 = 0, m3 = 0;
  for (auto& r : m) for (int v : r) { double d = v - mu; m2 += d * d; m3 += d * d * d; ++n; }
  if (n < 2) return 0.0; m2 /= n; m3 /= n;
  return m2 == 0.0 ? 0.0 : m3 / std::pow(std::sqrt(m2), 3.0);
}
double matKurt(const Mat& m)
{
  double mu = matMean(m); long long n = 0; double m2 = 0, m4 = 0;
  for (auto& r : m) for (int v : r) { double d = v - mu; m2 += d * d; m4 += d * d * d * d; ++n; }
  if (n < 2) return 0.0; m2 /= n; m4 /= n;
  return m2 == 0.0 ? 0.0 : m4 / (m2 * m2);
}
Mat sobel(const Mat& in)
{
  static const int Gx[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
  static const int Gy[3][3] = {{1,2,1},{0,0,0},{-1,-2,-1}};
  int H = (int)in.size(), W = H ? (int)in[0].size() : 0;
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
Mat laplacian(const Mat& in)
{
  static const int K[3][3] = {{0,1,0},{1,-4,1},{0,1,0}};
  int H = (int)in.size(), W = H ? (int)in[0].size() : 0;
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

// VTM RA(默认 GOP=32) 的 QP-value，与 yuvFeatureExtract/merger 完全一致
int gopToTid(int gop)
{
  if (gop == 0) return 0; if (gop == 16) return 1;
  if (gop == 8 || gop == 24) return 2;
  if (gop % 2 == 1) return 5; if (gop % 4 != 0) return 4; return 3;
}
int qpOffset(int baseQp, int tid)
{
  static const int o22[6]={-3,0,1,4,6,7}, o27[6]={-3,1,2,5,7,9},
                   o32[6]={-3,2,3,6,8,9}, o37[6]={-3,3,4,6,8,9};
  const int* t = baseQp==22?o22: baseQp==27?o27: baseQp==32?o32: baseQp==37?o37: nullptr;
  return (t && tid>=0 && tid<=5) ? t[tid] : 0;
}
int getQpValue(int poc, int baseQp) { return baseQp + qpOffset(baseQp, gopToTid(poc % 32)); }

// 小端二进制读取
int32_t rdI32(std::istream& f) { int32_t v=0; f.read((char*)&v,4); return v; }
double  rdF64(std::istream& f) { double  v=0; f.read((char*)&v,8); return v; }
std::string rdStr(std::istream& f) { int32_t n=rdI32(f); std::string s(n,'\0'); if(n>0) f.read(&s[0],n); return s; }
} // namespace

// ---------------- 单例 / 加载 ----------------
QtForestPredictor& QtForestPredictor::instance()
{
  static QtForestPredictor inst;
  return inst;
}

QtForestPredictor::QtForestPredictor()
{
  const char* env = std::getenv("QT_FOREST_MODEL");
  if (!env || !*env)
  {
    std::cerr << "[QTPRED] 未设置 QT_FOREST_MODEL，QT 预测早停关闭（编码行为同原版）\n";
    return;
  }
  // 置信度门控阈值 / 是否启用下界强制分（可用环境变量覆盖）
  if (const char* c = std::getenv("QT_FOREST_CONF"))  { try { m_confThreshold = std::stod(c); } catch (...) {} }
  if (const char* fEnv = std::getenv("QT_FOREST_FORCE")) { m_forceSplit = (std::string(fEnv) != "0"); }
  load(env);
}

void QtForestPredictor::load(const std::string& path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "[QTPRED] 打不开模型文件: " << path << "（已关闭）\n"; return; }
  char magic[4] = {0}; f.read(magic, 4);
  if (std::string(magic, 4) != "QTF1") { std::cerr << "[QTPRED] 模型魔数错误（已关闭）\n"; return; }

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
  if (!f.good()) { std::cerr << "[QTPRED] 模型读取不完整（已关闭）\n"; m_models.clear(); return; }
  m_enabled = true;
  std::cerr << "[QTPRED] 已加载 " << m_models.size() << " 个 QP 模型: " << path << "\n";
}

const QtForestPredictor::Model* QtForestPredictor::modelForQp(int baseQp) const
{
  for (auto& m : m_models) if (m.baseQp == baseQp) return &m;
  return nullptr;
}

// ---------------- 推理 ----------------
int QtForestPredictor::predictQtMax(const Mat& blk, int baseQp, int poc, double* conf) const
{
  if (conf) *conf = 0.0;
  if (!m_enabled) return -1;
  const Model* md = modelForQp(baseQp);
  if (!md || blk.empty() || blk[0].empty()) return -1;

  // 预计算可能用到的空域特征（按需，但 12 个矩/滤波特征都很便宜，一次算好放表里）
  std::map<std::string, double> fb;
  fb["mean"] = matMean(blk);  fb["std"] = matStd(blk);
  fb["skew"] = matSkew(blk);  fb["kurt"] = matKurt(blk);
  Mat sob = sobel(blk);
  fb["Sobel_mean"]=matMean(sob); fb["Sobel_std"]=matStd(sob);
  fb["Sobel_skew"]=matSkew(sob); fb["Sobel_kurt"]=matKurt(sob);
  Mat lap = laplacian(blk);
  fb["Lap_mean"]=matMean(lap); fb["Lap_std"]=matStd(lap);
  fb["Lap_skew"]=matSkew(lap); fb["Lap_kurt"]=matKurt(lap);

  // 按模型特征顺序组装输入向量
  std::vector<double> x;
  x.reserve(md->featNames.size());
  for (const std::string& name : md->featNames)
  {
    if (name.size() >= 3 && name.compare(name.size()-2, 2, "QP") == 0)
    {
      // QP-value 列，如 "22QP"
      int b = std::atoi(name.c_str());
      x.push_back((double)getQpValue(poc, b));
    }
    else
    {
      auto it = fb.find(name);
      if (it == fb.end()) { std::cerr << "[QTPRED] 不支持的特征: " << name << "\n"; return -1; }
      x.push_back(it->second);
    }
  }

  return predictClassFromRaw(x, baseQp, conf);
}

int QtForestPredictor::predictClassFromRaw(const std::vector<double>& x, int baseQp, double* conf) const
{
  if (conf) *conf = 0.0;
  if (!m_enabled) return -1;
  const Model* md = modelForQp(baseQp);
  if (!md || x.size() != md->featNames.size()) return -1;

  // 随机森林：累加各树叶子的类别概率，取 argmax（与 sklearn predict 一致）
  const int nc = (int)md->classes.size();
  std::vector<double> acc(nc, 0.0);
  for (const Tree& tr : md->trees)
  {
    int node = 0;
    while (tr.left[node] != -1)            // 非叶子
    {
      int fidx = tr.feat[node];
      node = (x[fidx] <= tr.thr[node]) ? tr.left[node] : tr.right[node];
    }
    for (int c = 0; c < nc; ++c) acc[c] += tr.prob[node][c];
  }
  int best = 0; double total = 0.0;
  for (int c = 0; c < nc; ++c) { total += acc[c]; if (acc[c] > acc[best]) best = c; }
  if (conf && total > 0.0) *conf = acc[best] / total;   // 预测类别的概率(置信度)
  return md->classes[best];
}
