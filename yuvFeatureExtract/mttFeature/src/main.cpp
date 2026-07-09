// mttFeatureBuilder —— 给采样到的 MTT 块配空域特征，做成 TrainingMtt.csv
//
// 输入：Data/labelMtt/<seq>_QP<qp>.csv（解码采样得到的 MTT 划分标签）
//       + 各序列源 YUV（信息取自 collector 的 config.yaml：path/width/height/bitdepth）
// 处理：对每个标签块(poc,x,y,w,h)裁出亮度块，算 ②方向性 + ③划分代价代理 + ④形状/QP；
//       再算该块所在 128×128 CTU 的 ①复杂度/能量 作为上下文（每 CTU 缓存一次）。
// 输出：Data/TrainingData/TrainingMtt.csv —— labelMtt 原列 + 追加特征列。
//
// 用法：mttFeatureBuilder [config.yaml] [--labels-dir DIR] [--out FILE]
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "MiniYaml/MiniYaml.h"
#include "YuvGrayLib/YuvReader.h"
#include "YuvGrayLib/GrayFrame.h"
#include "SpatialFeatureLib/Matrix.h"
#include "SpatialFeatureLib/Moments.h"
#include "SpatialFeatureLib/Filters.h"
#include "MttFeatureLib/MttFeature.h"

namespace fs = std::filesystem;

namespace {

struct SeqCfg {
  std::string name, path;
  int width = 0, height = 0, bitdepth = 8;
  std::string chroma = "420";
  int framerate = 0;       // 用于推 IntraPeriod
  int intraPeriod = 0;     // 显式 intra_period（>0 时优先）
};

// RA GOP 的 L0[0] 参考 POC = 去掉最低置位比特（二叉左父），与 collector/解码器口径一致
inline int l0PocOf(int poc) { return poc - (poc & (-poc)); }

// IntraPeriod：显式 intra_period > 按帧率取 ≥framerate 的最小 gopSize 倍数 > gopSize
inline int deriveIntraPeriod(const SeqCfg& s, int gopSize) {
  if (s.intraPeriod > 0) return s.intraPeriod;
  if (s.framerate > 0 && gopSize > 0)
    return ((s.framerate + gopSize - 1) / gopSize) * gopSize;
  return gopSize > 0 ? gopSize : 1;
}

// 取一行 CSV 的第 i 个字段（0 起）
std::string field(const std::string& line, int idx) {
  int cur = 0; size_t start = 0;
  for (size_t i = 0; i <= line.size(); ++i) {
    if (i == line.size() || line[i] == ',') {
      if (cur == idx) return line.substr(start, i - start);
      ++cur; start = i + 1;
    }
  }
  return "";
}
int toInt(const std::string& s) { try { return std::stoi(s); } catch (...) { return 0; } }

// 在帧上裁出 [y0,y0+h) × [x0,x0+w) 的亮度块
sf::Mat cropBlock(const yg::GrayFrame& fr, int x0, int y0, int w, int h) {
  sf::Mat m(h, std::vector<int>(w, 0));
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) m[y][x] = (int)fr.at(x0 + x, y0 + y);
  return m;
}

// ① CTU 级复杂度/能量（6 个）：mean,std,lapMean,lapStd,sobelMean,sobelStd
struct CtuCtx { double mean=0,std=0,lapMean=0,lapStd=0,sobelMean=0,sobelStd=0; };
CtuCtx computeCtuCtx(const sf::Mat& ctu) {
  CtuCtx c;
  c.mean = sf::mean(ctu); c.std = sf::stddev(ctu);
  sf::Mat lap = sf::laplacian(ctu); c.lapMean = sf::mean(lap); c.lapStd = sf::stddev(lap);
  sf::Mat sob = sf::sobel(ctu);     c.sobelMean = sf::mean(sob); c.sobelStd = sf::stddev(sob);
  return c;
}

// 残差特征列：在残差块 D=(当前块−L0同位块) 上算
//   · 矩（能量）：res_mean/std/skew/kurt —— 残差大小→是否/多深划分
//   · 方向性②：res_dirEx/Ey/Ratio/rowMeanVar/colMeanVar —— 残差边的朝向→横/竖
//   · 划分代理③：res_tbDiff/lrDiff/ttHDiff/ttVDiff —— 残差能量落在哪条切线→BT/TT（最关键）
std::string residualHeader() {
  return "res_mean,res_std,res_skew,res_kurt,"
         "res_dirEx,res_dirEy,res_dirRatio,res_rowMeanVar,res_colMeanVar,"
         "res_tbDiff,res_lrDiff,res_ttHDiff,res_ttVDiff";
}
std::string residualFields(const sf::Mat& cur, const sf::Mat& ref) {
  const int H = (int)cur.size(), W = H ? (int)cur[0].size() : 0;
  sf::Mat d(H, std::vector<int>(W, 0));
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) d[y][x] = cur[y][x] - ref[y][x];
  mf::DirFeatures dir = mf::directionalFeatures(d);
  mf::SplitProxy  sp  = mf::splitProxyFeatures(d);
  std::ostringstream o;
  o << sf::mean(d) << ',' << sf::stddev(d) << ',' << sf::skewness(d) << ',' << sf::kurtosis(d) << ','
    << dir.ex << ',' << dir.ey << ',' << dir.ratio << ',' << dir.rowMeanVar << ',' << dir.colMeanVar << ','
    << sp.tbDiff << ',' << sp.lrDiff << ',' << sp.ttHDiff << ',' << sp.ttVDiff;
  return o.str();
}

} // namespace

int main(int argc, char** argv) {
  std::string cfgPath = "collector/config/config.yaml";
  std::string labelsDir = "../Data/labelMtt";
  std::string outPath = "../Data/TrainingData/TrainingMtt.csv";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto nx = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
    if      (a == "--labels-dir") labelsDir = nx();
    else if (a == "--out")        outPath = nx();
    else if (a == "--config")     cfgPath = nx();
    else if (!a.empty() && a[0] != '-') cfgPath = a;
  }

  myaml::Document doc = myaml::parseFile(cfgPath);
  if (!doc.ok) { std::cerr << "配置解析失败: " << doc.error << " (" << cfgPath << ")\n"; return 1; }
  const int internalBd = myaml::getInt(doc.globals, "internal_bitdepth", 10);
  const int gopSize = myaml::getInt(doc.globals, "gop_size", 32);
  const bool collectResidual = myaml::getBool(doc.globals, "collect_residual", true);

  std::vector<SeqCfg> seqs;
  for (const auto& s : doc.sequences) {
    SeqCfg c;
    c.name = myaml::getStr(s, "name");
    c.path = myaml::getStr(s, "path");
    c.width = myaml::getInt(s, "width", 0);
    c.height = myaml::getInt(s, "height", 0);
    c.bitdepth = myaml::getInt(s, "bitdepth", 8);
    c.chroma = myaml::getStr(s, "chroma_format", myaml::getStr(s, "chroma", "420"));
    c.framerate = myaml::getInt(s, "framerate", 0);
    c.intraPeriod = myaml::getInt(s, "intra_period", 0);
    if (!c.name.empty()) seqs.push_back(c);
  }
  if (seqs.empty()) { std::cerr << "配置里没有 sequences\n"; return 1; }

  fs::create_directories(fs::path(outPath).parent_path());
  std::ofstream out(outPath, std::ios::out | std::ios::trunc);
  if (!out) { std::cerr << "打不开输出: " << outPath << "\n"; return 1; }
  // 表头：labelMtt 原列 + 特征列 + CTU 上下文列
  out << "sequence,POC_ID,pos_x,pos_y,w,h,qtDepth,mttDepth,splitMode,QP,"
      << mf::featureHeader() << ","
      << "ctuMean,ctuStd,ctuLapMean,ctuLapStd,ctuSobelMean,ctuSobelStd";
  if (collectResidual) out << "," << residualHeader();
  out << "\n";

  long grand = 0; int nSeqUsed = 0, nMissing = 0;
  for (const auto& sc : seqs) {
    // 该序列的 4 个 QP 标签文件
    std::vector<std::string> files;
    for (int qp : {22, 27, 32, 37}) {
      std::string f = labelsDir + "/" + sc.name + "_QP" + std::to_string(qp) + ".csv";
      if (fs::exists(f)) files.push_back(f);
    }
    if (files.empty()) continue;
    if (sc.path.empty() || !fs::exists(sc.path)) {
      std::cout << "[" << sc.name << "] 跳过（缺 YUV: " << sc.path << "）\n"; ++nMissing; continue;
    }

    yg::YuvReader reader(sc.path, sc.width, sc.height, sc.bitdepth,
                         yg::parseChromaFormat(sc.chroma), internalBd);
    if (!reader.ok()) { std::cout << "[" << sc.name << "] YUV 打开失败\n"; ++nMissing; continue; }

    std::unordered_map<int, yg::GrayFrame> frameCache;      // poc -> 亮度帧
    std::unordered_map<uint64_t, CtuCtx>  ctuCache;          // (poc,cx,cy) -> CTU 上下文
    const int intraPeriod = deriveIntraPeriod(sc, gopSize);  // 用于排除 I 帧（无 L0）
    long seqRows = 0;

    for (const auto& f : files) {
      std::ifstream in(f);
      std::string line; std::getline(in, line);             // 跳表头
      while (std::getline(in, line)) {
        if (line.empty()) continue;
        const int poc = toInt(field(line, 1));
        const int x = toInt(field(line, 2)), y = toInt(field(line, 3));
        const int w = toInt(field(line, 4)), h = toInt(field(line, 5));
        const int baseQp = toInt(field(line, 9));
        if (w <= 0 || h <= 0) continue;
        if (x + w > sc.width || y + h > sc.height) continue; // 越界保护（理论上已被排除）
        if (collectResidual && intraPeriod > 0 && (poc % intraPeriod) == 0)
          continue;                                          // I 帧无 L0，整行丢弃

        // 取帧（缓存）
        auto it = frameCache.find(poc);
        if (it == frameCache.end()) {
          yg::GrayFrame fr;
          if (!reader.readLumaFrame(poc, fr)) continue;       // 该 POC 读不到就跳过
          it = frameCache.emplace(poc, std::move(fr)).first;
        }
        const yg::GrayFrame& fr = it->second;

        // 块级 ②③④
        sf::Mat blk = cropBlock(fr, x, y, w, h);
        mf::DirFeatures   d  = mf::directionalFeatures(blk);
        mf::SplitProxy    sp = mf::splitProxyFeatures(blk);
        mf::ShapeFeatures sh = mf::shapeFeatures(w, h, poc, baseQp);

        // CTU 级 ①（缓存：同 CTU 多个块共用）
        const int cx0 = (x / 128) * 128, cy0 = (y / 128) * 128;
        const int cw = std::min(128, sc.width - cx0), ch = std::min(128, sc.height - cy0);
        uint64_t key = (uint64_t(uint32_t(poc)) << 32) ^ (uint64_t(cx0) << 16) ^ uint64_t(cy0);
        auto ci = ctuCache.find(key);
        if (ci == ctuCache.end()) {
          sf::Mat ctu = cropBlock(fr, cx0, cy0, cw, ch);
          ci = ctuCache.emplace(key, computeCtuCtx(ctu)).first;
        }
        const CtuCtx& c = ci->second;

        out << line << ',' << mf::featureFields(d, sp, sh) << ','
            << c.mean << ',' << c.std << ',' << c.lapMean << ',' << c.lapStd << ','
            << c.sobelMean << ',' << c.sobelStd;
        if (collectResidual) {
          // L0 同位块（缓存 L0 帧）；残差 D = 当前块 − L0块
          const int l0 = l0PocOf(poc);
          auto li = frameCache.find(l0);
          if (li == frameCache.end()) {
            yg::GrayFrame lf;
            if (reader.readLumaFrame(l0, lf))
              li = frameCache.emplace(l0, std::move(lf)).first;
          }
          if (li != frameCache.end()) {
            sf::Mat refBlk = cropBlock(li->second, x, y, w, h);
            out << ',' << residualFields(blk, refBlk);
          } else {
            // 读不到 L0（极少数边界），补 0 占位，保持列数一致
            out << ",0,0,0,0,0,0,0,0,0,0,0,0,0";
          }
        }
        out << '\n';
        ++seqRows;
      }
    }
    std::cout << "[" << sc.name << "] " << seqRows << " 行（" << files.size() << " 个QP文件）\n";
    grand += seqRows; ++nSeqUsed;
  }

  std::cout << "\n完成。" << nSeqUsed << " 个序列，共 " << grand << " 行 -> " << outPath
            << (nMissing ? ("（" + std::to_string(nMissing) + " 个序列缺YUV/标签已跳过）") : "") << "\n";
  return 0;
}
