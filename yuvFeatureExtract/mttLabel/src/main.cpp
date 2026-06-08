// mttLabelBuilder —— MTT 标签编排器
//
// 解码 binEncode202 的 RA 码流，按「随机 POC 子集 × 随机 block 比例」采样导出 MTT 划分标签。
// 熵解码本身必须走 VTM 解码器（DecoderApp + #MTTDUMP 钩子）；本程序只负责：
//   ① 读 mtt.yaml（序列/帧数/QP/采样参数/解码器路径）
//   ② 为每个序列挑随机 POC（同序列各 QP 用同一组 POC，便于对照）
//   ③ 给 DecoderApp 设好 MTT_DUMP_* 环境变量并驱动它，对每个 (序列,QP) 落一个 CSV
//   ④ 汇总每个文件写出的行数
//
// 输出：<out_dir>/<seq>_QP<qp>.csv，列：
//   sequence,POC_ID,pos_x,pos_y,w,h,qtDepth,mttDepth,splitMode,QP
//
// 用法：
//   mttLabelBuilder [config/mtt.yaml] [--decoder PATH] [--bin-dir DIR] [--out-dir DIR] [--dry-run]
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "MiniYaml/MiniYaml.h"
#include "MttSampleLib/MttSample.h"

namespace fs = std::filesystem;

namespace {

struct Cfg {
  std::string binDir   = "Data/binEncode202";   // 相对 cwd；通常由 Makefile --bin-dir 覆盖
  std::string outDir   = "Data/labelMtt";        // 相对 cwd；通常由 Makefile --out-dir 覆盖
  std::string decoder;                 // DecoderApp 路径（必填，可由 yaml 或 --decoder 给）
  std::string suffix   = "_RA_str";    // bin 命名：<seq><suffix><qp>.bin
  std::vector<int> qps = {22, 27, 32, 37};
  int      pocCount    = 8;            // 每序列随机采的 POC 个数（<=0 表示全部 POC）
  double   blockRatio  = 0.03;         // 每个 MTT 节点被写出的概率
  uint64_t seed        = 42;
  bool     dryRun      = false;
};

long countDataRows(const std::string& path)
{
  std::ifstream f(path);
  if (!f) return -1;
  long n = 0; std::string line;
  std::getline(f, line);               // 跳表头
  while (std::getline(f, line)) if (!line.empty()) ++n;
  return n;
}

// 设置/清空一个环境变量（空值 → 清除，让解码器走默认）
void setEnv(const char* k, const std::string& v)
{
  if (v.empty()) ::unsetenv(k);
  else           ::setenv(k, v.c_str(), 1);
}

} // namespace

int main(int argc, char** argv)
{
  std::string cfgPath = "config/mtt.yaml";
  std::string ovDecoder, ovBinDir, ovOutDir;
  bool ovDry = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
    if      (a == "--decoder") ovDecoder = next();
    else if (a == "--bin-dir") ovBinDir = next();
    else if (a == "--out-dir") ovOutDir = next();
    else if (a == "--dry-run") ovDry = true;
    else if (a == "--config")  cfgPath = next();
    else if (!a.empty() && a[0] != '-') cfgPath = a;   // 位置参数 = 配置路径
  }

  myaml::Document doc = myaml::parseFile(cfgPath);
  if (!doc.ok) { std::cerr << "配置解析失败: " << doc.error << " (" << cfgPath << ")\n"; return 1; }

  using namespace myaml;
  Cfg c;
  c.binDir     = getStr(doc.globals, "bin_dir", c.binDir);
  c.outDir     = getStr(doc.globals, "out_dir", c.outDir);
  c.decoder    = getStr(doc.globals, "decoder", c.decoder);
  c.suffix     = getStr(doc.globals, "bin_suffix", c.suffix);
  c.pocCount   = getInt(doc.globals, "poc_count", c.pocCount);
  c.blockRatio = getDouble(doc.globals, "block_ratio", c.blockRatio);
  c.seed       = (uint64_t)getInt(doc.globals, "random_seed", (int)c.seed);
  if (has(doc.globals, "qps")) c.qps = mttsample::splitInts(getStr(doc.globals, "qps", ""));

  if (!ovDecoder.empty()) c.decoder = ovDecoder;
  if (!ovBinDir.empty())  c.binDir  = ovBinDir;
  if (!ovOutDir.empty())  c.outDir  = ovOutDir;
  if (ovDry)              c.dryRun  = true;

  if (doc.sequences.empty()) { std::cerr << "配置里没有 sequences\n"; return 1; }
  if (c.decoder.empty() && !c.dryRun) {
    std::cerr << "未指定解码器路径（yaml decoder: 或 --decoder PATH）。\n"
                 "提示：根目录 `make mtt-label` 会自动构建并传入 DecoderApp。\n";
    return 1;
  }

  std::error_code ec; fs::create_directories(c.outDir, ec);
  std::cout << "解码器=" << (c.decoder.empty() ? "(dry-run)" : c.decoder)
            << " | bin=" << c.binDir << " | out=" << c.outDir
            << " | QP=" << mttsample::joinInts(c.qps)
            << " | pocCount=" << c.pocCount << " | blockRatio=" << c.blockRatio
            << " | seed=" << c.seed << (c.dryRun ? " | DRY-RUN" : "") << "\n"
            << "序列数=" << doc.sequences.size() << "\n\n";

  long grand = 0; int nFiles = 0, nMissing = 0;
  for (const auto& s : doc.sequences) {
    const std::string name = getStr(s, "name");
    const int frames = getInt(s, "frames", 0);
    if (name.empty()) continue;

    const uint64_t seqSeed = mttsample::seedForSequence(c.seed, name);
    const std::vector<int> pocs = mttsample::pickPocs(frames, c.pocCount, seqSeed);
    const std::string pocsStr = mttsample::joinInts(pocs);   // 空 → 解码器采全部 POC

    std::cout << "[" << name << "] frames=" << frames
              << " 采样POC=" << (pocs.empty() ? std::string("ALL") : pocsStr) << "\n";

    for (int qp : c.qps) {
      const std::string bin = c.binDir + "/" + name + c.suffix + std::to_string(qp) + ".bin";
      const std::string out = c.outDir + "/" + name + "_QP" + std::to_string(qp) + ".csv";
      if (!fs::exists(bin)) { std::cout << "    QP" << qp << " 跳过（缺 " << bin << "）\n"; ++nMissing; continue; }

      const std::string cmd = "\"" + c.decoder + "\" -b \"" + bin + "\" -o /dev/null > /dev/null 2>&1";
      if (c.dryRun) {
        std::cout << "    [dry] MTT_DUMP_OUT=" << out << " MTT_DUMP_SEQ=" << name
                  << " MTT_DUMP_QP=" << qp << " MTT_DUMP_POCS=" << (pocsStr.empty() ? "ALL" : pocsStr)
                  << " MTT_DUMP_BLOCK_RATIO=" << c.blockRatio << " MTT_DUMP_SEED=" << c.seed << "\n"
                  << "          " << cmd << "\n";
        ++nFiles;
        continue;
      }

      setEnv("MTT_DUMP_OUT", out);
      setEnv("MTT_DUMP_SEQ", name);
      setEnv("MTT_DUMP_QP", std::to_string(qp));
      setEnv("MTT_DUMP_POCS", pocsStr);
      setEnv("MTT_DUMP_BLOCK_RATIO", std::to_string(c.blockRatio));
      setEnv("MTT_DUMP_SEED", std::to_string(c.seed));

      const int rc = std::system(cmd.c_str());
      const long rows = countDataRows(out);
      if (rc != 0 || rows < 0) {
        std::cout << "    QP" << qp << " 解码失败 rc=" << rc << "（" << bin << "）\n";
      } else {
        std::cout << "    QP" << qp << " -> " << rows << " 行  " << out << "\n";
        grand += rows; ++nFiles;
      }
    }
  }

  std::cout << "\n完成。写出 " << nFiles << " 个文件，共 " << grand << " 行"
            << (nMissing ? ("（" + std::to_string(nMissing) + " 个 bin 缺失已跳过）") : "") << "\n";
  return 0;
}
