// yuvLabelMerger —— 把 labelCtu 的 QT 深度标签与 tempYuvFeature 的空域特征合并
//
// 对每个同名序列，按 (POC_ID,pos_x,pos_y) 内连接：
//   - 每序列中间表 -> <temp>/<name>.csv（键 + 标签 + 特征）
//   - 拼合并去掉含空字段(NaN)的行 -> <out> (Data/TrainingData/Training.csv)
// --keep-intermediate 默认 yes（保留每序列中间表）；--delete-intermediate 则删除之。
//
// 用法：
//   yuvLabelMerger --labels DIR --features DIR --out FILE --temp DIR [--delete-intermediate]
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string rstrip(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n' || s.back() == ' '))
        s.pop_back();
    return s;
}

// 按逗号切分，保留空字段
static std::vector<std::string> splitCsv(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.push_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

static int idxOf(const std::vector<std::string>& h, const std::string& name) {
    for (size_t i = 0; i < h.size(); ++i) if (h[i] == name) return (int)i;
    return -1;
}

// ===== VTM RA(默认 GOP=32) 的 QP-value 推导 =====
// 仿照 VTM-VVC：sliceQP = baseQP + 时域层(TID)相关的偏移；TID 由 POC%32 决定。
static const int BASE_QPS[4] = {22, 27, 32, 37};

// GOP 内位置(gop=poc%32) → 时域层 TID（与 VTM RA GOP32 层级一致）
static int gopToTid(int gop) {
    if (gop == 0)              return 0;
    if (gop == 16)             return 1;
    if (gop == 8 || gop == 24) return 2;
    if (gop % 2 == 1)          return 5;   // 奇数位置：最高时域层
    if (gop % 4 != 0)          return 4;   // 2,6,10,...
    return 3;                              // 4,12,20,28
}

// (baseQP, TID) → QP 偏移（取自 VTM RA 各基础 QP 的时域层偏移，与解码出的 PocInfo 一致）
static int qpOffset(int baseQp, int tid) {
    static const int off22[6] = {-3, 0, 1, 4, 6, 7};
    static const int off27[6] = {-3, 1, 2, 5, 7, 9};
    static const int off32[6] = {-3, 2, 3, 6, 8, 9};
    static const int off37[6] = {-3, 3, 4, 6, 8, 9};
    const int* t = (baseQp == 22) ? off22 : (baseQp == 27) ? off27
                 : (baseQp == 32) ? off32 : (baseQp == 37) ? off37 : nullptr;
    if (!t || tid < 0 || tid > 5) return 0;
    return t[tid];
}

static int getQpValue(int poc, int baseQp) {
    return baseQp + qpOffset(baseQp, gopToTid(poc % 32));
}

static int parseIntSafe(const std::string& s) {
    try { return std::stoi(s); } catch (...) { return 0; }
}

int main(int argc, char** argv) {
    // 默认相对路径（相对运行时 cwd）；通常由 Makefile 用 --labels/--features/--out/--temp 显式覆盖
    std::string labelsDir = "Data/labelCtu";
    std::string featDir   = "Data/temp/tempYuvFeature";
    std::string outFile   = "Data/TrainingData/Training.csv";
    std::string tempDir   = "Data/temp/trainMerged";
    bool keepIntermediate = true;   // 默认保留中间文件

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? std::string(argv[++i]) : std::string(); };
        if      (a == "--labels")               labelsDir = next();
        else if (a == "--features")             featDir   = next();
        else if (a == "--out")                  outFile   = next();
        else if (a == "--temp")                 tempDir   = next();
        else if (a == "--delete-intermediate")  keepIntermediate = false;
        else if (a == "--keep-intermediate")    keepIntermediate = (next() != "0");
    }

    const std::vector<std::string> KEYS = {"POC_ID", "pos_x", "pos_y"};

    if (!fs::is_directory(labelsDir) || !fs::is_directory(featDir)) {
        std::cerr << "[错误] 标签或特征目录不存在\n"; return 1;
    }

    // 发现同名序列
    std::set<std::string> labelNames, featNames;
    for (auto& e : fs::directory_iterator(labelsDir))
        if (e.is_regular_file() && e.path().extension() == ".csv")
            labelNames.insert(e.path().stem().string());
    for (auto& e : fs::directory_iterator(featDir))
        if (e.is_regular_file() && e.path().extension() == ".csv")
            featNames.insert(e.path().stem().string());
    std::vector<std::string> seqs;
    for (auto& n : labelNames) if (featNames.count(n)) seqs.push_back(n);
    std::sort(seqs.begin(), seqs.end());
    if (seqs.empty()) { std::cerr << "[错误] 没有同时具备标签和特征的序列\n"; return 1; }

    fs::create_directories(tempDir);
    fs::create_directories(fs::path(outFile).parent_path());

    std::ofstream train(outFile);
    if (!train) { std::cerr << "[错误] 无法写出 " << outFile << "\n"; return 1; }
    bool headerWritten = false;
    std::vector<std::string> gLabelCols, gFeatCols;
    long totalKept = 0, totalDropped = 0;

    std::cout << "发现 " << seqs.size() << " 个可合并序列\n";
    printf("%-18s %10s %10s %8s %10s\n", "sequence", "labelCTU", "featCTU", "match", "dropNaN");
    std::cout << std::string(60, '-') << "\n";

    for (const auto& name : seqs) {
        // ---- 读标签 ----
        std::ifstream lf(labelsDir + "/" + name + ".csv");
        std::string line;
        if (!std::getline(lf, line)) continue;
        auto lh = splitCsv(rstrip(line));
        int lp = idxOf(lh, "POC_ID"), lx = idxOf(lh, "pos_x"), ly = idxOf(lh, "pos_y");
        if (lp < 0 || lx < 0 || ly < 0) { std::cerr << "  [跳过] " << name << " 标签缺键列\n"; continue; }
        std::vector<int> lvalIdx; std::vector<std::string> lcols;
        for (size_t i = 0; i < lh.size(); ++i)
            if ((int)i != lp && (int)i != lx && (int)i != ly) { lvalIdx.push_back((int)i); lcols.push_back(lh[i]); }
        std::map<std::string, std::vector<std::string>> labelMap;
        long labCount = 0;
        while (std::getline(lf, line)) {
            line = rstrip(line); if (line.empty()) continue;
            auto f = splitCsv(line);
            int mx = std::max(lp, std::max(lx, ly));
            if ((int)f.size() <= mx) continue;
            std::string key = f[lp] + "," + f[lx] + "," + f[ly];
            std::vector<std::string> vals;
            for (int idx : lvalIdx) vals.push_back(idx < (int)f.size() ? f[idx] : "");
            labelMap[key] = vals; ++labCount;
        }

        // ---- 读特征 ----
        std::ifstream ff(featDir + "/" + name + ".csv");
        if (!std::getline(ff, line)) continue;
        auto fh = splitCsv(rstrip(line));
        int fp = idxOf(fh, "POC_ID"), fx = idxOf(fh, "pos_x"), fy = idxOf(fh, "pos_y");
        if (fp < 0 || fx < 0 || fy < 0) { std::cerr << "  [跳过] " << name << " 特征缺键列\n"; continue; }
        std::vector<int> fvalIdx; std::vector<std::string> fcols;
        for (size_t i = 0; i < fh.size(); ++i)
            if ((int)i != fp && (int)i != fx && (int)i != fy) { fvalIdx.push_back((int)i); fcols.push_back(fh[i]); }

        if (gLabelCols.empty()) gLabelCols = lcols;
        if (gFeatCols.empty())  gFeatCols  = fcols;

        // ---- 每序列中间表 ----
        std::ofstream pf(tempDir + "/" + name + ".csv");
        pf << "POC_ID,pos_x,pos_y";
        for (auto& c : lcols) pf << "," << c;
        for (int b = 0; b < 4; ++b) pf << "," << BASE_QPS[b] << "QP";   // QP-value 列
        for (auto& c : fcols) pf << "," << c;
        pf << "\n";

        long match = 0, dropped = 0, featCount = 0;
        while (std::getline(ff, line)) {
            line = rstrip(line); if (line.empty()) continue;
            auto f = splitCsv(line); ++featCount;
            int mx = std::max(fp, std::max(fx, fy));
            if ((int)f.size() <= mx) continue;
            std::string key = f[fp] + "," + f[fx] + "," + f[fy];
            auto it = labelMap.find(key);
            if (it == labelMap.end()) continue;     // 未匹配
            ++match;
            const auto& lvals = it->second;
            std::vector<std::string> fvals;
            for (int idx : fvalIdx) fvals.push_back(idx < (int)f.size() ? f[idx] : "");

            // QP-value：由 POC 推导的 4 个基础 QP 下的实际 slice QP
            int poc = parseIntSafe(f[fp]);
            std::string qpStr;
            for (int b = 0; b < 4; ++b) qpStr += "," + std::to_string(getQpValue(poc, BASE_QPS[b]));

            pf << key;
            for (auto& v : lvals) pf << "," << v;
            pf << qpStr;
            for (auto& v : fvals) pf << "," << v;
            pf << "\n";

            bool hasNaN = false;
            for (auto& v : lvals) if (v.empty()) { hasNaN = true; break; }
            if (!hasNaN) for (auto& v : fvals) if (v.empty()) { hasNaN = true; break; }
            if (hasNaN) { ++dropped; continue; }

            if (!headerWritten) {
                train << "sequence,POC_ID,pos_x,pos_y";
                for (auto& c : gLabelCols) train << "," << c;
                for (int b = 0; b < 4; ++b) train << "," << BASE_QPS[b] << "QP";
                for (auto& c : gFeatCols)  train << "," << c;
                train << "\n"; headerWritten = true;
            }
            train << name << "," << key;
            for (auto& v : lvals) train << "," << v;
            train << qpStr;
            for (auto& v : fvals) train << "," << v;
            train << "\n";
            ++totalKept;
        }
        pf.close();
        totalDropped += dropped;
        printf("%-18s %10ld %10ld %8ld %10ld\n", name.c_str(), labCount, featCount, match, dropped);
        if (!keepIntermediate) { std::error_code ec; fs::remove(tempDir + "/" + name + ".csv", ec); }
    }
    train.close();
    std::cout << std::string(60, '-') << "\n";
    std::cout << "合并训练总表 -> " << outFile << "  (保留 " << totalKept
              << " 行, 丢弃含NaN行 " << totalDropped << ")\n";
    if (keepIntermediate) std::cout << "每序列中间表 -> " << tempDir << "/<name>.csv\n";
    else                  std::cout << "(已删除每序列中间表)\n";
    return 0;
}
