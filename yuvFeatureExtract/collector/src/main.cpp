// yuvFeatureCollector —— 从 YUV 批量提取 128x128 CTU 的空域特征，输出对齐 128csv 的 CSV
//
// 流程：读 YAML 配置 → 逐序列：按文件大小/配置确定总帧数 → 自动排除无用帧 →
//        预估可生成行数(可用帧数 × 每帧CTU数) → “随机帧再抽CTU”采样 →
//        提取 18 个空域特征 → 写 <output_dir>/<name>.csv
//
// 用法：  yuvFeatureCollector [config.yaml]
#include "MiniYaml/MiniYaml.h"
#include "YuvGrayLib/YuvReader.h"
#include "YuvGrayLib/GrayFrame.h"
#include "YuvGrayLib/CtuSplitter.h"
#include "YuvGrayLib/FrameFilter.h"
#include "SpatialFeatureLib/CtuFeatures.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static std::string baseName(const std::string& path) {
    std::string b = fs::path(path).filename().string();
    size_t dot = b.find_last_of('.');
    return (dot == std::string::npos) ? b : b.substr(0, dot);
}

struct GlobalCfg {
    std::string outputDir = "./output";
    int    ctuSize = 128;
    int    internalBitDepth = 10;
    int    defaultCtuPerYuv = 10;
    double defaultSampleRatio = 0.0;   // >0 时按“预估可生成×比例”采样（优先于固定条数）
    unsigned seed = 42;
    std::string chroma = "420";
    yg::UselessFrameConfig uf;
};

// 处理单个序列；返回采集到的行数
static long processSequence(const myaml::KV& s, const GlobalCfg& g) {
    using namespace myaml;
    std::string path = getStr(s, "path");
    if (path.empty()) { std::cerr << "  [跳过] 序列缺少 path\n"; return 0; }
    std::string name = has(s, "name") ? getStr(s, "name") : baseName(path);
    int width  = getInt(s, "width", 0);
    int height = getInt(s, "height", 0);
    int bitdepth = getInt(s, "bitdepth", 8);
    std::string chroma = has(s, "chroma_format") ? getStr(s, "chroma_format") : g.chroma;
    std::cout << "== " << name << " (" << width << "x" << height
              << ", " << bitdepth << "bit, " << chroma << ") ==\n";
    if (width <= 0 || height <= 0) { std::cerr << "  [跳过] 宽高非法\n"; return 0; }

    yg::YuvReader reader(path, width, height, bitdepth,
                         yg::parseChromaFormat(chroma), g.internalBitDepth);
    if (!reader.ok()) { std::cerr << "  [跳过] 无法打开/读取: " << path << "\n"; return 0; }

    int frames = has(s, "frames") ? getInt(s, "frames", 0) : reader.inferredFrameCount();
    if (frames <= 0) { std::cerr << "  [跳过] 帧数为 0\n"; return 0; }

    // 自动排除无用帧
    std::vector<int> excluded;
    std::vector<int> usable = yg::detectUsableFrames(reader, frames, g.uf, &excluded);

    int ctusPF = yg::ctusPerFrame(width, height, g.ctuSize);
    long long maxRows = static_cast<long long>(usable.size()) * ctusPF;

    // 采样目标：按优先级决定用“比例”还是“固定条数”
    //   1) 序列级 sample_ratio>0       → ceil(预估可生成 × 比例)
    //   2) 序列级 ctu_per_yuv          → 固定条数
    //   3) 全局 default_sample_ratio>0 → ceil(预估可生成 × 比例)
    //   4) 全局 default_ctu_per_yuv    → 固定条数
    double seqRatio = has(s, "sample_ratio") ? getDouble(s, "sample_ratio", 0.0) : -1.0;
    long long targetReq;
    std::string mode;
    if (seqRatio > 0.0) {
        targetReq = static_cast<long long>(std::ceil(maxRows * seqRatio));
        mode = "比例";
    } else if (has(s, "ctu_per_yuv")) {
        targetReq = getInt(s, "ctu_per_yuv", g.defaultCtuPerYuv);
        mode = "固定";
    } else if (g.defaultSampleRatio > 0.0) {
        targetReq = static_cast<long long>(std::ceil(maxRows * g.defaultSampleRatio));
        mode = "比例";
    } else {
        targetReq = g.defaultCtuPerYuv;
        mode = "固定";
    }
    long target = static_cast<long>(std::min<long long>(targetReq, maxRows));

    std::cout << "  总帧=" << frames << " 无用帧=" << excluded.size()
              << " 可用帧=" << usable.size()
              << " 每帧CTU=" << ctusPF
              << " 预估可生成=" << maxRows
              << " 采样方式=" << mode
              << " 目标采样=" << target << "\n";
    if (!excluded.empty()) {
        std::cout << "  排除帧索引:";
        for (size_t i = 0; i < excluded.size() && i < 20; ++i) std::cout << ' ' << excluded[i];
        if (excluded.size() > 20) std::cout << " ...";
        std::cout << "\n";
    }
    if (target <= 0 || ctusPF <= 0) { std::cerr << "  [跳过] 无可采样CTU\n"; return 0; }

    // 采样：每条样本独立地“随机选一帧 + 随机选一个CTU位置”，按 (帧, CTU位置) 去重
    std::mt19937 rng(g.seed ^ static_cast<unsigned>(std::hash<std::string>{}(name)));
    std::vector<yg::CtuPos> ctuList = yg::enumerateFullCtus(width, height, g.ctuSize);
    std::uniform_int_distribution<size_t> frameDist(0, usable.size() - 1);
    std::uniform_int_distribution<size_t> ctuDist(0, ctuList.size() - 1);

    std::set<std::pair<int, int>> usedPairs;   // 已抽过的 (帧索引, CTU索引)
    std::map<int, yg::GrayFrame> frameCache;   // 已读帧缓存

    struct Row { int poc, x, y; std::string fields; };
    std::vector<Row> rows;
    rows.reserve(target);

    long collected = 0;
    long guard = 0, guardMax = static_cast<long>(maxRows) + static_cast<long>(target) * 64 + 64;
    while (collected < target && guard < guardMax) {
        ++guard;
        int frame = usable[frameDist(rng)];          // 随机帧
        int ci    = static_cast<int>(ctuDist(rng));  // 随机CTU位置
        if (!usedPairs.insert({frame, ci}).second) continue; // (帧,位置) 已抽过则重试

        // 读帧（带缓存）
        auto it = frameCache.find(frame);
        if (it == frameCache.end()) {
            yg::GrayFrame gf;
            if (!reader.readLumaFrame(frame, gf)) continue;
            it = frameCache.emplace(frame, std::move(gf)).first;
        }
        const yg::GrayFrame& gf = it->second;

        yg::CtuPos p = ctuList[ci];
        sf::Mat block = gf.extractBlock(p.x, p.y, g.ctuSize);
        sf::CtuFeatures feat = sf::computeCtuFeatures(block);
        rows.push_back({frame, p.x, p.y, feat.toCsvFields()});
        ++collected;
    }

    // 按 (POC, y, x) 排序，输出整洁
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
        if (a.poc != b.poc) return a.poc < b.poc;
        if (a.y != b.y) return a.y < b.y;
        return a.x < b.x;
    });

    fs::create_directories(g.outputDir);
    std::string outPath = (fs::path(g.outputDir) / (name + ".csv")).string();
    std::ofstream out(outPath);
    if (!out) { std::cerr << "  [错误] 无法写出: " << outPath << "\n"; return 0; }
    out << "POC_ID,pos_x,pos_y," << sf::CtuFeatures::csvHeader() << "\n";
    for (const auto& r : rows)
        out << r.poc << ',' << r.x << ',' << r.y << ',' << r.fields << "\n";

    std::cout << "  -> 写出 " << collected << " 条到 " << outPath << "\n";
    return collected;
}

int main(int argc, char** argv) {
    std::string configPath = (argc >= 2) ? argv[1] : "config/config.yaml";
    std::cout << "读取配置: " << configPath << "\n";

    myaml::Document doc = myaml::parseFile(configPath);
    if (!doc.ok) { std::cerr << "配置解析失败: " << doc.error << "\n"; return 1; }

    using namespace myaml;
    GlobalCfg g;
    g.outputDir        = getStr(doc.globals, "output_dir", g.outputDir);
    g.ctuSize          = getInt(doc.globals, "ctu_size", g.ctuSize);
    g.internalBitDepth = getInt(doc.globals, "internal_bitdepth", g.internalBitDepth);
    g.defaultCtuPerYuv = getInt(doc.globals, "default_ctu_per_yuv", g.defaultCtuPerYuv);
    g.defaultSampleRatio = getDouble(doc.globals, "default_sample_ratio", 0.0);
    g.seed             = static_cast<unsigned>(getInt(doc.globals, "random_seed", 42));
    g.chroma           = getStr(doc.globals, "chroma_format", g.chroma);
    g.uf.enable               = getBool(doc.globals, "detect_useless_frames", true);
    g.uf.blankStdThreshold    = getDouble(doc.globals, "blank_std_threshold", 1.0);
    g.uf.duplicateMadThreshold= getDouble(doc.globals, "duplicate_mad_threshold", 0.5);
    g.uf.pixelStride          = getInt(doc.globals, "detect_pixel_stride", 8);

    if (doc.sequences.empty()) { std::cerr << "配置里没有 sequences\n"; return 1; }

    std::cout << "输出目录: " << g.outputDir << " | CTU=" << g.ctuSize
              << " | 内部位深=" << g.internalBitDepth
              << " | 默认每YUV采样=" << g.defaultCtuPerYuv
              << " | 序列数=" << doc.sequences.size() << "\n\n";

    long total = 0;
    for (const auto& s : doc.sequences) total += processSequence(s, g);

    std::cout << "\n完成。共写出 " << total << " 条 CTU 特征。\n";
    return 0;
}
