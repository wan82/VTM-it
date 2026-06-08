#include "YuvGrayLib/FrameFilter.h"
#include <cmath>
#include <cstdlib>

namespace yg {

// 在抽样像素上计算一帧的均值与标准差
static void frameStats(const GrayFrame& f, int stride, double& mean, double& std) {
    if (stride < 1) stride = 1;
    double sum = 0.0, sumSq = 0.0;
    long long n = 0;
    const int W = f.width(), H = f.height();
    for (int y = 0; y < H; y += stride)
        for (int x = 0; x < W; x += stride) {
            int v = f.at(x, y);
            sum += v; sumSq += static_cast<double>(v) * v; ++n;
        }
    if (n == 0) { mean = std = 0.0; return; }
    mean = sum / n;
    double var = sumSq / n - mean * mean;
    std = std::sqrt(var < 0.0 ? 0.0 : var);
}

// 在抽样像素上计算两帧的平均绝对差（MAD）
static double frameMad(const GrayFrame& a, const GrayFrame& b, int stride) {
    if (stride < 1) stride = 1;
    if (a.width() != b.width() || a.height() != b.height()) return 1e30;
    double sum = 0.0;
    long long n = 0;
    const int W = a.width(), H = a.height();
    for (int y = 0; y < H; y += stride)
        for (int x = 0; x < W; x += stride) {
            sum += std::abs(static_cast<int>(a.at(x, y)) - static_cast<int>(b.at(x, y)));
            ++n;
        }
    return n == 0 ? 1e30 : sum / n;
}

std::vector<int> detectUsableFrames(const YuvReader& reader, int frameCount,
                                    const UselessFrameConfig& cfg,
                                    std::vector<int>* excludedOut) {
    std::vector<int> usable;
    if (frameCount <= 0) return usable;

    if (!cfg.enable) {
        for (int i = 0; i < frameCount; ++i) usable.push_back(i);
        return usable;
    }

    GrayFrame cur, prevValid;
    bool havePrev = false;
    for (int i = 0; i < frameCount; ++i) {
        if (!reader.readLumaFrame(i, cur)) break; // 读不到（文件尾）

        double mean = 0, std = 0;
        frameStats(cur, cfg.pixelStride, mean, std);

        bool isBlank = (std < cfg.blankStdThreshold);
        bool isDup = havePrev &&
                     (frameMad(cur, prevValid, cfg.pixelStride) < cfg.duplicateMadThreshold);

        if (isBlank || isDup) {
            if (excludedOut) excludedOut->push_back(i);
            continue; // 跳过无用帧，且不更新 prevValid
        }

        usable.push_back(i);
        prevValid = cur;
        havePrev = true;
    }
    return usable;
}

} // namespace yg
