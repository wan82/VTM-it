#include "SpatialFeatureLib/VarianceTests.h"
#include <algorithm>
#include <cmath>
#include <utility>

namespace sf {

// 把方阵切成 4 个子块（左上/右上/左下/右下），各自展平为一维向量。
// 返回 false 表示输入非法（非方阵 / 边长为奇 / 太小）。
static bool splitQuadrants(const Mat& m, std::vector<std::vector<double>>& sub) {
    size_t size = m.size();
    if (size == 0 || m[0].size() != size) return false; // 必须方阵
    if (size % 2 != 0) return false;                     // 边长需偶数
    size_t half = size / 2;
    if (half == 0) return false;

    sub.assign(4, {});
    for (size_t bi = 0; bi < 2; ++bi)
        for (size_t bj = 0; bj < 2; ++bj) {
            auto& g = sub[bi * 2 + bj];
            g.reserve(half * half);
            for (size_t y = bi * half; y < (bi + 1) * half; ++y)
                for (size_t x = bj * half; x < (bj + 1) * half; ++x)
                    g.push_back(static_cast<double>(m[y][x]));
        }
    return true;
}

static double vecMean(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / v.size();
}

double levene(const Mat& m) {
    std::vector<std::vector<double>> sub;
    if (!splitQuadrants(m, sub)) return 0.0;

    // 常量子块 → 返回 0（与参考实现一致）
    for (const auto& g : sub) {
        if (g.empty()) return 0.0;
        bool allSame = true;
        for (double v : g) if (v != g[0]) { allSame = false; break; }
        if (allSame) return 0.0;
    }

    // 每组到组均值的绝对偏差 zij
    std::vector<std::vector<double>> absDev(4);
    for (size_t i = 0; i < 4; ++i) {
        double gm = vecMean(sub[i]);
        absDev[i].reserve(sub[i].size());
        for (double v : sub[i]) absDev[i].push_back(std::abs(v - gm));
    }

    // 总体均值 z_bar
    std::vector<double> all;
    for (const auto& d : absDev) all.insert(all.end(), d.begin(), d.end());
    double overall = vecMean(all);

    size_t k = 4, N = all.size();
    double numerator = 0.0, denominator = 0.0;
    for (size_t i = 0; i < k; ++i) {
        double ni = static_cast<double>(absDev[i].size());
        double zi = vecMean(absDev[i]);
        numerator += ni * (zi - overall) * (zi - overall);
        for (double zij : absDev[i]) denominator += (zij - zi) * (zij - zi);
    }
    if (denominator == 0.0) return 0.0;
    return (static_cast<double>(N - k) * numerator) / ((k - 1) * denominator);
}

double kruskalWallis(const Mat& m) {
    std::vector<std::vector<double>> sub;
    if (!splitQuadrants(m, sub)) return 0.0;

    // 收集所有样本并记录组号
    std::vector<std::pair<double, int>> samples;
    for (int i = 0; i < 4; ++i) {
        if (sub[i].empty()) return 0.0;
        for (double v : sub[i]) {
            if (std::isnan(v) || std::isinf(v)) return 0.0;
            samples.emplace_back(v, i);
        }
    }
    if (samples.empty()) return 0.0;

    std::sort(samples.begin(), samples.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    // 简单秩 1..N（不做并列校正，与参考实现一致）
    std::vector<double> rankSum(4, 0.0);
    std::vector<long long> nGroup(4, 0);
    for (size_t i = 0; i < samples.size(); ++i) {
        int g = samples[i].second;
        rankSum[g] += static_cast<double>(i + 1);
        nGroup[g]++;
    }
    for (long long n : nGroup) if (n == 0) return 0.0;

    double N = static_cast<double>(samples.size());
    double sumTerm = 0.0;
    for (int i = 0; i < 4; ++i)
        sumTerm += rankSum[i] * rankSum[i] / static_cast<double>(nGroup[i]);

    return (12.0 / (N * (N + 1.0))) * sumTerm - 3.0 * (N + 1.0);
}

} // namespace sf
