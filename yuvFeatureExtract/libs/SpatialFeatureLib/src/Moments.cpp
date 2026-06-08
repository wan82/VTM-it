#include "SpatialFeatureLib/Moments.h"
#include <cmath>

namespace sf {

double mean(const Mat& m) {
    double sum = 0.0;
    long long count = 0;
    for (const auto& row : m) {
        for (int v : row) { sum += v; ++count; }
    }
    return count == 0 ? 0.0 : sum / static_cast<double>(count);
}

double stddev(const Mat& m) {
    double sum = 0.0, sumSq = 0.0;
    long long count = 0;
    for (const auto& row : m) {
        for (int v : row) {
            sum += v;
            sumSq += static_cast<double>(v) * v;
            ++count;
        }
    }
    if (count == 0) return 0.0;
    double mu = sum / count;
    double var = sumSq / count - mu * mu;
    return std::sqrt(var < 0.0 ? 0.0 : var); // 防浮点误差导致的负方差
}

// 把矩阵展平并返回均值，同时累积中心矩用
static double flattenMean(const Mat& m, long long& N) {
    double sum = 0.0;
    N = 0;
    for (const auto& row : m) {
        for (int v : row) { sum += v; ++N; }
    }
    return N == 0 ? 0.0 : sum / static_cast<double>(N);
}

double skewness(const Mat& m) {
    long long N = 0;
    double mu = flattenMean(m, N);
    if (N < 2) return 0.0;
    double m2 = 0.0, m3 = 0.0;
    for (const auto& row : m) {
        for (int v : row) {
            double d = v - mu;
            m2 += d * d;
            m3 += d * d * d;
        }
    }
    m2 /= N; m3 /= N;
    if (m2 == 0.0) return 0.0;
    return m3 / std::pow(std::sqrt(m2), 3.0);
}

double kurtosis(const Mat& m) {
    long long N = 0;
    double mu = flattenMean(m, N);
    if (N < 2) return 0.0;
    double m2 = 0.0, m4 = 0.0;
    for (const auto& row : m) {
        for (int v : row) {
            double d = v - mu;
            m2 += d * d;
            m4 += d * d * d * d;
        }
    }
    m2 /= N; m4 /= N;
    if (m2 == 0.0) return 0.0;
    return m4 / (m2 * m2);
}

} // namespace sf
