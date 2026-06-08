#include "SpatialFeatureLib/Filters.h"
#include <cmath>
#include <cstdlib>

namespace sf {

// Sobel 卷积核（与 w_cuProcess.h 中的 Gx/Gy 一致）
static const int Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};
static const int Gy[3][3] = {
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1}
};

// Laplacian 4-邻域核
static const int Lap[3][3] = {
    {0,  1, 0},
    {1, -4, 1},
    {0,  1, 0}
};

Mat sobel(const Mat& input) {
    if (input.empty() || input[0].empty()) return {};
    const int H = static_cast<int>(input.size());
    const int W = static_cast<int>(input[0].size());
    Mat out(H, std::vector<int>(W, 0));
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int sx = 0, sy = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx) {
                    int p = input[y + ky][x + kx];
                    sx += p * Gx[ky + 1][kx + 1];
                    sy += p * Gy[ky + 1][kx + 1];
                }
            out[y][x] = static_cast<int>(std::lround(std::sqrt(
                static_cast<double>(sx) * sx + static_cast<double>(sy) * sy)));
        }
    }
    return out;
}

Mat laplacian(const Mat& input) {
    if (input.empty() || input[0].empty()) return {};
    const int H = static_cast<int>(input.size());
    const int W = static_cast<int>(input[0].size());
    Mat out(H, std::vector<int>(W, 0));
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
            int acc = 0;
            for (int ky = -1; ky <= 1; ++ky)
                for (int kx = -1; kx <= 1; ++kx)
                    acc += input[y + ky][x + kx] * Lap[ky + 1][kx + 1];
            out[y][x] = std::abs(acc); // 取绝对响应，和 Sobel 的“幅值”语义一致
        }
    }
    return out;
}

} // namespace sf
