// GrayFrame.h —— 一帧 10bit 灰度图像矩阵
// 统一约定：无论源是 8bit 还是 10bit，读入后都左移到 VTM 默认的 10bit（值域 0..1023）。
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include "SpatialFeatureLib/Matrix.h"

namespace yg {

class GrayFrame {
public:
    GrayFrame() = default;
    GrayFrame(int w, int h) : width_(w), height_(h), data_(static_cast<size_t>(w) * h, 0) {}

    int width()  const { return width_; }
    int height() const { return height_; }
    bool empty() const { return data_.empty(); }

    uint16_t  at(int x, int y) const { return data_[static_cast<size_t>(y) * width_ + x]; }
    uint16_t& at(int x, int y)       { return data_[static_cast<size_t>(y) * width_ + x]; }

    const std::vector<uint16_t>& samples() const { return data_; }
    std::vector<uint16_t>&       samples()       { return data_; }

    // 取以 (x0,y0) 为左上角、边长 size 的方块为 sf::Mat（要求完全在画面内）
    sf::Mat extractBlock(int x0, int y0, int size) const;

private:
    int width_ = 0, height_ = 0;
    std::vector<uint16_t> data_; // 行优先，10bit
};

} // namespace yg
