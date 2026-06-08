// FrameFilter.h —— 自动检测并排除“没有实际用处”的帧
// 覆盖两种常见情形：
//   1) 空白/近常量帧：全帧标准差极低（全黑/全灰/填充帧）
//   2) 重复帧：与前一“有效”帧几乎逐像素相同（MAD 极低）
// 为提速，统计在像素抽样（stride）下进行，对“整帧异常”这类判断足够灵敏。
#pragma once
#include <vector>
#include <string>
#include "YuvGrayLib/YuvReader.h"

namespace yg {

struct UselessFrameConfig {
    bool   enable = true;
    double blankStdThreshold = 1.0;     // 全帧 std 低于此值视为空白帧
    double duplicateMadThreshold = 0.5; // 与前一有效帧 MAD 低于此值视为重复帧
    int    pixelStride = 8;             // 检测时的像素抽样步长（>=1）
};

// 扫描 [0, frameCount) 的所有帧，返回“有效（可用）帧”的索引列表。
// excludedOut（可选）回填被判为无用的帧索引，便于打印核对。
std::vector<int> detectUsableFrames(const YuvReader& reader, int frameCount,
                                    const UselessFrameConfig& cfg,
                                    std::vector<int>* excludedOut = nullptr);

} // namespace yg
