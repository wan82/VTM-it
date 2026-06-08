// CtuSplitter.h —— 把一帧切成 ctuSize x ctuSize 的 CTU；边缘不足 ctuSize 的直接丢弃
#pragma once
#include <vector>

namespace yg {

struct CtuPos { int x; int y; }; // CTU 左上角像素坐标

// 枚举一帧内所有“完整”CTU 的左上角坐标（行优先），右/下边缘不足 ctuSize 的丢弃。
std::vector<CtuPos> enumerateFullCtus(int frameWidth, int frameHeight, int ctuSize);

// 每帧完整 CTU 个数
inline int ctusPerFrame(int frameWidth, int frameHeight, int ctuSize) {
    return (frameWidth / ctuSize) * (frameHeight / ctuSize);
}

} // namespace yg
