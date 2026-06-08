#include "YuvGrayLib/CtuSplitter.h"

namespace yg {

std::vector<CtuPos> enumerateFullCtus(int frameWidth, int frameHeight, int ctuSize) {
    std::vector<CtuPos> out;
    if (ctuSize <= 0) return out;
    for (int y = 0; y + ctuSize <= frameHeight; y += ctuSize)
        for (int x = 0; x + ctuSize <= frameWidth; x += ctuSize)
            out.push_back({x, y});
    return out;
}

} // namespace yg
