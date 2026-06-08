#include "YuvGrayLib/GrayFrame.h"

namespace yg {

sf::Mat GrayFrame::extractBlock(int x0, int y0, int size) const {
    sf::Mat block(size, std::vector<int>(size, 0));
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x)
            block[y][x] = static_cast<int>(at(x0 + x, y0 + y));
    return block;
}

} // namespace yg
