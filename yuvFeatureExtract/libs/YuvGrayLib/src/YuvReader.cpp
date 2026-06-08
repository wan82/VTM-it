#include "YuvGrayLib/YuvReader.h"
#include <fstream>
#include <cstdio>

namespace yg {

ChromaFormat parseChromaFormat(const std::string& s) {
    if (s == "400") return ChromaFormat::C400;
    if (s == "422") return ChromaFormat::C422;
    if (s == "444") return ChromaFormat::C444;
    return ChromaFormat::C420; // 默认 4:2:0
}

static double chromaSampleFactor(ChromaFormat c) {
    // 色度采样数 = luma * factor（两个色度分量合计）
    switch (c) {
        case ChromaFormat::C400: return 0.0;   // 仅亮度
        case ChromaFormat::C420: return 0.5;   // 2 * (1/4)
        case ChromaFormat::C422: return 1.0;   // 2 * (1/2)
        case ChromaFormat::C444: return 2.0;   // 2 * 1
    }
    return 0.5;
}

YuvReader::YuvReader(const std::string& path, int width, int height,
                     int inputBitDepth, ChromaFormat chroma, int internalBitDepth)
    : path_(path), width_(width), height_(height),
      inputBitDepth_(inputBitDepth), internalBitDepth_(internalBitDepth),
      chroma_(chroma) {
    std::ifstream f(path_, std::ios::binary | std::ios::ate);
    if (f && width_ > 0 && height_ > 0) {
        fileBytes_ = static_cast<long long>(f.tellg());
        ok_ = fileBytes_ > 0;
    }
}

long long YuvReader::frameSamples() const {
    return lumaSamples() + static_cast<long long>(lumaSamples() * chromaSampleFactor(chroma_));
}

long long YuvReader::frameBytes() const {
    int bytesPerSample = (inputBitDepth_ > 8) ? 2 : 1;
    return frameSamples() * bytesPerSample;
}

int YuvReader::inferredFrameCount() const {
    long long fb = frameBytes();
    if (fb <= 0) return 0;
    return static_cast<int>(fileBytes_ / fb);
}

bool YuvReader::readLumaFrame(int idx, GrayFrame& out) const {
    if (!ok_ || idx < 0) return false;
    const int bytesPerSample = (inputBitDepth_ > 8) ? 2 : 1;
    const long long lumaBytes = lumaSamples() * bytesPerSample;
    const long long offset = static_cast<long long>(idx) * frameBytes();

    std::ifstream f(path_, std::ios::binary);
    if (!f) return false;
    f.seekg(offset, std::ios::beg);
    if (!f) return false;

    std::vector<unsigned char> buf(static_cast<size_t>(lumaBytes));
    f.read(reinterpret_cast<char*>(buf.data()), lumaBytes);
    if (f.gcount() != lumaBytes) return false; // 不足一帧亮度

    out = GrayFrame(width_, height_);
    const int shift = internalBitDepth_ - inputBitDepth_; // 8→10 时为 2
    const int maxVal = (1 << internalBitDepth_) - 1;
    auto& dst = out.samples();

    if (bytesPerSample == 1) {
        for (size_t i = 0; i < dst.size(); ++i) {
            int v = static_cast<int>(buf[i]);
            v = (shift >= 0) ? (v << shift) : (v >> (-shift));
            dst[i] = static_cast<uint16_t>(v > maxVal ? maxVal : v);
        }
    } else {
        for (size_t i = 0; i < dst.size(); ++i) {
            int v = static_cast<int>(buf[2 * i]) | (static_cast<int>(buf[2 * i + 1]) << 8); // 小端
            v = (shift >= 0) ? (v << shift) : (v >> (-shift));
            dst[i] = static_cast<uint16_t>(v > maxVal ? maxVal : v);
        }
    }
    return true;
}

} // namespace yg
