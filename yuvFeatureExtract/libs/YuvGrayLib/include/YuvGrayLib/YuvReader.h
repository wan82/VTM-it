// YuvReader.h —— 按给定参数从裸 YUV 文件按帧读取亮度（灰度）
// 裸 YUV 没有文件头，所有参数（宽高/位深/色度格式）必须由外部（YAML）给定。
#pragma once
#include <string>
#include <cstdint>
#include "YuvGrayLib/GrayFrame.h"

namespace yg {

enum class ChromaFormat { C400, C420, C422, C444 };

ChromaFormat parseChromaFormat(const std::string& s); // "400"/"420"/"422"/"444"

class YuvReader {
public:
    // inputBitDepth: 源位深(8 或 10)；internalBitDepth: 目标位深(默认 10，对齐 VTM)
    YuvReader(const std::string& path, int width, int height,
              int inputBitDepth, ChromaFormat chroma,
              int internalBitDepth = 10);

    bool ok() const { return ok_; }
    long long fileBytes() const { return fileBytes_; }

    // 按文件大小推断的总帧数（floor）
    int inferredFrameCount() const;

    // 读取第 idx 帧的亮度到 out（已转成 internalBitDepth）。失败返回 false。
    bool readLumaFrame(int idx, GrayFrame& out) const;

    int width()  const { return width_; }
    int height() const { return height_; }

private:
    long long lumaSamples()  const { return static_cast<long long>(width_) * height_; }
    long long frameSamples() const;            // 亮度+色度的总采样数
    long long frameBytes()   const;            // 一帧的字节数

    std::string path_;
    int width_ = 0, height_ = 0;
    int inputBitDepth_ = 8;
    int internalBitDepth_ = 10;
    ChromaFormat chroma_ = ChromaFormat::C420;
    long long fileBytes_ = 0;
    bool ok_ = false;
};

} // namespace yg
