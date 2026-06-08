// MttSample.h —— MTT 标签采样的纯函数库（无第三方依赖）
//
// 这里只放「采样计划 + 小工具」这类纯逻辑：给定帧数/采样数/种子，挑出要采的 POC 列表；
// 真正的熵解码由 VTM 解码器完成（见 DecoderLib/MttLabelDumper），由编排程序 setenv 后驱动。
// 之所以独立成库，是为后续「针对采到的 MTT block 再提取 YUV 空域特征」复用同一套采样口径。
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mttsample {

// 把序列名混进基础种子，使不同序列得到不同但可复现的 POC 抽样
uint64_t seedForSequence(uint64_t baseSeed, const std::string& name);

// 从 [0, frames) 里确定性地随机挑 count 个不同的 POC（升序返回）。
//   frames<=0 或 count<=0 → 返回空（表示「全部 POC」，交给 block 比例控制数据量）
//   count>=frames        → 返回 0..frames-1 全部
std::vector<int> pickPocs(int frames, int count, uint64_t seed);

// 把整型列表拼成逗号串（喂给解码器的 MTT_DUMP_POCS）
std::string joinInts(const std::vector<int>& v);

// 拆 "22,27,32,37" 这类逗号串为整型列表
std::vector<int> splitInts(const std::string& csv);

} // namespace mttsample
