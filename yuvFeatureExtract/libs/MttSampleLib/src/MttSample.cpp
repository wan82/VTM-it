#include "MttSampleLib/MttSample.h"

#include <algorithm>
#include <sstream>

namespace mttsample {

namespace {
// splitmix64：可复现的确定性随机源
struct SplitMix64 {
  uint64_t s;
  explicit SplitMix64(uint64_t seed) : s(seed) {}
  uint64_t next() {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
};
} // namespace

uint64_t seedForSequence(uint64_t baseSeed, const std::string& name)
{
  uint64_t h = baseSeed ^ 0xCBF29CE484222325ULL;
  for (unsigned char c : name) h = (h ^ c) * 0x100000001B3ULL;
  return h;
}

std::vector<int> pickPocs(int frames, int count, uint64_t seed)
{
  std::vector<int> out;
  if (frames <= 0 || count <= 0) return out;          // 空 → 全部 POC
  if (count >= frames) {
    out.resize(frames);
    for (int i = 0; i < frames; ++i) out[i] = i;
    return out;
  }
  // Fisher–Yates 部分洗牌，取前 count 个，确定性
  std::vector<int> idx(frames);
  for (int i = 0; i < frames; ++i) idx[i] = i;
  SplitMix64 rng(seed);
  for (int i = 0; i < count; ++i) {
    int j = i + (int)(rng.next() % (uint64_t)(frames - i));
    std::swap(idx[i], idx[j]);
  }
  out.assign(idx.begin(), idx.begin() + count);
  std::sort(out.begin(), out.end());
  return out;
}

std::string joinInts(const std::vector<int>& v)
{
  std::ostringstream os;
  for (size_t i = 0; i < v.size(); ++i) { if (i) os << ','; os << v[i]; }
  return os.str();
}

std::vector<int> splitInts(const std::string& csv)
{
  std::vector<int> out;
  std::stringstream ss(csv); std::string tok;
  while (std::getline(ss, tok, ',')) {
    // 去空白
    size_t a = tok.find_first_not_of(" \t");
    size_t b = tok.find_last_not_of(" \t");
    if (a == std::string::npos) continue;
    try { out.push_back(std::stoi(tok.substr(a, b - a + 1))); } catch (...) {}
  }
  return out;
}

} // namespace mttsample
