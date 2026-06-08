// ============================================================================
// #MTTDUMP  MttLabelDumper 实现（自定义新增文件）
// 见 MttLabelDumper.h 的环境变量约定。默认（不设 MTT_DUMP_OUT）彻底关闭。
// ============================================================================
#include "MttLabelDumper.h"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "CommonLib/CodingStructure.h"   // CodingStructure / Slice / Partitioner / UnitArea

namespace
{
// splitmix64：把整型键打散成 [0,1) 均匀数，结果可复现、与解码顺序无关
double hashUnit(uint64_t x)
{
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  x =  x ^ (x >> 31);
  return (x >> 11) * (1.0 / 9007199254740992.0);   // 取高 53 位 → [0,1)
}
uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001B3ULL + 0xCBF29CE484222325ULL; }

// PartSplit → 训练用的简短名；非 MTT（QUAD 等）返回 nullptr 表示跳过
const char* mttSplitName(PartSplit s)
{
  switch (s)
  {
    case CU_DONT_SPLIT: return "NO";
    case CU_HORZ_SPLIT: return "BTH";
    case CU_VERT_SPLIT: return "BTV";
    case CU_TRIH_SPLIT: return "TTH";
    case CU_TRIV_SPLIT: return "TTV";
    default:            return nullptr;   // QUAD / TU 划分等：不是 MTT 决策
  }
}
} // namespace

MttLabelDumper& MttLabelDumper::instance()
{
  static MttLabelDumper inst;
  return inst;
}

MttLabelDumper::MttLabelDumper()
{
  const char* out = std::getenv("MTT_DUMP_OUT");
  if (!out || !*out)
  {
    return;   // 未设 → 关闭，解码行为与原版一致（不打印，避免污染正常解码输出）
  }
  if (const char* s = std::getenv("MTT_DUMP_SEQ"))         m_seq = s;
  if (const char* q = std::getenv("MTT_DUMP_QP"))          { try { m_qp = std::stoi(q); } catch (...) {} }
  if (const char* r = std::getenv("MTT_DUMP_BLOCK_RATIO")) { try { m_blockRatio = std::stod(r); } catch (...) {} }
  if (const char* sd = std::getenv("MTT_DUMP_SEED"))       { try { m_seed = (uint64_t)std::stoull(sd); } catch (...) {} }
  if (const char* p = std::getenv("MTT_DUMP_POCS"))
  {
    std::stringstream ss(p); std::string tok;
    while (std::getline(ss, tok, ','))
    {
      if (tok.empty()) continue;
      try { m_pocs.insert(std::stoi(tok)); } catch (...) {}
    }
    m_allPocs = m_pocs.empty();
  }

  m_ofs.open(out, std::ios::out | std::ios::trunc);
  if (!m_ofs)
  {
    std::cerr << "[MTTDUMP] 打不开输出文件: " << out << "（已关闭）\n";
    return;
  }
  m_ofs << "sequence,POC_ID,pos_x,pos_y,w,h,qtDepth,mttDepth,splitMode,QP\n";
  m_enabled = true;
  std::cerr << "[MTTDUMP] 已启用 → " << out
            << " | seq=" << m_seq << " qp=" << m_qp
            << " blockRatio=" << m_blockRatio
            << " pocs=" << (m_allPocs ? std::string("ALL") : std::to_string(m_pocs.size()))
            << " seed=" << m_seed << "\n";
}

MttLabelDumper::~MttLabelDumper()
{
  if (m_ofs.is_open()) m_ofs.close();
}

void MttLabelDumper::maybeDump(const CodingStructure& cs, const Partitioner& partitioner, PartSplit splitMode)
{
  if (!m_enabled) return;
  if (!isLuma(partitioner.chType)) return;            // 只采亮度

  const char* name = mttSplitName(splitMode);
  if (!name) return;                                  // QUAD / 非 MTT 节点：跳过

  const int poc = cs.slice->getPOC();
  if (!m_allPocs && m_pocs.find(poc) == m_pocs.end()) return;   // POC 不在采样集

  const Area& a = partitioner.currArea().Y();         // 亮度块坐标/尺寸
  const int x = a.x, y = a.y, w = a.width, h = a.height;

  // 跳过越过图像边界的块（VVC 在边界会隐式强制划分，不是真正的决策；与采集器丢边缘一致）
  const int picW = cs.pps->getPicWidthInLumaSamples();
  const int picH = cs.pps->getPicHeightInLumaSamples();
  if (x + w > picW || y + h > picH) return;

  // 按 (seed,poc,x,y,w,h) 哈希做 block 采样，可复现、与解码顺序无关
  if (m_blockRatio < 1.0)
  {
    uint64_t key = m_seed;
    key = mix(key, (uint64_t)poc);
    key = mix(key, ((uint64_t)x << 16) ^ (uint64_t)y);
    key = mix(key, ((uint64_t)w << 16) ^ (uint64_t)h);
    if (hashUnit(key) >= m_blockRatio) return;
  }

  m_ofs << m_seq << ',' << poc << ',' << x << ',' << y << ','
        << w << ',' << h << ','
        << partitioner.currQtDepth << ',' << partitioner.currMtDepth << ','
        << name << ',' << m_qp << '\n';
}
