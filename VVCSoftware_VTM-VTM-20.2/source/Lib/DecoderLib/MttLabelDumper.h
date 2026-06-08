// ============================================================================
// #MTTDUMP  MttLabelDumper —— 解码时按采样把 MTT 划分标签写成 CSV（自定义新增文件）
//
// 目的：从 binEncode202 的 RA 码流里解出 MTT（BT/TT）划分，做成训练标签。整流解码是
//   必须的（CABAC 熵码只有解码器能还原划分树），但训练只需要很小一部分，所以这里在
//   解码遍历到每个 MTT 阶段节点时按「随机 POC 子集 × 随机 block 比例」采样写出。
//
// 完全由环境变量控制，**不设 MTT_DUMP_OUT 就彻底关闭、解码行为与原版一致**：
//   MTT_DUMP_OUT          输出 CSV 路径（未设 → 关闭）
//   MTT_DUMP_SEQ          序列名（写进 sequence 列，便于训练合并）
//   MTT_DUMP_QP           基础 QP（写进 QP 列；一个 bin 对应一个 QP）
//   MTT_DUMP_POCS         采样的 POC 列表，逗号分隔（空/未设 → 全部 POC）
//   MTT_DUMP_BLOCK_RATIO  每个 MTT 节点被写出的概率 0..1（默认 1.0）
//   MTT_DUMP_SEED         采样随机种子（默认 42；与坐标哈希，结果可复现、与解码顺序无关）
//
// 采样的「节点」= QT 叶子及其下的每个 BT/TT 决策节点（即 splitMode≠QUAD 的节点），
//   会跳过图像边界的隐式强制划分（那不是真正的决策）。仅亮度通道。
//   每行：sequence,POC_ID,pos_x,pos_y,w,h,qtDepth,mttDepth,splitMode,QP
//   splitMode ∈ {NO, BTH, BTV, TTH, TTV}
// ============================================================================
#pragma once

#include <cstdint>
#include <fstream>
#include <set>
#include <string>

#include "CommonLib/UnitPartitioner.h"

class CodingStructure;

class MttLabelDumper
{
public:
  static MttLabelDumper& instance();

  bool enabled() const { return m_enabled; }

  // 在 CABACReader::coding_tree() 解出 splitMode 后调用一次。内部自带全部过滤与采样。
  void maybeDump(const CodingStructure& cs, const Partitioner& partitioner, PartSplit splitMode);

private:
  MttLabelDumper();
  ~MttLabelDumper();

  bool          m_enabled = false;
  std::ofstream m_ofs;
  std::string   m_seq;
  int           m_qp        = 0;
  double        m_blockRatio = 1.0;
  uint64_t      m_seed       = 42;
  bool          m_allPocs    = true;
  std::set<int> m_pocs;
};
