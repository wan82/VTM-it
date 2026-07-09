// ============================================================================
// #MTTPRED  MttForestPredictor —— 在编码器内用训练好的森林预测 MTT 划分模式(splitMode)
//
// 与 QtForestPredictor 平行、独立的 MTT 预测器（原版 VTM 没有）。
//   · 模型由 modelTrain/trainMtt.py 训练时直接序列化成 mttForest.bin（格式 "QTF1"，
//     与 qtForest.bin 同一套；按基础 QP 分 4 个模型）。
//   · 运行时由环境变量 MTT_FOREST_MODEL 指定 bin 路径；未设或加载失败 → 关闭，
//     编码行为与原版完全一致。
//   · 对一个 w×h 的亮度块提取 22 个特征（②方向性 + ③划分代价代理 + ④形状/QP +
//     ①该块所在 128 CTU 的复杂度上下文），喂入对应基础 QP 的森林，预测 splitMode：
//        NO=0, BTH=1, BTV=2, TTH=3, TTV=4
//     EncCu 据此在 MTT 决策节点剪掉与预测相悖的 BT/TT 模式（置信度门控）。
//
// 环境变量：
//   MTT_FOREST_MODEL   bin 路径（未设 → 关闭）
//   MTT_FOREST_CONF    置信度门控阈值（默认 0.7）
//   MTT_FOREST_PRUNE_NO 预测为 NO(不分) 且确信时是否跳过全部 BT/TT（默认 1）
//   MTT_FOREST_PRUNE   "orient"(默认,只剪相反方向) / "exact"(只留预测那一种 BTT)
// ============================================================================
#pragma once

#include <cstdint>
#include <string>
#include <vector>

class MttForestPredictor
{
public:
  static MttForestPredictor& instance();

  bool enabled() const { return m_enabled; }
  bool hasModelForQp(int baseQp) const { return modelForQp(baseQp) != nullptr; }

  double confThreshold()  const { return m_confThreshold; }   // MTT_FOREST_CONF
  bool   pruneNoEnabled() const { return m_pruneNo; }         // MTT_FOREST_PRUNE_NO
  bool   pruneExact()     const { return m_pruneExact; }      // MTT_FOREST_PRUNE=exact

  using Mat = std::vector<std::vector<int>>;

  // 算 CTU 级 ① 上下文(6维: mean,std,lapMean,lapStd,sobelMean,sobelStd)。
  // ctu: 该块所在 128×128 CTU（裁剪到图像内）的 10bit 亮度块。
  void computeCtuCtx(const Mat& ctu, double out6[6]) const;

  // 预测 splitMode 码(0..4) + 置信度。
  //   blk: 当前 w×h 亮度块(10bit)；ctuCtx6: 该块所在 CTU 的 ① 上下文（computeCtuCtx 产物）。
  //   不可用返回 -1；conf 回填预测类别概率(0..1)。
  int predictSplit(const Mat& blk, const double ctuCtx6[6], int w, int h,
                   int qtDepth, int mttDepth, int poc, int baseQp, double* conf) const;

  // 按对应 QP 模型的 featNames 顺序组装 22 维特征向量（供一致性自测；不可用返回空）。
  std::vector<double> featuresFor(const Mat& blk, const double ctuCtx6[6], int w, int h,
                                  int qtDepth, int mttDepth, int poc, int baseQp) const;

private:
  MttForestPredictor();

  struct Tree
  {
    std::vector<int>                 left, right, feat;
    std::vector<double>              thr;
    std::vector<std::vector<double>> prob;
  };
  struct Model
  {
    int                       baseQp = 0;
    std::vector<std::string>  featNames;
    std::vector<int>          classes;
    std::vector<Tree>         trees;
  };

  void load(const std::string& path);
  const Model* modelForQp(int baseQp) const;
  int  predictFromRaw(const std::vector<double>& x, int baseQp, double* conf) const;

  bool               m_enabled       = false;
  double             m_confThreshold = 0.7;
  bool               m_pruneNo       = true;
  bool               m_pruneExact    = false;
  std::vector<Model> m_models;
};
