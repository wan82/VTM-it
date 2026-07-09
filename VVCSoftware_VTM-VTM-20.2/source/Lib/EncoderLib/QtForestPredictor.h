// ============================================================================
// #QTPRED  QtForestPredictor —— 在编码器内用训练好的随机森林预测 CTU 的 QT 最大深度
//
// 这是嵌入到 VTM 的“快速 CU 划分”推理模块（自定义新增文件，原版 VTM 没有）。
//   · 模型由 modelTrain 训练时直接序列化成 qtForest.bin（不经 joblib 转换）；
//   · 运行时通过环境变量 QT_FOREST_MODEL 指定该 bin 文件路径；未设或加载失败 → 关闭，
//     编码行为与原版完全一致。
//   · 对一个 128x128 的亮度 CTU 提取 [Sobel_mean,Sobel_std,Lap_std,std,Lap_mean]
//     等空域特征，再结合按 VTM 推导的 QP-value，喂入对应基础 QP 的随机森林，得到
//     预测的 qt_max（0..），EncCu 据此在该层级终止 QT 划分。
// ============================================================================
#pragma once

#include <string>
#include <vector>

class QtForestPredictor
{
public:
  static QtForestPredictor& instance();          // 懒加载单例（线程安全的只读模型）

  bool enabled() const { return m_enabled; }
  bool hasModelForQp(int baseQp) const { return modelForQp(baseQp) != nullptr; }

  double confThreshold() const { return m_confThreshold; } // 置信度门控阈值(QT_FOREST_CONF)
  bool   forceSplitEnabled() const { return m_forceSplit; }// 下界强制分(QT_FOREST_FORCE)

  // blk: 128x128 亮度块（必须已缩放到 10bit，与训练一致）；baseQp ∈ {22,27,32,37}；poc 当前帧 POC
  // l0blk: L0 参考帧同位 128x128 重建块（同样10bit）。非空时算 res_* 残差特征(当前−L0)；
  //        空则不提供 res_*（若模型用到 res_* 会因缺特征返回 -1）。
  // 返回预测 qt_max（0..）；不可用时返回 -1。conf 回填预测类别的概率(置信度, 0..1)。
  int predictQtMax(const std::vector<std::vector<int>>& blk, int baseQp, int poc,
                   double* conf = nullptr,
                   const std::vector<std::vector<int>>& l0blk = {}) const;

  // 直接用已组装好的特征向量预测类别（顺序同模型 featNames，末尾含 QP-value）。
  // 供单元测试与高级调用；不可用返回 -1。conf 回填置信度。
  int predictClassFromRaw(const std::vector<double>& x, int baseQp, double* conf = nullptr) const;

private:
  QtForestPredictor();

  struct Tree
  {
    std::vector<int>                 left, right, feat;  // sklearn: 叶子 left==-1, feat==-2
    std::vector<double>              thr;                // 分裂阈值（x[feat] <= thr → 左）
    std::vector<std::vector<double>> prob;               // 每个节点的类别概率（叶子用）
  };
  struct Model
  {
    int                       baseQp = 0;
    std::vector<std::string>  featNames;   // 特征顺序（含末尾的 QP-value 列名，如 "22QP"）
    std::vector<int>          classes;     // 类别标签（qt_max 取值）
    std::vector<Tree>         trees;
  };

  void load(const std::string& path);
  const Model* modelForQp(int baseQp) const;

  bool               m_enabled = false;
  double             m_confThreshold = 0.7;   // 置信度门控阈值（环境变量 QT_FOREST_CONF）
  bool               m_forceSplit = true;     // 是否启用下界强制分（环境变量 QT_FOREST_FORCE）
  std::vector<Model> m_models;
};
