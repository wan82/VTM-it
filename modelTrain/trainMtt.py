#!/usr/bin/env python3
"""
trainMtt —— MTT 划分模式（splitMode 5 分类）训练。

与 QT(train.py) 分开、独立的 MTT 模型：读 TrainingMtt.csv，按基础 QP 分 4 个模型，
预测每个 MTT 节点的 splitMode ∈ {NO,BTH,BTV,TTH,TTV}。
  · 按序列 GroupKFold(5) 评估（无内容泄漏）；
  · 类别严重不平衡（NO~64%），统一用 balanced 样本权重，主指标看 macro-F1 与逐类 F1；
  · 算法级选择：在 决策树/随机森林 里挑「4 个 QP 平均 macro-F1」最高者（朴素贝叶斯仅作参考，
    不可序列化成森林 bin）；
  · 训练时直接导出 C++(VTM) 可读的 mttForest.bin（复用 train.py 的序列化，格式 "QTF1"）。

splitMode → 类别码（C++ 侧据此映射回 PartSplit）：NO=0, BTH=1, BTV=2, TTH=3, TTV=4
"""
import argparse
import datetime as dt
import json
import os
import struct
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (confusion_matrix, classification_report,
                             f1_score, accuracy_score)
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_sample_weight

# 复用 QT 训练里「直接导出 C++ 可读森林 bin」的序列化（保证与 QtForestPredictor 格式一致）
from train import forest_block_bytes, plot_cm

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT  = os.path.join(_ROOT, "Data/TrainingData/TrainingMtt.csv")
DEFAULT_OUT    = os.path.join(_ROOT, "Data/ModelResults")       # mttForest.bin + mttModels/*.json
DEFAULT_REPORT = os.path.join(_ROOT, "Report/MttTraining")      # report.md + 混淆矩阵

ID_COLS = ["sequence", "POC_ID", "pos_x", "pos_y"]
LABEL = "splitMode"
QP_COL = "QP"
BASE_QPS = [22, 27, 32, 37]
SPLIT_CODE = {"NO": 0, "BTH": 1, "BTV": 2, "TTH": 3, "TTV": 4}
CODE_NAME = {v: k for k, v in SPLIT_CODE.items()}

# 候选特征（除 ID/label/baseQP 外的列；②③④ + CTU 上下文 + 形状/深度 + 时域残差）
FEATURES = ["w", "h", "qtDepth", "mttDepth",
            "qpValue", "aspectLog2", "areaLog2",
            "dirEx", "dirEy", "dirRatio", "rowMeanVar", "colMeanVar",
            "tbDiff", "lrDiff", "ttHDiff", "ttVDiff",
            "ctuMean", "ctuStd", "ctuLapMean", "ctuLapStd", "ctuSobelMean", "ctuSobelStd"]

# 时域残差特征（res_*）：在残差块 D=(当前块−L0同位块) 上算。
#   对 MTT 关键的是「划分代理」res_tbDiff/lrDiff/ttHDiff/ttVDiff —— 残差能量落在哪条切线 → BT/TT。
#   若 TrainingMtt.csv 含这些列则自动纳入；旧数据集缺列时自动忽略（见 select_features）。
RES_FEATURES = ["res_mean", "res_std", "res_skew", "res_kurt",
                "res_dirEx", "res_dirEy", "res_dirRatio", "res_rowMeanVar", "res_colMeanVar",
                "res_tbDiff", "res_lrDiff", "res_ttHDiff", "res_ttVDiff"]


def select_features(df):
    """实际可用特征 = FEATURES + (数据里存在的 RES_FEATURES)。兼容无残差的旧数据集。"""
    feats = [c for c in FEATURES if c in df.columns]
    feats += [c for c in RES_FEATURES if c in df.columns]
    return feats


def make_models(seed, rf_estimators, min_leaf):
    # class_weight=balanced：树模型直接平衡；NB 无 class_weight，用 sample_weight 统一处理
    return {
        "NaiveBayes":   GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=seed, min_samples_leaf=min_leaf,
                                               class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=rf_estimators, min_samples_leaf=min_leaf,
                                               random_state=seed, n_jobs=-1, class_weight="balanced"),
    }


def fit_one(model, X, y):
    sw = compute_sample_weight("balanced", y)
    try:
        model.fit(X, y, sample_weight=sw)
    except TypeError:
        model.fit(X, y)
    return model


def cv_eval(model_factory, X, y, groups, classes, folds):
    """GroupKFold 折外预测，返回 (oof_pred, macroF1, weightedF1, acc)。"""
    yp = np.empty_like(y)
    gkf = GroupKFold(n_splits=folds)
    for tr, te in gkf.split(X, y, groups):
        m = model_factory()
        fit_one(m, X[tr], y[tr])
        yp[te] = m.predict(X[te])
    macro = f1_score(y, yp, average="macro", labels=classes, zero_division=0)
    wf1 = f1_score(y, yp, average="weighted", labels=classes, zero_division=0)
    return yp, macro, wf1, accuracy_score(y, yp)


def rank_features_mi(df, feats, seed):
    sub = df.sample(12000, random_state=seed) if len(df) > 12000 else df
    mi = mutual_info_classif(sub[feats], sub["code"], random_state=seed)
    order = sorted(zip(feats, mi), key=lambda t: t[1], reverse=True)
    return order  # [(feat, mi), ...]


def main():
    ap = argparse.ArgumentParser(description="训练 MTT splitMode 5 分类（按 QP 分 4 模型）")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT, help="模型产物目录(mttForest.bin + mttModels/*.json)")
    ap.add_argument("--report", default=DEFAULT_REPORT, help="报告目录")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rf-estimators", type=int, default=100)
    ap.add_argument("--rf-min-leaf", type=int, default=50, help="限制 RF/决策树规模")
    ap.add_argument("--no-res", action="store_true", help="不使用 res_* 残差特征（用于 A/B 对照）")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[错误] 找不到输入: {args.input}（先 make mttFeature 生成 TrainingMtt.csv）")
    os.makedirs(args.out, exist_ok=True)
    figdir = os.path.join(args.report, "figures")
    os.makedirs(figdir, exist_ok=True)
    mdir = os.path.join(args.out, "mttModels")
    os.makedirs(mdir, exist_ok=True)

    df = pd.read_csv(args.input)
    n0 = len(df)
    feats = select_features(df)          # FEATURES + 数据里存在的 res_*（兼容旧数据集）
    if args.no_res:                      # A/B 对照：剔掉残差特征，只留空域/形状
        feats = [c for c in feats if c not in RES_FEATURES]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feats + [LABEL, QP_COL])
    df = df[df[LABEL].isin(SPLIT_CODE)].copy()
    df["code"] = df[LABEL].map(SPLIT_CODE).astype(int)
    df.reset_index(drop=True, inplace=True)
    print(f"[特征] 共 {len(feats)} 个："
          f"{len([c for c in feats if c in RES_FEATURES])} 个残差 res_* + 其余空域/形状")

    report = {"input": args.input, "rows_raw": n0, "rows_clean": len(df),
              "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "features": feats}

    # 特征 MI 排名（仅供报告/参考；默认用全部特征）
    mi_order = rank_features_mi(df, feats, args.seed)

    classes = [0, 1, 2, 3, 4]
    metric_rows = []          # metrics.csv
    fig_index = {}            # qp -> {model -> 相对图路径}
    chosen_scores = {"DecisionTree": [], "RandomForest": [], "NaiveBayes": []}
    perqp = {}                # qp -> 评估细节

    for qp in BASE_QPS:
        sub = df[df[QP_COL] == qp]
        if len(sub) < 50:
            print(f"[跳过] QP{qp} 样本太少({len(sub)})"); continue
        X = sub[feats].values
        y = sub["code"].values
        groups = sub["sequence"].values
        fig_index[qp] = {}
        for mname in ["NaiveBayes", "DecisionTree", "RandomForest"]:
            factory = lambda mn=mname: make_models(args.seed, args.rf_estimators, args.rf_min_leaf)[mn]
            yp, macro, wf1, acc = cv_eval(factory, X, y, groups, classes, args.folds)
            cm = confusion_matrix(y, yp, labels=classes)
            names = [CODE_NAME[c] for c in classes]
            figpath = os.path.join(figdir, f"cm_QP{qp}_{mname}.png")
            plot_cm(cm, names, f"QP{qp} - {mname}", figpath)
            fig_index[qp][mname] = os.path.relpath(figpath, args.report)
            per = f1_score(y, yp, average=None, labels=classes, zero_division=0)
            metric_rows.append({"QP": qp, "model": mname, "macroF1": round(macro, 4),
                                "weightedF1": round(wf1, 4), "acc": round(acc, 4),
                                **{f"F1_{CODE_NAME[c]}": round(float(per[i]), 3) for i, c in enumerate(classes)}})
            chosen_scores[mname].append(macro)
            print(f"  QP{qp} {mname:12} macroF1={macro:.4f} wF1={wf1:.4f} acc={acc:.4f}")
        perqp[qp] = {"n": int(len(sub)),
                     "dist": {CODE_NAME[c]: int((y == c).sum()) for c in classes}}

    # 算法级选择：只在可序列化的 DT/RF 里挑（4 QP 平均 macro-F1 最高）
    avg = {m: (float(np.mean(s)) if s else -1) for m, s in chosen_scores.items()}
    best_algo = max(["DecisionTree", "RandomForest"], key=lambda m: avg[m])
    report["avg_macroF1"] = {m: round(v, 4) for m, v in avg.items()}
    report["chosen_algo"] = best_algo

    # 用选中的算法在每个 QP 的全量数据上重训，序列化进 mttForest.bin
    blocks = bytearray()
    blocks += b"QTF1" + struct.pack("<i", 0)  # 占位：magic + nModels（后填）
    nModels = 0
    for qp in BASE_QPS:
        sub = df[df[QP_COL] == qp]
        if len(sub) < 50:
            continue
        model = make_models(args.seed, args.rf_estimators, args.rf_min_leaf)[best_algo]
        fit_one(model, sub[feats].values, sub["code"].values)
        blk = forest_block_bytes(qp, feats, model)
        if blk is None:
            print(f"[警告] QP{qp} {best_algo} 非树模型，跳过序列化"); continue
        blocks += blk; nModels += 1
        json.dump({"baseQp": qp, "algo": best_algo, "features": feats,
                   "classes": classes, "code_name": CODE_NAME,
                   "n": int(len(sub))},
                  open(os.path.join(mdir, f"QP{qp}.json"), "w"), ensure_ascii=False, indent=2)
    blocks[4:8] = struct.pack("<i", nModels)
    forest_path = os.path.join(args.out, "mttForest.bin")
    with open(forest_path, "wb") as f:
        f.write(blocks)

    # ---- 报告 ----
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(args.report, "metrics.csv"), index=False)
    write_report(args, report, mi_order, metric_rows, fig_index, perqp,
                 forest_path, nModels)

    print(f"\n选中算法: {best_algo} | 平均 macroF1: {report['avg_macroF1']}")
    print(f"模型 -> {forest_path}（{nModels} 个 QP 块）")
    print(f"报告 -> {os.path.join(args.report, 'report.md')} | 指标 -> {os.path.join(args.report,'metrics.csv')}")


def write_report(args, report, mi_order, metric_rows, fig_index, perqp, forest_path, nModels):
    L = []
    A = L.append
    A(f"# MTT 划分模式训练报告（splitMode 5 分类）\n")
    A(f"- 生成时间：{report['time']}")
    A(f"- 输入：`{report['input']}`")
    A(f"- 清洗后样本：{report['rows_clean']} / {report['rows_raw']}")
    A(f"- 评估：按序列 GroupKFold(5)，balanced 样本权重；主指标 **macro-F1**（类别极不平衡，不看 accuracy）")
    A(f"- **选中算法：{report['chosen_algo']}**（4 个 QP 平均 macro-F1：{report['avg_macroF1']}）")
    A(f"- 类别码：NO=0, BTH=1, BTV=2, TTH=3, TTV=4\n")

    A("## 各 QP 样本与类别分布\n")
    A("| QP | 样本 | NO | BTH | BTV | TTH | TTV |")
    A("|----|------|----|-----|-----|-----|-----|")
    for qp in BASE_QPS:
        if qp not in perqp: continue
        d = perqp[qp]["dist"]
        A(f"| {qp} | {perqp[qp]['n']} | {d['NO']} | {d['BTH']} | {d['BTV']} | {d['TTH']} | {d['TTV']} |")
    A("")

    A("## 交叉验证指标（macro-F1 / weighted-F1 / acc + 逐类 F1）\n")
    cols = ["QP", "model", "macroF1", "weightedF1", "acc",
            "F1_NO", "F1_BTH", "F1_BTV", "F1_TTH", "F1_TTV"]
    A("| " + " | ".join(cols) + " |")
    A("|" + "|".join(["----"] * len(cols)) + "|")
    for r in metric_rows:
        A("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    A("")

    A("## 特征互信息排名（参考，默认用全部特征）\n")
    A("| 特征 | MI |")
    A("|------|----|")
    for f, v in mi_order:
        A(f"| {f} | {v:.4f} |")
    A("")

    A("## 混淆矩阵（折外预测，行=真值 列=预测）\n")
    for qp in BASE_QPS:
        if qp not in fig_index: continue
        A(f"### QP{qp}\n")
        for mname in ["NaiveBayes", "DecisionTree", "RandomForest"]:
            if mname in fig_index[qp]:
                A(f"![{qp}-{mname}]({fig_index[qp][mname]})")
        A("")

    A("## 模型产物\n")
    A(f"- `mttForest.bin`（{nModels} 个 QP 的森林合一，格式 \"QTF1\"，与 QtForestPredictor 同），"
      f"编码时单独加载用于 MTT 逐节点剪枝；")
    A(f"- 每 QP 元数据 `mttModels/QP<qp>.json`。")
    A(f"- 朴素贝叶斯仅作对照，不导出（森林 bin 只存树模型）。\n")

    with open(os.path.join(args.report, "report.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
