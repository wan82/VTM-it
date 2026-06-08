#!/usr/bin/env python3
"""
modelTrain —— 读取 Training.csv，做数据清洗 + 特征筛选，
按 QP 分别训练 朴素贝叶斯 / 决策树 / 随机森林（多分类 qt_max 0–4），
用 GroupKFold(5, 按序列分组) 评估，生成混淆矩阵图与完整 Markdown 报告。

用法：
  python train.py
  python train.py --input Training.csv --out DIR [--corr-thr 0.95] [--top-k 0] [--seed 42]
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
                             accuracy_score, f1_score)
from sklearn.feature_selection import mutual_info_classif

ID_COLS = ["sequence", "POC_ID", "pos_x", "pos_y"]
LABEL_COLS = ["22qt_max", "27qt_max", "32qt_max", "37qt_max"]
# QP-value 列（由 POC 按 VTM RA 推导的实际 slice QP）：默认当作元数据，不进特征集，
# 以保持现有 18 个空域特征不变。如需把匹配的 QP-value 当特征用，可后续扩展。
QP_VALUE_COLS = ["22QP", "27QP", "32QP", "37QP"]

# 仓库根目录：由本脚本位置推导（modelTrain 的父目录），换机器/clone 后无需改路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(_ROOT, "Data/TrainingData/Training.csv")
DEFAULT_OUT = os.path.join(_ROOT, "Data/ModelResults")            # 模型产物(bin/json)
DEFAULT_REPORT = os.path.join(_ROOT, "Report/ModelTraining")      # 报告(md/csv/图)


def make_models(seed, rf_estimators=100, min_leaf=50):
    return {
        "NaiveBayes":   GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=seed, min_samples_leaf=min_leaf),
        "RandomForest": RandomForestClassifier(n_estimators=rf_estimators, min_samples_leaf=min_leaf,
                                               random_state=seed, n_jobs=-1),
    }


# ---------------- 训练时直接产出 C++(VTM) 可读的森林字节（小端二进制） ----------------
def _tree_bytes(tree, n_classes):
    nn = int(tree.node_count)
    buf = bytearray(struct.pack("<i", nn))
    cl, cr, feat, thr = tree.children_left, tree.children_right, tree.feature, tree.threshold
    for n in range(nn):
        buf += struct.pack("<iiid", int(cl[n]), int(cr[n]), int(feat[n]), float(thr[n]))
    val = tree.value
    for n in range(nn):
        v = np.asarray(val[n][0], dtype=float); s = v.sum()
        p = (v / s) if s > 0 else np.zeros(n_classes)
        for c in range(n_classes):
            buf += struct.pack("<d", float(p[c]))
    return buf


def forest_block_bytes(base_qp, features, model):
    """把一个已拟合的树/森林模型序列化成 qtForest.bin 的一个模型块；非树模型返回 None。"""
    if hasattr(model, "estimators_"):
        trees = [e.tree_ for e in model.estimators_]
    elif hasattr(model, "tree_"):
        trees = [model.tree_]
    else:
        return None
    classes = [int(c) for c in model.classes_]
    nc = len(classes)
    buf = bytearray()
    buf += struct.pack("<i", int(base_qp))
    buf += struct.pack("<i", len(features))
    for name in features:
        b = name.encode("utf-8"); buf += struct.pack("<i", len(b)) + b
    buf += struct.pack("<i", nc)
    for c in classes:
        buf += struct.pack("<i", c)
    buf += struct.pack("<i", len(trees))
    for t in trees:
        buf += _tree_bytes(t, nc)
    return buf


# ---------------- 数据清洗 ----------------
def clean_data(df, feature_cols, report):
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + LABEL_COLS).reset_index(drop=True)
    n_dup = int(df.duplicated(subset=feature_cols + LABEL_COLS).sum())
    for c in LABEL_COLS:
        df[c] = df[c].round().astype(int)
    report["rows_raw"] = n0
    report["rows_clean"] = len(df)
    report["rows_dropped"] = n0 - len(df)
    report["dup_rows"] = n_dup           # 仅统计，不删除
    return df


# ---------------- 特征重要性排名（去冗余 + 互信息） ----------------
def rank_features(df, feature_cols, corr_thr, report):
    X = df[feature_cols]
    # 1) 去零方差 / 近常量
    stds = X.std()
    near_const = stds[stds < 1e-9].index.tolist()
    feats = [c for c in feature_cols if c not in near_const]
    # 2) 去高相关冗余（|corr| > 阈值的成对，丢后者）
    corr = X[feats].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_corr = [c for c in upper.columns if (upper[c] > corr_thr).any()]
    candidates = [c for c in feats if c not in drop_corr]
    # 3) 互信息（每个 QP 一组 + 4 个 QP 平均），kNN 估计大数据时抽样加速
    mi_df = df.sample(8000, random_state=0) if len(df) > 8000 else df
    mi = {}
    for lab in LABEL_COLS:
        m = mutual_info_classif(mi_df[candidates], mi_df[lab], random_state=0)
        mi[lab] = dict(zip(candidates, [float(v) for v in m]))
    mi_avg = {f: float(np.mean([mi[lab][f] for lab in LABEL_COLS])) for f in candidates}
    ranked = sorted(candidates, key=lambda f: mi_avg[f], reverse=True)
    report["dropped_near_constant"] = near_const
    report["dropped_high_corr"] = drop_corr
    report["mi_per_qp"] = mi
    report["mi_avg"] = mi_avg
    report["ranked"] = ranked
    return ranked, mi_avg


# ---------------- 包裹式前向选择（按重要性逐个加，找甜蜜点） ----------------
def wrapper_select(df, ranked, qp_feat_map, args, report):
    """按 MI 重要性从高到低逐个加特征，每步评估 3 个模型 × 4 个 QP 的平均 Weighted-F1；
       取‘性能达到峰值−容差’的最小特征数作为甜蜜点（兼顾性能与精简）。"""
    work = df
    if args.wrapper_sample and 0 < args.wrapper_sample < len(df):
        work = df.sample(args.wrapper_sample, random_state=args.seed).reset_index(drop=True)
    groups = work["sequence"].values
    ys = {qp: work[qp].values.astype(int) for qp in LABEL_COLS}

    curve = []   # (k, 新加入的特征, 平均weighted_f1)
    for k in range(1, len(ranked) + 1):
        subset = ranked[:k]
        scores = []
        for qp in LABEL_COLS:
            X = work[subset + [qp_feat_map[qp]]].values
            y = ys[qp]
            for model in make_models(args.seed, args.wrapper_rf, args.rf_min_leaf).values():
                gkf = GroupKFold(n_splits=args.wrapper_folds)
                yp = np.empty_like(y)
                for tr, te in gkf.split(X, y, groups):
                    model.fit(X[tr], y[tr]); yp[te] = model.predict(X[te])
                scores.append(f1_score(y, yp, average="weighted"))
        curve.append((k, ranked[k - 1], float(np.mean(scores))))
        print(f"    wrapper k={k:<2} +{ranked[k-1]:<12} 平均wF1={curve[-1][2]:.4f}")

    best = max(c[2] for c in curve)
    chosen_k = next(k for (k, _, p) in curve if p >= best - args.wrapper_tol)
    selected = ranked[:chosen_k]
    report["wrapper_curve"] = curve
    report["wrapper_best"] = best
    report["wrapper_chosen_k"] = chosen_k
    return selected


def plot_wrapper(curve, chosen_k, path):
    ks = [c[0] for c in curve]; ps = [c[2] for c in curve]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, ps, "-o", ms=4)
    ax.axvline(chosen_k, color="red", ls="--", label=f"sweet spot k={chosen_k}")
    ax.set_xlabel("number of features (by MI rank)")
    ax.set_ylabel("mean Weighted-F1 (3 models x 4 QP)")
    ax.set_title("Wrapper forward feature selection")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------- 混淆矩阵作图 ----------------
def plot_cm(cm, classes, title, path):
    cm = cm.astype(int)
    rowsum = cm.sum(axis=1, keepdims=True).astype(float)
    rowsum[rowsum == 0] = 1.0
    cmn = cm / rowsum
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]*100:.0f}%", ha="center", va="center",
                    fontsize=7, color="white" if cmn[i, j] > 0.6 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ---------------- 报告渲染 ----------------
def class_dist_table(df):
    lines = ["| QP | " + " | ".join(f"qt={k}" for k in range(5)) + " | 合计 |",
             "|----|" + "----|" * 6]
    for lab in LABEL_COLS:
        vc = df[lab].value_counts()
        row = [str(int(vc.get(k, 0))) for k in range(5)]
        lines.append(f"| {lab} | " + " | ".join(row) + f" | {len(df)} |")
    return "\n".join(lines)


def mi_table(report):
    order = report.get("ranked") or sorted(report["mi_avg"],
                                           key=lambda f: report["mi_avg"][f], reverse=True)
    head = "| 特征 | 平均MI | " + " | ".join(LABEL_COLS) + " |"
    sep = "|------|--------|" + "------|" * len(LABEL_COLS)
    lines = [head, sep]
    for f in order:
        cells = " | ".join(f"{report['mi_per_qp'][lab][f]:.3f}" for lab in LABEL_COLS)
        lines.append(f"| {f} | {report['mi_avg'][f]:.3f} | {cells} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="训练 NB/DT/RF 并出混淆矩阵与报告")
    ap.add_argument("--input", default=DEFAULT_INPUT, help="Training.csv 路径")
    ap.add_argument("--out", default=DEFAULT_OUT, help="模型产物目录(qtForest.bin + models/*.json)")
    ap.add_argument("--report", default=DEFAULT_REPORT, help="报告目录(report.md/metrics.csv/figures)")
    ap.add_argument("--corr-thr", type=float, default=0.95, help="高相关剔除阈值")
    ap.add_argument("--top-k", type=int, default=0, help=">0 时直接按互信息取前K个(跳过wrapper)")
    ap.add_argument("--folds", type=int, default=5, help="交叉验证折数")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--rf-estimators", type=int, default=100, help="随机森林树数")
    ap.add_argument("--rf-min-leaf", type=int, default=50,
                    help="树/森林叶子最小样本数(同时限制DT和RF规模)。默认50：模型小而快、可直接嵌入")
    ap.add_argument("--max-rows", type=int, default=0, help=">0 时随机抽样这么多行(加速/调试)")
    # —— 包裹式前向特征选择 ——
    ap.add_argument("--no-wrapper", action="store_true", help="关闭 wrapper，直接用全部去冗余后的特征")
    ap.add_argument("--wrapper-tol", type=float, default=0.003,
                    help="甜蜜点容差：取性能达到峰值−tol 的最小特征数")
    ap.add_argument("--wrapper-rf", type=int, default=40, help="wrapper 搜索时的随机森林树数")
    ap.add_argument("--wrapper-folds", type=int, default=3, help="wrapper 搜索时的折数")
    ap.add_argument("--wrapper-sample", type=int, default=10000,
                    help="wrapper 搜索时的抽样行数(0=全量)")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"[错误] 找不到输入: {args.input}")
    figdir = os.path.join(args.report, "figures")   # 图归报告目录
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)

    df = pd.read_csv(args.input)
    if args.max_rows and args.max_rows > 0 and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ID_COLS + LABEL_COLS + QP_VALUE_COLS]
    if "sequence" not in df.columns:
        sys.exit("[错误] Training.csv 缺少 sequence 列（GroupKFold 需要分组）")

    report = {}
    df = clean_data(df, feature_cols, report)

    # 特征：先去冗余 + 按互信息排名，再用 wrapper 前向选择找甜蜜点
    qp_feat_map = dict(zip(LABEL_COLS, QP_VALUE_COLS))
    ranked, mi_avg = rank_features(df, feature_cols, args.corr_thr, report)
    if args.top_k and args.top_k > 0:
        selected = ranked[:args.top_k]
        report["fs_method"] = f"top-{args.top_k}(按互信息)"
    elif not args.no_wrapper:
        print("包裹式前向特征选择(按互信息排名逐个加)…")
        selected = wrapper_select(df, ranked, qp_feat_map, args, report)
        plot_wrapper(report["wrapper_curve"], report["wrapper_chosen_k"],
                     os.path.join(figdir, "featureWrapper.png"))
        report["fs_method"] = f"wrapper 甜蜜点 k={report['wrapper_chosen_k']}"
    else:
        selected = list(ranked)
        report["fs_method"] = "全部去冗余后特征"
    report["selected_features"] = selected

    # 每个 QP 的特征 = 选出的空域特征 + 该 QP 自己的 QP-value（匹配的那一个）
    qp_feat_map = dict(zip(LABEL_COLS, QP_VALUE_COLS))
    print(f"清洗后 {len(df)} 行 | 空域特征 {len(feature_cols)} → 选用 {len(selected)} 个，"
          f"每个 QP 再加 1 个匹配 QP-value")
    print(f"分组 5 折(按序列, 共 {df['sequence'].nunique()} 个序列)")

    groups = df["sequence"].values

    # 评估
    metric_rows = []       # 给 metrics.csv
    fig_index = {}         # (qp) -> {model: figpath}
    saved_models = {}      # qp -> dict(model, bin, json)
    mdir = os.path.join(args.out, "models")
    os.makedirs(mdir, exist_ok=True)
    for qp in LABEL_COLS:
        y = df[qp].values.astype(int)
        classes = sorted(np.unique(y).tolist())
        feats_qp = selected + [qp_feat_map[qp]]   # 空域特征 + 匹配 QP-value
        X = df[feats_qp].values
        fig_index[qp] = {}
        for mname, model in make_models(args.seed, args.rf_estimators, args.rf_min_leaf).items():
            # 手动 GroupKFold：单趟训练即可同时得到折外预测和每折准确率
            gkf = GroupKFold(n_splits=args.folds)
            y_pred = np.empty_like(y)
            fold_acc = []
            for tr, te in gkf.split(X, y, groups):
                model.fit(X[tr], y[tr])
                p = model.predict(X[te])
                y_pred[te] = p
                fold_acc.append(accuracy_score(y[te], p))
            accs = np.array(fold_acc)
            acc = accuracy_score(y, y_pred)
            mf1 = f1_score(y, y_pred, average="macro")
            wf1 = f1_score(y, y_pred, average="weighted")
            cm = confusion_matrix(y, y_pred, labels=classes)
            figpath = os.path.join(figdir, f"cm_{qp}_{mname}.png")
            plot_cm(cm, classes, f"{qp} - {mname}", figpath)
            fig_index[qp][mname] = os.path.relpath(figpath, args.report)
            metric_rows.append({
                "QP": qp, "model": mname,
                "acc": round(acc, 4),
                "cv_acc_mean": round(float(accs.mean()), 4),
                "cv_acc_std": round(float(accs.std()), 4),
                "macro_f1": round(mf1, 4),
                "weighted_f1": round(wf1, 4),
            })
            print(f"  {qp:<10} {mname:<13} acc={acc:.4f}  "
                  f"cvAcc={accs.mean():.4f}±{accs.std():.4f}  "
                  f"macroF1={mf1:.4f}  wF1={wf1:.4f}")

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(args.report, "metrics.csv"), index=False)

    # —— 模型选择：在算法层面进行 ——
    # 比较每种算法在 4 个 QP 上的平均 Weighted-F1，平均最好的那一种作为最终算法；
    # 然后用该算法在每个 QP 上各训一个模型保存（而不是每个 QP 各挑各的）。
    algo_mean = metrics_df.groupby("model")["weighted_f1"].mean()
    chosen = algo_mean.idxmax()
    print("\n各算法跨 4 个 QP 的平均 Weighted-F1: " +
          ", ".join(f"{m}={algo_mean[m]:.4f}" for m in algo_mean.index))
    print(f"最终选用算法: {chosen}")

    if chosen == "NaiveBayes":
        print("[警告] 最终算法是朴素贝叶斯，无法导出为森林 bin；"
              "如需嵌入 VTM，请用树/森林类算法（可 --no-wrapper 或调参使 RF/DT 胜出）。")

    saved_models = {}
    forest_blocks = []
    for qp in LABEL_COLS:
        y = df[qp].values.astype(int)
        feats_qp = selected + [qp_feat_map[qp]]           # 空域特征 + 匹配 QP-value
        X = df[feats_qp].values
        model = make_models(args.seed, args.rf_estimators, args.rf_min_leaf)[chosen]
        model.fit(X, y)                                   # 全量数据重训
        base_qp = int(qp.replace("qt_max", ""))
        blk = forest_block_bytes(base_qp, feats_qp, model)   # 直接序列化成 C++ 可读字节
        if blk is not None:
            forest_blocks.append(blk)
        row = metrics_df[(metrics_df.QP == qp) & (metrics_df.model == chosen)].iloc[0]
        meta = {
            "qp": qp, "baseQp": base_qp, "model": chosen,
            "features": feats_qp,                         # 特征顺序（嵌入时按此喂入；末尾为匹配QP-value）
            "classes": [int(c) for c in model.classes_],
            "cv_accuracy": float(row["acc"]),
            "cv_weighted_f1": float(row["weighted_f1"]),
            "cv_macro_f1": float(row["macro_f1"]),
            "trained_on_rows": int(len(X)),
        }
        with open(os.path.join(mdir, f"{qp}.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        saved_models[qp] = {"model": chosen, "wf1": float(row["weighted_f1"])}
        print(f"  -> [{qp}] {chosen} (wF1={row['weighted_f1']:.4f}) 已序列化")

    # 训练时直接产出 C++(VTM) 可读的森林模型，无需任何转换步骤
    forest_path = os.path.join(args.out, "qtForest.bin")
    if forest_blocks:
        with open(forest_path, "wb") as f:
            f.write(b"QTF1")
            f.write(struct.pack("<i", len(forest_blocks)))
            for b in forest_blocks:
                f.write(b)
        print(f"\nVTM 模型 -> {forest_path}  ({os.path.getsize(forest_path)/1024:.1f} KB)")
        print(f"编码时设置：export QT_FOREST_MODEL={forest_path}")

    write_report(args, df, report, selected, metrics_df, fig_index,
                 saved_models, algo_mean, chosen)
    print(f"报告 -> {os.path.join(args.report, 'report.md')}  | 指标 -> {os.path.join(args.report, 'metrics.csv')}")
    print(f"模型 -> {os.path.join(args.out, 'qtForest.bin')}  | 元数据 -> {mdir}/<QP>.json")
    print(f"混淆矩阵 -> {figdir}/cm_<QP>_<model>.png  (共 {len(metric_rows)} 张)")


def write_report(args, df, report, selected, metrics_df, fig_index,
                 saved_models, algo_mean, chosen):
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = []
    A = lines.append
    A(f"# 模型训练报告 —— QT 划分深度预测\n")
    A(f"> 生成时间 {now} · 输入 `{os.path.basename(args.input)}` · "
      f"GroupKFold({args.folds}, 按序列) · 多分类 qt_max(0–4)\n")

    A("## 1. 数据概览\n")
    A(f"- 原始行数：{report['rows_raw']}")
    A(f"- 清洗后行数：{report['rows_clean']}（丢弃含 NaN/Inf 行 {report['rows_dropped']}，"
      f"重复行 {report['dup_rows']} 仅统计未删）")
    A(f"- 序列数（分组用）：{df['sequence'].nunique()}\n")
    A("各 QP 下 qt_max 类别分布：\n")
    A(class_dist_table(df) + "\n")

    A("## 2. 数据清洗\n")
    A("- 把 ±Inf 视作缺失，连同 NaN 一并按特征/标签列删除（Training.csv 已基本干净，此处兜底）。")
    A("- 标签列四舍五入取整为类别。重复行仅统计，不删除（不同 CTU 偶有完全相同的特征也属正常）。\n")

    A("## 3. 特征筛选\n")
    nc = report["dropped_near_constant"]; hc = report["dropped_high_corr"]
    A(f"- 预处理：剔除近常量特征 {nc if nc else '无'}；"
      f"剔除高相关冗余（|corr|>{args.corr_thr}） {hc if hc else '无'}。")
    A(f"- 方法：**{report.get('fs_method','')}**。最终选用 **{len(selected)}** 个空域特征。")
    A(f"- 选用空域特征（{len(selected)} 个）：{selected}")
    A("- **QP-value 为必选特征，不参与 wrapper 选择**：每个 QP 的模型固定额外加入它自己的"
      " QP-value（22qt_max→22QP，27→27QP，32→32QP，37→37QP；与 VTM 部署时的用法一致），"
      "wrapper 只在空域特征里挑。")
    A(f"- **因此每个 QP 的模型最终只用 {len(selected)} 个空域特征 + 1 个匹配 QP-value "
      f"= {len(selected)+1} 个特征**，实际编码时计算开销很小。\n")

    if "wrapper_curve" in report:
        ck = report["wrapper_chosen_k"]; cb = report["wrapper_best"]
        A("### 3.1 包裹式前向选择（甜蜜点）\n")
        A("按平均互信息从高到低逐个加入特征，每步用 3 个模型 × 4 个 QP 的"
          "平均 Weighted-F1 评估；取**性能达到峰值−容差**的最小特征数为甜蜜点。\n")
        A(f"峰值平均 Weighted-F1 = {cb:.4f}，容差 = {args.wrapper_tol}，"
          f"**甜蜜点 k = {ck}**（即只保留前 {ck} 个特征）。\n")
        A("![wrapper](figures/featureWrapper.png)\n")
        A("| k | 新加入特征 | 平均 Weighted-F1 | 是否保留 |")
        A("|---|------------|------------------|----------|")
        for (k, feat, p) in report["wrapper_curve"]:
            keep = "✅" if k <= ck else ""
            A(f"| {k} | {feat} | {p:.4f} | {keep} |")
        A("")

    A("全部候选特征对标签的互信息（越大越有判别力，按平均 MI 降序）：\n")
    A(mi_table(report) + "\n")

    A("## 4. 模型评估（每个 QP × 3 个模型）\n")
    A("指标均为 5 折分组交叉验证的折外（out-of-fold）结果：\n")
    for qp in LABEL_COLS:
        A(f"### {qp}\n")
        sub = metrics_df[metrics_df.QP == qp].copy()
        A("| 模型 | 准确率 | CV准确率(均值±std) | Macro-F1 | Weighted-F1 |")
        A("|------|--------|--------------------|----------|-------------|")
        for _, r in sub.iterrows():
            A(f"| {r['model']} | {r['acc']:.4f} | {r['cv_acc_mean']:.4f}±{r['cv_acc_std']:.4f} "
              f"| {r['macro_f1']:.4f} | {r['weighted_f1']:.4f} |")
        best = sub.loc[sub["weighted_f1"].idxmax(), "model"]
        A(f"\n该 QP 下 Weighted-F1 最高：**{best}**\n")
        A("混淆矩阵（行=真实，列=预测，格内为计数与行归一化%）：\n")
        for mname in fig_index[qp]:
            A(f"- {mname}：![{qp}-{mname}]({fig_index[qp][mname]})")
        A("")

    A("## 5. 汇总\n")
    piv = metrics_df.pivot(index="QP", columns="model", values="weighted_f1")
    A("各 QP × 模型的 Weighted-F1：\n")
    A("| QP | " + " | ".join(piv.columns) + " |")
    A("|----|" + "----|" * len(piv.columns))
    for qp in piv.index:
        A(f"| {qp} | " + " | ".join(f"{piv.loc[qp, m]:.4f}" for m in piv.columns) + " |")
    overall_best = metrics_df.groupby("model")["weighted_f1"].mean().idxmax()
    A(f"\n- 跨 4 个 QP 平均 Weighted-F1 最高的模型：**{overall_best}**")
    A("- 注意：本评估按**序列分组**交叉验证，反映对“没见过的视频”的泛化；"
      "若改成随机分层折，指标通常会更高但偏乐观。\n")

    A("## 6. 模型选择与已保存模型（供嵌入 VTM）\n")
    A("**选择策略**：在**算法层面**选——比较每种算法在 4 个 QP 上的平均 Weighted-F1，"
      "平均最高的那一种作为最终算法（而非每个 QP 各挑各的，保证部署时只用一种模型）。\n")
    A("各算法跨 4 个 QP 的平均 Weighted-F1：\n")
    A("| 算法 | 平均 Weighted-F1 |")
    A("|------|------------------|")
    for m in algo_mean.sort_values(ascending=False).index:
        mark = " ✅ 最终选用" if m == chosen else ""
        A(f"| {m} | {algo_mean[m]:.4f}{mark} |")
    A(f"\n**最终算法：{chosen}**（叶子最小样本数 = {args.rf_min_leaf}，已限制规模）。"
      "用它在每个 QP 上各训一个模型、在**全量数据**上重训，并**训练时直接序列化**进一个"
      "C++(VTM) 可读的森林文件：\n")
    A("| QP | 模型 | Weighted-F1 |")
    A("|----|------|-------------|")
    for qp in LABEL_COLS:
        s = saved_models.get(qp)
        if s:
            A(f"| {qp} | {s['model']} | {s['wf1']:.4f} |")
    A(f"\n- **VTM 模型**：`qtForest.bin`（4 个 QP 的森林合一），编码时设 "
      "`QT_FOREST_MODEL` 指向它即可。")
    A("- 每个 QP 另有 `models/<qp>.json` 记录特征顺序/类别/指标（仅供人看，VTM 只读 bin）。")
    A("- 特征顺序末尾即匹配的 QP-value；嵌入推理时按此顺序喂入。\n")

    with open(os.path.join(args.report, "report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
