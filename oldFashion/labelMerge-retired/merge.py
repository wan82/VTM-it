#!/usr/bin/env python3
"""
labelMerge —— 把 label-ctu 的 QT 深度标签与 temp-yuvFeature 的空域特征合并成训练数据。

输入：
  label-ctu/<name>.csv       列: POC_ID,pos_x,pos_y,22qt_max,27qt_max,32qt_max,37qt_max
  temp-yuvFeature/<name>.csv 列: POC_ID,pos_x,pos_y, + 18 个空域特征
合并（按 (POC_ID,pos_x,pos_y) 内连接）：
  → TrainingData/<name>.csv   每序列训练表（键 + 标签 + 特征）
  → TrainingData/Training.csv 全部序列拼合并去掉含 NaN 的行（首列 sequence）—— 最终训练集

用法：
  python merge.py
  python merge.py --labels DIR --features DIR --out DIR --no-combined
"""
import argparse
import os
import sys

import pandas as pd

KEYS = ["POC_ID", "pos_x", "pos_y"]

DEFAULT_LABELS = "/Users/w/Documents/Dev/VTM-it/Data/label-ctu"
DEFAULT_FEATURES = "/Users/w/Documents/Dev/VTM-it/Data/temp/temp-yuvFeature"
DEFAULT_OUT = "/Users/w/Documents/Dev/VTM-it/Data/TrainingData"


def _try_remove(path):
    """尽力删除文件，失败时静默忽略。"""
    try:
        os.remove(path)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="合并 label-ctu 标签与 temp-yuvFeature 特征")
    ap.add_argument("--labels", default=DEFAULT_LABELS,
                    help="label-ctu 目录(每序列一个含 4 个 qtmax 的标签 CSV)")
    ap.add_argument("--features", default=DEFAULT_FEATURES, help="特征目录")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出目录(训练表 + Training.csv)")
    ap.add_argument("--no-combined", action="store_true", help="不生成合并总表 Training.csv")
    args = ap.parse_args()

    for d in (args.labels, args.features):
        if not os.path.isdir(d):
            sys.exit(f"[错误] 目录不存在: {d}")
    os.makedirs(args.out, exist_ok=True)

    # 序列 = label-ctu 与 features 下同名的 CSV
    label_names = {f[:-4] for f in os.listdir(args.labels) if f.endswith(".csv")}
    seqs = sorted(n for n in label_names
                  if os.path.exists(os.path.join(args.features, f"{n}.csv")))
    only_labels = sorted(label_names - set(seqs))
    if not seqs:
        sys.exit("[错误] 没有同时具备标签和特征的序列")

    print(f"发现 {len(seqs)} 个可合并序列")
    print(f"{'序列':<18}{'标签CTU':>10}{'特征CTU':>10}{'匹配':>8}{'未匹配特征':>12}{'缺标签':>8}")
    print("-" * 66)

    combined = []
    total_match = 0

    for name in seqs:
        labels = pd.read_csv(os.path.join(args.labels, f"{name}.csv"))
        feat = pd.read_csv(os.path.join(args.features, f"{name}.csv"))

        label_cols = [c for c in labels.columns if c not in KEYS]
        feat_cols = [c for c in feat.columns if c not in KEYS]

        final = feat.merge(labels, on=KEYS, how="inner")
        final = final[KEYS + label_cols + feat_cols]

        n_unmatched = len(feat) - len(final)                            # 有特征但无标签
        n_missing = int(final[label_cols].isna().any(axis=1).sum())     # 某个 qtmax 缺

        final.to_csv(os.path.join(args.out, f"{name}.csv"), index=False)
        if not args.no_combined:
            c = final.copy()
            c.insert(0, "sequence", name)
            combined.append(c)

        total_match += len(final)
        print(f"{name:<18}{len(labels):>10}{len(feat):>10}{len(final):>8}"
              f"{n_unmatched:>12}{n_missing:>8}")

    print("-" * 66)
    print(f"合计匹配训练样本: {total_match}")
    if only_labels:
        print(f"[提示] 有标签但无特征(已跳过): {', '.join(only_labels)}")

    if not args.no_combined and combined:
        all_df = pd.concat(combined, ignore_index=True)
        clean = all_df.dropna()                       # 去掉任意含 NaN 的行
        n_dropped = len(all_df) - len(clean)
        out_path = os.path.join(args.out, "Training.csv")
        clean.to_csv(out_path, index=False)
        _try_remove(os.path.join(args.out, "dataForTrain.csv"))   # 清理旧版总表
        print(f"合并训练总表 -> {out_path}  "
              f"({len(clean)} 行, {clean.shape[1]} 列; 丢弃含NaN行 {n_dropped})")

    print(f"\n训练表(每序列) -> {args.out}/<name>.csv")


if __name__ == "__main__":
    main()
