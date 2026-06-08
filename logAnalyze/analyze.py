#!/usr/bin/env python3
"""
analyze —— 对比 fastTest(QT早停) 与原始编码日志，算 BD-rate 与 Time Reduction。

做法（参考 GOP/CTC 的逐 POC 对齐思路）：
  · 从每个 log 解析每个 POC 的 bits / Y-PSNR / ET(秒)；
  · fastTest 只编了 POC 0..N（默认 33 帧），原始日志编了整段序列 → **只取两边都有的 POC** 对齐；
  · 每序列每 QP 在对齐的 POC 集合上聚合：
       bitrate(kbps) = (Σbits / 帧数) × frameRate / 1000   （frameRate 取自 per-sequence cfg）
       Y-PSNR        = 对齐 POC 的平均
       ET            = 对齐 POC 的 ET 之和
  · 4 个 QP → BD-rate(原始为锚, fast 为测试)；每 QP 的 Time Reduction = (ET原 − ET快)/ET原。

用法：python analyze.py [--fast DIR] [--orig DIR] [--cfg DIR] [--out DIR]
"""
import argparse
import os
import re
import sys

import numpy as np

QPS = [22, 27, 32, 37]
# POC <n> ... ) <bits> bits [Y <psnr> dB ...] ... [ET <sec> ]
POC_RE = re.compile(
    r"POC\s+(\d+)\b.*?\)\s+(\d+)\s+bits\s+\[Y\s+([\d.]+)\s+dB.*?\[ET\s+([\d.]+)\s*\]")
LOG_RE = re.compile(r"^(.*)_QP(\d+)\.log$", re.IGNORECASE)
ALT_RE = re.compile(r"^(.*)_q(\d+)\.log$")

# 仓库根目录：由本脚本位置推导（logAnalyze 的父目录），换机器/clone 后无需改路径
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEF_FAST = os.path.join(_ROOT, "Data/Log/FastTest")
DEF_ORIG = os.path.join(_ROOT, "Data/LogOriginalEnc")
DEF_CFG  = os.path.join(_ROOT, "perSequenceCfg")
DEF_OUT  = os.path.join(_ROOT, "Report/Analysis")


def parse_log(path):
    """log -> {poc: (bits, ypsnr, et)}"""
    d = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = POC_RE.search(line)
            if m:
                d[int(m.group(1))] = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
    return d


def frame_rate(cfg_dir, seq):
    p = os.path.join(cfg_dir, f"{seq}.cfg")
    if os.path.isfile(p):
        for line in open(p, errors="ignore"):
            m = re.match(r"\s*FrameRate\s*:\s*([\d.]+)", line)
            if m:
                return float(m.group(1))
    return 0.0


def aggregate(d, pocs, fr):
    n = len(pocs)
    if n == 0:
        return None
    bits = sum(d[p][0] for p in pocs)
    y = sum(d[p][1] for p in pocs) / n
    et = sum(d[p][2] for p in pocs)
    kbps = (bits / n) * fr / 1000.0
    return {"kbps": kbps, "ypsnr": y, "et": et, "frames": n}


def bd_rate(R_a, P_a, R_t, P_t):
    """Bjontegaard Delta Rate(%)：锚=a, 测试=t。正=测试码率更高(更差)。"""
    R_a, P_a = np.asarray(R_a, float), np.asarray(P_a, float)
    R_t, P_t = np.asarray(R_t, float), np.asarray(P_t, float)
    lR_a, lR_t = np.log10(R_a), np.log10(R_t)
    pa = np.polyfit(P_a, lR_a, 3)
    pt = np.polyfit(P_t, lR_t, 3)
    lo = max(P_a.min(), P_t.min())
    hi = min(P_a.max(), P_t.max())
    if hi <= lo:
        return float("nan")
    ia = np.polyval(np.polyint(pa), hi) - np.polyval(np.polyint(pa), lo)
    it = np.polyval(np.polyint(pt), hi) - np.polyval(np.polyint(pt), lo)
    return (10 ** ((it - ia) / (hi - lo)) - 1) * 100.0


def discover(fast_dir):
    seqs = {}
    for fn in os.listdir(fast_dir):
        m = LOG_RE.match(fn) or ALT_RE.match(fn)
        if m:
            seqs.setdefault(m.group(1), set()).add(int(m.group(2)))
    return seqs


def main():
    ap = argparse.ArgumentParser(description="BD-rate / Time-Reduction 分析")
    ap.add_argument("--fast", default=DEF_FAST, help="fastTest 日志目录")
    ap.add_argument("--orig", default=DEF_ORIG, help="原始编码日志目录")
    ap.add_argument("--cfg",  default=DEF_CFG,  help="per-sequence cfg 目录(取 FrameRate)")
    ap.add_argument("--out",  default=DEF_OUT,  help="输出目录")
    args = ap.parse_args()

    for d in (args.fast, args.orig):
        if not os.path.isdir(d):
            sys.exit(f"[错误] 目录不存在: {d}")
    os.makedirs(args.out, exist_ok=True)

    seqs = discover(args.fast)
    if not seqs:
        sys.exit(f"[错误] {args.fast} 里没有 <Seq>_QP<qp>.log 日志")

    rows = []          # 汇总：每序列
    detail_lines = []  # 每序列每QP明细
    for seq in sorted(seqs):
        fr = frame_rate(args.cfg, seq)
        if fr <= 0:
            print(f"[提示] {seq}: 未取到 FrameRate，bitrate 用 1.0 占位（不影响 BD-rate）")
            fr = 1.0
        Ra = Pa = Rt = Pt = None
        ra, pa, rt, pt, trs = [], [], [], [], []
        ok = True
        for qp in QPS:
            ff = os.path.join(args.fast, f"{seq}_QP{qp}.log")
            of = os.path.join(args.orig, f"{seq}_QP{qp}.log")
            if not (os.path.isfile(ff) and os.path.isfile(of)):
                ok = False
                break
            df, do = parse_log(ff), parse_log(of)
            common = sorted(set(df) & set(do))            # 对齐的 POC
            af, ao = aggregate(df, common, fr), aggregate(do, common, fr)
            if not af or not ao or ao["et"] <= 0:
                ok = False
                break
            tr = (af["et"] - ao["et"]) / ao["et"] * 100.0   # 负值=编码时间下降(更快)
            ra.append(ao["kbps"]); pa.append(ao["ypsnr"])
            rt.append(af["kbps"]); pt.append(af["ypsnr"])
            trs.append(tr)
            detail_lines.append(
                f"| {seq} | {qp} | {len(common)} | {ao['kbps']:.2f} | {ao['ypsnr']:.4f} | "
                f"{rt[-1]:.2f} | {pt[-1]:.4f} | {ao['et']:.1f} | {af['et']:.1f} | {tr:.2f}% |")
        if not ok or len(ra) < 4:
            print(f"[跳过] {seq}: 缺少 4 个 QP 的对齐日志")
            continue
        bd = bd_rate(ra, pa, rt, pt)
        tr_avg = float(np.mean(trs))
        rows.append((seq, bd, tr_avg))
        print(f"{seq:<16} BD-rate = {bd:+.2f}%   平均 Time-Reduction = {tr_avg:+.2f}%")

    if not rows:
        sys.exit("[错误] 没有可对比的序列（确认 fastTest 与原始日志命名一致：<Seq>_QP<qp>.log）")

    # 写报告
    out_md = os.path.join(args.out, "analysis.md")
    L = []
    L.append("# fastTest 分析报告（QT 早停 vs 原始编码）\n")
    L.append("> BD-rate：原始为锚、fastTest 为测试，正值=码率升高（变差）；"
             "Time-Reduction：**负值=编码时间下降（更快）**。逐 POC 对齐（仅取两边都有的 POC）。\n")
    L.append("## 汇总\n")
    L.append("| 序列 | BD-rate | 平均 Time-Reduction |")
    L.append("|------|---------|---------------------|")
    for seq, bd, tr in rows:
        L.append(f"| {seq} | {bd:+.2f}% | {tr:+.2f}% |")
    if len(rows) > 1:
        L.append(f"| **平均** | {np.mean([r[1] for r in rows]):+.2f}% | "
                 f"{np.mean([r[2] for r in rows]):+.2f}% |")
    L.append("\n## 逐 QP 明细\n")
    L.append("| 序列 | QP | 对齐帧 | 原kbps | 原Y-PSNR | 快kbps | 快Y-PSNR | 原ET(s) | 快ET(s) | TR |")
    L.append("|------|----|--------|--------|----------|--------|----------|---------|---------|-----|")
    L += detail_lines
    L.append("")
    with open(out_md, "w") as f:
        f.write("\n".join(L) + "\n")

    out_csv = os.path.join(args.out, "bdSummary.csv")
    with open(out_csv, "w") as f:
        f.write("sequence,bd_rate_percent,time_reduction_percent\n")
        for seq, bd, tr in rows:
            f.write(f"{seq},{bd:.4f},{tr:.4f}\n")

    print(f"\n报告 -> {out_md}\n汇总 -> {out_csv}")


if __name__ == "__main__":
    main()
