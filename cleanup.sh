#!/usr/bin/env bash
# ============================================================
# cleanup.sh —— 清理 VTM-it 里的无用/可重生文件
# 在仓库根目录运行：  bash cleanup.sh
# 只删「构建产物 / 陈旧虚拟环境 / 系统垃圾 / 测试输出」，不碰源码、数据、报告。
# oldFashion/ 按约定完全不动。
# ============================================================
set -u
cd "$(dirname "$0")"
ROOT="$(pwd)"
echo "清理目录: $ROOT"
echo "（oldFashion/ 不动；官方 VTM 源码不动，仅清其 build 产物）"
echo

before=$(du -sk . 2>/dev/null | cut -f1)

say() { printf '  - %s\n' "$1"; }

echo "[1/6] 系统垃圾 .DS_Store（排除 oldFashion）"
find . -path ./oldFashion -prune -o -name .DS_Store -print -delete 2>/dev/null | sed 's/^/      /' | head

echo "[2/6] 陈旧的每项目虚拟环境 modelTrain/.venv（已被根目录共享 .venv 取代，约 336MB）"
[ -d modelTrain/.venv ] && { rm -rf modelTrain/.venv && say "已删 modelTrain/.venv"; } || say "无（已清）"

echo "[3/6] Python 缓存 __pycache__ / *.pyc（排除 oldFashion 与各 .venv）"
find . -path ./oldFashion -prune -o -path '*/.venv' -prune -o -name __pycache__ -type d -print -exec rm -rf {} + 2>/dev/null | sed 's/^/      /' | head
find . -path ./oldFashion -prune -o -name '*.pyc' -print -delete 2>/dev/null | head -0

echo "[4/6] C++ 构建产物 yuvFeatureExtract/build（可由 make 重生）"
[ -d yuvFeatureExtract/build ] && { rm -rf yuvFeatureExtract/build && say "已删 yuvFeatureExtract/build"; } || say "无"

echo "[5/6] 子项目临时测试输出（output/ 回退目录、outTest/ 合成自测、空 figures）"
[ -d yuvFeatureExtract/output ]   && { rm -rf yuvFeatureExtract/output   && say "已删 yuvFeatureExtract/output"; }
[ -d yuvFeatureExtract/outTest ]  && { rm -rf yuvFeatureExtract/outTest  && say "已删 yuvFeatureExtract/outTest"; }
[ -d Data/ModelResults/figures ]  && { rmdir Data/ModelResults/figures 2>/dev/null && say "已删空目录 Data/ModelResults/figures"; }

echo "[6/6] 可选（默认保留，去掉前面的 # 即生效）："
echo "      # rm -rf VVCSoftware_VTM-VTM-20.2/build      # 32MB，但重建 VTM 较慢"
echo "      # rm -f  VVCSoftware_VTM-VTM-20.2.zip         # 8.4MB，官方源码已解压，冗余"
# rm -rf VVCSoftware_VTM-VTM-20.2/build
# rm -f  VVCSoftware_VTM-VTM-20.2.zip

after=$(du -sk . 2>/dev/null | cut -f1)
echo
echo "完成。占用 $((before/1024)) MB → $((after/1024)) MB（释放约 $(((before-after)/1024)) MB）"
echo "提示：这些路径已写进 .gitignore，之后不会再被 git 跟踪。"
