#!/usr/bin/env bash
# 创建 venv 并安装依赖（pandas/numpy/scikit-learn/matplotlib）
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo ""
echo "完成。运行：./run.sh   或   source .venv/bin/activate && python train.py --help"
