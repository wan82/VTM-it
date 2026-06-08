#!/usr/bin/env bash
# 创建 venv 并安装依赖（pandas）
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
echo ""
echo "完成。运行方式："
echo "  ./run.sh                       # 用默认路径合并"
echo "  source .venv/bin/activate && python merge.py --help"
