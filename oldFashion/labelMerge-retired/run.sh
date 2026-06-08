#!/usr/bin/env bash
# 没有 venv 就先建；然后用 venv 里的 python 跑 merge.py（参数透传）
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  ./setup.sh
fi
./.venv/bin/python merge.py "$@"
