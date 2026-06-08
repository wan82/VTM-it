#!/usr/bin/env python3
"""
[已弃用] 不再需要单独的 joblib→bin 转换步骤。

训练（train.py / `make train`）会**直接产出** C++(VTM) 可读的森林模型
`Data/ModelResults/qtForest.bin`。本文件保留仅为兼容，可删除。
"""
import sys

if __name__ == "__main__":
    sys.exit("已弃用：请直接用 `make train`，它会直接生成 qtForest.bin（无需本脚本）。")
