# Makefile —— 对比 fastTest 与原始编码日志，算 BD-rate / Time-Reduction
#   make            建/复用共享 venv → 分析 → 报告
#   路径可覆盖：make FAST=/路径 ORIG=/路径 OUT=/路径

# 仓库根目录：由本 Makefile 自身位置推导（其父目录），换机器/clone 后无需改路径
THIS_MK := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT ?= $(abspath $(dir $(THIS_MK))/..)
DATA ?= $(ROOT)/Data
FAST ?= $(DATA)/Log/FastTest
ORIG ?= $(DATA)/LogOriginalEnc
CFG  ?= $(ROOT)/perSequenceCfg
OUT  ?= $(ROOT)/Report/Analysis

# 共享 venv（与 modelTrain 共用 VTM-it/.venv）
VENV  := ../.venv
PY    := $(VENV)/bin/python
STAMP := $(VENV)/.req-analyze

.PHONY: all analyze setup clean

all: analyze

setup: $(STAMP)

$(STAMP): requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -r requirements.txt
	@touch $@

analyze: setup
	$(PY) analyze.py --fast "$(FAST)" --orig "$(ORIG)" --cfg "$(CFG)" --out "$(OUT)"

clean:
	@echo "共享 venv 在 $(VENV)，请用根目录 make distclean 删除"
