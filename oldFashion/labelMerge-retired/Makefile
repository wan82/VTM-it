# Makefile —— 检测中间件 temp-yuvFeature 是否有数据，有则与 label-ctu 合并到 TrainingData
#
#   make            建venv(首次) → 若 temp-yuvFeature 有特征CSV则合并
#   make setup      只建 venv 装依赖
#   make merge      执行合并（同 make all）
#   make clean      删除 venv
#   make distclean  删除 venv 和 TrainingData 输出
#
# 路径可在命令行覆盖，例如：
#   make FEATURES=/path/to/temp/temp-yuvFeature OUT=/path/to/TrainingData

DATA     ?= /Users/w/Documents/Dev/VTM-it/Data
FEATURES ?= $(DATA)/temp/temp-yuvFeature
LABELS   ?= $(DATA)/label-ctu
OUT      ?= $(DATA)/TrainingData

VENV := .venv
PY   := $(VENV)/bin/python

.PHONY: all merge setup clean distclean

all: merge

setup: $(VENV)/.installed

$(VENV)/.installed: requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@touch $@

# 检测到 temp-yuvFeature 里有 *.csv 才合并，否则提示先跑采集器
merge: setup
	@if ls $(FEATURES)/*.csv >/dev/null 2>&1; then \
		echo "[检测到中间件数据] 合并 $(FEATURES) × label-ctu -> $(OUT)"; \
		$(PY) merge.py --features "$(FEATURES)" --labels "$(LABELS)" --out "$(OUT)"; \
	else \
		echo "[跳过] $(FEATURES) 里没有特征CSV。请先运行采集器："; \
		echo "       cd ../yuvFeatureExtract && make run"; \
	fi

clean:
	rm -rf $(VENV)

distclean: clean
	rm -rf "$(OUT)"
