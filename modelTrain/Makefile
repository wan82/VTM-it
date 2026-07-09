# Makefile —— 训练 NB/DT/RF 并生成混淆矩阵与报告
#
#   make            建venv(首次) → 若 Training.csv 存在则训练并出报告
#   make setup      只建 venv 装依赖
#   make train      执行 QT 训练（同 make all）
#   make train-mtt  执行 MTT 训练（splitMode 5分类，按QP分4模型 → mttForest.bin）
#   make clean      删除 venv
#   make distclean  删除 venv 和 ModelResults 输出
#
# 路径/参数可覆盖：make INPUT=/path/Training.csv OUT=/path TOPK=10 CORR=0.9

# 仓库根目录：由本 Makefile 自身位置推导（其父目录），换机器/clone 后无需改路径
THIS_MK  := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT     ?= $(abspath $(dir $(THIS_MK))/..)
DATA     ?= $(ROOT)/Data
INPUT    ?= $(DATA)/TrainingData/Training.csv
OUT      ?= $(DATA)/ModelResults
REPORT   ?= $(ROOT)/Report/ModelTraining
CORR     ?= 0.95
TOPK     ?= 0
FOLDS    ?= 5
SEED     ?= 42
RFMINLEAF?= 50                    # 限制 RF/DT 规模，模型小而快、可直接嵌入

# MTT 训练参数（独立模型）
INPUT_MTT  ?= $(DATA)/TrainingData/TrainingMtt.csv
REPORT_MTT ?= $(ROOT)/Report/MttTraining

# 共享 venv（所有 Python 子项目共用 VTM-it/.venv），每个子项目用各自的标记安装依赖
VENV  := ../.venv
PY    := $(VENV)/bin/python
STAMP := $(VENV)/.req-modeltrain

.PHONY: all train train-mtt setup clean distclean

all: train

setup: $(STAMP)

$(STAMP): requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -q --upgrade pip
	$(VENV)/bin/pip install -q -r requirements.txt
	@touch $@

train: setup
	@if [ -f "$(INPUT)" ]; then \
		echo "[训练] $(INPUT) -> 模型:$(OUT)  报告:$(REPORT)"; \
		$(PY) train.py --input "$(INPUT)" --out "$(OUT)" --report "$(REPORT)" \
			--corr-thr $(CORR) --top-k $(TOPK) --folds $(FOLDS) --seed $(SEED) \
			--rf-min-leaf $(RFMINLEAF); \
	else \
		echo "[跳过] 找不到 $(INPUT)。请先用合并生成 Training.csv（根目录 make merge）"; \
	fi
	@echo "（训练已直接产出 VTM 可读的 $(OUT)/qtForest.bin，无需额外导出步骤）"

# MTT 训练：splitMode 5 分类，按 QP 分 4 模型，直接导出 $(OUT)/mttForest.bin
train-mtt: setup
	@if [ -f "$(INPUT_MTT)" ]; then \
		echo "[MTT训练] $(INPUT_MTT) -> 模型:$(OUT)/mttForest.bin  报告:$(REPORT_MTT)"; \
		$(PY) trainMtt.py --input "$(INPUT_MTT)" --out "$(OUT)" --report "$(REPORT_MTT)" \
			--folds $(FOLDS) --seed $(SEED) --rf-min-leaf $(RFMINLEAF); \
	else \
		echo "[跳过] 找不到 $(INPUT_MTT)。请先 make mttFeature 生成 TrainingMtt.csv"; \
	fi

clean:
	@echo "共享 venv 在 $(VENV)，请用根目录 make distclean 删除"

distclean:
	rm -rf "$(OUT)"
