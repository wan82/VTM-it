# Makefile —— 包装 CMake 构建（make 启动），按模块分别运行
#   make            配置并编译（Release）
#   make run        运行采集器：YUV → 特征(tempYuvFeature)
#   make merge      运行合并器：特征 × labelCtu → Training.csv
#   make data       run + merge 一条龙
#   make mtt-label  解码 binEncode202 → 采样 MTT 标签 → Data/labelMtt（自动构建 DecoderApp）
#   make debug      Debug 构建
#   make clean      删除 build 目录

BUILD_DIR  ?= build
CONFIG     ?= collector/config/config.yaml
BUILD_TYPE ?= Release
JOBS       ?= 4

# 仓库根目录：由本 Makefile 自身位置推导（其父目录），换机器/clone 后无需改路径
THIS_MK     := $(abspath $(lastword $(MAKEFILE_LIST)))
ROOT        ?= $(abspath $(dir $(THIS_MK))/..)

# 合并器路径参数
DATA        ?= $(ROOT)/Data
LABELS      ?= $(DATA)/labelCtu
FEATURES    ?= $(DATA)/temp/tempYuvFeature
TRAINING    ?= $(DATA)/TrainingData/Training.csv
TEMP_MERGED ?= $(DATA)/temp/trainMerged
KEEP        ?= yes      # 是否保留每序列中间表（yes/no），默认 yes

# MTT 标签编排参数
VTM_DIR    ?= $(ROOT)/VVCSoftware_VTM-VTM-20.2
MTT_CONFIG ?= mttLabel/config/mtt.yaml
BIN_DIR    ?= $(DATA)/binEncode202
LABELMTT   ?= $(DATA)/labelMtt

.PHONY: all build run merge data mtt-label debug clean

all: build

build:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build $(BUILD_DIR) -j $(JOBS)

# 采集器：YUV → 特征中间件（可用 CONFIG=path/to.yaml 覆盖）
run: build
	$(BUILD_DIR)/collector/yuvFeatureCollector $(CONFIG)

# 合并器：特征 × 标签 → Training.csv（KEEP=no 则删除中间表）
merge: build
	@KEEPFLAG=""; if [ "$(KEEP)" = "no" ]; then KEEPFLAG="--delete-intermediate"; fi; \
	$(BUILD_DIR)/merger/yuvLabelMerger \
		--labels "$(LABELS)" --features "$(FEATURES)" \
		--out "$(TRAINING)" --temp "$(TEMP_MERGED)" $$KEEPFLAG; \
	find "$(dir $(TRAINING))" -maxdepth 1 -name '*.csv' ! -name 'Training.csv' -delete 2>/dev/null || true

# 一条龙：采集 + 合并
data: run merge

# MTT 标签：构建 DecoderApp（带 #MTTDUMP 钩子）+ mttLabelBuilder，解码采样导出标签
#   解码本身必须走 VTM 解码器；本目标自动构建并把其路径传给编排器。
#   采样参数（poc_count/block_ratio/seed/qps）改 $(MTT_CONFIG) 即可。
mtt-label: build
	@echo "[mtt-label] 构建 VTM DecoderApp ..."
	cmake -S "$(VTM_DIR)" -B "$(VTM_DIR)/build" -DCMAKE_BUILD_TYPE=Release
	cmake --build "$(VTM_DIR)/build" --target DecoderApp -j $(JOBS)
	@DEC=$$(find "$(VTM_DIR)" -type f -name 'DecoderApp*' -perm -u+x 2>/dev/null | head -1); \
	if [ -z "$$DEC" ]; then echo "[错误] 没找到 DecoderApp 二进制"; exit 1; fi; \
	echo "[mtt-label] 解码器: $$DEC"; \
	$(BUILD_DIR)/mttLabel/mttLabelBuilder "$(MTT_CONFIG)" \
		--decoder "$$DEC" --bin-dir "$(BIN_DIR)" --out-dir "$(LABELMTT)"

debug:
	$(MAKE) build BUILD_TYPE=Debug

clean:
	rm -rf $(BUILD_DIR)
