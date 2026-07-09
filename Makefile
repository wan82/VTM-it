# ============================================================
# VTM-it 顶层 Makefile —— 一站式控制两个子项目，无需 cd
#   yuvFeatureExtract (C++) ：特征提取 + 标签合并
#   modelTrain        (Py)  ：模型训练 + 报告
#
# 常用：
#   make extract     提取：YUV → 特征中间件(tempYuvFeature)
#   make merge       汇总：特征 × 标签 → Training.csv (含 QP-value)
#   make data        extract + merge 一条龙
#   make train       训练：Training.csv → 模型 + 报告 + 最佳模型bin
#   make all         data + train 全流程
#   make build       仅编译 C++ 子项目
#   make clean       清理两个子项目的 build / venv
#   make help        查看全部命令
#
# 参数透传（命令行变量会自动传给子项目），例如：
#   make merge KEEP=no                 # 合并后不保留中间表
#   make train CORR=0.9 TOPK=8         # 改特征筛选参数
#   make train RF_ESTIMATORS=200       # 注意：子项目参数名见各自 Makefile
# ============================================================

EXTRACT_DIR := yuvFeatureExtract
TRAIN_DIR   := modelTrain
ANALYZE_DIR := logAnalyze

# ---- fastTest 参数（可覆盖）----
VTM_DIR ?= VVCSoftware_VTM-VTM-20.2
RA_CFG  ?= $(VTM_DIR)/cfg/encoder_randomaccess_vtm.cfg
SEQ_CFG ?= $(VTM_DIR)/cfg/per-sequence/BQSquare.cfg
YUV     ?= /Users/w/wan_data/yuv/BQSquare_416x240_60.yuv
SEQNAME ?= BQSquare
FRAMES  ?= 33
LOGDIR  ?= Data/Log/FastTest
MODEL   ?= Data/ModelResults/qtForest.bin

.DEFAULT_GOAL := help
.PHONY: help all build extract merge data train trainMtt fastTest analyze mttLabel mttFeature clean distclean

help:
	@echo "VTM-it 顶层命令（在根目录直接用）："
	@echo "  make build     编译 C++ 子项目 ($(EXTRACT_DIR))"
	@echo "  make extract   提取特征：YUV → Data/temp/tempYuvFeature"
	@echo "  make merge     汇总：特征 × labelCtu → Data/TrainingData/Training.csv"
	@echo "  make data      extract + merge"
	@echo "  make mttLabel  解码 binEncode202 → 采样 MTT 划分标签 → Data/labelMtt"
	@echo "  make mttFeature labelMtt × 源YUV → 块特征 → Data/TrainingData/TrainingMtt.csv"
	@echo "  make trainMtt  训练 MTT splitMode 5分类(按QP分4模型) → Data/ModelResults/mttForest.bin"
	@echo "  make train     训练→模型 qtForest.bin→Data/ModelResults；报告→Report/ModelTraining"
	@echo "  make all       data + train（全流程）"
	@echo "  make fastTest  编译VTM并对BQSquare编码33帧(QP22/27/32/37)，日志→Data/Log/FastTest"
	@echo "  make analyze   对比 fastTest 与原始日志，算 BD-rate/Time-Reduction，报告→Report/Analysis"
	@echo "  make clean     清理子项目构建产物"
	@echo "  make distclean clean + 删除共享 venv 和训练产物"

# ---- C++ 子项目：提取与合并 ----
build:
	$(MAKE) -C $(EXTRACT_DIR) build

extract:
	$(MAKE) -C $(EXTRACT_DIR) run

merge:
	$(MAKE) -C $(EXTRACT_DIR) merge

data:
	$(MAKE) -C $(EXTRACT_DIR) data

# ---- Python 子项目：训练（直接产出 VTM 可读的 qtForest.bin）----
train:
	$(MAKE) -C $(TRAIN_DIR) train

# ---- Python 子项目：分析 BD-rate / Time-Reduction ----
analyze:
	$(MAKE) -C $(ANALYZE_DIR) analyze

# ---- MTT 标签：解码 binEncode202 采样导出（C++ 模块，自动构建 DecoderApp）----
mttLabel:
	$(MAKE) -C $(EXTRACT_DIR) mtt-label

# ---- MTT 块特征：labelMtt × 源YUV → TrainingMtt.csv ----
mttFeature:
	$(MAKE) -C $(EXTRACT_DIR) mtt-feature

# ---- MTT 模型训练：splitMode 5分类 → mttForest.bin ----
trainMtt:
	$(MAKE) -C $(TRAIN_DIR) train-mtt

# ---- 全流程 ----
all: data train

# ---- fastTest：编译 VTM 编码器 + 对 BQSquare 编码一组 GOP(POC 0–32, 33帧) ----
#   QP 22/27/32/37 各编码一次，日志存 Log/FastTest/BQSquare_q<QP>.log
#   若 Data/ModelResults/qtForest.bin 存在则启用 QT 早停；否则为原版编码
#   YUV 路径可覆盖：make fastTest YUV=/Users/w/wan_data/yuv/BQSquare_416x240_60.yuv
fastTest:
	@echo "[fastTest] 编译 VTM EncoderApp ..."
	cmake -S $(VTM_DIR) -B $(VTM_DIR)/build -DCMAKE_BUILD_TYPE=Release
	cmake --build $(VTM_DIR)/build --target EncoderApp -j 4
	@ENC=$$(find "$(VTM_DIR)" -type f -name EncoderApp -perm -u+x 2>/dev/null | head -1); \
	if [ -z "$$ENC" ]; then echo "[错误] 没找到 EncoderApp 二进制"; exit 1; fi; \
	if [ ! -f "$(YUV)" ]; then \
	  echo "[错误] 找不到 YUV: $(YUV)"; \
	  echo "       用 make fastTest YUV=/你的路径/BQSquare_416x240_60.yuv 指定"; exit 1; fi; \
	mkdir -p "$(LOGDIR)"; \
	MODELABS=""; if [ -f "$(MODEL)" ]; then MODELABS="$$(cd "$$(dirname "$(MODEL)")" && pwd)/$$(basename "$(MODEL)")"; fi; \
	echo "[fastTest] 编码器: $$ENC"; \
	if [ -n "$$MODELABS" ]; then echo "[fastTest] QT早停模型: $$MODELABS"; \
	else echo "[fastTest] 未找到 qtForest.bin → 原版编码（先 make train 可启用早停）"; fi; \
	for q in 22 27 32 37; do \
	  echo "[fastTest] 编码 QP=$$q ($(FRAMES) 帧) ..."; \
	  QT_FOREST_MODEL="$$MODELABS" "$$ENC" \
	    -c "$(RA_CFG)" -c "$(SEQ_CFG)" \
	    -i "$(YUV)" -f $(FRAMES) -q $$q \
	    -b /dev/null -o /dev/null \
	    > "$(LOGDIR)/$(SEQNAME)_QP$$q.log" 2>&1 ; \
	  echo "  -> $(LOGDIR)/$(SEQNAME)_QP$$q.log"; \
	done; \
	echo "[fastTest] 完成。日志在 $(LOGDIR)/"

# ---- 清理 ----
clean:
	-$(MAKE) -C $(EXTRACT_DIR) clean

distclean: clean
	-$(MAKE) -C $(TRAIN_DIR) distclean
	rm -rf .venv          # 共享 venv
