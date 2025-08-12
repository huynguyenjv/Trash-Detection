# Makefile cho Trash Detection Project
# Author: Huy Nguyen

SHELL := /bin/bash
VENV = trash_detection_env

.PHONY: help install setup data train evaluate demo clean all quick-test usage

# Default target
help:
	@echo "🗑️  Trash Detection Project Commands"
	@echo "=================================="
	@echo "quick-test       - Test nhanh với pre-trained model"
	@echo "usage            - Hiện hướng dẫn chi tiết"
	@echo "install          - Cài đặt dependencies"
	@echo "setup            - Setup project (install + kaggle setup)"
	@echo "data             - Chạy data preprocessing"
	@echo "train            - Train model"
	@echo "evaluate         - Đánh giá model"
	@echo "demo             - Chạy demo detection"
	@echo "all              - Chạy toàn bộ pipeline"
	@echo "clean            - Dọn dẹp files"
	@echo ""
	@echo "🚀 Quick start: make quick-test"

# Test nhanh
quick-test:
	@echo "🚀 Running quick detection test..."
	@source $(VENV)/bin/activate && python test_detection.py

# Hướng dẫn sử dụng
usage:
	@echo "📖 Showing usage guide..."
	@source $(VENV)/bin/activate && python USAGE.py

# Cài đặt dependencies
install:
	@echo "📦 Cài đặt dependencies..."
	@source $(VENV)/bin/activate && pip install -r requirements.txt
	@echo "✅ Dependencies đã được cài đặt!"

# Setup dự án
setup: install
	@echo "🔧 Setup dự án..."
	@if [ ! -f ~/.kaggle/kaggle.json ]; then \
		echo "⚠️  Cần setup Kaggle API key!"; \
		echo "1. Tạo file ~/.kaggle/kaggle.json"; \
		echo "2. Thêm nội dung: {\"username\": \"your_username\", \"key\": \"your_key\"}"; \
		echo "3. Chạy: chmod 600 ~/.kaggle/kaggle.json"; \
		exit 1; \
	fi
	@echo "✅ Project setup hoàn thành!"

# Data preprocessing
data:
	@echo "🔄 Bắt đầu data preprocessing..."
	python run_pipeline.py --step preprocessing
	@echo "✅ Data preprocessing hoàn thành!"

# Training
train:
	@echo "🚂 Bắt đầu training..."
	python run_pipeline.py --step training
	@echo "✅ Training hoàn thành!"

# Evaluation
evaluate:
	@echo "📊 Đánh giá model..."
	python run_pipeline.py --step evaluation
	@echo "✅ Evaluation hoàn thành!"

# Demo
demo:
	@echo "🎬 Chạy demo detection..."
	python run_pipeline.py --step demo

# Full pipeline
all:
	@echo "🚀 Chạy toàn bộ pipeline..."
	python run_pipeline.py
	@echo "✅ Pipeline hoàn thành!"

# Quick detection commands
detect-webcam:
	@echo "📹 Khởi động webcam detection..."
	cd src && python detect.py --mode webcam --source 0

detect-image:
	@echo "🖼️  Image detection (cần chỉ định --source)"
	@echo "Sử dụng: make detect-image SOURCE=path/to/image.jpg"
	@if [ -z "$(SOURCE)" ]; then \
		echo "❌ Cần chỉ định SOURCE=path/to/image.jpg"; \
		exit 1; \
	fi
	cd src && python detect.py --mode image --source $(SOURCE)

# Clean up
clean:
	@echo "🧹 Dọn dẹp files..."
	rm -rf data/raw/*
	rm -rf runs/
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -f *.log
	rm -rf evaluation_results/
	@echo "✅ Đã dọn dẹp!"

# Development commands
dev-install:
	@echo "🛠️  Cài đặt development dependencies..."
	pip install -r requirements.txt
	pip install jupyter notebook ipython

jupyter:
	@echo "📓 Khởi động Jupyter Notebook..."
	jupyter notebook notebooks/

# Model commands
download-pretrained:
	@echo "⬇️  Download pretrained YOLOv8 models..."
	cd models && \
	wget -nc https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt && \
	wget -nc https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
	@echo "✅ Pretrained models đã download!"

# Check system
check-gpu:
	@echo "🔍 Kiểm tra GPU..."
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

check-deps:
	@echo "🔍 Kiểm tra dependencies..."
	python -c "import cv2, torch, ultralytics; print('✅ Tất cả dependencies OK!')"

# Training với custom config
train-fast:
	@echo "🚂 Training nhanh (YOLOv8n, 25 epochs)..."
	cd src && python train.py --model yolov8n.pt --epochs 25

train-accurate:
	@echo "🚂 Training độ chính xác cao (YOLOv8m, 100 epochs)..."
	cd src && python train.py --model yolov8m.pt --epochs 100

# Benchmark
benchmark:
	@echo "⏱️  Benchmark model performance..."
	cd src && python -c "\
import time; \
from detect import TrashDetector, DetectionConfig; \
config = DetectionConfig(); \
detector = TrashDetector(config); \
import cv2; \
import numpy as np; \
dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8); \
times = []; \
for i in range(10): \
    start = time.time(); \
    results = detector.model(dummy_img, verbose=False); \
    times.append(time.time() - start); \
print(f'Average inference time: {np.mean(times)*1000:.1f}ms'); \
print(f'FPS: {1/np.mean(times):.1f}'); \
"

# Show project status
status:
	@echo "📊 Project Status"
	@echo "=================="
	@echo "Data:"
	@if [ -d "data/processed" ]; then echo "  ✅ Processed data exists"; else echo "  ❌ No processed data"; fi
	@echo "Model:"
	@if [ -f "models/trash_detection_best.pt" ]; then echo "  ✅ Trained model exists"; else echo "  ❌ No trained model"; fi
	@echo "Results:"
	@if [ -d "evaluation_results" ]; then echo "  ✅ Evaluation results exist"; else echo "  ❌ No evaluation results"; fi
