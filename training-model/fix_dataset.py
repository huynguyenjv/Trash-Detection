#!/usr/bin/env python3
"""
Script sửa chữa dataset detection
"""
import os
import yaml
import shutil
from pathlib import Path
from collections import defaultdict
import random

def fix_dataset():
    """Sửa chữa dataset detection"""
    
    dataset_path = Path("data/processed/detection")
    
    print("🔧 FIXING DATASET ISSUES")
    print("="*50)
    
    # 1. Fix mismatch giữa images và labels
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = dataset_path / f"images/{split}"
        label_dir = dataset_path / f"labels/{split}"
        
        if not (img_dir.exists() and label_dir.exists()):
            continue
            
        # Lấy list images và labels
        img_files = set()
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            img_files.update([f.stem for f in img_dir.glob(ext)])
        
        label_files = set([f.stem for f in label_dir.glob('*.txt')])
        
        print(f"📂 {split.upper()}:")
        print(f"   Images: {len(img_files)}")
        print(f"   Labels: {len(label_files)}")
        
        # Remove orphan labels
        orphan_labels = label_files - img_files
        if orphan_labels:
            print(f"   🗑️  Removing {len(orphan_labels)} orphan labels")
            for label in orphan_labels:
                (label_dir / f"{label}.txt").unlink()
        
        # Remove images without labels
        missing_labels = img_files - label_files
        if missing_labels:
            print(f"   🗑️  Removing {len(missing_labels)} images without labels")
            for img in missing_labels:
                for ext in ['jpg', 'jpeg', 'png']:
                    img_file = img_dir / f"{img}.{ext}"
                    if img_file.exists():
                        img_file.unlink()
                        break
    
    # 2. Phân tích lại sau khi fix
    print("\n📊 DATASET AFTER FIXING:")
    print("="*30)
    
    config_file = dataset_path / "dataset.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    for split in splits:
        img_dir = dataset_path / f"images/{split}"
        label_dir = dataset_path / f"labels/{split}"
        
        if img_dir.exists() and label_dir.exists():
            img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
            label_count = len(list(label_dir.glob('*.txt')))
            
            print(f"{split}: {img_count} images, {label_count} labels")

def create_balanced_config():
    """Tạo config training để handle class imbalance"""
    
    # Class weights để balance classes
    # Class "other" có weight thấp, các class khác có weight cao
    class_weights = {
        0: 5.0,   # cardboard (2%)
        1: 8.0,   # glass (1%) 
        2: 3.0,   # metal (4%)
        3: 10.0,  # organic (1%)
        4: 0.5,   # other (60%) - weight thấp
        5: 4.0,   # paper (3%)
        6: 1.5    # plastic (27%)
    }
    
    # Tạo config mới với focus loss và class weights
    balanced_config = {
        # Model config
        'model': 'yolov8n.pt',
        'epochs': 300,
        'batch': 8,
        'imgsz': 640,
        'device': 0,
        
        # Learning
        'optimizer': 'SGD',
        'lr0': 0.01,
        'lrf': 0.1,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights để handle imbalance
        'box': 7.5,
        'cls': 1.0,
        'dfl': 1.5,
        
        # Detection thresholds
        'conf': 0.001,
        'iou': 0.7,
        
        # Augmentation - reduced để tránh overfitting trên class thiểu số
        'hsv_h': 0.005,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'degrees': 3.0,
        'translate': 0.05,
        'scale': 0.25,
        'shear': 1.0,
        'perspective': 0.0001,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.8,  # Giữ mosaic để tăng diversity
        'mixup': 0.2,   # Thêm mixup
        'copy_paste': 0.1,
        
        # Early stopping
        'patience': 50,
        'save_period': 10,
        
        # Validation
        'val': True,
        'split': 'val',
        'save_json': True,
        'save_hybrid': False,
        'half': False,
        'dnn': False,
        
        # Class weights
        'cls_weights': class_weights
    }
    
    # Lưu config mới
    with open('training_config_balanced.yaml', 'w') as f:
        yaml.dump(balanced_config, f, default_flow_style=False, indent=2)
    
    print("💾 Đã tạo training_config_balanced.yaml với class weights")
    
    return balanced_config

def suggest_improvements():
    """Đưa ra các khuyến nghị cải thiện"""
    
    print("\n💡 KHUYẾN NGHỊ CẢI THIỆN DATASET:")
    print("="*40)
    
    print("1. 🎯 CLASS IMBALANCE:")
    print("   - Sử dụng config balanced với class weights")
    print("   - Tăng augmentation cho minority classes")
    print("   - Sử dụng focal loss thay cross-entropy")
    
    print("\n2. 📸 DATA COLLECTION:")
    print("   - Thu thập thêm data cho: glass, organic, cardboard")
    print("   - Giảm số lượng class 'other' hoặc chia nhỏ ra")
    print("   - Đảm bảo quality annotations")
    
    print("\n3. 🔧 TRAINING STRATEGY:")
    print("   - Train với learning rate thấp (0.001-0.005)")
    print("   - Sử dụng transfer learning từ pretrained model")
    print("   - Áp dụng progressive resizing")
    print("   - Monitor validation loss carefully")
    
    print("\n4. 📊 EVALUATION:")
    print("   - Sử dụng per-class metrics")
    print("   - Focus vào recall cho minority classes")
    print("   - Confusion matrix analysis")

if __name__ == "__main__":
    fix_dataset()
    create_balanced_config()
    suggest_improvements()