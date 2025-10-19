#!/usr/bin/env python3
"""
Script kiểm tra chất lượng dataset detection
"""
import os
import yaml
from pathlib import Path
import json

def analyze_dataset():
    """Phân tích dataset detection"""
    
    # Load dataset config
    dataset_path = Path("data/processed/detection")
    config_file = dataset_path / "dataset.yaml"
    
    if not config_file.exists():
        print("❌ Dataset config không tìm thấy!")
        return
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔍 PHÂN TÍCH DATASET DETECTION")
    print("="*50)
    
    # Check splits
    splits = ['train', 'val', 'test']
    for split in splits:
        if split in config:
            img_dir = dataset_path / config[split]
            label_dir = dataset_path / config[split].replace('images', 'labels')
            
            if img_dir.exists() and label_dir.exists():
                img_count = len(list(img_dir.glob('*.jpg'))) + len(list(img_dir.glob('*.png')))
                label_count = len(list(label_dir.glob('*.txt')))
                
                print(f"📂 {split.upper()}: {img_count} images, {label_count} labels")
                
                if img_count != label_count:
                    print(f"   ⚠️  Mismatch: {img_count} images vs {label_count} labels")
                
                # Analyze label distribution
                class_counts = [0] * config['nc']
                total_objects = 0
                
                for label_file in label_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            if line.strip():
                                cls_id = int(line.split()[0])
                                if 0 <= cls_id < config['nc']:
                                    class_counts[cls_id] += 1
                                    total_objects += 1
                
                print(f"   📊 Total objects: {total_objects}")
                print(f"   📊 Objects per image: {total_objects/img_count:.2f}")
                
                print(f"   📊 Class distribution:")
                for i, (name, count) in enumerate(zip(config['names'].values(), class_counts)):
                    percentage = (count/total_objects)*100 if total_objects > 0 else 0
                    print(f"      {i}: {name}: {count} ({percentage:.1f}%)")
                
                print()
            else:
                print(f"❌ {split} directory không tồn tại!")
    
    print("\n💡 KHUYẾN NGHỊ:")
    
    # Check cho imbalanced classes
    if total_objects > 0:
        min_class_pct = min([(count/total_objects)*100 for count in class_counts if count > 0])
        max_class_pct = max([(count/total_objects)*100 for count in class_counts if count > 0])
        
        if max_class_pct / min_class_pct > 10:
            print("⚠️  Dataset imbalanced! Cân nhắc:")
            print("   - Tăng augmentation cho class thiểu số")
            print("   - Sử dụng class weights")
            print("   - Focal loss")
    
    # Check objects per image
    if img_count > 0:
        avg_objects = total_objects / img_count
        if avg_objects < 1:
            print("⚠️  Quá ít objects per image (<1):")
            print("   - Kiểm tra lại annotations")
            print("   - Tăng mosaic/mixup augmentation")
        elif avg_objects < 2:
            print("⚠️  Ít objects per image (<2):")
            print("   - Cân nhắc tăng mosaic augmentation")
            print("   - Kiểm tra missing annotations")

if __name__ == "__main__":
    analyze_dataset()