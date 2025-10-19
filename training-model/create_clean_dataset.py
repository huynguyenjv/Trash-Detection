#!/usr/bin/env python3
"""
Script để dọn dẹp và tạo lại dataset từ scratch với tỷ lệ mới
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import argparse

def find_valid_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Tìm tất cả valid image-label pairs trong data directory
    """
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')
    
    # Thu thập tất cả files từ tất cả subdirectories
    all_image_files = []
    all_label_files = []
    
    # Tìm images
    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                all_image_files.append(os.path.join(root, file))
    
    # Tìm labels  
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                all_label_files.append(os.path.join(root, file))
    
    # Match image-label pairs
    valid_pairs = []
    for img_path in all_image_files:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Tìm label file tương ứng
        label_path = None
        for lbl_path in all_label_files:
            lbl_name = os.path.splitext(os.path.basename(lbl_path))[0]
            if img_name == lbl_name:
                label_path = lbl_path
                break
        
        # Kiểm tra file tồn tại và có content
        if label_path and os.path.exists(img_path) and os.path.exists(label_path):
            if os.path.getsize(img_path) > 0 and os.path.getsize(label_path) > 0:
                valid_pairs.append((img_path, label_path))
    
    print(f"Found {len(valid_pairs)} valid image-label pairs")
    return valid_pairs

def analyze_class_distribution(valid_pairs: List[Tuple[str, str]]) -> Dict[str, int]:
    """
    Phân tích phân phối class trong dataset
    """
    class_counts = {}
    
    for img_path, label_path in valid_pairs:
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # class_id x y w h
                        class_id = int(parts[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except Exception as e:
            print(f"Error reading {label_path}: {e}")
    
    return class_counts

def create_clean_dataset(valid_pairs: List[Tuple[str, str]], output_dir: str, 
                        train_split: float, val_split: float, test_split: float):
    """
    Tạo dataset clean với tỷ lệ chia mới
    """
    # Shuffle data
    random.shuffle(valid_pairs)
    
    n_total = len(valid_pairs)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    print(f"Creating clean dataset:")
    print(f"Total: {n_total}")
    print(f"Train: {n_train} ({n_train/n_total:.1%})")
    print(f"Val: {n_val} ({n_val/n_total:.1%})")
    print(f"Test: {n_test} ({n_test/n_total:.1%})")
    
    # Chia data
    train_pairs = valid_pairs[:n_train]
    val_pairs = valid_pairs[n_train:n_train + n_val]
    test_pairs = valid_pairs[n_train + n_val:]
    
    splits = {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }
    
    # Xóa output directory cũ
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Tạo directories mới
    for split_name, pairs in splits.items():
        images_dir = os.path.join(output_dir, 'images', split_name)
        labels_dir = os.path.join(output_dir, 'labels', split_name)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Copy files
        for i, (img_path, label_path) in enumerate(pairs):
            # Tạo tên file mới với format consistent
            ext = os.path.splitext(img_path)[1]
            new_name = f"{split_name}_{i:05d}"
            
            new_img_path = os.path.join(images_dir, new_name + ext)
            new_label_path = os.path.join(labels_dir, new_name + '.txt')
            
            shutil.copy2(img_path, new_img_path)
            shutil.copy2(label_path, new_label_path)
        
        print(f"Created {split_name}: {len(pairs)} samples")
    
    return splits

def create_dataset_yaml(output_dir: str, class_names: List[str]):
    """
    Tạo dataset.yaml file
    """
    dataset_config = {
        'path': output_dir,
        'train': os.path.join(output_dir, 'images', 'train'),
        'val': os.path.join(output_dir, 'images', 'val'),
        'test': os.path.join(output_dir, 'images', 'test'),
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = os.path.join(output_dir, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"Created dataset.yaml at {yaml_path}")
    return yaml_path

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Tạo lại dataset clean từ scratch')
    parser.add_argument('--input_dir', default='/home/huynguyen/source/Trash-Detection/data/processed', 
                       help='Input data directory')
    parser.add_argument('--output_dir', default='/home/huynguyen/source/Trash-Detection/data/processed/clean', 
                       help='Output clean dataset directory')
    parser.add_argument('--train_split', type=float, default=0.6, help='Train split ratio')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.3, help='Test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Validate splits
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0, got {total}")
    
    print("=" * 50)
    print("TẠO LẠI DATASET CLEAN")
    print("=" * 50)
    
    # Find valid pairs
    print(f"Scanning {args.input_dir} for valid data...")
    valid_pairs = find_valid_pairs(args.input_dir)
    
    if len(valid_pairs) == 0:
        print("No valid data found!")
        return
    
    # Analyze class distribution
    print("\\nAnalyzing class distribution...")
    class_counts = analyze_class_distribution(valid_pairs)
    print("Class distribution:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  Class {class_id}: {count} samples")
    
    # Define class names (TACO dataset mapping)
    class_names = [
        'cardboard',    # 0
        'glass',        # 1
        'metal',        # 2
        'organic',      # 3
        'other',        # 4
        'paper',        # 5
        'plastic'       # 6
    ]
    
    # Create clean dataset
    print(f"\\nCreating clean dataset at {args.output_dir}...")
    splits = create_clean_dataset(
        valid_pairs, args.output_dir, 
        args.train_split, args.val_split, args.test_split
    )
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(args.output_dir, class_names)
    
    # Final stats
    print("\\n" + "=" * 50)
    print("DATASET CLEAN HOÀN THÀNH")
    print("=" * 50)
    print(f"Output: {args.output_dir}")
    print(f"Config: {yaml_path}")
    print(f"Train: {len(splits['train'])} samples")
    print(f"Val: {len(splits['val'])} samples") 
    print(f"Test: {len(splits['test'])} samples")
    print(f"Total: {sum(len(pairs) for pairs in splits.values())} samples")

if __name__ == "__main__":
    main()