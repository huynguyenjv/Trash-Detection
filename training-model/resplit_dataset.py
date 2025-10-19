#!/usr/bin/env python3
"""
Script để chia lại dataset theo tỷ lệ mới: Train 60%, Val 10%, Test 30%
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple
import argparse

def load_config():
    """Load config từ training_config.yaml"""
    config_path = "configs/training_config.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_file_pairs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """
    Lấy danh sách các cặp image-label file có matching
    Returns: List of (image_path, label_path) tuples
    """
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(images_dir).glob(ext))
    
    valid_pairs = []
    for img_path in image_files:
        # Tìm label file tương ứng
        label_name = img_path.stem + '.txt'
        label_path = Path(labels_dir) / label_name
        
        # Check if both files exist
        if label_path.exists() and img_path.exists():
            valid_pairs.append((str(img_path), str(label_path)))
        else:
            if not label_path.exists():
                print(f"Warning: No label found for {img_path}")
            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
    
    return valid_pairs

def split_dataset(file_pairs: List[Tuple[str, str]], train_split: float, val_split: float, test_split: float):
    """
    Chia dataset theo tỷ lệ mới
    """
    # Kiểm tra tổng tỷ lệ = 1.0
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Tổng tỷ lệ phải = 1.0, hiện tại = {total}")
    
    # Shuffle dữ liệu
    random.shuffle(file_pairs)
    
    n_total = len(file_pairs)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val  # Phần còn lại
    
    print(f"Tổng số samples: {n_total}")
    print(f"Train: {n_train} ({n_train/n_total:.1%})")
    print(f"Val: {n_val} ({n_val/n_total:.1%})")
    print(f"Test: {n_test} ({n_test/n_total:.1%})")
    
    # Chia dataset
    train_pairs = file_pairs[:n_train]
    val_pairs = file_pairs[n_train:n_train + n_val]
    test_pairs = file_pairs[n_train + n_val:]
    
    return {
        'train': train_pairs,
        'val': val_pairs,
        'test': test_pairs
    }

def copy_files(file_pairs: List[Tuple[str, str]], target_images_dir: str, target_labels_dir: str):
    """
    Copy files to target directories
    """
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    copied_count = 0
    for img_path, label_path in file_pairs:
        # Check if files exist
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        if not os.path.exists(label_path):
            print(f"Warning: Label not found: {label_path}")
            continue
            
        # Copy image
        img_name = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(target_images_dir, img_name))
        
        # Copy label
        label_name = os.path.basename(label_path)
        shutil.copy2(label_path, os.path.join(target_labels_dir, label_name))
        
        copied_count += 1
    
    print(f"Copied {copied_count} valid pairs")

def backup_current_split(data_dir: str):
    """
    Backup current dataset split
    """
    backup_dir = os.path.join(data_dir, "backup_old_split")
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    
    # Backup train, val, test directories
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            backup_split_dir = os.path.join(backup_dir, split)
            shutil.copytree(split_dir, backup_split_dir)
            print(f"Backed up {split} to {backup_split_dir}")

def resplit_detection_dataset():
    """
    Chia lại TACO detection dataset
    """
    config = load_config()
    data_dir = config['datasets']['taco']['processed_dir']
    
    train_split = config['datasets']['taco']['train_split']
    val_split = config['datasets']['taco']['val_split'] 
    test_split = config['datasets']['taco']['test_split']
    
    print(f"Chia lại detection dataset: {train_split:.1%} / {val_split:.1%} / {test_split:.1%}")
    print(f"Dataset dir: {data_dir}")
    
    # Backup current split
    backup_current_split(data_dir)
    
    # Thu thập tất cả file pairs từ các split hiện tại
    all_pairs = []
    for split in ['train', 'val', 'test']:
        images_dir = os.path.join(data_dir, 'images', split)
        labels_dir = os.path.join(data_dir, 'labels', split)
        
        if os.path.exists(images_dir) and os.path.exists(labels_dir):
            pairs = get_file_pairs(images_dir, labels_dir)
            all_pairs.extend(pairs)
            print(f"Found {len(pairs)} pairs in {split}")
    
    print(f"Total pairs collected: {len(all_pairs)}")
    
    # Chia lại dataset
    new_splits = split_dataset(all_pairs, train_split, val_split, test_split)
    
    # Xóa directories cũ
    for split in ['train', 'val', 'test']:
        split_images_dir = os.path.join(data_dir, 'images', split)
        split_labels_dir = os.path.join(data_dir, 'labels', split)
        if os.path.exists(split_images_dir):
            shutil.rmtree(split_images_dir)
        if os.path.exists(split_labels_dir):
            shutil.rmtree(split_labels_dir)
    
    # Tạo directories mới và copy files
    for split_name, pairs in new_splits.items():
        images_dir = os.path.join(data_dir, 'images', split_name)
        labels_dir = os.path.join(data_dir, 'labels', split_name)
        
        copy_files(pairs, images_dir, labels_dir)
        print(f"Created {split_name} with {len(pairs)} samples")
    
    # Update dataset.yaml
    dataset_yaml_path = os.path.join(data_dir, 'dataset.yaml')
    if os.path.exists(dataset_yaml_path):
        with open(dataset_yaml_path, 'r') as f:
            dataset_config = yaml.safe_load(f)
        
        # Update paths
        dataset_config['train'] = os.path.join(data_dir, 'images', 'train')
        dataset_config['val'] = os.path.join(data_dir, 'images', 'val')
        dataset_config['test'] = os.path.join(data_dir, 'images', 'test')
        
        with open(dataset_yaml_path, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Updated {dataset_yaml_path}")

def resplit_classification_dataset():
    """
    Chia lại classification dataset (nếu có)
    """
    config = load_config()
    data_dir = config['datasets']['trashnet']['processed_dir']
    
    if not os.path.exists(data_dir):
        print(f"Classification dataset not found at {data_dir}")
        return
    
    train_split = config['datasets']['trashnet']['train_split']
    val_split = config['datasets']['trashnet']['val_split']
    test_split = config['datasets']['trashnet']['test_split']
    
    print(f"Chia lại classification dataset: {train_split:.1%} / {val_split:.1%} / {test_split:.1%}")
    print(f"Dataset dir: {data_dir}")
    
    # Thu thập tất cả files từ các class directories
    all_files_by_class = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            for class_name in os.listdir(split_dir):
                class_dir = os.path.join(split_dir, class_name)
                if os.path.isdir(class_dir):
                    if class_name not in all_files_by_class:
                        all_files_by_class[class_name] = []
                    
                    # Thu thập tất cả image files
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        files = list(Path(class_dir).glob(ext))
                        all_files_by_class[class_name].extend([str(f) for f in files])
    
    # Chia lại cho từng class
    new_splits = {'train': {}, 'val': {}, 'test': {}}
    
    for class_name, files in all_files_by_class.items():
        random.shuffle(files)
        n_total = len(files)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        new_splits['train'][class_name] = files[:n_train]
        new_splits['val'][class_name] = files[n_train:n_train + n_val]
        new_splits['test'][class_name] = files[n_train + n_val:]
        
        print(f"Class {class_name}: {len(new_splits['train'][class_name])}/{len(new_splits['val'][class_name])}/{len(new_splits['test'][class_name])}")
    
    # Xóa directories cũ
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)
    
    # Tạo directories mới và copy files
    for split_name, classes in new_splits.items():
        for class_name, files in classes.items():
            class_dir = os.path.join(data_dir, split_name, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for file_path in files:
                file_name = os.path.basename(file_path)
                shutil.copy2(file_path, os.path.join(class_dir, file_name))

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Chia lại dataset theo tỷ lệ mới')
    parser.add_argument('--detection', action='store_true', help='Chia lại detection dataset')
    parser.add_argument('--classification', action='store_true', help='Chia lại classification dataset')
    parser.add_argument('--all', action='store_true', help='Chia lại cả hai datasets')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    if args.all or args.detection:
        print("=" * 50)
        print("CHIA LẠI DETECTION DATASET")
        print("=" * 50)
        resplit_detection_dataset()
    
    if args.all or args.classification:
        print("=" * 50)
        print("CHIA LẠI CLASSIFICATION DATASET")
        print("=" * 50)
        resplit_classification_dataset()
    
    if not any([args.detection, args.classification, args.all]):
        print("Vui lòng chọn --detection, --classification, hoặc --all")
        parser.print_help()

if __name__ == "__main__":
    main()