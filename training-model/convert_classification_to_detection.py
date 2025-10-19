#!/usr/bin/env python3
"""
Script để convert garbage classification dataset sang detection format với labels
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import argparse

def map_class_name_to_id(class_name: str) -> int:
    """
    Map class name to ID (TACO style)
    """
    class_mapping = {
        'cardboard': 0,
        'glass': 1,
        'metal': 2,
        'biological': 3,  # organic
        'trash': 4,       # other  
        'paper': 5,
        'plastic': 6,
        'battery': 4,     # other
        'clothes': 4,     # other
        'shoes': 4        # other
    }
    return class_mapping.get(class_name, 4)  # default to 'other'

def create_full_image_label(class_id: int) -> str:
    """
    Tạo label cho full image (whole image is the object)
    Format: class_id x_center y_center width height (normalized)
    """
    # Full image có center tại (0.5, 0.5) và size (1.0, 1.0)
    return f"{class_id} 0.5 0.5 1.0 1.0\\n"

def process_classification_dataset(input_dir: str, output_dir: str, 
                                 train_split: float, val_split: float, test_split: float):
    """
    Convert classification dataset to detection format
    """
    print(f"Processing classification dataset from {input_dir}")
    
    # Collect all images by class
    all_images = []
    class_stats = {}
    
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_id = map_class_name_to_id(class_name)
        image_files = []
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(class_dir).glob(ext))
        
        for img_path in image_files:
            all_images.append((str(img_path), class_id, class_name))
        
        class_stats[class_name] = len(image_files)
        print(f"Class {class_name} -> ID {class_id}: {len(image_files)} images")
    
    print(f"\\nTotal images: {len(all_images)}")
    print("Class distribution:")
    for class_name, count in class_stats.items():
        print(f"  {class_name}: {count} ({count/len(all_images):.1%})")
    
    # Shuffle and split
    random.shuffle(all_images)
    
    n_total = len(all_images)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val
    
    train_images = all_images[:n_train]
    val_images = all_images[n_train:n_train + n_val]
    test_images = all_images[n_train + n_val:]
    
    splits = {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }
    
    print(f"\\nSplit sizes:")
    print(f"Train: {len(train_images)} ({len(train_images)/n_total:.1%})")
    print(f"Val: {len(val_images)} ({len(val_images)/n_total:.1%})")
    print(f"Test: {len(test_images)} ({len(test_images)/n_total:.1%})")
    
    # Create output structure
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    for split_name, split_images in splits.items():
        images_dir = os.path.join(output_dir, 'images', split_name)
        labels_dir = os.path.join(output_dir, 'labels', split_name)
        
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        for i, (img_path, class_id, class_name) in enumerate(split_images):
            # Create new filename
            ext = os.path.splitext(img_path)[1]
            new_name = f"{split_name}_{i:05d}"
            
            new_img_path = os.path.join(images_dir, new_name + ext)
            new_label_path = os.path.join(labels_dir, new_name + '.txt')
            
            # Copy image
            shutil.copy2(img_path, new_img_path)
            
            # Create label file
            label_content = create_full_image_label(class_id)
            with open(new_label_path, 'w') as f:
                f.write(label_content)
        
        print(f"Created {split_name}: {len(split_images)} samples")
    
    return splits

def create_dataset_yaml(output_dir: str):
    """
    Create dataset.yaml for YOLO
    """
    class_names = [
        'cardboard',    # 0
        'glass',        # 1  
        'metal',        # 2
        'biological',   # 3 (organic)
        'other',        # 4 (trash, battery, clothes, shoes)
        'paper',        # 5
        'plastic'       # 6
    ]
    
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
    parser = argparse.ArgumentParser(description='Convert classification to detection dataset')
    parser.add_argument('--input_dir', 
                       default='/home/huynguyen/source/Trash-Detection/data/raw/garbage-dataset',
                       help='Input classification dataset directory')
    parser.add_argument('--output_dir', 
                       default='/home/huynguyen/source/Trash-Detection/data/processed/detection',
                       help='Output detection dataset directory')
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
    
    print("=" * 60)
    print("CONVERT CLASSIFICATION TO DETECTION DATASET")
    print("=" * 60)
    
    # Process dataset
    splits = process_classification_dataset(
        args.input_dir, args.output_dir,
        args.train_split, args.val_split, args.test_split
    )
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(args.output_dir)
    
    print("\\n" + "=" * 60)
    print("CONVERSION COMPLETED")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Dataset config: {yaml_path}")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Total samples: {sum(len(split) for split in splits.values())}")

if __name__ == "__main__":
    main()