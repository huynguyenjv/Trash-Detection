#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balance Dataset for Training

Mô tả:
    Script để balance lại dataset bị imbalanced:
    - Undersample class đa số (BIODEGRADABLE)
    - Giữ nguyên các class thiểu số
    - Tạo dataset mới đã balanced

Author: Huy Nguyen
Date: December 2025
"""

import os
import shutil
import random
import logging
from pathlib import Path
from collections import defaultdict
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def analyze_dataset(labels_dir: Path) -> dict:
    """Phân tích phân bố classes trong dataset"""
    class_counts = defaultdict(int)
    image_classes = defaultdict(set)  # image -> set of classes
    
    for label_file in labels_dir.glob("*.txt"):
        image_name = label_file.stem
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    class_id = int(parts[0])
                    class_counts[class_id] += 1
                    image_classes[image_name].add(class_id)
    
    return dict(class_counts), dict(image_classes)


def balance_dataset(
    source_dir: Path,
    output_dir: Path,
    target_ratio: float = 0.25,  # Class 0 sẽ chiếm tối đa 25% 
    seed: int = 42
):
    """
    Balance dataset bằng cách undersample class đa số
    
    Args:
        source_dir: Thư mục dataset gốc (chứa train/valid/test)
        output_dir: Thư mục output cho dataset balanced
        target_ratio: Tỷ lệ tối đa cho class đa số
        seed: Random seed
    """
    random.seed(seed)
    
    logger.info("="*60)
    logger.info("BALANCING DATASET")
    logger.info("="*60)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        src_images = source_dir / split / 'images'
        src_labels = source_dir / split / 'labels'
        
        if not src_images.exists():
            logger.warning(f"Split {split} not found, skipping...")
            continue
        
        dst_images = output_dir / split / 'images'
        dst_labels = output_dir / split / 'labels'
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)
        
        # Analyze current distribution
        class_counts, image_classes = analyze_dataset(src_labels)
        
        logger.info(f"\n=== {split.upper()} SPLIT ===")
        logger.info("Original distribution:")
        total = sum(class_counts.values())
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            pct = (count / total) * 100
            logger.info(f"  Class {class_id}: {count} ({pct:.1f}%)")
        
        # Tìm images chỉ chứa class 0 (để undersample)
        # và images chứa các class khác (giữ nguyên)
        class0_only_images = []
        mixed_or_other_images = []
        
        for image_name, classes in image_classes.items():
            if classes == {0}:  # Chỉ có class 0
                class0_only_images.append(image_name)
            else:  # Có class khác hoặc mixed
                mixed_or_other_images.append(image_name)
        
        logger.info(f"\nImages with only class 0: {len(class0_only_images)}")
        logger.info(f"Images with other classes: {len(mixed_or_other_images)}")
        
        # Tính số images class 0 cần giữ lại
        # Để đạt target_ratio, cần: class0_new / total_new = target_ratio
        # total_new = mixed_images + class0_keep
        # Giả sử mixed_images đóng góp x annotations cho class 0
        # Ta cần: class0_keep_annotations / (total_other + class0_keep_annotations) <= target_ratio
        
        # Đơn giản hơn: giữ lại 1/4 số images chỉ có class 0
        keep_ratio = 0.15  # Giữ 15% images chỉ có class 0
        num_keep = int(len(class0_only_images) * keep_ratio)
        
        # Random sample
        random.shuffle(class0_only_images)
        kept_class0_images = class0_only_images[:num_keep]
        
        # Tổng images sẽ copy
        final_images = set(mixed_or_other_images + kept_class0_images)
        
        logger.info(f"\nAfter balancing:")
        logger.info(f"  Keeping {len(kept_class0_images)} class-0-only images (from {len(class0_only_images)})")
        logger.info(f"  Total images: {len(final_images)}")
        
        # Copy files
        copied = 0
        for image_name in final_images:
            # Find image file
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                src_img = src_images / f"{image_name}{ext}"
                if src_img.exists():
                    # Copy image
                    shutil.copy2(src_img, dst_images / src_img.name)
                    # Copy label
                    src_lbl = src_labels / f"{image_name}.txt"
                    if src_lbl.exists():
                        shutil.copy2(src_lbl, dst_labels / src_lbl.name)
                    copied += 1
                    break
        
        logger.info(f"  Copied {copied} images")
        
        # Verify new distribution
        new_class_counts, _ = analyze_dataset(dst_labels)
        new_total = sum(new_class_counts.values())
        
        logger.info(f"\nNew distribution:")
        for class_id in sorted(new_class_counts.keys()):
            count = new_class_counts[class_id]
            pct = (count / new_total) * 100 if new_total > 0 else 0
            logger.info(f"  Class {class_id}: {count} ({pct:.1f}%)")
    
    # Copy data.yaml và sửa path
    src_yaml = source_dir / 'data.yaml'
    dst_yaml = output_dir / 'data.yaml'
    
    with open(src_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    data_config['path'] = str(output_dir.absolute())
    
    with open(dst_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    logger.info(f"\n✅ Balanced dataset saved to: {output_dir}")
    logger.info(f"   data.yaml: {dst_yaml}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Balance imbalanced dataset")
    parser.add_argument("--source", type=str, default="data/garbage_detection",
                       help="Source dataset directory")
    parser.add_argument("--output", type=str, default="data/garbage_detection_balanced",
                       help="Output balanced dataset directory")
    parser.add_argument("--keep-ratio", type=float, default=0.15,
                       help="Ratio of class-0-only images to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    balance_dataset(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        seed=args.seed
    )


if __name__ == "__main__":
    main()
