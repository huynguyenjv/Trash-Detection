#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare Classification Dataset from Detection Dataset

Mô tả:
    Tạo dataset classification từ detection dataset bằng cách:
    - Crop các bounding boxes từ ảnh gốc
    - Lưu theo cấu trúc ImageNet (class_name/image.jpg)
    - Phù hợp để train YOLOv8-cls

    Input: Detection dataset (YOLOv8 format)
    Output: Classification dataset (ImageFolder format)

Author: Huy Nguyen
Email: huynguyen@example.com
Date: December 2025
Version: 1.0.0
License: MIT
"""

import os
import cv2
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import random
import yaml

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetectionToClassificationConverter:
    """Chuyển đổi detection dataset sang classification dataset"""
    
    def __init__(
        self,
        detection_data_yaml: str,
        output_dir: str,
        img_size: Tuple[int, int] = (224, 224),
        min_crop_size: int = 32,
        padding_ratio: float = 0.1
    ):
        self.detection_data_yaml = Path(detection_data_yaml)
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.min_crop_size = min_crop_size
        self.padding_ratio = padding_ratio
        
        # Load detection config
        with open(self.detection_data_yaml, 'r') as f:
            self.detection_config = yaml.safe_load(f)
        
        # Get classes
        self.classes = self.detection_config.get('names', {})
        if isinstance(self.classes, list):
            self.classes = {i: name for i, name in enumerate(self.classes)}
        
        logger.info(f"Loaded {len(self.classes)} classes: {list(self.classes.values())}")
    
    def create_directories(self):
        """Tạo cấu trúc thư mục output"""
        for split in ['train', 'val', 'test']:
            for class_name in self.classes.values():
                (self.output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created directory structure at: {self.output_dir}")
    
    def parse_yolo_label(self, label_path: Path, img_width: int, img_height: int) -> List[Dict]:
        """Parse YOLO format label file"""
        bboxes = []
        
        if not label_path.exists():
            return bboxes
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    
                    # Convert normalized to absolute coordinates
                    x1 = int((cx - w/2) * img_width)
                    y1 = int((cy - h/2) * img_height)
                    x2 = int((cx + w/2) * img_width)
                    y2 = int((cy + h/2) * img_height)
                    
                    # Clamp to image bounds
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(img_width, x2)
                    y2 = min(img_height, y2)
                    
                    bboxes.append({
                        'class_id': class_id,
                        'class_name': self.classes.get(class_id, 'unknown'),
                        'bbox': (x1, y1, x2, y2)
                    })
        
        return bboxes
    
    def crop_and_save(self, img: any, bbox: Tuple[int, int, int, int], 
                      output_path: Path) -> bool:
        """Crop bounding box từ ảnh và lưu"""
        x1, y1, x2, y2 = bbox
        
        # Add padding
        w, h = x2 - x1, y2 - y1
        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)
        
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(img.shape[1], x2 + pad_w)
        y2 = min(img.shape[0], y2 + pad_h)
        
        # Check minimum size
        if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
            return False
        
        # Crop
        crop = img[y1:y2, x1:x2]
        
        # Resize to target size
        crop_resized = cv2.resize(crop, self.img_size)
        
        # Save
        cv2.imwrite(str(output_path), crop_resized)
        return True
    
    def process_split(self, split: str) -> Dict[str, int]:
        """Xử lý một split (train/val/test)"""
        stats = {class_name: 0 for class_name in self.classes.values()}
        
        # Get paths from detection config
        detection_base = self.detection_data_yaml.parent
        
        # Determine split folder name
        split_folder = split
        if split == 'val':
            split_folder = 'valid'  # Roboflow uses 'valid' instead of 'val'
        
        images_dir = detection_base / split_folder / 'images'
        labels_dir = detection_base / split_folder / 'labels'
        
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return stats
        
        # Process each image
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        logger.info(f"Processing {len(image_files)} images in {split} split...")
        
        for img_path in image_files:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Get corresponding label file
            label_path = labels_dir / (img_path.stem + '.txt')
            
            # Parse bounding boxes
            bboxes = self.parse_yolo_label(label_path, img_width, img_height)
            
            # Crop and save each bbox
            for i, bbox_info in enumerate(bboxes):
                class_name = bbox_info['class_name']
                bbox = bbox_info['bbox']
                
                # Generate output filename
                output_filename = f"{img_path.stem}_crop{i}.jpg"
                output_path = self.output_dir / split / class_name / output_filename
                
                # Crop and save
                if self.crop_and_save(img, bbox, output_path):
                    stats[class_name] += 1
        
        return stats
    
    def convert(self) -> Dict[str, Dict[str, int]]:
        """Chạy chuyển đổi toàn bộ dataset"""
        logger.info("="*60)
        logger.info("CONVERTING DETECTION DATASET TO CLASSIFICATION")
        logger.info("="*60)
        
        # Create directories
        self.create_directories()
        
        # Process each split
        all_stats = {}
        for split in ['train', 'val', 'test']:
            logger.info(f"\nProcessing {split} split...")
            stats = self.process_split(split)
            all_stats[split] = stats
            
            total = sum(stats.values())
            logger.info(f"{split}: {total} crops created")
            for class_name, count in stats.items():
                if count > 0:
                    logger.info(f"  {class_name}: {count}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("CONVERSION COMPLETE!")
        logger.info("="*60)
        
        total_train = sum(all_stats['train'].values())
        total_val = sum(all_stats['val'].values())
        total_test = sum(all_stats['test'].values())
        
        logger.info(f"Train: {total_train} crops")
        logger.info(f"Val: {total_val} crops")
        logger.info(f"Test: {total_test} crops")
        logger.info(f"Total: {total_train + total_val + total_test} crops")
        logger.info(f"Output: {self.output_dir}")
        
        return all_stats


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert detection dataset to classification")
    parser.add_argument("--detection-yaml", type=str, 
                       default="data/garbage_detection/data.yaml",
                       help="Path to detection dataset YAML")
    parser.add_argument("--output", type=str,
                       default="data/garbage_classification",
                       help="Output directory for classification dataset")
    parser.add_argument("--img-size", type=int, default=224,
                       help="Target image size for crops")
    parser.add_argument("--min-crop", type=int, default=32,
                       help="Minimum crop size (skip smaller boxes)")
    parser.add_argument("--padding", type=float, default=0.1,
                       help="Padding ratio around bounding boxes")
    
    args = parser.parse_args()
    
    converter = DetectionToClassificationConverter(
        detection_data_yaml=args.detection_yaml,
        output_dir=args.output,
        img_size=(args.img_size, args.img_size),
        min_crop_size=args.min_crop,
        padding_ratio=args.padding
    )
    
    converter.convert()


if __name__ == "__main__":
    main()
