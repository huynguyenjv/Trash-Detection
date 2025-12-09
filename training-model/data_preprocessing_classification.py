#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing for Image Classification - Trash Detection System

Mô tả:
    Tiền xử lý dataset cho Image Classification:
    - Xử lý Garbage-Dataset từ Kaggle
    - Tạo cấu trúc thư mục ImageNet-style
    - Chia train/val/test stratified
    - Resize và chuẩn hóa ảnh

Author: Huy Nguyen
Email: huynguyen.job2003@gmail.com
Date: October 2025
Version: 1.0.0
License: MIT
"""

import os
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import argparse

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_preprocessing_classification.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationConfig:
    """Cấu hình cho data preprocessing classification"""
    # Dataset paths
    raw_data_dir: Path = Path("../data/raw")
    processed_data_dir: Path = Path("data/processed")
    
    # Split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    
    # Image processing
    image_size: Tuple[int, int] = (224, 224)
    image_formats: List[str] = None
    
    # Output settings
    create_yaml: bool = True
    copy_images: bool = True
    
    # Random seed
    random_seed: int = 42
    
    def __post_init__(self):
        if self.image_formats is None:
            self.image_formats = ['.jpg', '.jpeg', '.png', '.bmp']


class GarbageDataProcessor:
    """Processor for Garbage dataset for trash classification."""
    
    def __init__(self, raw_data_dir: Path, processed_data_dir: Path):
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.classification_dir = processed_data_dir / "classification"
        
        # Garbage dataset paths - using existing local dataset
        self.garbage_dir = raw_data_dir / "garbage-dataset"
        
        # Class mapping for unified classification - updated for garbage dataset
        self.class_mapping = {
            "battery": "other", 
            "biological": "organic",
            "cardboard": "cardboard",
            "clothes": "other",
            "glass": "glass", 
            "metal": "metal",
            "paper": "paper",
            "plastic": "plastic",
            "shoes": "other",
            "trash": "other"
        }
        
        # Unified classes
        self.unified_classes = list(set(self.class_mapping.values()))
        self.unified_classes.sort()
    
    def check_garbage_dataset(self) -> bool:
        """Check if Garbage dataset exists locally."""
        try:
            if self.garbage_dir.exists():
                # Check if class directories exist
                class_dirs = [d for d in self.garbage_dir.iterdir() if d.is_dir()]
                if len(class_dirs) > 0:
                    logger.info(f"Garbage dataset found at {self.garbage_dir}")
                    logger.info(f"Found {len(class_dirs)} class directories")
                    return True
                else:
                    logger.error(f"No class directories found in {self.garbage_dir}")
                    return False
            else:
                logger.error(f"Garbage dataset not found at {self.garbage_dir}")
                return False
            
        except Exception as e:
            logger.error(f"Error checking Garbage dataset: {e}")
            return False
    
    def get_class_statistics(self) -> Dict[str, int]:
        """Get statistics for each class in the dataset."""
        stats = {}
        total_images = 0
        
        try:
            for class_dir in self.garbage_dir.iterdir():
                if class_dir.is_dir():
                    # Count images in this class
                    image_files = []
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_files.extend(class_dir.glob(f"*{ext}"))
                        image_files.extend(class_dir.glob(f"*{ext.upper()}"))
                    
                    count = len(image_files)
                    stats[class_dir.name] = count
                    total_images += count
                    
                    logger.info(f"Class '{class_dir.name}': {count} images")
            
            logger.info(f"Total images: {total_images}")
            return stats
            
        except Exception as e:
            logger.error(f"Error getting class statistics: {e}")
            return {}
    
    def create_unified_dataset(self, config: ClassificationConfig) -> bool:
        """Create unified dataset with train/val/test splits."""
        try:
            # Create output directories
            for split in ['train', 'val', 'test']:
                for unified_class in self.unified_classes:
                    output_dir = self.classification_dir / split / unified_class
                    output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each original class
            all_files_by_unified_class = {cls: [] for cls in self.unified_classes}
            
            for class_dir in self.garbage_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                
                original_class = class_dir.name
                if original_class not in self.class_mapping:
                    logger.warning(f"Unknown class: {original_class}, skipping...")
                    continue
                
                unified_class = self.class_mapping[original_class]
                
                # Get all image files in this class
                image_files = []
                for ext in config.image_formats:
                    image_files.extend(class_dir.glob(f"*{ext}"))
                    image_files.extend(class_dir.glob(f"*{ext.upper()}"))
                
                # Add to unified class collection
                for img_file in image_files:
                    all_files_by_unified_class[unified_class].append({
                        'file_path': img_file,
                        'original_class': original_class
                    })
                
                logger.info(f"Mapped {len(image_files)} images from '{original_class}' to '{unified_class}'")
            
            # Split data for each unified class
            random.seed(config.random_seed)
            dataset_stats = {'train': {}, 'val': {}, 'test': {}}
            
            for unified_class, files in all_files_by_unified_class.items():
                if not files:
                    logger.warning(f"No files found for unified class: {unified_class}")
                    continue
                
                # Shuffle files
                random.shuffle(files)
                
                # Calculate split indices
                total_files = len(files)
                train_end = int(total_files * config.train_ratio)
                val_end = train_end + int(total_files * config.val_ratio)
                
                # Split files
                train_files = files[:train_end]
                val_files = files[train_end:val_end]
                test_files = files[val_end:]
                
                # Copy files to appropriate directories
                splits = {
                    'train': train_files,
                    'val': val_files, 
                    'test': test_files
                }
                
                for split_name, split_files in splits.items():
                    dataset_stats[split_name][unified_class] = len(split_files)
                    
                    if config.copy_images:
                        for i, file_info in enumerate(split_files):
                            src_path = file_info['file_path']
                            dst_path = self.classification_dir / split_name / unified_class / f"{unified_class}_{i:06d}{src_path.suffix}"
                            
                            try:
                                shutil.copy2(src_path, dst_path)
                            except Exception as e:
                                logger.error(f"Error copying {src_path} to {dst_path}: {e}")
                
                logger.info(f"Class '{unified_class}': {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
            
            # Save dataset statistics
            stats_file = self.classification_dir / "dataset_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'class_mapping': self.class_mapping,
                    'unified_classes': self.unified_classes,
                    'dataset_stats': dataset_stats,
                    'total_files': sum(len(files) for files in all_files_by_unified_class.values())
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset statistics saved to {stats_file}")
            
            # Create dataset.yaml for YOLOv8 classification
            if config.create_yaml:
                self.create_dataset_yaml()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating unified dataset: {e}")
            return False
    
    def create_dataset_yaml(self) -> bool:
        """Create dataset.yaml file for YOLOv8 classification."""
        try:
            yaml_content = {
                'path': str(self.classification_dir.absolute()),
                'train': 'train',
                'val': 'val',
                'test': 'test',
                'names': {i: name for i, name in enumerate(self.unified_classes)}
            }
            
            yaml_file = self.classification_dir / "dataset.yaml"
            
            # Write YAML manually to ensure proper format
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(f"# Trash Classification Dataset\n")
                f.write(f"path: {yaml_content['path']}\n")
                f.write(f"train: {yaml_content['train']}\n")
                f.write(f"val: {yaml_content['val']}\n")
                f.write(f"test: {yaml_content['test']}\n")
                f.write(f"\n# Classes\n")
                f.write(f"names:\n")
                for i, name in yaml_content['names'].items():
                    f.write(f"  {i}: {name}\n")
            
            logger.info(f"Dataset YAML created: {yaml_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating dataset YAML: {e}")
            return False
    
    def validate_dataset(self) -> bool:
        """Validate the processed dataset."""
        try:
            validation_results = {
                'splits': {},
                'total_images': 0,
                'classes': self.unified_classes
            }
            
            for split in ['train', 'val', 'test']:
                split_dir = self.classification_dir / split
                if not split_dir.exists():
                    logger.error(f"Split directory not found: {split_dir}")
                    return False
                
                split_stats = {}
                split_total = 0
                
                for unified_class in self.unified_classes:
                    class_dir = split_dir / unified_class
                    if class_dir.exists():
                        image_count = len(list(class_dir.glob("*.*")))
                        split_stats[unified_class] = image_count
                        split_total += image_count
                    else:
                        logger.warning(f"Class directory not found: {class_dir}")
                        split_stats[unified_class] = 0
                
                validation_results['splits'][split] = {
                    'classes': split_stats,
                    'total': split_total
                }
                validation_results['total_images'] += split_total
                
                logger.info(f"Split '{split}': {split_total} images")
            
            # Save validation results
            validation_file = self.classification_dir / "validation_results.json"
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dataset validation completed. Total images: {validation_results['total_images']}")
            logger.info(f"Validation results saved to {validation_file}")
            
            return validation_results['total_images'] > 0
            
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False
    
    def run_preprocessing(self, config: ClassificationConfig) -> bool:
        """Run the complete preprocessing pipeline."""
        try:
            logger.info("Starting Garbage dataset preprocessing for classification...")
            
            # Step 1: Check if dataset exists
            if not self.check_garbage_dataset():
                return False
            
            # Step 2: Get class statistics
            logger.info("\n=== Dataset Statistics ===")
            class_stats = self.get_class_statistics()
            if not class_stats:
                return False
            
            # Step 3: Create unified dataset with splits
            logger.info("\n=== Creating Unified Dataset ===")
            if not self.create_unified_dataset(config):
                return False
            
            # Step 4: Validate dataset
            logger.info("\n=== Validating Dataset ===")
            if not self.validate_dataset():
                return False
            
            logger.info("\n=== Classification Dataset Preprocessing Completed Successfully! ===")
            return True
            
        except Exception as e:
            logger.error(f"Error in preprocessing pipeline: {e}")
            return False


def main():
    """Main function for running classification data preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess Garbage dataset for trash classification")
    parser.add_argument("--raw-data-dir", type=str, default="../data/raw",
                        help="Path to raw data directory")
    parser.add_argument("--processed-data-dir", type=str, default="data/processed", 
                        help="Path to processed data directory")
    parser.add_argument("--train-ratio", type=float, default=0.7,
                        help="Training data ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation data ratio")
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Test data ratio")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy images, just create directory structure")
    
    args = parser.parse_args()
    
    # Create configuration
    config = ClassificationConfig(
        raw_data_dir=Path(args.raw_data_dir),
        processed_data_dir=Path(args.processed_data_dir),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        copy_images=not args.no_copy
    )
    
    # Initialize processor
    processor = GarbageDataProcessor(config.raw_data_dir, config.processed_data_dir)
    
    # Run preprocessing
    success = processor.run_preprocessing(config)
    
    if success:
        logger.info("Classification data preprocessing completed successfully!")
        return 0
    else:
        logger.error("Classification data preprocessing failed!")
        return 1


if __name__ == "__main__":
    exit(main())