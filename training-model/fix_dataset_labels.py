#!/usr/bin/env python3
"""
Script to fix corrupted label files in the YOLO detection dataset.
Cleans up invalid characters and validates dataset structure.
"""

import os
import re
import shutil
from pathlib import Path

def is_valid_yolo_label(line):
    """Check if a line contains a valid YOLO label format."""
    parts = line.strip().split()
    if len(parts) != 5:
        return False
    
    try:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])
        
        # Check if values are in valid ranges
        if class_id < 0:
            return False
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            return False
        if not (0 < width <= 1 and 0 < height <= 1):
            return False
        
        return True
    except ValueError:
        return False

def fix_label_file(file_path):
    """Fix a single label file by removing invalid characters."""
    try:
        # Read file in binary mode to handle the \n corruption properly
        with open(file_path, 'rb') as f:
            content_bytes = f.read()
        
        # Convert to string and fix the literal \n issue
        content = content_bytes.decode('utf-8', errors='ignore')
        
        # Fix the specific corruption pattern: literal \n instead of newline
        content = content.replace('\\n', '\n')
        
        # Remove any remaining % characters or other corruption
        content = re.sub(r'[%]', '', content)
        
        # Clean up any extra whitespace and normalize
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line and is_valid_yolo_label(line):
                lines.append(line)
        
        if not lines:
            return False
        
        # Write back clean content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        
        return True
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def validate_dataset(dataset_path):
    """Validate the dataset structure and report statistics."""
    print("\n📊 VALIDATING DATASET")
    print("="*40)
    
    dataset_path = Path(dataset_path)
    print(f"Validating dataset at {dataset_path}")
    
    splits = ['train', 'val', 'test']
    total_images = 0
    total_labels = 0
    
    for split in splits:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if images_dir.exists() and labels_dir.exists():
            images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            labels = list(labels_dir.glob('*.txt'))
            
            print(f"{split}: {len(images)} images, {len(labels)} labels")
            
            if len(images) == len(labels):
                print(f"  ✅ {split} is valid")
            else:
                print(f"  ❌ {split} mismatch: {len(images)} images vs {len(labels)} labels")
            
            total_images += len(images)
            total_labels += len(labels)
    
    return total_images, total_labels

def main():
    """Main function to fix dataset labels."""
    # Dataset path
    dataset_path = Path("/home/huynguyen/source/Trash-Detection/data/processed/detection")
    
    # Clear YOLO cache
    cache_files = [
        dataset_path / 'train.cache',
        dataset_path / 'val.cache', 
        dataset_path / 'test.cache'
    ]
    
    for cache_file in cache_files:
        if cache_file.exists():
            cache_file.unlink()
            print(f"Removed cache: {cache_file}")
    
    # Process all label files
    labels_dirs = [
        dataset_path / 'labels' / 'train',
        dataset_path / 'labels' / 'val',
        dataset_path / 'labels' / 'test'
    ]
    
    total_files = 0
    fixed_files = 0
    invalid_files = 0
    
    for labels_dir in labels_dirs:
        if labels_dir.exists():
            for label_file in labels_dir.glob('*.txt'):
                total_files += 1
                if fix_label_file(label_file):
                    fixed_files += 1
                else:
                    invalid_files += 1
    
    print(f"Fixed {fixed_files} label files")
    print(f"Found {invalid_files} invalid files")
    
    # Validate dataset
    validate_dataset(dataset_path)
    
    print("\n✅ Dataset fix completed!")

if __name__ == "__main__":
    main()