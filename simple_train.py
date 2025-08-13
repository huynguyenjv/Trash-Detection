#!/usr/bin/env python3
"""
Simple training script cho Trash Detection với demo data
"""

import os
import logging
from pathlib import Path
from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_dataset():
    """Tạo demo dataset structure"""
    print("🔧 Tạo demo dataset structure...")
    
    # Tạo thư mục dataset
    base_path = Path("data/demo_dataset")
    
    # Tạo cấu trúc thư mục
    for split in ['train', 'val']:
        for folder in ['images', 'labels']:
            (base_path / split / folder).mkdir(parents=True, exist_ok=True)
    
    # Tạo dataset.yaml
    dataset_yaml = f"""
path: {base_path.absolute()}
train: train/images
val: val/images
test: val/images

nc: 6
names: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
"""
    
    yaml_path = base_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(dataset_yaml.strip())
    
    print(f"✅ Created dataset.yaml at: {yaml_path}")
    
    # Tạo một số file ảnh demo (empty files)
    import numpy as np
    from PIL import Image
    
    # Tạo vài ảnh demo
    for i in range(3):
        # Random image
        img_array = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save train images
        img.save(base_path / 'train' / 'images' / f'demo_{i}.jpg')
        # Empty label file
        with open(base_path / 'train' / 'labels' / f'demo_{i}.txt', 'w') as f:
            f.write('0 0.5 0.5 0.2 0.2\n')  # class_id x_center y_center width height
        
        # Save val images  
        img.save(base_path / 'val' / 'images' / f'demo_val_{i}.jpg')
        with open(base_path / 'val' / 'labels' / f'demo_val_{i}.txt', 'w') as f:
            f.write('1 0.3 0.3 0.1 0.1\n')
    
    print("✅ Created demo images and labels")
    
    return yaml_path

def simple_train():
    """Train với demo dataset"""
    try:
        # Tạo demo dataset
        dataset_yaml = create_demo_dataset()
        
        print("🚀 Bắt đầu training với YOLO...")
        
        # Load pre-trained model
        model = YOLO('yolov8n.pt')
        
        # Training với minimal config
        results = model.train(
            data=str(dataset_yaml),
            epochs=5,  # Chỉ 5 epochs để test
            batch=4,   # Batch size nhỏ
            imgsz=640,
            device='auto',
            verbose=True,
            project='runs/train',
            name='trash_demo'
        )
        
        print("✅ Training hoàn thành!")
        print(f"📊 Results: {results}")
        
        # Lưu model
        model_path = Path("models")
        model_path.mkdir(exist_ok=True)
        
        # Copy best weights
        best_path = Path("runs/train/trash_demo/weights/best.pt")
        if best_path.exists():
            import shutil
            final_path = model_path / "demo_best.pt" 
            shutil.copy2(best_path, final_path)
            print(f"✅ Model saved to: {final_path}")
            return str(final_path)
        else:
            print("⚠️ Best weights not found")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

def test_trained_model(model_path: str):
    """Test model đã train"""
    try:
        print(f"🧪 Testing model: {model_path}")
        
        model = YOLO(model_path)
        
        # Test với ảnh demo
        results = model("data/demo_dataset/val/images/demo_val_0.jpg")
        
        print("✅ Model test thành công!")
        print(f"📊 Detected {len(results[0].boxes)} objects" if results[0].boxes else "📊 No objects detected")
        
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    print("🗑️ SIMPLE TRASH DETECTION TRAINER")
    print("="*40)
    
    # Train model
    model_path = simple_train()
    
    if model_path:
        # Test model
        test_trained_model(model_path)
        
        print("\n✅ HOÀN THÀNH!")
        print(f"📁 Model: {model_path}")
        print("🚀 Bây giờ bạn có thể:")
        print("   - Thêm ảnh thật vào data/demo_dataset/")
        print("   - Chạy detection với model này")
        print("   - Setup Kaggle dataset để training với data thật")
