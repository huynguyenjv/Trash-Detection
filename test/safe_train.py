#!/usr/bin/env python3
"""
Training script với memory-safe configuration
"""

import os
import torch
from ultralytics import YOLO
from pathlib import Path

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

def safe_train():
    """Training với cấu hình an toàn"""
    try:
        # Clear memory trước khi bắt đầu
        clear_gpu_memory()
        
        # Kiểm tra GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"🚀 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            print("⚠️ Sử dụng CPU - training sẽ chậm")
        
        # Load model nhẹ nhất
        print("📦 Loading YOLOv8 nano model...")
        model = YOLO('yolov8n.pt')
        
        # Dataset path
        data_yaml = "../data/processed/dataset.yaml"
        if not Path(data_yaml).exists():
            raise FileNotFoundError(f"Dataset not found: {data_yaml}")
        
        print(f"📊 Dataset: {data_yaml}")
        
        # Training với memory-safe config
        print("🏋️ Starting training...")
        results = model.train(
            data=data_yaml,
            epochs=50,          # Training với 50 epochs
            batch=4,            # Batch size nhỏ
            imgsz=416,          # Image size nhỏ hơn (thay vì 640)
            device='auto',
            workers=1,          # Ít workers
            verbose=True,
            project='runs/train',
            name='trash_safe',
            
            # Memory optimization
            amp=True,           # Mixed precision
            cache=False,        # Không cache images trong RAM
            single_cls=False,
            
            # Giảm data augmentation để tiết kiệm memory
            mosaic=0.5,         # Giảm mosaic
            mixup=0.0,          # Tắt mixup
            copy_paste=0.0,     # Tắt copy-paste
            
            # Learning settings
            lr0=0.01,
            patience=5,         # Early stopping
            save_period=5,      # Save ít thường xuyên hơn
        )
        
        print("✅ Training completed!")
        
        # Lưu model
        best_path = Path("../runs/train/trash_safe/weights/best.pt")
        if best_path.exists():
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            import shutil
            final_path = models_dir / "trash_safe_best.pt"
            shutil.copy2(best_path, final_path)
            print(f"💾 Model saved: {final_path}")
            
            return str(final_path)
        else:
            print("⚠️ Best weights not found")
            return None
            
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ GPU Memory Error: {e}")
        print("💡 Thử giảm batch size hoặc image size:")
        print("   - batch=2")
        print("   - imgsz=320")
        print("   - Hoặc sử dụng CPU: device='cpu'")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_model(model_path):
    """Test model after training"""
    try:
        print(f"🧪 Testing model: {model_path}")
        
        model = YOLO(model_path)
        
        # Validate on test set
        results = model.val(
            data="../data/processed/dataset.yaml",
            split='test',
            batch=1,
            device='auto',
            verbose=False
        )
        
        print("📊 Test Results:")
        print(f"   mAP50: {results.box.map50:.3f}")
        print(f"   mAP50-95: {results.box.map:.3f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        return None

if __name__ == "__main__":
    print("🗑️ SAFE TRASH DETECTION TRAINING")
    print("="*50)
    
    # Set memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Train model
    model_path = safe_train()
    
    if model_path:
        # Test model
        test_model(model_path)
        
        print("\n✅ HOÀN THÀNH!")
        print(f"📁 Model: {model_path}")
        print("🎯 Để training lâu hơn, sửa epochs=50 trong script")
    else:
        print("\n❌ Training failed")
        print("💡 Thử chạy với CPU: device='cpu'")
