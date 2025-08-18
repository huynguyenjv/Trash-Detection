#!/usr/bin/env python3
"""
Test script to check PyTorch model loading fix
"""

import os
import sys
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# Set environment variable to fix PyTorch 2.6+ issue
os.environ['TORCH_WEIGHTS_ONLY'] = 'False'

# Monkey patch torch.load to always use weights_only=False
original_load = torch.load

def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
print("✅ Applied torch.load patch")

try:
    from ultralytics import YOLO
    print("✅ Successfully imported ultralytics")
    
    # Try loading YOLOv8n model
    print("📥 Attempting to load YOLOv8n model...")
    model = YOLO('yolov8n.pt')
    print("✅ Successfully loaded YOLOv8n model!")
    
    # Test a simple prediction
    import numpy as np
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy_image, verbose=False)
    print(f"✅ Model prediction test successful! Got {len(results)} results")
    
    # Check if custom model exists
    model_paths_to_check = [
        '../models/final.pt',
        '../../models/final.pt',
        '../../../models/final.pt',
        'D:/MasterUIT/Trash-Detection/models/final.pt'
    ]
    
    custom_model_loaded = False
    for model_path in model_paths_to_check:
        if os.path.exists(model_path):
            print(f"📥 Found custom model: {model_path}")
            try:
                custom_model = YOLO(model_path)
                print(f"✅ Successfully loaded custom model: {model_path}")
                custom_model_loaded = True
                break
            except Exception as e:
                print(f"❌ Failed to load custom model {model_path}: {e}")
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    if not custom_model_loaded:
        print("ℹ️  No custom model found, will use YOLOv8n default")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Restore original torch.load
    torch.load = original_load
    print("🔄 Restored original torch.load")
