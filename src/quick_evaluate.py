#!/usr/bin/env python3
"""
Quick Evaluate Script - Limited test images for fast evaluation
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def quick_evaluate(model_path="../models/best.pt", max_images=50):
    """Đánh giá nhanh với số lượng ảnh giới hạn"""
    print("⚡ QUICK EVALUATION")
    print("=" * 50)
    
    # Load model
    print(f"📦 Loading model: {model_path}")
    model = YOLO(model_path)
    class_names = list(model.names.values())
    print(f"🏷️ Classes ({len(class_names)}): {class_names}")
    
    # Paths
    test_images_dir = Path("../data/processed/images/test")
    test_labels_dir = Path("../data/processed/labels/test")
    
    # Get test images
    image_files = list(test_images_dir.glob("*.jpg"))[:max_images]
    print(f"🖼️ Testing on {len(image_files)} images (limited from {len(list(test_images_dir.glob('*.jpg')))})")
    
    ground_truth = []
    predictions = []
    correct_predictions = 0
    total_predictions = 0
    
    # Process images
    for i, img_path in enumerate(image_files):
        if i % 10 == 0:
            print(f"   Processing {i+1}/{len(image_files)}...")
        
        try:
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Predict
            results = model(image, conf=0.25, device="cpu", verbose=False)
            
            # Get predicted class
            pred_class = None
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                max_conf_idx = np.argmax(confidences)
                class_id = class_ids[max_conf_idx]
                
                if 0 <= class_id < len(class_names):
                    pred_class = class_names[class_id]
            
            # Get ground truth
            label_file = test_labels_dir / f"{img_path.stem}.txt"
            gt_class = None
            
            if label_file.exists():
                with open(label_file, 'r') as f:
                    line = f.readline().strip()
                    if line:
                        class_id = int(line.split()[0])
                        if 0 <= class_id < len(class_names):
                            gt_class = class_names[class_id]
            
            # Store results
            if gt_class is not None and pred_class is not None:
                ground_truth.append(gt_class)
                predictions.append(pred_class)
                total_predictions += 1
                
                if gt_class == pred_class:
                    correct_predictions += 1
            
        except Exception as e:
            print(f"⚠️ Error processing {img_path.name}: {e}")
            continue
    
    print(f"\n📊 RESULTS:")
    print(f"   Total samples: {total_predictions}")
    print(f"   Correct: {correct_predictions}")
    print(f"   Accuracy: {correct_predictions/total_predictions:.2%}" if total_predictions > 0 else "   Accuracy: N/A")
    
    if len(ground_truth) > 0:
        # Classification report
        print(f"\n📋 CLASSIFICATION REPORT:")
        report = classification_report(ground_truth, predictions, output_dict=True)
        
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                print(f"   {class_name:12s}: P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1-score']:.3f}")
        
        # Overall metrics
        print(f"\n📈 OVERALL:")
        print(f"   Accuracy: {report['accuracy']:.3f}")
        print(f"   Macro avg: P={report['macro avg']['precision']:.3f} R={report['macro avg']['recall']:.3f} F1={report['macro avg']['f1-score']:.3f}")
        
        # Simple confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=class_names)
        
        print(f"\n🎯 CONFUSION MATRIX (Top predictions):")
        df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
        
        # Show only classes that appear in test
        appearing_classes = set(ground_truth + predictions)
        df_cm_filtered = df_cm.loc[list(appearing_classes), list(appearing_classes)]
        print(df_cm_filtered)
        
        # Save simple plot
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_cm_filtered, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Quick Eval - {total_predictions} samples)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            
            save_path = "quick_evaluation_confusion_matrix.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n💾 Saved confusion matrix: {save_path}")
            
            # Show plot (comment out if running without display)
            # plt.show()
            
        except Exception as e:
            print(f"⚠️ Could not create plot: {e}")
    
    else:
        print("❌ No valid predictions found!")
    
    print(f"\n✅ Quick evaluation completed!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Quick Model Evaluation")
    parser.add_argument("--model", type=str, default="../models/best.pt", help="Model path")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum images to test")
    
    args = parser.parse_args()
    
    quick_evaluate(args.model, args.max_images)
