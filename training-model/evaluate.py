#!/usr/bin/env python3
"""
Comprehensive Evaluation System for Trash Detection Pipeline
Evaluate cả Detection Model và Classification Model một cách riêng biệt và kết hợp

Features:
- Detection Model Evaluation: mAP, precision, recall, F1-score
- Classification Model Evaluation: accuracy, precision, recall, F1-score  
- Pipeline Evaluation: end-to-end performance
- Detailed analysis và visualization

Author: Huy Nguyen
Date: September 2025
"""

import os
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import json
import yaml
import time
from datetime import datetime
import argparse

import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, average_precision_score,
    roc_curve, auc
)
from PIL import Image
import tqdm

# Import pipeline components
from detect import TrashDetectionPipeline, PipelineConfig, DetectionResult

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Cấu hình cho evaluation"""
    # Model paths
    detection_model_path: str = "models/detection/best.pt"
    classification_model_path: str = "models/classification/best.pt"
    
    # Dataset paths
    detection_data_yaml: str = "data/detection/processed/dataset_detection.yaml"
    classification_data_yaml: str = "data/classification/processed/dataset_classification.yaml"
    
    # Evaluation settings
    detection_conf_thresholds: List[float] = None
    detection_iou_threshold: float = 0.5
    classification_conf_threshold: float = 0.5
    
    # Device
    device: str = "auto"
    
    # Output
    results_dir: Path = Path("results/evaluation")
    experiment_name: str = "evaluation_v1"
    
    # Visualization
    save_plots: bool = True
    show_plots: bool = True
    
    def __post_init__(self):
        """Initialize default values"""
        if self.detection_conf_thresholds is None:
            self.detection_conf_thresholds = [0.1, 0.25, 0.5, 0.75]
        
        self.results_dir.mkdir(parents=True, exist_ok=True)


class DetectionEvaluator:
    """Evaluator cho detection model"""
    
    def __init__(self, model_path: str, data_yaml: str, device: str = "auto"):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Load dataset info
        with open(data_yaml, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        
        self.class_names = self.dataset_info['names']
        logger.info(f"Detection evaluator initialized with {len(self.class_names)} classes")
    
    def evaluate_model(self, conf_threshold: float = 0.25) -> Dict[str, Any]:
        """Evaluate detection model sử dụng YOLO's built-in evaluation"""
        try:
            logger.info(f"Evaluating detection model with conf={conf_threshold}")
            
            # Run YOLO evaluation
            results = self.model.val(
                data=self.data_yaml,
                split='test',
                conf=conf_threshold,
                iou=0.5,
                device=self.device,
                save_json=True,
                save_hybrid=True,
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': 2 * float(results.box.mp) * float(results.box.mr) / (float(results.box.mp) + float(results.box.mr)) if (float(results.box.mp) + float(results.box.mr)) > 0 else 0,
                'conf_threshold': conf_threshold,
            }
            
            # Per-class metrics nếu có
            if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
                per_class_ap50 = results.box.ap50.cpu().numpy()
                metrics['per_class_mAP50'] = {
                    self.class_names[i]: float(ap) for i, ap in enumerate(per_class_ap50)
                }
            
            logger.info(f"Detection Results - mAP50: {metrics['mAP50']:.4f}, mAP50-95: {metrics['mAP50_95']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating detection model: {e}")
            return {}
    
    def evaluate_multiple_thresholds(self, conf_thresholds: List[float]) -> Dict[str, Any]:
        """Evaluate với multiple confidence thresholds"""
        try:
            logger.info(f"Evaluating detection model with {len(conf_thresholds)} thresholds")
            
            all_results = {}
            for conf in conf_thresholds:
                results = self.evaluate_model(conf)
                all_results[conf] = results
            
            # Tạo summary
            summary = {
                'threshold_results': all_results,
                'best_threshold': max(all_results.keys(), key=lambda k: all_results[k].get('f1_score', 0)),
                'threshold_comparison': {
                    metric: {str(conf): results.get(metric, 0) for conf, results in all_results.items()}
                    for metric in ['mAP50', 'mAP50_95', 'precision', 'recall', 'f1_score']
                }
            }
            
            logger.info(f"Best threshold: {summary['best_threshold']} (F1: {all_results[summary['best_threshold']].get('f1_score', 0):.4f})")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in multi-threshold evaluation: {e}")
            return {}


class ClassificationEvaluator:
    """Evaluator cho classification model"""
    
    def __init__(self, model_path: str, data_yaml: str, device: str = "auto"):
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.device = device
        
        # Load model
        self.model = YOLO(model_path)
        self.model.to(device)
        
        # Load dataset info
        with open(data_yaml, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
        
        self.class_names = self.dataset_info['names']
        logger.info(f"Classification evaluator initialized with {len(self.class_names)} classes")
    
    def evaluate_model(self) -> Dict[str, Any]:
        """Evaluate classification model"""
        try:
            logger.info("Evaluating classification model")
            
            # Run YOLO evaluation
            results = self.model.val(
                data=self.data_yaml,
                split='test',
                device=self.device,
                plots=True,
                verbose=True
            )
            
            # Extract metrics
            metrics = {
                'top1_accuracy': float(results.top1),
                'top5_accuracy': float(results.top5) if hasattr(results, 'top5') else None,
            }
            
            logger.info(f"Classification Results - Top1: {metrics['top1_accuracy']:.4f}")
            if metrics['top5_accuracy']:
                logger.info(f"Top5: {metrics['top5_accuracy']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating classification model: {e}")
            return {}
    
    def detailed_evaluation(self) -> Dict[str, Any]:
        """Detailed evaluation với confusion matrix và per-class metrics"""
        try:
            logger.info("Running detailed classification evaluation")
            
            # Test directory
            test_dir = Path(self.dataset_info['path']) / 'test'
            
            all_predictions = []
            all_true_labels = []
            all_probabilities = []
            
            # Process từng class
            for class_idx, class_name in enumerate(self.class_names):
                class_dir = test_dir / class_name
                if not class_dir.exists():
                    continue
                
                # Get image files
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                image_files = [f for f in class_dir.iterdir() 
                             if f.suffix.lower() in image_extensions]
                
                logger.info(f"Evaluating {len(image_files)} images for class '{class_name}'")
                
                for img_path in tqdm.tqdm(image_files, desc=f"Class {class_name}"):
                    # Predict
                    results = self.model(str(img_path), verbose=False)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        if result.probs is not None:
                            probs = result.probs
                            
                            # Top-1 prediction
                            predicted_class_idx = probs.top1
                            all_predictions.append(predicted_class_idx)
                            all_true_labels.append(class_idx)
                            
                            # All probabilities
                            all_probs = probs.data.cpu().numpy()
                            all_probabilities.append(all_probs)
            
            if not all_predictions:
                logger.warning("No predictions generated")
                return {}
            
            # Convert to numpy arrays
            all_predictions = np.array(all_predictions)
            all_true_labels = np.array(all_true_labels)
            all_probabilities = np.array(all_probabilities)
            
            # Confusion matrix
            cm = confusion_matrix(all_true_labels, all_predictions)
            
            # Classification report
            report = classification_report(
                all_true_labels, all_predictions,
                target_names=self.class_names,
                output_dict=True,
                zero_division=0
            )
            
            # Per-class metrics
            per_class_accuracy = np.diag(cm) / np.sum(cm, axis=1)
            per_class_accuracy = np.nan_to_num(per_class_accuracy)
            
            # Multi-class ROC AUC nếu có đủ classes
            roc_auc_scores = {}
            if len(self.class_names) > 2:
                try:
                    from sklearn.preprocessing import label_binarize
                    y_true_bin = label_binarize(all_true_labels, classes=range(len(self.class_names)))
                    
                    for i, class_name in enumerate(self.class_names):
                        if np.sum(y_true_bin[:, i]) > 0:  # Class có samples
                            fpr, tpr, _ = roc_curve(y_true_bin[:, i], all_probabilities[:, i])
                            roc_auc_scores[class_name] = auc(fpr, tpr)
                except Exception as e:
                    logger.warning(f"Could not compute ROC AUC: {e}")
            
            evaluation_results = {
                'confusion_matrix': cm.tolist(),
                'classification_report': report,
                'per_class_accuracy': per_class_accuracy.tolist(),
                'overall_accuracy': report['accuracy'],
                'macro_precision': report['macro avg']['precision'],
                'macro_recall': report['macro avg']['recall'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_precision': report['weighted avg']['precision'],
                'weighted_recall': report['weighted avg']['recall'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'roc_auc_scores': roc_auc_scores,
                'total_samples': len(all_predictions),
            }
            
            logger.info("Detailed Classification Results:")
            logger.info(f"  Overall Accuracy: {evaluation_results['overall_accuracy']:.4f}")
            logger.info(f"  Macro F1-Score: {evaluation_results['macro_f1']:.4f}")
            logger.info(f"  Weighted F1-Score: {evaluation_results['weighted_f1']:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in detailed classification evaluation: {e}")
            return {}


class PipelineEvaluator:
    """Evaluator cho toàn bộ detection pipeline"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize pipeline
        pipeline_config = PipelineConfig(
            detection_model_path=config.detection_model_path,
            classification_model_path=config.classification_model_path,
            device=config.device
        )
        self.pipeline = TrashDetectionPipeline(pipeline_config)
        
        logger.info("Pipeline evaluator initialized")
    
    def evaluate_on_test_images(self, test_images_dir: str) -> Dict[str, Any]:
        """Evaluate pipeline trên test images"""
        try:
            logger.info(f"Evaluating pipeline on test images: {test_images_dir}")
            
            test_dir = Path(test_images_dir)
            if not test_dir.exists():
                raise FileNotFoundError(f"Test directory not found: {test_dir}")
            
            # Find all images
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(test_dir.rglob(f"*{ext}")))
                image_files.extend(list(test_dir.rglob(f"*{ext.upper()}")))
            
            if not image_files:
                logger.warning(f"No images found in {test_dir}")
                return {}
            
            logger.info(f"Found {len(image_files)} test images")
            
            # Process images
            all_results = []
            processing_times = []
            
            for img_path in tqdm.tqdm(image_files, desc="Processing images"):
                try:
                    # Load image
                    frame = cv2.imread(str(img_path))
                    if frame is None:
                        continue
                    
                    start_time = time.time()
                    
                    # Process with pipeline
                    annotated_frame, detections, performance_info = self.pipeline.process_frame(frame)
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # Store results
                    result = {
                        'image_path': str(img_path),
                        'detections': len(detections),
                        'classified_objects': sum(1 for d in detections if d.classification_result is not None),
                        'processing_time': processing_time,
                        'fps': 1.0 / processing_time if processing_time > 0 else 0,
                        'detection_details': [
                            {
                                'bbox': det.bbox,
                                'detection_confidence': det.confidence,
                                'detection_class': det.class_name,
                                'classification_class': det.classification_result.class_name if det.classification_result else None,
                                'classification_confidence': det.classification_result.confidence if det.classification_result else None,
                            }
                            for det in detections
                        ]
                    }
                    all_results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Error processing {img_path}: {e}")
                    continue
            
            # Calculate statistics
            total_detections = sum(r['detections'] for r in all_results)
            total_classified = sum(r['classified_objects'] for r in all_results)
            
            pipeline_stats = {
                'total_images': len(all_results),
                'total_detections': total_detections,
                'total_classified': total_classified,
                'classification_rate': total_classified / total_detections if total_detections > 0 else 0,
                'avg_detections_per_image': total_detections / len(all_results) if all_results else 0,
                'avg_processing_time': np.mean(processing_times) if processing_times else 0,
                'avg_fps': np.mean([r['fps'] for r in all_results]) if all_results else 0,
                'min_fps': min([r['fps'] for r in all_results]) if all_results else 0,
                'max_fps': max([r['fps'] for r in all_results]) if all_results else 0,
                'detailed_results': all_results
            }
            
            logger.info("Pipeline Evaluation Results:")
            logger.info(f"  Total images: {pipeline_stats['total_images']}")
            logger.info(f"  Total detections: {pipeline_stats['total_detections']}")
            logger.info(f"  Classification rate: {pipeline_stats['classification_rate']:.2%}")
            logger.info(f"  Average FPS: {pipeline_stats['avg_fps']:.2f}")
            
            return pipeline_stats
            
        except Exception as e:
            logger.error(f"Error in pipeline evaluation: {e}")
            return {}


class ComprehensiveEvaluator:
    """Main evaluator class kết hợp tất cả evaluation components"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        
        # Initialize sub-evaluators
        self.detection_evaluator = DetectionEvaluator(
            config.detection_model_path,
            config.detection_data_yaml,
            config.device
        )
        
        self.classification_evaluator = ClassificationEvaluator(
            config.classification_model_path,
            config.classification_data_yaml,
            config.device
        )
        
        self.pipeline_evaluator = PipelineEvaluator(config)
        
        logger.info("Comprehensive evaluator initialized")
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Chạy toàn bộ evaluation pipeline"""
        try:
            logger.info("=== STARTING COMPREHENSIVE EVALUATION ===")
            
            evaluation_results = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'detection_model': self.config.detection_model_path,
                    'classification_model': self.config.classification_model_path,
                    'device': self.config.device,
                    'experiment_name': self.config.experiment_name,
                }
            }
            
            # 1. Detection Model Evaluation
            logger.info("1. Evaluating Detection Model...")
            detection_results = self.detection_evaluator.evaluate_multiple_thresholds(
                self.config.detection_conf_thresholds
            )
            evaluation_results['detection_evaluation'] = detection_results
            
            # 2. Classification Model Evaluation
            logger.info("2. Evaluating Classification Model...")
            classification_basic = self.classification_evaluator.evaluate_model()
            classification_detailed = self.classification_evaluator.detailed_evaluation()
            
            evaluation_results['classification_evaluation'] = {
                'basic_metrics': classification_basic,
                'detailed_metrics': classification_detailed
            }
            
            # 3. Pipeline Evaluation (nếu có test images)
            # Sử dụng test images từ classification dataset
            classification_test_dir = Path(self.classification_evaluator.dataset_info['path']) / 'test'
            if classification_test_dir.exists():
                logger.info("3. Evaluating Full Pipeline...")
                pipeline_results = self.pipeline_evaluator.evaluate_on_test_images(str(classification_test_dir))
                evaluation_results['pipeline_evaluation'] = pipeline_results
            
            # 4. Generate visualizations
            if self.config.save_plots:
                logger.info("4. Generating Evaluation Plots...")
                self.generate_evaluation_plots(evaluation_results)
            
            # 5. Save results
            self.save_evaluation_results(evaluation_results)
            
            # 6. Print summary
            self.print_evaluation_summary(evaluation_results)
            
            logger.info("=== COMPREHENSIVE EVALUATION COMPLETED ===")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive evaluation: {e}")
            raise
    
    def generate_evaluation_plots(self, results: Dict[str, Any]) -> None:
        """Generate evaluation visualization plots"""
        try:
            logger.info("Generating evaluation plots...")
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Detection threshold comparison plot
            if 'detection_evaluation' in results and 'threshold_comparison' in results['detection_evaluation']:
                self._plot_detection_thresholds(results['detection_evaluation']['threshold_comparison'])
            
            # 2. Classification confusion matrix
            if 'classification_evaluation' in results:
                detailed = results['classification_evaluation'].get('detailed_metrics', {})
                if 'confusion_matrix' in detailed:
                    self._plot_classification_confusion_matrix(detailed)
            
            # 3. Pipeline performance plots
            if 'pipeline_evaluation' in results:
                self._plot_pipeline_performance(results['pipeline_evaluation'])
            
            # 4. Combined metrics comparison
            self._plot_combined_metrics(results)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
    
    def _plot_detection_thresholds(self, threshold_data: Dict[str, Dict[str, float]]) -> None:
        """Plot detection performance vs confidence thresholds"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Detection Model - Confidence Threshold Analysis', fontsize=16)
            
            thresholds = [float(k) for k in list(threshold_data['mAP50'].keys())]
            
            metrics = ['mAP50', 'mAP50_95', 'precision', 'recall']
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                values = [threshold_data[metric][str(t)] for t in thresholds]
                
                ax.plot(thresholds, values, marker='o', linewidth=2, markersize=6)
                ax.set_xlabel('Confidence Threshold')
                ax.set_ylabel(metric)
                ax.set_title(f'{metric} vs Confidence Threshold')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, 1)
            
            plt.tight_layout()
            
            plot_path = self.config.results_dir / f"{self.config.experiment_name}_detection_thresholds.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detection threshold plot saved: {plot_path}")
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting detection thresholds: {e}")
    
    def _plot_classification_confusion_matrix(self, detailed_metrics: Dict[str, Any]) -> None:
        """Plot classification confusion matrix"""
        try:
            cm = np.array(detailed_metrics['confusion_matrix'])
            class_names = self.classification_evaluator.class_names
            
            # Normalized confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            
            # Raw counts
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ax=ax1)
            ax1.set_title('Confusion Matrix (Counts)')
            ax1.set_ylabel('True Label')
            ax1.set_xlabel('Predicted Label')
            
            # Normalized
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=class_names,
                       yticklabels=class_names,
                       ax=ax2)
            ax2.set_title('Confusion Matrix (Normalized)')
            ax2.set_ylabel('True Label')
            ax2.set_xlabel('Predicted Label')
            
            plt.tight_layout()
            
            plot_path = self.config.results_dir / f"{self.config.experiment_name}_classification_cm.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification confusion matrix saved: {plot_path}")
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {e}")
    
    def _plot_pipeline_performance(self, pipeline_data: Dict[str, Any]) -> None:
        """Plot pipeline performance metrics"""
        try:
            if 'detailed_results' not in pipeline_data:
                return
            
            results = pipeline_data['detailed_results']
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Pipeline Performance Analysis', fontsize=16)
            
            # FPS distribution
            fps_values = [r['fps'] for r in results]
            axes[0, 0].hist(fps_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_xlabel('FPS')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('FPS Distribution')
            axes[0, 0].axvline(np.mean(fps_values), color='red', linestyle='--', label=f'Mean: {np.mean(fps_values):.2f}')
            axes[0, 0].legend()
            
            # Detections per image
            det_counts = [r['detections'] for r in results]
            axes[0, 1].hist(det_counts, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_xlabel('Detections per Image')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Detections Distribution')
            axes[0, 1].axvline(np.mean(det_counts), color='red', linestyle='--', label=f'Mean: {np.mean(det_counts):.1f}')
            axes[0, 1].legend()
            
            # Processing time vs detections
            proc_times = [r['processing_time'] for r in results]
            axes[1, 0].scatter(det_counts, proc_times, alpha=0.6, color='orange')
            axes[1, 0].set_xlabel('Number of Detections')
            axes[1, 0].set_ylabel('Processing Time (s)')
            axes[1, 0].set_title('Processing Time vs Detections')
            
            # Classification rate
            class_rates = [r['classified_objects'] / r['detections'] if r['detections'] > 0 else 0 for r in results]
            axes[1, 1].hist(class_rates, bins=20, alpha=0.7, color='salmon', edgecolor='black')
            axes[1, 1].set_xlabel('Classification Rate')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Classification Rate Distribution')
            axes[1, 1].axvline(np.mean(class_rates), color='red', linestyle='--', label=f'Mean: {np.mean(class_rates):.2f}')
            axes[1, 1].legend()
            
            plt.tight_layout()
            
            plot_path = self.config.results_dir / f"{self.config.experiment_name}_pipeline_performance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pipeline performance plot saved: {plot_path}")
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting pipeline performance: {e}")
    
    def _plot_combined_metrics(self, results: Dict[str, Any]) -> None:
        """Plot combined metrics comparison"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            metrics_data = []
            
            # Detection metrics
            if 'detection_evaluation' in results:
                det_data = results['detection_evaluation']
                if 'best_threshold' in det_data:
                    best_results = det_data['threshold_results'][det_data['best_threshold']]
                    metrics_data.extend([
                        ('Detection mAP@50', best_results.get('mAP50', 0)),
                        ('Detection mAP@50-95', best_results.get('mAP50_95', 0)),
                        ('Detection Precision', best_results.get('precision', 0)),
                        ('Detection Recall', best_results.get('recall', 0)),
                        ('Detection F1', best_results.get('f1_score', 0)),
                    ])
            
            # Classification metrics
            if 'classification_evaluation' in results:
                cls_data = results['classification_evaluation']
                if 'detailed_metrics' in cls_data:
                    detailed = cls_data['detailed_metrics']
                    metrics_data.extend([
                        ('Classification Accuracy', detailed.get('overall_accuracy', 0)),
                        ('Classification Precision', detailed.get('macro_precision', 0)),
                        ('Classification Recall', detailed.get('macro_recall', 0)),
                        ('Classification F1', detailed.get('macro_f1', 0)),
                    ])
            
            if metrics_data:
                labels, values = zip(*metrics_data)
                
                bars = ax.barh(labels, values, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
                ax.set_xlabel('Score')
                ax.set_title('Model Performance Comparison')
                ax.set_xlim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{value:.3f}', va='center', ha='left')
            
            plt.tight_layout()
            
            plot_path = self.config.results_dir / f"{self.config.experiment_name}_combined_metrics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Combined metrics plot saved: {plot_path}")
            
            if self.config.show_plots:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            logger.error(f"Error plotting combined metrics: {e}")
    
    def save_evaluation_results(self, results: Dict[str, Any]) -> None:
        """Save evaluation results to JSON"""
        try:
            # Main results file
            results_path = self.config.results_dir / f"{self.config.experiment_name}_results.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Evaluation results saved: {results_path}")
            
            # Summary file
            summary = self._create_evaluation_summary(results)
            summary_path = self.config.results_dir / f"{self.config.experiment_name}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Evaluation summary saved: {summary_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _create_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create evaluation summary"""
        summary = {
            'experiment_name': self.config.experiment_name,
            'timestamp': results.get('timestamp'),
            'models': results.get('config', {}),
        }
        
        # Detection summary
        if 'detection_evaluation' in results:
            det_data = results['detection_evaluation']
            if 'best_threshold' in det_data:
                best_results = det_data['threshold_results'][det_data['best_threshold']]
                summary['detection_summary'] = {
                    'best_threshold': det_data['best_threshold'],
                    'mAP50': best_results.get('mAP50', 0),
                    'mAP50_95': best_results.get('mAP50_95', 0),
                    'f1_score': best_results.get('f1_score', 0),
                }
        
        # Classification summary
        if 'classification_evaluation' in results:
            cls_data = results['classification_evaluation']
            if 'detailed_metrics' in cls_data:
                detailed = cls_data['detailed_metrics']
                summary['classification_summary'] = {
                    'overall_accuracy': detailed.get('overall_accuracy', 0),
                    'macro_f1': detailed.get('macro_f1', 0),
                    'weighted_f1': detailed.get('weighted_f1', 0),
                }
        
        # Pipeline summary
        if 'pipeline_evaluation' in results:
            pipe_data = results['pipeline_evaluation']
            summary['pipeline_summary'] = {
                'avg_fps': pipe_data.get('avg_fps', 0),
                'classification_rate': pipe_data.get('classification_rate', 0),
                'total_images_processed': pipe_data.get('total_images', 0),
            }
        
        return summary
    
    def print_evaluation_summary(self, results: Dict[str, Any]) -> None:
        """Print evaluation summary to console"""
        try:
            print("\n" + "="*50)
            print("COMPREHENSIVE EVALUATION SUMMARY")
            print("="*50)
            
            # Detection results
            if 'detection_evaluation' in results:
                det_data = results['detection_evaluation']
                if 'best_threshold' in det_data:
                    best_results = det_data['threshold_results'][det_data['best_threshold']]
                    print(f"\n📊 DETECTION MODEL PERFORMANCE:")
                    print(f"   Best Threshold: {det_data['best_threshold']}")
                    print(f"   mAP@50: {best_results.get('mAP50', 0):.4f}")
                    print(f"   mAP@50-95: {best_results.get('mAP50_95', 0):.4f}")
                    print(f"   Precision: {best_results.get('precision', 0):.4f}")
                    print(f"   Recall: {best_results.get('recall', 0):.4f}")
                    print(f"   F1-Score: {best_results.get('f1_score', 0):.4f}")
            
            # Classification results
            if 'classification_evaluation' in results:
                cls_data = results['classification_evaluation']
                if 'detailed_metrics' in cls_data:
                    detailed = cls_data['detailed_metrics']
                    print(f"\n🎯 CLASSIFICATION MODEL PERFORMANCE:")
                    print(f"   Overall Accuracy: {detailed.get('overall_accuracy', 0):.4f}")
                    print(f"   Macro Precision: {detailed.get('macro_precision', 0):.4f}")
                    print(f"   Macro Recall: {detailed.get('macro_recall', 0):.4f}")
                    print(f"   Macro F1-Score: {detailed.get('macro_f1', 0):.4f}")
                    print(f"   Weighted F1-Score: {detailed.get('weighted_f1', 0):.4f}")
                    print(f"   Total Test Samples: {detailed.get('total_samples', 0)}")
            
            # Pipeline results
            if 'pipeline_evaluation' in results:
                pipe_data = results['pipeline_evaluation']
                print(f"\n⚡ PIPELINE PERFORMANCE:")
                print(f"   Average FPS: {pipe_data.get('avg_fps', 0):.2f}")
                print(f"   Classification Rate: {pipe_data.get('classification_rate', 0):.2%}")
                print(f"   Total Images Processed: {pipe_data.get('total_images', 0)}")
                print(f"   Total Objects Detected: {pipe_data.get('total_detections', 0)}")
                print(f"   Average Objects per Image: {pipe_data.get('avg_detections_per_image', 0):.1f}")
            
            print(f"\n📁 Results saved to: {self.config.results_dir}")
            print("="*50)
            
        except Exception as e:
            logger.error(f"Error printing summary: {e}")


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Comprehensive Trash Detection Evaluation")
    parser.add_argument("--detection-model", type=str, default="models/detection/best.pt",
                       help="Path to detection model")
    parser.add_argument("--classification-model", type=str, default="models/classification/best.pt",
                       help="Path to classification model")
    parser.add_argument("--detection-data", type=str, default="data/detection/processed/dataset_detection.yaml",
                       help="Detection dataset YAML")
    parser.add_argument("--classification-data", type=str, default="data/classification/processed/dataset_classification.yaml",
                       help="Classification dataset YAML")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--results-dir", type=str, default="results/evaluation",
                       help="Results directory")
    parser.add_argument("--experiment-name", type=str, default="evaluation_v1",
                       help="Experiment name")
    parser.add_argument("--no-plots", action="store_true",
                       help="Disable plot generation")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't show plots interactively")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = EvaluationConfig(
            detection_model_path=args.detection_model,
            classification_model_path=args.classification_model,
            detection_data_yaml=args.detection_data,
            classification_data_yaml=args.classification_data,
            device=args.device,
            results_dir=Path(args.results_dir),
            experiment_name=args.experiment_name,
            save_plots=not args.no_plots,
            show_plots=not args.no_show
        )
        
        # Khởi tạo evaluator
        evaluator = ComprehensiveEvaluator(config)
        
        # Chạy evaluation
        results = evaluator.run_full_evaluation()
        
        logger.info("Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
