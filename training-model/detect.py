#!/usr/bin/env python3
"""
Real-time Trash Detection Pipeline - 2-Stage YOLOv8 System
Kết hợp Detection Model + Classification Model với threading optimization

Pipeline:
1. YOLOv8 Detection Model → phát hiện trash objects (bounding boxes)
2. YOLOv8 Classification Model → phân loại trash types cho từng detected object
3. Threading optimization cho real-time performance

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
import time
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import torch

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trash_detection.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Kết quả detection cho một object"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    classification_result: Optional[Dict[str, Any]] = None


@dataclass 
class ClassificationResult:
    """Kết quả classification"""
    class_id: int
    class_name: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class PipelineConfig:
    """Cấu hình cho detection pipeline"""
    # Model paths
    detection_model_path: str = "models/detection/best.pt"
    classification_model_path: str = "models/classification/best.pt"
    
    # Detection settings
    detection_conf_threshold: float = 0.25
    detection_iou_threshold: float = 0.45
    detection_img_size: int = 640
    
    # Classification settings
    classification_img_size: int = 224
    classification_conf_threshold: float = 0.5
    
    # Threading settings
    max_workers: int = 4
    queue_size: int = 100
    
    # Device settings
    device: str = "auto"
    
    # Output settings
    save_results: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    line_thickness: int = 2
    
    # Performance settings
    skip_classification_below: float = 0.3  # Skip classification for low-confidence detections
    batch_classification: bool = True
    
    def __post_init__(self):
        """Validate configuration"""
        if not Path(self.detection_model_path).exists():
            raise FileNotFoundError(f"Detection model not found: {self.detection_model_path}")
        if not Path(self.classification_model_path).exists():
            raise FileNotFoundError(f"Classification model not found: {self.classification_model_path}")


class TrashDetectionPipeline:
    """Pipeline chính cho trash detection và classification"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detection_model: Optional[YOLO] = None
        self.classification_model: Optional[YOLO] = None
        
        # Threading components
        self.classification_queue = Queue(maxsize=config.queue_size)
        self.result_queue = Queue(maxsize=config.queue_size)
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Performance tracking
        self.performance_stats = {
            'detection_times': [],
            'classification_times': [],
            'total_frames': 0,
            'fps_history': []
        }
        
        # Setup device
        self._setup_device()
        
        # Load models
        self._load_models()
        
        # Get class names
        self._load_class_names()
        
        logger.info("TrashDetectionPipeline initialized successfully")
    
    def _setup_device(self) -> None:
        """Setup device cho inference"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Using Apple MPS")
            else:
                self.device = "cpu"
                logger.info("Using CPU")
        else:
            self.device = self.config.device
        
        logger.info(f"Inference device: {self.device}")
    
    def _load_models(self) -> None:
        """Load detection và classification models"""
        try:
            # Load detection model
            logger.info(f"Loading detection model: {self.config.detection_model_path}")
            self.detection_model = YOLO(self.config.detection_model_path)
            self.detection_model.to(self.device)
            
            # Load classification model  
            logger.info(f"Loading classification model: {self.config.classification_model_path}")
            self.classification_model = YOLO(self.config.classification_model_path)
            self.classification_model.to(self.device)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def _load_class_names(self) -> None:
        """Load class names từ models"""
        try:
            # Detection class names
            self.detection_classes = self.detection_model.names
            logger.info(f"Detection classes ({len(self.detection_classes)}): {list(self.detection_classes.values())}")
            
            # Classification class names
            self.classification_classes = self.classification_model.names
            logger.info(f"Classification classes ({len(self.classification_classes)}): {list(self.classification_classes.values())}")
            
        except Exception as e:
            logger.error(f"Error loading class names: {e}")
            raise
    
    def detect_objects(self, frame: np.ndarray) -> List[DetectionResult]:
        """Phát hiện trash objects trong frame"""
        try:
            start_time = time.time()
            
            # Run detection
            results = self.detection_model(
                frame,
                conf=self.config.detection_conf_threshold,
                iou=self.config.detection_iou_threshold,
                imgsz=self.config.detection_img_size,
                device=self.device,
                verbose=False
            )
            
            detection_time = time.time() - start_time
            self.performance_stats['detection_times'].append(detection_time)
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Bounding box coordinates
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        
                        # Confidence và class
                        confidence = float(boxes.conf[i].cpu().numpy())
                        class_id = int(boxes.cls[i].cpu().numpy())
                        class_name = self.detection_classes.get(class_id, f"class_{class_id}")
                        
                        detection = DetectionResult(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in object detection: {e}")
            return []
    
    def classify_object(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[ClassificationResult]:
        """Phân loại trash type cho detected object"""
        try:
            start_time = time.time()
            
            # Crop object từ frame
            x1, y1, x2, y2 = bbox
            
            # Validate bbox
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            # Crop và resize
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return None
            
            # Run classification
            results = self.classification_model(
                cropped,
                imgsz=self.config.classification_img_size,
                device=self.device,
                verbose=False
            )
            
            classification_time = time.time() - start_time
            self.performance_stats['classification_times'].append(classification_time)
            
            # Parse results
            if results and len(results) > 0:
                result = results[0]
                if result.probs is not None:
                    probs = result.probs
                    
                    # Top prediction
                    top1_idx = probs.top1
                    top1_conf = float(probs.top1conf.cpu().numpy())
                    class_name = self.classification_classes.get(top1_idx, f"class_{top1_idx}")
                    
                    # All probabilities
                    all_probs = probs.data.cpu().numpy()
                    prob_dict = {
                        self.classification_classes.get(i, f"class_{i}"): float(prob)
                        for i, prob in enumerate(all_probs)
                    }
                    
                    return ClassificationResult(
                        class_id=top1_idx,
                        class_name=class_name,
                        confidence=top1_conf,
                        probabilities=prob_dict
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error in classification: {e}")
            return None
    
    def classify_objects_batch(self, frame: np.ndarray, detections: List[DetectionResult]) -> None:
        """Phân loại multiple objects sử dụng threading"""
        if not detections:
            return
        
        # Filter detections theo confidence threshold
        high_conf_detections = [
            det for det in detections 
            if det.confidence >= self.config.skip_classification_below
        ]
        
        if not high_conf_detections:
            return
        
        # Submit classification tasks
        futures = []
        for detection in high_conf_detections:
            future = self.executor.submit(
                self.classify_object, 
                frame, 
                detection.bbox
            )
            futures.append((detection, future))
        
        # Collect results
        for detection, future in futures:
            try:
                classification_result = future.result(timeout=1.0)  # 1 second timeout
                detection.classification_result = classification_result
                
            except Exception as e:
                logger.warning(f"Classification failed for detection: {e}")
                detection.classification_result = None
    
    def draw_annotations(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Vẽ annotations lên frame"""
        try:
            annotated_frame = frame.copy()
            
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Màu sắc theo class
                if detection.classification_result:
                    # Sử dụng classification result
                    label = f"{detection.classification_result.class_name}"
                    conf = detection.classification_result.confidence
                    color = self._get_color_for_class(detection.classification_result.class_name)
                else:
                    # Sử dụng detection result
                    label = detection.class_name
                    conf = detection.confidence
                    color = self._get_color_for_class(detection.class_name)
                
                # Vẽ bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.config.line_thickness)
                
                # Tạo label text
                if self.config.show_confidence:
                    if detection.classification_result:
                        label_text = f"{label} {conf:.2f} (Det: {detection.confidence:.2f})"
                    else:
                        label_text = f"{label} {conf:.2f}"
                else:
                    label_text = label
                
                # Vẽ label background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Vẽ text
                cv2.putText(
                    annotated_frame,
                    label_text,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
                
                # Vẽ thêm thông tin classification nếu có
                if detection.classification_result and len(detection.classification_result.probabilities) > 1:
                    # Hiển thị top-3 probabilities
                    sorted_probs = sorted(
                        detection.classification_result.probabilities.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    
                    for i, (class_name, prob) in enumerate(sorted_probs):
                        if i == 0:  # Skip top-1 (already shown)
                            continue
                        prob_text = f"{class_name}: {prob:.2f}"
                        cv2.putText(
                            annotated_frame,
                            prob_text,
                            (x1, y2 + 15 + i * 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1
                        )
            
            return annotated_frame
            
        except Exception as e:
            logger.error(f"Error drawing annotations: {e}")
            return frame
    
    def _get_color_for_class(self, class_name: str) -> Tuple[int, int, int]:
        """Lấy màu sắc cho class"""
        # Simple hash-based color generation
        hash_val = hash(class_name) % 360
        import colorsys
        rgb = colorsys.hsv_to_rgb(hash_val / 360.0, 0.7, 0.9)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR format
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[DetectionResult], Dict[str, Any]]:
        """Xử lý một frame hoàn chỉnh"""
        try:
            start_time = time.time()
            
            # 1. Object Detection
            detections = self.detect_objects(frame)
            
            # 2. Classification (with threading)
            if detections and self.config.batch_classification:
                self.classify_objects_batch(frame, detections)
            elif detections:
                # Sequential classification
                for detection in detections:
                    if detection.confidence >= self.config.skip_classification_below:
                        classification_result = self.classify_object(frame, detection.bbox)
                        detection.classification_result = classification_result
            
            # 3. Draw annotations
            annotated_frame = self.draw_annotations(frame, detections)
            
            # 4. Calculate performance metrics
            total_time = time.time() - start_time
            fps = 1.0 / total_time if total_time > 0 else 0
            self.performance_stats['fps_history'].append(fps)
            self.performance_stats['total_frames'] += 1
            
            # Keep only recent history
            if len(self.performance_stats['fps_history']) > 100:
                self.performance_stats['fps_history'] = self.performance_stats['fps_history'][-100:]
            
            # Performance info
            performance_info = {
                'fps': fps,
                'avg_fps': np.mean(self.performance_stats['fps_history']),
                'total_time': total_time,
                'detection_count': len(detections),
                'classified_count': sum(1 for d in detections if d.classification_result is not None)
            }
            
            return annotated_frame, detections, performance_info
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, [], {}
    
    def process_video(self, input_source: Union[str, int], output_path: Optional[str] = None) -> None:
        """Xử lý video từ file hoặc webcam"""
        try:
            logger.info(f"Starting video processing: {input_source}")
            
            # Mở video source
            if isinstance(input_source, str) and not input_source.isdigit():
                cap = cv2.VideoCapture(input_source)
                logger.info(f"Processing video file: {input_source}")
            else:
                cap = cv2.VideoCapture(int(input_source) if isinstance(input_source, str) else input_source)
                logger.info(f"Processing webcam: {input_source}")
            
            if not cap.isOpened():
                raise ValueError(f"Cannot open video source: {input_source}")
            
            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
            
            # Setup video writer nếu cần
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                logger.info(f"Saving output to: {output_path}")
            
            # Processing loop
            frame_count = 0
            total_detections = 0
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    
                    # Process frame
                    annotated_frame, detections, performance_info = self.process_frame(frame)
                    total_detections += len(detections)
                    
                    # Save frame nếu cần
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Display frame
                    cv2.imshow('Trash Detection', annotated_frame)
                    
                    # Performance overlay
                    if performance_info:
                        info_text = f"FPS: {performance_info['fps']:.1f} | Detections: {performance_info['detection_count']}"
                        cv2.putText(annotated_frame, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Log progress
                    if frame_count % (fps * 5) == 0:  # Every 5 seconds
                        logger.info(f"Processed {frame_count}/{total_frames} frames, "
                                   f"FPS: {performance_info.get('avg_fps', 0):.1f}, "
                                   f"Total detections: {total_detections}")
                    
                    # Quit nếu nhấn 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User requested quit")
                        break
                        
            except KeyboardInterrupt:
                logger.info("Processing interrupted by user")
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            # Final statistics
            logger.info("=== VIDEO PROCESSING COMPLETED ===")
            logger.info(f"Total frames processed: {frame_count}")
            logger.info(f"Total detections: {total_detections}")
            logger.info(f"Average FPS: {np.mean(self.performance_stats['fps_history']):.2f}")
            
            if self.performance_stats['detection_times']:
                avg_detection_time = np.mean(self.performance_stats['detection_times'])
                logger.info(f"Average detection time: {avg_detection_time:.3f}s")
            
            if self.performance_stats['classification_times']:
                avg_classification_time = np.mean(self.performance_stats['classification_times'])
                logger.info(f"Average classification time: {avg_classification_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error in video processing: {e}")
            raise
        finally:
            # Cleanup threading
            self.executor.shutdown(wait=True)
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Xử lý một ảnh"""
        try:
            logger.info(f"Processing image: {image_path}")
            
            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Process frame
            annotated_frame, detections, performance_info = self.process_frame(frame)
            
            # Save result nếu cần
            if output_path:
                cv2.imwrite(output_path, annotated_frame)
                logger.info(f"Result saved to: {output_path}")
            
            # Show result
            cv2.imshow('Trash Detection Result', annotated_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Return results
            results = {
                'image_path': image_path,
                'output_path': output_path,
                'detections': [
                    {
                        'bbox': detection.bbox,
                        'detection_confidence': detection.confidence,
                        'detection_class': detection.class_name,
                        'classification_result': {
                            'class_name': detection.classification_result.class_name,
                            'confidence': detection.classification_result.confidence,
                            'probabilities': detection.classification_result.probabilities
                        } if detection.classification_result else None
                    }
                    for detection in detections
                ],
                'performance': performance_info,
                'summary': {
                    'total_objects': len(detections),
                    'classified_objects': sum(1 for d in detections if d.classification_result is not None)
                }
            }
            
            logger.info(f"Image processing completed: {len(detections)} objects detected")
            return results
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def save_results_json(self, results: Dict[str, Any], output_path: str) -> None:
        """Lưu kết quả ra JSON file"""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to JSON: {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def main():
    """Hàm main"""
    parser = argparse.ArgumentParser(description="Trash Detection Pipeline")
    parser.add_argument("--detection-model", type=str, default="models/detection/best.pt",
                       help="Path to detection model")
    parser.add_argument("--classification-model", type=str, default="models/classification/best.pt",
                       help="Path to classification model")
    parser.add_argument("--source", type=str, default="0",
                       help="Input source (video file, image file, or webcam index)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output path for results")
    parser.add_argument("--conf-det", type=float, default=0.25,
                       help="Detection confidence threshold")
    parser.add_argument("--conf-cls", type=float, default=0.5,
                       help="Classification confidence threshold")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--no-display", action="store_true",
                       help="Disable display")
    parser.add_argument("--save-json", type=str, default=None,
                       help="Save results to JSON file")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = PipelineConfig(
            detection_model_path=args.detection_model,
            classification_model_path=args.classification_model,
            detection_conf_threshold=args.conf_det,
            classification_conf_threshold=args.conf_cls,
            device=args.device
        )
        
        # Khởi tao pipeline
        pipeline = TrashDetectionPipeline(config)
        
        # Xác định input type
        source = args.source
        if source.isdigit():
            # Webcam
            logger.info("Processing webcam input")
            pipeline.process_video(int(source), args.output)
            
        elif Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # Image file
            logger.info("Processing image file")
            results = pipeline.process_image(source, args.output)
            
            # Save JSON results nếu cần
            if args.save_json:
                pipeline.save_results_json(results, args.save_json)
                
        else:
            # Video file
            logger.info("Processing video file")
            pipeline.process_video(source, args.output)
        
        logger.info("Processing completed successfully!")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
