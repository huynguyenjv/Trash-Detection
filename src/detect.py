"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detect - Phát hiện rác thải real-time với YOLOv8

Mô tả:
    Module này thực hiện phát hiện rác thải:
    - Single image detection
    - Video stream detection
    - Webcam real-time detection
    - Batch processing với multi-threading
    - Vẽ bounding box và hiển thị kết quả

Author: Huy Nguyen
Email: huynguyen@example.com
Date: August 2025
Version: 1.0.0
License: MIT
"""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from queue import Queue

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import torch

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DetectionConfig:
    """Cấu hình detection"""
    # Model
    model_path: str = "../models/final.pt"
    
    # Detection parameters - Giảm confidence threshold để nhận diện nhiều object hơn
    conf_threshold: float = 0.15  # Confidence threshold (giảm từ 0.25 xuống 0.15)
    iou_threshold: float = 0.4    # IoU threshold for NMS (giảm để tránh suppress quá nhiều)
    max_detections: int = 100     # Maximum detections per image
    
    # Image/Video processing
    input_size: int = 640         # Input size for model
    
    # Real-time settings
    fps_limit: int = 30           # Maximum FPS for real-time detection
    buffer_size: int = 3          # Frame buffer size
    
    # Visualization
    line_thickness: int = 2       # Bounding box line thickness (giảm để nhìn rõ hơn)
    font_size: float = 0.6        # Text font size (giảm để không che object)
    
    # Multi-scale detection
    augment: bool = True          # Sử dụng test time augmentation
    agnostic_nms: bool = False    # Class-agnostic NMS
    
    # Colors for different classes (BGR format)
    colors: Dict[str, Tuple[int, int, int]] = None
    
    # Device
    device: str = "auto"


class TrashDetector:
    """Class chính để thực hiện detection"""
    
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model: Optional[YOLO] = None
        self.class_names: List[str] = []
        self.colors: Dict[str, Tuple[int, int, int]] = {}
        
        # Setup device
        self._setup_device()
        
        # Load model
        self.load_model()
        
        # Setup colors
        self._setup_colors()
    
    def _setup_device(self) -> None:
        """Setup device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info(f"Sử dụng GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = "cpu"
                logger.info("Sử dụng CPU")
        else:
            self.device = self.config.device
    
    def load_model(self) -> None:
        """Load trained model"""
        try:
            if not Path(self.config.model_path).exists():
                raise FileNotFoundError(f"Không tìm thấy model: {self.config.model_path}")
            
            logger.info(f"Đang load model: {self.config.model_path}")
            self.model = YOLO(self.config.model_path)
            
            # Get class names
            self.class_names = list(self.model.names.values())
            logger.info(f"Model đã load thành công với {len(self.class_names)} classes")
            logger.info(f"Classes: {', '.join(self.class_names)}")
            
        except Exception as e:
            logger.error(f"Lỗi khi load model: {e}")
            raise
    
    def _setup_colors(self) -> None:
        """Setup colors cho từng class"""
        if self.config.colors is not None:
            self.colors = self.config.colors
        else:
            # Tạo màu tự động
            np.random.seed(42)  # Đảm bảo màu consistent
            for class_name in self.class_names:
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                self.colors[class_name] = color
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh trước khi đưa vào model
        
        Args:
            image: Ảnh đầu vào (BGR format)
            
        Returns:
            Ảnh đã được tiền xử lý
        """
        # Convert BGR to RGB (YOLO expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        return image_rgb
    
    def postprocess_detections(self, results, original_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Xử lý kết quả detection
        
        Args:
            results: Kết quả từ YOLO model
            original_shape: Shape của ảnh gốc (height, width)
            
        Returns:
            List các detection đã được xử lý
        """
        detections = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            scores = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            
            for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
                if score >= self.config.conf_threshold:
                    x1, y1, x2, y2 = box
                    
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(score),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    }
                    
                    detections.append(detection)
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Vẽ bounding boxes và labels lên ảnh
        
        Args:
            image: Ảnh gốc
            detections: List các detection
            
        Returns:
            Ảnh đã được vẽ detection
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Vẽ bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.config.line_thickness)
            
            # Tạo label text
            label = f"{class_name}: {confidence:.2f}"
            
            # Tính toán kích thước text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_size, 2
            )
            
            # Vẽ background cho text
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            
            # Vẽ text
            cv2.putText(
                result_image,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_size,
                (255, 255, 255),
                2
            )
        
        return result_image
    
    def draw_enhanced_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Vẽ bounding boxes và labels với style cải tiến
        
        Args:
            image: Ảnh gốc
            detections: List các detection
            
        Returns:
            Ảnh đã được vẽ detection với style đẹp hơn
        """
        result_image = image.copy()
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            x1, y1, x2, y2 = bbox
            color = self.colors.get(class_name, (0, 255, 0))
            
            # Vẽ bounding box với shadow effect
            shadow_color = tuple(max(0, c - 50) for c in color)
            cv2.rectangle(result_image, (x1+2, y1+2), (x2+2, y2+2), shadow_color, self.config.line_thickness)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, self.config.line_thickness)
            
            # Vẽ corner markers để làm nổi bật
            corner_length = 15
            cv2.line(result_image, (x1, y1), (x1 + corner_length, y1), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x1, y1), (x1, y1 + corner_length), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x2, y1), (x2 - corner_length, y1), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x2, y1), (x2, y1 + corner_length), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x1, y2), (x1 + corner_length, y2), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x1, y2), (x1, y2 - corner_length), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x2, y2), (x2 - corner_length, y2), color, self.config.line_thickness + 1)
            cv2.line(result_image, (x2, y2), (x2, y2 - corner_length), color, self.config.line_thickness + 1)
            
            # Tạo label text với confidence
            label = f"{class_name} {confidence:.2f}"
            
            # Tính toán kích thước text
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.config.font_size, 2
            )
            
            # Vẽ background cho text với gradient effect
            label_y = y1 - text_height - baseline - 5
            if label_y < 0:
                label_y = y2 + text_height + 5
            
            # Background với alpha blending
            overlay = result_image.copy()
            cv2.rectangle(
                overlay,
                (x1, label_y),
                (x1 + text_width + 10, label_y + text_height + baseline + 5),
                color,
                -1
            )
            cv2.addWeighted(overlay, 0.8, result_image, 0.2, 0, result_image)
            
            # Vẽ text
            cv2.putText(
                result_image,
                label,
                (x1 + 5, label_y + text_height + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_size,
                (255, 255, 255),
                2
            )
            
            # Vẽ ID số cho từng object
            id_text = f"#{i+1}"
            cv2.putText(
                result_image,
                id_text,
                (x2 - 30, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
        
        return result_image
    
    def draw_info_panel(self, image: np.ndarray, fps: float, detection_time: float, detections: List[Dict[str, Any]]) -> None:
        """
        Vẽ panel thông tin chi tiết lên ảnh
        
        Args:
            image: Ảnh để vẽ lên
            fps: FPS hiện tại
            detection_time: Thời gian detection
            detections: List các detection
        """
        height, width = image.shape[:2]
        
        # Background cho info panel
        panel_height = 120
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Thông tin cơ bản
        info_lines = [
            f"FPS: {fps:.1f} | Detection Time: {detection_time*1000:.1f}ms",
            f"Objects Detected: {len(detections)}",
            f"Resolution: {width}x{height}",
        ]
        
        # Đếm số lượng từng loại object
        if detections:
            class_counts = {}
            for detection in detections:
                class_name = detection['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            class_info = " | ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
            info_lines.append(f"Classes: {class_info}")
        
        # Vẽ text
        y_offset = 25
        for line in info_lines:
            cv2.putText(
                image,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
            y_offset += 25
        
        # Vẽ thanh confidence nếu có detection
        if detections:
            max_conf = max(det['confidence'] for det in detections)
            avg_conf = sum(det['confidence'] for det in detections) / len(detections)
            
            conf_text = f"Max Confidence: {max_conf:.3f} | Avg: {avg_conf:.3f}"
            cv2.putText(
                image,
                conf_text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                1
            )
    
    def detect_image(self, image_path: str, output_path: Optional[str] = None, show: bool = True) -> List[Dict[str, Any]]:
        """
        Thực hiện detection trên ảnh đơn lẻ
        
        Args:
            image_path: Đường dẫn đến ảnh đầu vào
            output_path: Đường dẫn lưu ảnh kết quả (optional)
            show: Có hiển thị ảnh kết quả không
            
        Returns:
            List các detection
        """
        try:
            # Đọc ảnh
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Không thể đọc ảnh: {image_path}")
            
            # Tiền xử lý
            processed_image = self.preprocess_image(image)
            
            # Detection
            results = self.model(
                processed_image,
                conf=self.config.conf_threshold,
                iou=self.config.iou_threshold,
                max_det=self.config.max_detections,
                device=self.device,
                verbose=False
            )
            
            # Xử lý kết quả
            detections = self.postprocess_detections(results, image.shape[:2])
            
            # Vẽ detection lên ảnh
            result_image = self.draw_detections(image, detections)
            
            # Lưu ảnh nếu được chỉ định
            if output_path:
                cv2.imwrite(output_path, result_image)
                logger.info(f"Đã lưu kết quả: {output_path}")
            
            # Hiển thị ảnh
            if show:
                cv2.imshow("Trash Detection", result_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            logger.info(f"Phát hiện {len(detections)} objects trong ảnh")
            for detection in detections:
                logger.info(f"  - {detection['class_name']}: {detection['confidence']:.3f}")
            
            return detections
            
        except Exception as e:
            logger.error(f"Lỗi khi detect ảnh: {e}")
            raise
    
    def detect_video_stream(self, source: int = 0, output_path: Optional[str] = None) -> None:
        """
        Thực hiện real-time detection trên video stream với cải thiện nhận diện
        
        Args:
            source: Nguồn video (0 cho webcam, đường dẫn file cho video)
            output_path: Đường dẫn lưu video kết quả (optional)
        """
        try:
            # Mở video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Không thể mở video source: {source}")
            
            # Cấu hình camera để có chất lượng tốt hơn
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer delay
            
            # Lấy thông tin video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Video stream: {width}x{height} @ {fps} FPS")
            
            # Setup video writer nếu cần
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Frame processing variables
            frame_count = 0
            start_time = time.time()
            display_fps = 0
            detection_history = []  # Lưu lịch sử detection để làm mượt
            
            logger.info("Bắt đầu real-time detection. Nhấn 'q' để thoát, 's' để screenshot.")
            
            while True:
                # Đọc frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Tính FPS thực tế
                if frame_count % 30 == 0:
                    current_time = time.time()
                    display_fps = 30 / (current_time - start_time)
                    start_time = current_time
                
                # Tiền xử lý frame - cải thiện chất lượng
                processed_frame = self.preprocess_image(frame)
                
                # Tăng cường độ tương phản và độ sáng nếu cần
                enhanced_frame = cv2.convertScaleAbs(processed_frame, alpha=1.1, beta=10)
                
                # Detection với multiple scales
                detection_start = time.time()
                results = self.model(
                    enhanced_frame,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_detections,
                    device=self.device,
                    verbose=False,
                    augment=self.config.augment,  # Sử dụng test time augmentation
                    agnostic_nms=self.config.agnostic_nms
                )
                detection_time = time.time() - detection_start
                
                # Xử lý kết quả
                detections = self.postprocess_detections(results, frame.shape[:2])
                
                # Lọc detection theo kích thước minimum (loại bỏ detection quá nhỏ)
                filtered_detections = []
                for detection in detections:
                    bbox = detection['bbox']
                    width_box = bbox[2] - bbox[0]
                    height_box = bbox[3] - bbox[1]
                    area = width_box * height_box
                    
                    # Chỉ giữ lại detection có diện tích đủ lớn
                    if area > 400:  # Min area 20x20 pixels
                        filtered_detections.append(detection)
                
                # Lưu lịch sử detection để tracking
                detection_history.append(filtered_detections)
                if len(detection_history) > 5:
                    detection_history.pop(0)
                
                # Vẽ detection với màu sắc phân biệt
                result_frame = self.draw_enhanced_detections(frame, filtered_detections)
                
                # Vẽ thông tin chi tiết
                self.draw_info_panel(result_frame, display_fps, detection_time, filtered_detections)
                
                # Lưu video nếu cần
                if writer:
                    writer.write(result_frame)
                
                # Hiển thị frame
                cv2.imshow("Real-time Trash Detection - Enhanced", result_frame)
                
                # Kiểm tra phím thoát
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):  # Screenshot
                    screenshot_name = f"detection_screenshot_{int(time.time())}.jpg"
                    cv2.imwrite(screenshot_name, result_frame)
                    logger.info(f"Đã lưu screenshot: {screenshot_name}")
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            logger.info("Đã dừng real-time detection")
            
        except Exception as e:
            logger.error(f"Lỗi khi detect video stream: {e}")
            raise


class ThreadedCamera:
    """Camera với threading để tối ưu performance"""
    
    def __init__(self, source: int = 0, buffer_size: int = 3):
        self.source = source
        self.buffer_size = buffer_size
        self.frame_queue = Queue(maxsize=buffer_size)
        self.capture = cv2.VideoCapture(source)
        self.running = False
        self.thread = None
    
    def start(self) -> None:
        """Bắt đầu capture thread"""
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.start()
    
    def _capture_frames(self) -> None:
        """Capture frames trong thread riêng"""
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    # Bỏ frame cũ nếu buffer đầy
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame)
                    except:
                        pass
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Lấy frame mới nhất"""
        try:
            return self.frame_queue.get_nowait()
        except:
            return None
    
    def stop(self) -> None:
        """Dừng capture thread"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.capture.release()


def main():
    """Hàm main với các tùy chọn cải thiện detection"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Trash Detection System")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True,
                       help="Chế độ detection")
    parser.add_argument("--source", type=str, default="0",
                       help="Nguồn đầu vào (đường dẫn ảnh/video hoặc ID webcam)")
    parser.add_argument("--output", type=str, default=None,
                       help="Đường dẫn lưu kết quả")
    parser.add_argument("--model", type=str, default="../models/final.pt",
                       help="Đường dẫn model weights")
    parser.add_argument("--conf", type=float, default=0.15,
                       help="Confidence threshold (giảm xuống 0.15 để nhận diện nhiều hơn)")
    parser.add_argument("--iou", type=float, default=0.4,
                       help="IoU threshold for NMS")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    parser.add_argument("--augment", action="store_true",
                       help="Sử dụng test time augmentation để tăng độ chính xác")
    parser.add_argument("--show-info", action="store_true", default=True,
                       help="Hiển thị panel thông tin chi tiết")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config với tham số tối ưu
        config = DetectionConfig(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            device=args.device,
            augment=args.augment
        )
        
        # Log thông tin cấu hình
        logger.info("=== ENHANCED TRASH DETECTION SYSTEM ===")
        logger.info(f"Model: {config.model_path}")
        logger.info(f"Confidence threshold: {config.conf_threshold}")
        logger.info(f"IoU threshold: {config.iou_threshold}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Augmentation: {config.augment}")
        logger.info("==========================================")
        
        # Khởi tạo detector
        detector = TrashDetector(config)
        
        # Thực hiện detection theo mode
        if args.mode == "image":
            logger.info(f"Đang detect ảnh: {args.source}")
            detections = detector.detect_image(args.source, args.output)
            logger.info(f"Hoàn thành! Phát hiện {len(detections)} objects")
        
        elif args.mode == "video":
            logger.info(f"Đang detect video: {args.source}")
            detector.detect_video_stream(args.source, args.output)
        
        elif args.mode == "webcam":
            source = int(args.source) if args.source.isdigit() else 0
            logger.info(f"Bắt đầu real-time detection với webcam {source}")
            logger.info("Hướng dẫn sử dụng:")
            logger.info("- Nhấn 'q' để thoát")
            logger.info("- Nhấn 's' để chụp screenshot")
            logger.info("- Di chuyển object vào camera để nhận diện")
            detector.detect_video_stream(source, args.output)
        
    except KeyboardInterrupt:
        logger.info("Người dùng dừng chương trình")
    except Exception as e:
        logger.error(f"Lỗi chương trình: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
