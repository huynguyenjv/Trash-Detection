"""
Script thực hiện real-time detection cho dự án Trash Detection

Author: Huy Nguyen
Date: August 2025
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
    model_path: str = "models/trash_detection_best.pt"
    
    # Detection parameters
    conf_threshold: float = 0.25  # Confidence threshold
    iou_threshold: float = 0.45   # IoU threshold for NMS
    max_detections: int = 100     # Maximum detections per image
    
    # Image/Video processing
    input_size: int = 640         # Input size for model
    
    # Real-time settings
    fps_limit: int = 30           # Maximum FPS for real-time detection
    buffer_size: int = 3          # Frame buffer size
    
    # Visualization
    line_thickness: int = 3       # Bounding box line thickness
    font_size: int = 0.8          # Text font size
    
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
        Thực hiện real-time detection trên video stream
        
        Args:
            source: Nguồn video (0 cho webcam, đường dẫn file cho video)
            output_path: Đường dẫn lưu video kết quả (optional)
        """
        try:
            # Mở video capture
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                raise ValueError(f"Không thể mở video source: {source}")
            
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
            fps_counter = 0
            display_fps = 0
            
            logger.info("Bắt đầu real-time detection. Nhấn 'q' để thoát.")
            
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
                
                # Tiền xử lý frame
                processed_frame = self.preprocess_image(frame)
                
                # Detection
                detection_start = time.time()
                results = self.model(
                    processed_frame,
                    conf=self.config.conf_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_detections,
                    device=self.device,
                    verbose=False
                )
                detection_time = time.time() - detection_start
                
                # Xử lý kết quả
                detections = self.postprocess_detections(results, frame.shape[:2])
                
                # Vẽ detection
                result_frame = self.draw_detections(frame, detections)
                
                # Vẽ thông tin FPS và detection time
                info_text = f"FPS: {display_fps:.1f} | Detection: {detection_time*1000:.1f}ms | Objects: {len(detections)}"
                cv2.putText(
                    result_frame,
                    info_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )
                
                # Lưu video nếu cần
                if writer:
                    writer.write(result_frame)
                
                # Hiển thị frame
                cv2.imshow("Real-time Trash Detection", result_frame)
                
                # Kiểm tra phím thoát
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
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
    """Hàm main"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Trash Detection")
    parser.add_argument("--mode", choices=["image", "video", "webcam"], required=True,
                       help="Chế độ detection")
    parser.add_argument("--source", type=str, default="0",
                       help="Nguồn đầu vào (đường dẫn ảnh/video hoặc ID webcam)")
    parser.add_argument("--output", type=str, default=None,
                       help="Đường dẫn lưu kết quả")
    parser.add_argument("--model", type=str, default="models/trash_detection_best.pt",
                       help="Đường dẫn model weights")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    try:
        # Khởi tạo config
        config = DetectionConfig(
            model_path=args.model,
            conf_threshold=args.conf,
            device=args.device
        )
        
        # Khởi tạo detector
        detector = TrashDetector(config)
        
        # Thực hiện detection theo mode
        if args.mode == "image":
            detector.detect_image(args.source, args.output)
        
        elif args.mode == "video":
            detector.detect_video_stream(args.source, args.output)
        
        elif args.mode == "webcam":
            source = int(args.source) if args.source.isdigit() else 0
            detector.detect_video_stream(source, args.output)
        
    except Exception as e:
        logger.error(f"Lỗi chương trình: {e}")
        raise


if __name__ == "__main__":
    main()
