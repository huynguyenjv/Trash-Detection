# Custom Instructions for GitHub Copilot
## Về Kho Lưu Trữ (Repository Overview)
Kho lưu trữ này nhằm phát triển một hệ thống nhận diện và phân loại rác thải theo thời gian thực. Hệ thống triển khai pipeline hai giai đoạn để đạt được sự cân bằng tối ưu giữa tốc độ và độ chính xác:

YOLOv8 Detection: Phát hiện vị trí tổng thể của vật thể rác (bounding box).

YOLOv8 Classification: Phân loại chi tiết loại rác (nhựa, giấy, kim loại, v.v.) sau khi đã cắt (crop) bounding box từ ảnh gốc.

Chúng ta sẽ fine-tune các mô hình YOLOv8 pre-trained riêng biệt cho từng tác vụ và kết hợp chúng trong quá trình inference real-time.

##  General Coding Instructions (Hướng Dẫn Mã Hóa Chung)
Tiêu chuẩn: Tuân thủ chặt chẽ chuẩn Python PEP 8.

Độ rõ ràng: Code phải rõ ràng, dễ đọc và có comment chi tiết (đặc biệt là docstrings cho các hàm và class).

Type Hinting: Bắt buộc sử dụng type hints cho tất cả các tham số và giá trị trả về của hàm.

Thư viện: Xử lý ảnh/video bằng OpenCV (cv2), huấn luyện và inference bằng PyTorch (ultralytics).

Logging & Error Handling: Triển khai logging hiệu quả và xử lý lỗi (try...except) mạnh mẽ.

Tối ưu Real-time: Sử dụng threading hoặc multiprocessing trong detect.py để tách biệt việc chụp frame và xử lý/inference.

Ngôn ngữ Phản hồi: Tiếng Việt.

## Module Structure (Cấu Trúc Module)
Tách biệt chức năng rõ ràng theo các module sau:
```
data_preprocessing_detection.py

data_preprocessing_classification.py

train_detection.py

train_classification.py

detect.py (Pipeline Detection + Classification)

evaluate.py
```

##  Building the Trash Detection + Classification System: Step-by-Step Guide
### 1. Project Setup and Data Preprocessing
#### 1.1. Detection Dataset (TACO/Other BB Dataset)
Script: data_preprocessing_detection.py

Chức năng:

Tải xuống và giải nén dataset (ví dụ: TACO).

Chuẩn hóa annotation sang định dạng YOLO (files .txt).

Phân chia dataset (Train/Validation/Test: 80% / 10% / 10%).

Tạo file cấu hình dataset_detection.yaml.

#### 1.2. Classification Dataset (TrashNet/Garbage Classification V2)
Script: data_preprocessing_classification.py

Chức năng:

Tải dataset từ Kaggle.

Đảm bảo cấu trúc thư mục đầu ra theo chuẩn: train/class_name/img.jpg, val/..., test/....

Tạo file cấu hình dataset_classification.yaml.

### 2. Model Training
#### 2.1. Detection Training
Script: train_detection.py

Mô hình: Load YOLOv8 object detection pre-trained (yolov8n.pt hoặc yolov8m.pt).

Cấu hình Huấn luyện:

Sử dụng dataset_detection.yaml.
```
epochs = 50, imgsz = 640, batch = 16.
```
Kích hoạt augmentations (mosaic, flip, color jitter).
```
Output: runs/detect/train/weights/best.pt.
```
#### 2.2. Classification Training
Script: train_classification.py

Mô hình: Load YOLOv8 image classification pre-trained (yolov8n-cls.pt).

Cấu hình Huấn luyện: Tương tự Detection.

Output: runs/classify/train/weights/best.pt.

### 3. Real-time Detection + Classification Pipeline
Script: detect.py

Load Model: Tải cả Detection Model và Classification Model đã huấn luyện.

Hàm Chính:
```
detect_image(path: str) -> None: Detect + Classify ảnh tĩnh.
```
detect_realtime(source: Union[str, int] = 0) -> None: Inference webcam/video (tối ưu threading).

Quy trình Pipeline:
```
Detection: Tìm bounding box và confidence.

Crop & Classification: Cắt ảnh theo BB, đưa qua Classification Model để lấy label và confidence phân loại.

Visualization: Hiển thị BB và nhãn kết hợp: <Detection Label> - <Classification Label>.
```
### 4. Evaluation
Script: evaluate.py

Đánh giá Detection: Tính toán và hiển thị các metrics: mAP@0.5, mAP@0.5:0.95, Precision, và Recall.

Đánh giá Classification: Tính toán và hiển thị: Overall Accuracy, và Confusion Matrix.

Ví dụ: Hiển thị prediction mẫu so với ground truth.