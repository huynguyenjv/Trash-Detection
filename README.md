# Dự án Trash Detection với YOLOv8

Dự án phát triển mô hình nhận diện rác thải thời gian thực sử dụng YOLOv8, được training trên dataset "Garbage Classification V2" từ Kaggle.

## 🎯 Mục tiêu dự án

- Phát triển mô hình AI có độ chính xác cao để nhận diện các loại rác thải
- Tối ưu hóa cho ứng dụng thời gian thực (real-time detection)
- Có thể triển khai trên edge devices và camera trực tiếp
- Đạt được sự cân bằng tối ưu giữa tốc độ và độ chính xác

## 🏗️ Kiến trúc dự án

```
Trash-Detection/
├── src/                          # Source code chính
│   ├── data_preprocessing.py     # Tiền xử lý dữ liệu
│   ├── train.py                 # Training model
│   ├── detect.py                # Real-time detection
│   └── evaluate.py              # Đánh giá model
├── data/                        # Dữ liệu
│   ├── raw/                     # Dataset gốc
│   └── processed/               # Dataset đã xử lý
│       ├── images/              # Ảnh train/val/test
│       ├── labels/              # Annotations YOLO format
│       └── dataset.yaml         # Cấu hình dataset
├── models/                      # Model weights
├── notebooks/                   # Jupyter notebooks
├── evaluation_results/          # Kết quả đánh giá
├── requirements.txt            # Dependencies
└── README.md                   # Tài liệu này
```

## 🚀 Cài đặt

### 1. Clone repository

```bash
git clone <repository-url>
cd Trash-Detection
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Kaggle API

Tạo file `~/.kaggle/kaggle.json` với nội dung:
```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_key"
}
```

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## 📊 Dataset

Dự án sử dụng dataset **"Garbage Classification V2"** từ Kaggle:
- **URL**: https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2
- **Loại**: Classification dataset (chuyển đổi thành object detection)
- **Classes**: Các loại rác thải khác nhau
- **Preprocessing**: Tự động tạo bounding boxes và convert sang YOLO format

## 🔄 Quy trình thực hiện

### Bước 1: Tiền xử lý dữ liệu

```bash
cd src
python data_preprocessing.py
```

Quy trình này sẽ:
- Tự động download dataset từ Kaggle
- Chuyển đổi từ classification sang object detection format
- Tạo bounding boxes (giả định object chiếm 80% diện tích ảnh)
- Convert annotations sang YOLO format
- Chia dataset thành train/val/test (80/10/10)
- Tạo file `dataset.yaml`

### Bước 2: Training model

```bash
cd src
python train.py
```

Cấu hình training:
- **Model**: YOLOv8n (fast) hoặc YOLOv8m (balanced)
- **Epochs**: 50
- **Batch size**: 16 (tự động điều chỉnh theo VRAM)
- **Image size**: 640x640
- **Data augmentation**: Mosaic, Mixup, flips, color adjustments

### Bước 3: Đánh giá model

```bash
cd src
python evaluate.py --model ../models/trash_detection_best.pt
```

Sẽ tạo ra:
- Confusion matrix
- Classification report
- Per-class performance plots
- Visualization của predictions
- Các metrics: mAP50, mAP50-95, Precision, Recall

### Bước 4: Real-time Detection

#### Detection trên ảnh đơn lẻ:
```bash
cd src
python detect.py --mode image --source path/to/image.jpg --output result.jpg
```

#### Real-time detection từ webcam:
```bash
cd src
python detect.py --mode webcam --source 0
```

#### Detection trên video:
```bash
cd src
python detect.py --mode video --source path/to/video.mp4 --output result.mp4
```

## 🎛️ Tham số cấu hình

### Data Preprocessing
- `bbox_coverage`: 0.8 (tỷ lệ bounding box so với ảnh)
- `train_ratio`: 0.8
- `val_ratio`: 0.1
- `test_ratio`: 0.1

### Training
- `epochs`: 50
- `batch_size`: 16
- `image_size`: 640
- `lr0`: 0.01 (learning rate ban đầu)
- `device`: "auto" (tự động chọn GPU/CPU)

### Detection
- `conf_threshold`: 0.25 (confidence threshold)
- `iou_threshold`: 0.45 (IoU threshold cho NMS)
- `max_detections`: 100

## 📈 Kết quả mong đợi

- **mAP50**: > 0.85
- **Real-time FPS**: > 20 FPS trên GPU, > 5 FPS trên CPU
- **Accuracy**: > 90% trên test set
- **Inference time**: < 50ms trên GPU

## 🔧 Tối ưu hóa

### GPU Memory
- **≥ 8GB VRAM**: Sử dụng YOLOv8m, batch_size=32
- **4-8GB VRAM**: Sử dụng YOLOv8n, batch_size=16
- **< 4GB VRAM**: batch_size=8

### Real-time Performance
- Sử dụng threading để tách frame reading và inference
- Buffer frames để tránh lag
- Tối ưu image preprocessing

## 📝 Logging

Tất cả scripts đều có logging chi tiết:
- `data_preprocessing.log`
- `training.log`  
- `evaluation.log`

## 🐛 Troubleshooting

### Lỗi thường gặp:

1. **CUDA out of memory**:
   - Giảm batch_size
   - Sử dụng model nhỏ hơn (YOLOv8n)

2. **Kaggle API error**:
   - Kiểm tra file `~/.kaggle/kaggle.json`
   - Verify API credentials

3. **OpenCV camera error**:
   - Thử các camera ID khác (0, 1, 2...)
   - Kiểm tra camera permissions

## 📚 Tài liệu tham khảo

- [YOLOv8 Official Documentation](https://docs.ultralytics.com/)
- [Research Paper: Real-time Recyclable Waste Detection Using YOLOv8](https://eprints.uad.ac.id/69140/1/13-Real-time%20Recyclable%20Waste%20Detection%20Using%20YOLOv8%20for%20Reverse%20Vending%20Machines.pdf)
- [Kaggle Dataset](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)

## 🤝 Đóng góp

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 👥 Tác giả

- **Huy Nguyen** - *Initial work*

## 🙏 Acknowledgments

- Ultralytics team cho YOLOv8
- Kaggle dataset contributors
- OpenCV community
