# Hướng dẫn sử dụng Data Preprocessing - Gộp nhiều Dataset

## Tổng quan

Script `data_preprocessing.py` mới được thiết kế để gộp nhiều dataset trash detection từ Kaggle thành một dataset thống nhất, sẵn sàng cho việc huấn luyện YOLOv8.

## Tính năng chính

- ✅ Tự động download nhiều dataset từ Kaggle
- ✅ Gộp các dataset có format YOLO khác nhau
- ✅ Ánh xạ class names thành master classes thống nhất
- ✅ Phân chia train/validation tự động
- ✅ Tạo file `data.yaml` chuẩn cho YOLOv8
- ✅ Báo cáo chi tiết về dataset

## Cài đặt

```bash
pip install pyyaml tqdm kaggle
```

## Cấu hình Kaggle API

1. Tạo tài khoản Kaggle và lấy API key
2. Tạo file `~/.config/kaggle/kaggle.json` với nội dung:
```json
{"username":"your_username","key":"your_api_key"}
```

## Dataset được hỗ trợ

1. **Drinking Waste Classification**
   - `arkadiyhacks/drinking-waste-classification`
   - Classes: Aluminium Can, Glass Bottle, Plastic Bottle...

2. **Household Trash Recycling**
   - `youssefelebiary/household-trash-recycling-dataset`
   - Classes: cardboard, glass, metal, paper, plastic, trash

3. **TACO Dataset YOLO Format**
   - `vencerlanz09/taco-dataset-yolo-format`
   - Classes: Bottle, Can, Plastic bag, Cardboard...

4. **Waste Classification YOLOv8**
   - `spellsharp/garbage-data`
   - Classes: biological, brown-glass, clothes, battery...

## Master Classes

Dataset gộp sẽ có 13 master classes:
- `bottle`, `can`, `cardboard`, `plastic_bag`
- `glass`, `paper`, `metal`, `organic`
- `plastic`, `battery`, `clothes`, `shoes`, `trash`

## Cách sử dụng

### 1. Sử dụng cơ bản

```python
from data_preprocessing import MultiDatasetConfig, MultiDatasetProcessor

# Khởi tạo với config mặc định
config = MultiDatasetConfig()
processor = MultiDatasetProcessor(config)

# Download datasets từ Kaggle
processor.download_datasets()

# Gộp tất cả datasets
processor.process_all_datasets()
```

### 2. Chạy từ command line

```bash
python data_preprocessing.py
```

### 3. Cấu hình tùy chỉnh

```python
config = MultiDatasetConfig()
config.source_datasets_path = Path("my_datasets") 
config.output_dataset_path = Path("my_output")
config.train_ratio = 0.85  # 85% train, 15% val
config.val_ratio = 0.15

processor = MultiDatasetProcessor(config)
```

### 4. Tùy chỉnh class mapping

```python
processor = MultiDatasetProcessor(config)

# Thêm ánh xạ tùy chỉnh
processor.class_mapping.update({
    'food_waste': 'organic',
    'aluminum_foil': 'metal',
    'plastic_container': 'plastic'
})

# Thêm master class mới
processor.master_classes.append('electronic')
processor.master_class_to_id = {
    name: idx for idx, name in enumerate(processor.master_classes)
}
```

## Cấu trúc thư mục

### Input (source_datasets/):
```
source_datasets/
├── drinking-waste-classification/
│   ├── images/
│   ├── labels/
│   └── data.yaml
├── household-trash-recycling/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── ...
```

### Output (merged_dataset/):
```
merged_dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
└── dataset_summary.json
```

## File data.yaml đầu ra

```yaml
path: /absolute/path/to/merged_dataset
train: images/train
val: images/val
nc: 13
names:
  - bottle
  - can
  - cardboard
  - plastic_bag
  - glass
  - paper
  - metal
  - organic
  - plastic
  - battery
  - clothes
  - shoes
  - trash
```

## Class Mapping Examples

| Original Class | Master Class |
|---------------|--------------|
| Aluminium Can | can |
| Glass Bottle | bottle |
| Plastic bag | plastic_bag |
| cardboard | cardboard |
| biological | organic |
| brown-glass | glass |
| clothes | clothes |
| battery | battery |

## Tùy chọn cấu hình

```python
@dataclass
class MultiDatasetConfig:
    source_datasets_path: Path = Path("source_datasets")  # Thư mục chứa datasets gốc
    output_dataset_path: Path = Path("merged_dataset")     # Thư mục output
    temp_path: Path = Path("temp_merged")                  # Thư mục tạm thời
    train_ratio: float = 0.8                              # Tỷ lệ train
    val_ratio: float = 0.2                                # Tỷ lệ validation
    
    kaggle_datasets: List[str] = [...]                    # Danh sách dataset Kaggle
```

## Demo

Chạy script demo để xem các tính năng:

```bash
python demo_data_preprocessing.py
```

Demo bao gồm:
1. Thiết lập thủ công
2. Quy trình đầy đủ
3. Tùy chỉnh ánh xạ class
4. Cấu trúc thư mục mẫu

## Logging

Tất cả hoạt động được ghi log vào:
- Console (real-time)
- File `data_preprocessing.log`

## Báo cáo

Script tạo file `dataset_summary.json` với thông tin:
- Tổng số classes và danh sách
- Class mapping được sử dụng
- Số lượng ảnh trong train/val split
- Thống kê chi tiết

## Troubleshooting

### Lỗi thường gặp:

1. **Không tìm thấy Kaggle API key**
   - Kiểm tra file `~/.config/kaggle/kaggle.json`
   - Đảm bảo permissions: `chmod 600 ~/.config/kaggle/kaggle.json`

2. **Không tìm thấy file yaml trong dataset**
   - Một số dataset có thể không có file yaml
   - Script sẽ cố gắng tìm từ cấu trúc thư mục

3. **Class không được map**
   - Kiểm tra log để xem class nào không được ánh xạ
   - Thêm vào `class_mapping` dictionary

4. **Không có ảnh nào được xử lý**
   - Kiểm tra cấu trúc thư mục dataset
   - Đảm bảo có thư mục `images/` và `labels/`

### Debug mode:

```python
import logging
logging.basicConfig(level=logging.DEBUG)  # Hiển thị chi tiết hơn
```

## Performance

- Xử lý khoảng 1000-2000 ảnh/phút
- Ram sử dụng: ~200-500MB
- Disk space: Khoảng 2x kích thước datasets gốc

## Tích hợp với YOLOv8

Sau khi gộp dataset, sử dụng trực tiếp với YOLOv8:

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train trên dataset đã gộp
model.train(
    data='merged_dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```
