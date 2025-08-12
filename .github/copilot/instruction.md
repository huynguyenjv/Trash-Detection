# Custom instructions for GitHub Copilot

## About this repository
This repository aims to develop a high-accuracy, real-time trash detection model. The model will be trained on the "Garbage Classification V2" dataset from Kaggle and optimized for deployment in real-world scenarios, such as on edge devices or in applications requiring live camera feeds. The primary goal is to accurately classify different types of garbage in real-time.

The core of this project is based on the principles and architecture outlined in the YOLO (You Only Look Once) research papers, specifically targeting a balance between speed and accuracy. We will leverage transfer learning from a pre-trained YOLOv8 model and fine-tune it on our specific garbage dataset.

## General coding instructions
- **Follow Python best practices (PEP 8)**. Write clean, readable, and well-commented code.
- **Use type hints** for all function definitions and complex variable declarations to improve code clarity and allow for static analysis.
- **Prioritize performance and efficiency**, especially for real-time detection components. This includes efficient data loading, optimized image preprocessing, and minimizing computational overhead during inference.
- **Write modular code**. Separate concerns into different files and modules (e.g., `data_preprocessing.py`, `model.py`, `train.py`, `detect.py`).
- **Implement comprehensive error handling** and logging to facilitate debugging and monitoring. Use the `logging` module instead of `print()` for non-trivial outputs.
- **All code must be framework-specific**. Use PyTorch for model building and training, and OpenCV for image/video processing.
- **Using Vietnamese to response answer**
- **This is paper you should to learn this** https://eprints.uad.ac.id/69140/1/13-Real-time%20Recyclable%20Waste%20Detection%20Using%20YOLOv8%20for%20Reverse%20Vending%20Machines.pdf

## Building the trash detection model: Step-by-step guide

### 1. Project Setup and Data Preprocessing
- **Goal**: Prepare the Kaggle dataset ("Garbage Classification V2") for training with YOLOv8.
- **Instructions**:
    - Create a script `data_preprocessing.py`.
    - This script should automatically download and extract the dataset from the provided Kaggle URL: `https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2`.
    - The dataset is structured into image classification folders. You need to convert this into a format suitable for object detection. This involves:
        - **Generating Bounding Box Annotations**: Since the original dataset is for classification, assume the object of interest covers a significant portion of the image. Generate bounding box labels for each image. The bounding box should tightly enclose the primary garbage item. You can start with a default (e.g., 80% of image dimensions centered) and suggest tools or methods for refinement if needed.
        - **Converting to YOLO format**: Create `.txt` annotation files for each image. Each file should contain one line per object in the format: `<class_id> <x_center> <y_center> <width> <height>`. All coordinates must be normalized (from 0 to 1).
    - Create a `dataset.yaml` file that defines the dataset structure, including the path to training and validation images and the list of class names.
    - Split the data into training, validation, and testing sets (e.g., 80-10-10 split). Ensure the split is stratified to maintain class distribution.

### 2. Model Training
- **Goal**: Fine-tune a pre-trained YOLOv8 model on our prepared dataset.
- **Instructions**:
    - Create a script `train.py`.
    - Use the `ultralytics` library to load a pre-trained YOLOv8 model (e.g., `yolov8n.pt` for speed or `yolov8m.pt` for a balance of speed and accuracy).
    - Initialize the training process using the `YOLO` class.
    - The training configuration should be clearly defined:
        - `data`: Path to `dataset.yaml`.
        - `epochs`: Start with 50 epochs.
        - `imgsz`: Use an image size of 640.
        - `batch`: Suggest a batch size (e.g., 16), but note that it may need adjustment based on available VRAM.
    - Implement data augmentation techniques suitable for this task, such as mosaic, mixup, random flips, and color space adjustments, using the built-in capabilities of the `ultralytics` training pipeline.
    - After training, the best-performing model weights (e.g., `best.pt`) should be saved automatically in a `runs/detect/train/weights/` directory.

### 3. Real-time Detection and Inference
- **Goal**: Use the trained model to perform real-time trash detection on video streams or individual images.
- **Instructions**:
    - Create a script `detect.py`.
    - This script should load the fine-tuned model weights (`best.pt`).
    - Implement functions for:
        - **Image Inference**: A function that takes an image path, runs detection, draws bounding boxes with class labels and confidence scores on the image, and displays or saves the result.
        - **Real-time Video Inference**: A function that captures video from a webcam (or a video file). For each frame, it should perform inference, draw the bounding boxes, and display the resulting video stream in real-time.
    - Optimize the real-time detection loop. Use techniques like threading to separate frame reading and inference to prevent lagging.
    - The output should be clear and informative, showing the detected object's class and the confidence level of the prediction.

### 4. Evaluation
- **Goal**: Evaluate the model's performance using standard object detection metrics.
- **Instructions**:
    - Create a script `evaluate.py`.
    - Load the validation or test set.
    - Run the model in validation mode to calculate metrics like Precision, Recall, and mean Average Precision (mAP50, mAP50-95).
    - The script should print a confusion matrix to help identify which classes the model is struggling with.
    - Visualize some prediction examples from the test set with their ground truth and predicted bounding boxes.

By following these structured instructions, Copilot should be able to assist in generating high-quality, organized, and effective code for building the entire real-time trash detection system.