# Underwater Acoustic Target Detection (YOLOv8)

Object detection pipeline for identifying underwater targets in acoustic/sonar imagery, built on the UATD dataset and YOLOv8.

## Overview

This project converts Pascal VOC XML annotations from the UATD dataset into YOLO format, trains a YOLOv8n model to detect 10 classes of underwater objects, and evaluates model performance through standard detection metrics and visualizations.

## Dataset

[UATD - Underwater Acoustic Target Detection Dataset](https://www.kaggle.com/datasets/mazenkhaled202201534/underwater-acoustic-target-detection-uatddataset)

## Classes

| ID | Class |
|----|-------|
| 0 | Ball |
| 1 | Circle Cage |
| 2 | Cube |
| 3 | Cylinder |
| 4 | Human Body |
| 5 | Metal Bucket |
| 6 | Plane |
| 7 | ROV |
| 8 | Square Cage |
| 9 | Tyre |

## Pipeline

1. **Data Preparation** – Parses XML annotations, converts bounding boxes to YOLO format (normalized x, y, w, h), and builds train/val splits with corresponding `images/` and `labels/` directories.
2. **Dataset Config** – Generates a `data.yaml` file defining paths, class count, and class names for YOLO training.
3. **Training** – Fine-tunes YOLOv8n (`yolov8n.pt`) on the prepared dataset for 20 epochs at 640×640 resolution.
4. **Inference** – Runs the trained model on sample validation images and visualizes detections in a grid layout.
5. **Evaluation** – Plots precision, recall, mAP@0.5, and mAP@0.5:0.95 curves over training epochs; visualizes per-class AP scores and the normalized confusion matrix.
6. **Benchmarking** – Reports inference latency (ms) and frames per second (FPS).

## Tech Stack

- Python
- Ultralytics YOLOv8
- PIL / Pillow
- pandas
- matplotlib, seaborn
- xml.etree.ElementTree

## Requirements

```bash
pip install ultralytics pillow lxml tqdm pandas matplotlib seaborn
```

## Usage

1. Download the UATD dataset and place it under your input directory.
2. Run the notebook to preprocess annotations and build the YOLO-format dataset.
3. Train the model:
```python
   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=20, imgsz=640, batch=8, device=0)
```
4. Evaluate results using the generated `results.csv` and confusion matrix outputs.

## Results

Training metrics, per-class AP scores, confusion matrix, and inference speed (FPS) are visualized at the end of the notebook for performance analysis.
