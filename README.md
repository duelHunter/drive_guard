# Vision-Based Vehicle Warning System for Traffic Sign and Pedestrian Detection

A classical computer vision system for real-time detection of traffic signs and pedestrians from dashcam video, with warning alerts for driver safety.

## Project Overview

This project implements a vision-based driver warning system that processes road-scene video frames and generates alerts when selected traffic signs (Stop, Speed Limit, Pedestrian Crossing) or pedestrians are detected. The system emphasizes classical image processing and computer vision techniques (~80% of the pipeline) over deep learning.

## Key Features

- **Traffic Sign Detection**: Color-based segmentation (HSV) + geometric filtering + HOG feature classification
- **Pedestrian Detection**: HOG descriptors with linear SVM classification
- **Preprocessing**: Denoising, color space conversion, CLAHE for illumination compensation
- **Real-time Warnings**: Visual overlays and audio alerts for high-risk events
- **Temporal Stability**: Optional Kalman filtering for tracking
- **Performance Evaluation**: Precision, recall, F1-score, FPS metrics

## Project Structure

```
vision_project/
├── src/                          # Source code modules
│   ├── __init__.py
│   ├── preprocessing.py          # Image preprocessing utilities
│   ├── sign_detection.py         # Traffic sign detection module
│   ├── sign_classification.py    # Traffic sign classification (HOG+SVM)
│   ├── pedestrian_detection.py   # Pedestrian detection (HOG+SVM)
│   ├── zebra_crossing.py         # Optional: Zebra crossing detection
│   ├── tracking.py               # Optional: Kalman filter tracking
│   ├── warning_system.py         # Warning logic and visualization
│   └── main.py                   # Main pipeline
├── data/
│   ├── datasets/                 # GTSRB, GTSDB, pedestrian datasets
│   └── videos/                   # Test dashcam videos
├── models/                        # Trained SVM models and templates
├── output/                        # Output videos and results
├── tests/                         # Unit tests and evaluation scripts
├── requirements.txt
├── README.md
└── config.py                     # Configuration parameters
```

## Installation

1. Clone or extract the project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Pipeline
```python
python src/main.py --video input_video.mp4 --output output_video.mp4
```

### Training Sign Classifier
```python
python src/sign_classification.py --train --dataset path/to/GTSRB
```

### Training Pedestrian Detector
```python
python src/pedestrian_detection.py --train --dataset path/to/pedestrian_data
```

## Methodology

### 1. Preprocessing
- Gaussian/bilateral filtering for denoising
- BGR to HSV conversion for robust color segmentation
- CLAHE for illumination compensation
- Canny edge detection (for zebra crossing)

### 2. Traffic Sign Detection
- Color thresholding in HSV space (red, blue, yellow)
- Morphological operations (opening, closing)
- Contour extraction and geometric filtering
- ROI normalization (64×64 or 96×96)

### 3. Traffic Sign Classification
- Template matching (baseline)
- HOG feature extraction + Linear SVM (preferred)
- Confidence thresholding to reduce false positives

### 4. Pedestrian Detection
- HOG descriptor with linear SVM
- Multi-scale detection (image pyramid)
- Search region optimization (lower 2/3 of frame)
- Non-maximum suppression (NMS)

### 5. Optional Extensions
- Zebra crossing detection using Hough lines
- Kalman filter for temporal stabilization and tracking

### 6. Warning System
- Risk assessment based on spatial position and size
- Debouncing for stable alerts
- Real-time visualization with bounding boxes
- Audio alerts for high-risk events

## Datasets

- **GTSRB** (German Traffic Sign Recognition Benchmark): 43 classes, ~39,000 training images
- **GTSDB** (German Traffic Sign Detection Benchmark): Annotated detection locations
- **Pedestrian datasets**: Public pedestrian images and dashcam videos

## Evaluation Metrics

- **Detection Quality**: Precision, Recall, F1-score
- **Real-time Performance**: FPS, latency per frame
- **Scenario Robustness**: Performance across day/dusk, shadows, rain

## Configuration

See `config.py` for tunable parameters:
- Color thresholds (HSV ranges)
- Morphological kernel sizes
- HOG parameters (cells per block, block size)
- Confidence thresholds
- NMS overlap thresholds

## References

[1] R. C. Gonzalez and R. E. Woods, Digital Image Processing. Pearson, 3rd ed., 2008.
[2] S. Maldonado-Bascón et al., "Road-sign detection and recognition based on support vector machines," IEEE TITS, vol. 8, no. 2, 2007.
[3] J. Stallkamp et al., "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition," Neural Networks, 2012.
[4] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," CVPR, 2005.
[5] P. Viola and M. Jones, "Rapid object detection using a boosted cascade of simple features," CVPR, 2001.

## License

Academic use only (University of Ruhuna, EE7204/EC7205)
