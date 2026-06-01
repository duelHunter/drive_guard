"""
QUICK START GUIDE - Vehicle Warning System
"""

# ============================================================================
# 1. INSTALLATION & SETUP
# ============================================================================

# Install dependencies:
# pip install -r requirements.txt

# ============================================================================
# 2. PROJECT STRUCTURE
# ============================================================================
"""
vision_project/
├── src/
│   ├── preprocessing.py          # Image preprocessing
│   ├── sign_detection.py         # Traffic sign detection
│   ├── sign_classification.py    # Sign classification (HOG+SVM)
│   ├── pedestrian_detection.py   # Pedestrian detection
│   ├── warning_system.py         # Warnings & visualization
│   ├── zebra_crossing.py         # Zebra crossing detection (optional)
│   ├── tracking.py               # Kalman filtering (optional)
│   ├── main.py                   # Main pipeline
│   └── __init__.py
├── data/
│   ├── datasets/                 # GTSRB, GTSDB, pedestrian data
│   └── videos/                   # Test videos
├── models/                        # Trained models
├── output/                        # Results
├── tests/
│   └── test_modules.py          # Unit tests
├── config.py                     # Configuration
├── README.md                     # Documentation
├── requirements.txt              # Dependencies
└── QUICKSTART.md                 # This file
"""

# ============================================================================
# 3. BASIC USAGE
# ============================================================================

# Process a video:
# python src/main.py --video input.mp4 --output output.mp4

# Process an image:
# python src/main.py --image input.jpg --output result.jpg

# Disable optional features:
# python src/main.py --video input.mp4 --no-zebra --no-tracking

# ============================================================================
# 4. PYTHON API USAGE
# ============================================================================

from src.main import VehicleWarningPipeline
import cv2

# Initialize pipeline
pipeline = VehicleWarningPipeline(enable_zebra=True, enable_tracking=True)

# Process video
pipeline.process_video('input.mp4', 'output.mp4', display=True)

# OR process image
result = pipeline.process_image('input.jpg')
cv2.imshow("Result", result)
cv2.waitKey(0)


# ============================================================================
# 5. USING INDIVIDUAL MODULES
# ============================================================================

from src.preprocessing import ImagePreprocessor
from src.sign_detection import TrafficSignDetector
from src.pedestrian_detection import PedestrianDetection
import cv2

# Read image
image = cv2.imread('road_scene.jpg')

# Preprocessing
preprocessor = ImagePreprocessor()
denoised, hsv = preprocessor.preprocess_frame(image, apply_clahe=True)

# Traffic sign detection
sign_detector = TrafficSignDetector()
sign_candidates = sign_detector.detect_all_signs(hsv, image, colors=['red', 'blue', 'yellow'])
print(f"Found {len(sign_candidates)} sign candidates")

# Pedestrian detection
ped_detector = PedestrianDetection()
pedestrians = ped_detector.detect(image, use_roi=True)
print(f"Found {len(pedestrians)} pedestrians")

# Visualize
for ped in pedestrians:
    x, y, w, h = ped
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Results", image)
cv2.waitKey(0)


# ============================================================================
# 6. RUNNING TESTS
# ============================================================================

# python tests/test_modules.py

# This will test all modules and verify they work correctly


# ============================================================================
# 7. CONFIGURATION
# ============================================================================

# Edit config.py to tune parameters:
# - Color thresholds (HSV ranges)
# - Morphological kernel sizes
# - HOG parameters
# - Alert thresholds
# - Risk zone boundaries

# Example: Adjusting color thresholds for red signs
"""
# In config.py:
RED_LOWER1 = (0, 100, 100)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (170, 100, 100)
RED_UPPER2 = (180, 255, 255)
"""


# ============================================================================
# 8. TRAINING CLASSIFIERS
# ============================================================================

# For HOG+SVM sign classification (once you have training data):
from src.sign_classification import HOGSignClassifier
import numpy as np

classifier = HOGSignClassifier()

# Prepare training data (features and labels)
# X_train: (N_samples, feature_size)
# y_train: (N_samples,) - class labels

classifier.train(X_train, y_train)
classifier.save_model('models/sign_classifier.pkl')

# Load model later
classifier.load_model('models/sign_classifier.pkl')


# ============================================================================
# 9. DATASET SETUP
# ============================================================================

# Download datasets:
# 1. GTSRB (German Traffic Sign Recognition Benchmark)
#    - 43 classes, 39,209 training images
#    - Download: http://benchmark.ini.rub.de/?section=gtsrb&subsection=news
#    - Place in: data/datasets/GTSRB/

# 2. GTSDB (German Traffic Sign Detection Benchmark)
#    - Full road scenes with annotations
#    - Download: http://www.cvlibs.net/datasets/traffic-sign-detection/
#    - Place in: data/datasets/GTSDB/

# 3. Pedestrian data
#    - INRIA Person Dataset
#    - Other public pedestrian datasets
#    - Place in: data/datasets/pedestrians/


# ============================================================================
# 10. TROUBLESHOOTING
# ============================================================================

"""
Q: No detections found
A: Check color thresholds in config.py. Test with debug_hsv.py

Q: Too many false positives
A: Increase confidence thresholds:
   - SVM_CONFIDENCE_THRESHOLD
   - TEMPLATE_THRESHOLD
   - Adjust NMS overlap threshold

Q: Slow performance
A: Disable optional features (--no-zebra, --no-tracking)
   Reduce frame resolution
   Lower processing FPS

Q: Missing pedestrians
A: May need to train custom detector if using non-standard pedestrian data
   Verify OpenCV HOG detector works with sample image

Q: Signs not classified correctly
A: Train HOG+SVM on GTSRB dataset
   Adjust color segmentation thresholds
   Check ROI normalization (config.ROI_SIZE)
"""


# ============================================================================
# 11. PERFORMANCE METRICS
# ============================================================================

"""
The system outputs:
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-score: 2 * (Precision * Recall) / (Precision + Recall)
- FPS: Frames per second (real-time performance)
- Latency: Time per frame in milliseconds

Goal: High F1-score (>0.8) and FPS >10 for real-time
"""


# ============================================================================
# 12. NEXT STEPS
# ============================================================================

"""
1. Run test suite:
   python tests/test_modules.py

2. Download datasets and place in data/datasets/

3. Train sign classifier on GTSRB:
   python train_sign_classifier.py

4. Test on sample video:
   python src/main.py --video data/videos/sample.mp4

5. Adjust parameters in config.py based on results

6. Evaluate performance:
   python evaluate.py

7. Create demo video and final report
"""

print("Quick Start Guide loaded successfully!")
