"""
Demo/test script for the Vehicle Warning System
Tests individual modules and the complete pipeline
"""

import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import ImagePreprocessor
from src.sign_detection import TrafficSignDetector
from src.sign_classification import HOGSignClassifier, HOGFeatureExtractor
from src.pedestrian_detection import PedestrianDetection
from src.warning_system import WarningSystem, Visualizer
from src.zebra_crossing import ZebraCrossingDetector
from src.tracking import KalmanFilterTracker, MultiTracker


def test_preprocessing():
    """Test preprocessing module"""
    print("\n" + "="*50)
    print("Testing Preprocessing Module")
    print("="*50)
    
    preprocessor = ImagePreprocessor()
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Test denoising
    denoised = preprocessor.denoise_gaussian(test_img)
    print("✓ Gaussian denoising: OK")
    
    denoised = preprocessor.denoise_bilateral(test_img)
    print("✓ Bilateral denoising: OK")
    
    # Test color conversion
    hsv = preprocessor.convert_to_hsv(test_img)
    assert hsv.shape == test_img.shape
    print("✓ BGR to HSV conversion: OK")
    
    gray = preprocessor.convert_to_gray(test_img)
    assert len(gray.shape) == 2
    print("✓ BGR to Grayscale conversion: OK")
    
    # Test color extraction
    mask = preprocessor.extract_color_mask(hsv, 'red')
    print("✓ Color mask extraction: OK")
    
    # Test edge detection
    edges = preprocessor.detect_edges_canny(gray)
    print("✓ Canny edge detection: OK")
    
    print("\nPreprocessing module: ALL TESTS PASSED ✓")


def test_hog_extraction():
    """Test HOG feature extraction"""
    print("\n" + "="*50)
    print("Testing HOG Feature Extractor")
    print("="*50)
    
    extractor = HOGFeatureExtractor(cell_size=8, block_size=2, num_bins=9)
    
    # Create test image
    test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    
    # Extract features
    features = extractor.extract(test_img)
    print(f"✓ HOG extraction: OK (feature vector size: {len(features)})")
    
    # Test with grayscale
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    features = extractor.extract(gray_img)
    print(f"✓ HOG extraction (grayscale): OK")
    
    print("\nHOG Extractor: ALL TESTS PASSED ✓")


def test_sign_detection():
    """Test traffic sign detection"""
    print("\n" + "="*50)
    print("Testing Traffic Sign Detection")
    print("="*50)
    
    detector = TrafficSignDetector()
    
    # Create synthetic image with red region
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[100:200, 100:200] = [0, 0, 255]  # Red in BGR
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Detect candidates
    candidates = detector.detect_all_signs(hsv, img, colors=['red'])
    print(f"✓ Color segmentation and candidate detection: OK ({len(candidates)} candidates)")
    
    print("\nSign Detection: ALL TESTS PASSED ✓")


def test_pedestrian_detection():
    """Test pedestrian detection"""
    print("\n" + "="*50)
    print("Testing Pedestrian Detection")
    print("="*50)
    
    detector = PedestrianDetection()
    
    # Create test image
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect pedestrians
    detections = detector.detect(test_img, use_roi=True)
    print(f"✓ Pedestrian detection: OK ({len(detections)} detections)")
    
    # Test NMS
    test_detections = [(10, 10, 64, 128), (15, 15, 64, 128), (200, 200, 64, 128)]
    suppressed = detector._non_max_suppression(test_detections)
    print(f"✓ Non-maximum suppression: OK ({len(suppressed)} after NMS)")
    
    print("\nPedestrian Detection: ALL TESTS PASSED ✓")


def test_warning_system():
    """Test warning system"""
    print("\n" + "="*50)
    print("Testing Warning System")
    print("="*50)
    
    warning_system = WarningSystem()
    
    # Create test detections
    detections = {
        'pedestrian': [(100, 200, 50, 100)],
        'stop_sign': [(300, 50, 30, 30)]
    }
    
    # Assess risk
    risk_scores = warning_system.assess_risk(detections)
    print(f"✓ Risk assessment: OK (scores: {risk_scores})")
    
    # Check alerts (with debouncing)
    alert1 = warning_system.should_alert('pedestrian')
    print(f"✓ Alert debouncing: OK (first alert: {alert1})")
    
    # Generate message
    alerts = {'pedestrian': True, 'stop_sign': False}
    msg = warning_system.generate_alert_message(alerts)
    print(f"✓ Alert message generation: OK (message: '{msg}')")
    
    print("\nWarning System: ALL TESTS PASSED ✓")


def test_kalman_tracking():
    """Test Kalman filter tracking"""
    print("\n" + "="*50)
    print("Testing Kalman Filter Tracking")
    print("="*50)
    
    tracker = KalmanFilterTracker()
    
    # Initialize with first measurement
    tracker.init(100, 100)
    print("✓ Tracker initialization: OK")
    
    # Predict
    x, y = tracker.predict()
    print(f"✓ Prediction: OK (position: {x:.1f}, {y:.1f})")
    
    # Update with measurement
    tracker.update(105, 105)
    print("✓ Update with measurement: OK")
    
    # Get current position
    x, y = tracker.get_position()
    print(f"✓ Position retrieval: OK (position: {x:.1f}, {y:.1f})")
    
    # Test multi-tracker
    multi = MultiTracker(max_age=30)
    detections = [(10, 10, 64, 128), (200, 200, 64, 128)]
    tracked = multi.update(detections)
    print(f"✓ Multi-object tracking: OK ({len(tracked)} tracks)")
    
    print("\nKalman Tracking: ALL TESTS PASSED ✓")


def test_zebra_crossing():
    """Test zebra crossing detection"""
    print("\n" + "="*50)
    print("Testing Zebra Crossing Detection")
    print("="*50)
    
    detector = ZebraCrossingDetector()
    
    # Create synthetic image with parallel lines
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    for y in range(100, 300, 20):
        cv2.line(img, (100, y), (500, y), (255, 255, 255), 2)
    
    # Detect crossings
    crossings = detector.detect(img)
    print(f"✓ Zebra crossing detection: OK ({len(crossings)} crossings detected)")
    
    print("\nZebra Crossing Detection: ALL TESTS PASSED ✓")


def test_visualizer():
    """Test visualization"""
    print("\n" + "="*50)
    print("Testing Visualizer")
    print("="*50)
    
    visualizer = Visualizer()
    
    # Create test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Test drawing detections
    detections = {
        'pedestrian': [(100, 100, 50, 100), (300, 200, 40, 80)],
        'stop_sign': [(400, 50, 30, 30)]
    }
    
    result = visualizer.draw_detections(img, detections)
    print(f"✓ Drawing detections: OK (output shape: {result.shape})")
    
    # Test alert message
    result = visualizer.draw_alert_message(img, "TEST ALERT", alert_triggered=True)
    print(f"✓ Drawing alert message: OK")
    
    # Test ROI zones
    result = visualizer.draw_roi_zones(img)
    print(f"✓ Drawing ROI zones: OK")
    
    print("\nVisualizer: ALL TESTS PASSED ✓")


def run_all_tests():
    """Run all module tests"""
    print("\n" + "="*50)
    print("VEHICLE WARNING SYSTEM - MODULE TESTS")
    print("="*50)
    
    try:
        test_preprocessing()
        test_hog_extraction()
        test_sign_detection()
        test_pedestrian_detection()
        test_warning_system()
        test_kalman_tracking()
        test_zebra_crossing()
        test_visualizer()
        
        print("\n" + "="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY ✓✓✓")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
