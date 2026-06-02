"""
Pedestrian detection module using HOG+SVM
"""

import cv2
import numpy as np
from typing import List, Tuple
from sklearn.svm import LinearSVC
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from sign_classification import HOGFeatureExtractor


class PedestrianDetection:
    """Pedestrian detection using HOG descriptor and SVM classifier"""
    
    def __init__(self):
        """Initialize pedestrian detector"""
        self.hog = cv2.HOGDescriptor()
        
        # Try to use pre-trained detector, otherwise use custom
        self.use_pretrained = True
        self.custom_svm = None
        self.is_custom_trained = False
        
        try:
            # Use OpenCV's pre-trained pedestrian detector
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except:
            self.use_pretrained = False
    
    def detect(self, image: np.ndarray, 
              scale_factor: float = None,
              min_neighbors: int = 1,
              use_roi: bool = True) -> List[Tuple[int, int, int, int]]:
        """
        Detect pedestrians in image using HOG+SVM
        
        Args:
            image: Input image
            scale_factor: Scale factor for detection pyramid (default: config.PED_SCALE_FACTOR)
            min_neighbors: Minimum neighbors for grouping detections
            use_roi: Restrict detection to road region of interest
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if scale_factor is None:
            scale_factor = config.PED_SCALE_FACTOR
        
        # Apply ROI if requested
        roi_image = image
        roi_offset = (0, 0)
        
        if use_roi:
            roi_y1, roi_y2 = config.PED_SEARCH_REGION[1], config.PED_SEARCH_REGION[3]
            roi_image = image[roi_y1:roi_y2, :]
            roi_offset = (0, roi_y1)
        
        # Detect pedestrians
        if self.use_pretrained:
            detections, weights = self.hog.detectMultiScale(
                roi_image,
                winStride=(4, 4),
                padding=(8, 8),
                scale=scale_factor
            )
        else:
            detections = self._custom_detect(roi_image, scale_factor)
        
        # Adjust for ROI offset
        if use_roi:
            detections = [(x, y + roi_offset[1], w, h) for x, y, w, h in detections]
        
        # Apply NMS
        detections = self._non_max_suppression(detections)
        
        return detections
    
    def _custom_detect(self, image: np.ndarray, scale_factor: float) -> List[Tuple[int, int, int, int]]:
        """
        Custom pedestrian detection using custom SVM
        
        Args:
            image: Input image
            scale_factor: Scale factor for pyramid
            
        Returns:
            List of detections
        """
        if not self.is_custom_trained or self.custom_svm is None:
            return []
        
        detections = []
        hog_extractor = HOGFeatureExtractor(
            cell_size=config.PED_HOG_CELL_SIZE,
            block_size=config.PED_HOG_BLOCK_SIZE,
            num_bins=config.PED_HOG_NUM_BINS
        )
        
        # Sliding window detection
        window_size = config.PED_HOG_WINDOW_SIZE  # (64, 128)
        stride = 8
        
        scale = 1.0
        while scale <= config.PED_MAX_SCALE:
            scaled_image = cv2.resize(
                image,
                (int(image.shape[1] * scale), int(image.shape[0] * scale))
            )
            
            # Sliding window
            for y in range(0, scaled_image.shape[0] - window_size[1], stride):
                for x in range(0, scaled_image.shape[1] - window_size[0], stride):
                    window = scaled_image[y:y+window_size[1], x:x+window_size[0]]
                    
                    if window.shape != window_size[::-1]:
                        continue
                    
                    # Extract features
                    features = hog_extractor.extract(window)
                    
                    # Predict
                    if self.custom_svm.predict(features.reshape(1, -1))[0] == 1:
                        # Scale back to original coordinates
                        x_orig = int(x / scale)
                        y_orig = int(y / scale)
                        w = int(window_size[0] / scale)
                        h = int(window_size[1] / scale)
                        detections.append((x_orig, y_orig, w, h))
            
            scale *= config.PED_SCALE_FACTOR
        
        return detections
    
    def _non_max_suppression(self, detections: List[Tuple[int, int, int, int]],
                            threshold: float = None) -> List[Tuple[int, int, int, int]]:
        """
        Apply non-maximum suppression to remove overlapping detections
        
        Args:
            detections: List of (x, y, w, h) detections
            threshold: IoU threshold for suppression
            
        Returns:
            Filtered detections
        """
        if threshold is None:
            threshold = config.NMS_OVERLAP_THRESHOLD
        
        if len(detections) == 0:
            return []
        
        # Convert to (x1, y1, x2, y2) format
        boxes = np.array([
            [d[0], d[1], d[0] + d[2], d[1] + d[3]]
            for d in detections
        ], dtype=np.float32)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by area (descending)
        order = np.argsort(-areas)
        
        keep = []
        while len(order) > 0:
            idx = order[0]
            keep.append(idx)
            
            if len(order) == 1:
                break
            
            # Calculate IoU with current box
            x1 = np.maximum(boxes[idx, 0], boxes[order[1:], 0])
            y1 = np.maximum(boxes[idx, 1], boxes[order[1:], 1])
            x2 = np.minimum(boxes[idx, 2], boxes[order[1:], 2])
            y2 = np.minimum(boxes[idx, 3], boxes[order[1:], 3])
            
            w = np.maximum(0, x2 - x1)
            h = np.maximum(0, y2 - y1)
            intersection = w * h
            
            union = areas[idx] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)
            
            # Keep boxes with IoU < threshold
            order = order[np.where(iou <= threshold)[0] + 1]
        
        # Convert back to (x, y, w, h)
        result = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            result.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return result
    
    def train_custom_detector(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train custom pedestrian detector
        
        Args:
            X_train: Training features
            y_train: Training labels (1 for pedestrian, 0 for non-pedestrian)
        """
        print("Training custom pedestrian detector...")
        self.custom_svm = LinearSVC(random_state=42, max_iter=5000)
        
        try:
            self.custom_svm.fit(X_train, y_train)
            self.is_custom_trained = True
            print("Custom detector trained successfully!")
        except Exception as e:
            print(f"Error during training: {e}")
            self.is_custom_trained = False
    
    def save_detector(self, path: str):
        """Save trained detector"""
        if self.custom_svm is None:
            print("No custom detector to save")
            return
        
        with open(path, 'wb') as f:
            pickle.dump({
                'svm': self.custom_svm,
                'is_trained': self.is_custom_trained
            }, f)
        print(f"Detector saved to {path}")
    
    def load_detector(self, path: str):
        """Load trained detector"""
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.custom_svm = data['svm']
                self.is_custom_trained = data['is_trained']
            print(f"Detector loaded from {path}")
        except Exception as e:
            print(f"Error loading detector: {e}")


def main():
    """Test pedestrian detection module"""
    print("Pedestrian detection module loaded successfully")


if __name__ == "__main__":
    main()
