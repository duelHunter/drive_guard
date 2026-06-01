"""
Traffic sign classification module
Implements template matching and HOG+SVM classification
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.svm import LinearSVC
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class HOGFeatureExtractor:
    """Extracts HOG (Histogram of Oriented Gradients) features from images"""
    
    def __init__(self, cell_size: int = None, block_size: int = None, num_bins: int = None):
        """
        Initialize HOG extractor
        
        Args:
            cell_size: Size of each cell (default: config.HOG_CELL_SIZE)
            block_size: Number of cells per block (default: config.HOG_BLOCK_SIZE)
            num_bins: Number of orientation bins (default: config.HOG_NUM_BINS)
        """
        self.cell_size = cell_size or config.HOG_CELL_SIZE
        self.block_size = block_size or config.HOG_BLOCK_SIZE
        self.num_bins = num_bins or config.HOG_NUM_BINS
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from image
        
        Args:
            image: Input image (grayscale or BGR)
            
        Returns:
            HOG feature vector
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute gradients
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        
        # Normalize angle to [0, num_bins)
        angle = (angle % 180) / (180 / self.num_bins)
        
        # Create histogram grid
        h, w = gray.shape
        num_cells_y = h // self.cell_size
        num_cells_x = w // self.cell_size
        
        # Initialize histogram grid
        hist_grid = np.zeros((num_cells_y, num_cells_x, self.num_bins))
        
        # Fill histograms
        for y in range(num_cells_y):
            for x in range(num_cells_x):
                y_start = y * self.cell_size
                y_end = y_start + self.cell_size
                x_start = x * self.cell_size
                x_end = x_start + self.cell_size
                
                cell_magnitude = magnitude[y_start:y_end, x_start:x_end]
                cell_angle = angle[y_start:y_end, x_start:x_end]
                
                # Weighted histogram
                for i in range(self.cell_size):
                    for j in range(self.cell_size):
                        bin_idx = int(cell_angle[i, j]) % self.num_bins
                        hist_grid[y, x, bin_idx] += cell_magnitude[i, j]
        
        # Normalize blocks and flatten
        features = self._normalize_blocks(hist_grid)
        
        return features
    
    def _normalize_blocks(self, hist_grid: np.ndarray) -> np.ndarray:
        """
        Normalize blocks of histograms (L2 normalization)
        
        Args:
            hist_grid: Histogram grid
            
        Returns:
            Flattened, normalized feature vector
        """
        num_cells_y, num_cells_x, _ = hist_grid.shape
        num_blocks_y = num_cells_y - self.block_size + 1
        num_blocks_x = num_cells_x - self.block_size + 1
        
        normalized_features = []
        
        for y in range(num_blocks_y):
            for x in range(num_blocks_x):
                block = hist_grid[y:y+self.block_size, x:x+self.block_size, :].flatten()
                
                # L2 normalization
                norm = np.sqrt(np.sum(block ** 2)) + 1e-6
                block = block / norm
                
                normalized_features.append(block)
        
        return np.concatenate(normalized_features)


class TemplateSignClassifier:
    """Classify signs using template matching"""
    
    def __init__(self, templates_dir: str = None):
        """
        Initialize template classifier
        
        Args:
            templates_dir: Directory containing template images
        """
        self.templates = {}
        self.threshold = config.TEMPLATE_THRESHOLD
        
        if templates_dir and os.path.exists(templates_dir):
            self.load_templates(templates_dir)
    
    def load_templates(self, templates_dir: str):
        """
        Load template images from directory
        
        Args:
            templates_dir: Directory path
        """
        for filename in os.listdir(templates_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(templates_dir, filename)
                template = cv2.imread(path, 0)  # Grayscale
                sign_name = os.path.splitext(filename)[0]
                
                # Resize to standard size
                template = cv2.resize(template, (config.ROI_SIZE, config.ROI_SIZE))
                self.templates[sign_name] = template
    
    def classify(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify sign using template matching
        
        Args:
            roi: Region of interest (sign image)
            
        Returns:
            Tuple of (class_name, confidence)
        """
        if not self.templates:
            return "Unknown", 0.0
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
        
        # Resize to standard size
        gray = cv2.resize(gray, (config.ROI_SIZE, config.ROI_SIZE))
        
        best_match = "Unknown"
        best_score = 0.0
        
        for name, template in self.templates.items():
            # Normalize
            roi_norm = (gray.astype(np.float32) - gray.min()) / (gray.max() - gray.min() + 1e-6)
            template_norm = (template.astype(np.float32) - template.min()) / (template.max() - template.min() + 1e-6)
            
            # Cross-correlation
            score = cv2.matchTemplate(roi_norm, template_norm, cv2.TM_CCORR_NORMED)
            score = float(score)
            
            if score > best_score:
                best_score = score
                best_match = name
        
        # Apply confidence threshold
        if best_score < self.threshold:
            return "Unknown", 0.0
        
        return best_match, best_score


class HOGSignClassifier:
    """Classify signs using HOG features + SVM"""
    
    def __init__(self):
        """Initialize classifier"""
        self.hog_extractor = HOGFeatureExtractor()
        self.svm = LinearSVC(random_state=42, max_iter=2000)
        self.classes = []
        self.is_trained = False
        self.threshold = config.SVM_CONFIDENCE_THRESHOLD
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train the SVM classifier
        
        Args:
            X_train: Training features (N x D)
            y_train: Training labels (N,)
        """
        print(f"Training HOG+SVM classifier with {len(X_train)} samples...")
        self.classes = np.unique(y_train)
        
        try:
            self.svm.fit(X_train, y_train)
            self.is_trained = True
            print("Training completed successfully!")
        except Exception as e:
            print(f"Error during training: {e}")
            self.is_trained = False
    
    def classify(self, roi: np.ndarray) -> Tuple[str, float]:
        """
        Classify sign using HOG+SVM
        
        Args:
            roi: Region of interest
            
        Returns:
            Tuple of (class_name, confidence)
        """
        if not self.is_trained:
            return "Untrained", 0.0
        
        # Extract HOG features
        features = self.hog_extractor.extract(roi)
        features = features.reshape(1, -1)
        
        # Predict
        prediction = self.svm.predict(features)[0]
        
        # Get confidence (distance to decision boundary for binary, max prob for multi-class)
        try:
            decision_function = self.svm.decision_function(features)[0]
            confidence = abs(decision_function)
        except:
            confidence = 1.0
        
        # Apply confidence threshold
        if confidence < self.threshold:
            return "Low Confidence", confidence
        
        return str(prediction), confidence
    
    def save_model(self, path: str):
        """
        Save trained model
        
        Args:
            path: File path
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'classes': self.classes,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """
        Load trained model
        
        Args:
            path: File path
        """
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.svm = data['svm']
                self.classes = data['classes']
                self.is_trained = data['is_trained']
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")


def main():
    """Test classification module"""
    print("Sign classification module loaded successfully")


if __name__ == "__main__":
    main()
