"""
Traffic sign detection module
Uses color segmentation, geometric filtering, and contour analysis
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SignCandidate:
    """Represents a potential traffic sign candidate"""
    
    def __init__(self, contour: np.ndarray, image: np.ndarray):
        """
        Initialize a sign candidate
        
        Args:
            contour: OpenCV contour
            image: Source image for ROI extraction
        """
        self.contour = contour
        self.image = image
        
        # Calculate properties
        self.area = cv2.contourArea(contour)
        self.perimeter = cv2.arcLength(contour, True)
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)
        self.bbox_area = self.w * self.h
        
        # Shape descriptors
        self.aspect_ratio = float(self.w) / self.h if self.h > 0 else 0
        self.extent = float(self.area) / self.bbox_area if self.bbox_area > 0 else 0
        self.circularity = (4 * np.pi * self.area) / (self.perimeter ** 2) if self.perimeter > 0 else 0
        
        # ROI
        self.roi = None
        self.roi_normalized = None
    
    def extract_roi(self, size: int = None) -> bool:
        """
        Extract and normalize region of interest
        
        Args:
            size: Target ROI size (config.ROI_SIZE if None)
            
        Returns:
            True if ROI extracted successfully
        """
        if size is None:
            size = config.ROI_SIZE
        
        # Extract ROI
        roi = self.image[self.y:self.y+self.h, self.x:self.x+self.w]
        
        if roi.size == 0:
            return False
        
        self.roi = roi
        # Resize to normalized size
        self.roi_normalized = cv2.resize(roi, (size, size))
        return True
    
    def __repr__(self):
        return f"SignCandidate(area={self.area:.1f}, AR={self.aspect_ratio:.2f}, extent={self.extent:.2f})"


class TrafficSignDetector:
    """
    Detects traffic sign candidates using color and geometric features
    """
    
    def __init__(self):
        """Initialize detector with configuration"""
        self.min_area = config.MIN_CONTOUR_AREA
        self.max_area = config.MAX_CONTOUR_AREA
        self.min_aspect = config.MIN_ASPECT_RATIO
        self.max_aspect = config.MAX_ASPECT_RATIO
        self.min_extent = config.MIN_EXTENT
        self.max_circularity = config.MAX_CIRCULARITY
    
    def detect_color_candidates(self, hsv_frame: np.ndarray, 
                               color: str) -> np.ndarray:
        """
        Detect candidates of a specific color
        
        Args:
            hsv_frame: HSV frame
            color: 'red', 'blue', or 'yellow'
            
        Returns:
            Binary mask of detected color
        """
        from preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        mask = preprocessor.extract_color_mask(hsv_frame, color)
        
        # Morphological cleanup
        mask = preprocessor.apply_morphological_ops(
            mask,
            operation='both',
            iterations=config.MORPH_ITERATIONS
        )
        
        return mask
    
    def extract_candidates(self, mask: np.ndarray, 
                          original_image: np.ndarray) -> List[SignCandidate]:
        """
        Extract sign candidates from binary mask
        
        Args:
            mask: Binary mask of potential signs
            original_image: Original image for ROI extraction
            
        Returns:
            List of SignCandidate objects
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        
        for contour in contours:
            candidate = SignCandidate(contour, original_image)
            
            # Geometric filtering
            if not self._passes_geometric_filter(candidate):
                continue
            
            # Extract and normalize ROI
            if candidate.extract_roi():
                candidates.append(candidate)
        
        return candidates
    
    def _passes_geometric_filter(self, candidate: SignCandidate) -> bool:
        """
        Check if candidate passes geometric filtering criteria
        
        Args:
            candidate: SignCandidate object
            
        Returns:
            True if passes filter, False otherwise
        """
        # Area check
        if not (self.min_area <= candidate.area <= self.max_area):
            return False
        
        # Aspect ratio check
        if not (self.min_aspect <= candidate.aspect_ratio <= self.max_aspect):
            return False
        
        # Extent check (compactness)
        if candidate.extent < self.min_extent:
            return False
        
        # Circularity check (to filter very circular objects)
        if candidate.circularity > self.max_circularity:
            return False
        
        return True
    
    def detect_all_signs(self, hsv_frame: np.ndarray, 
                        original_image: np.ndarray,
                        colors: List[str] = None) -> List[SignCandidate]:
        """
        Detect all traffic sign candidates (all colors)
        
        Args:
            hsv_frame: HSV frame
            original_image: Original BGR image
            colors: List of colors to detect (default: red, blue, yellow)
            
        Returns:
            List of all detected candidates
        """
        if colors is None:
            colors = ['red', 'blue', 'yellow']
        
        all_candidates = []
        
        for color in colors:
            try:
                mask = self.detect_color_candidates(hsv_frame, color)
                candidates = self.extract_candidates(mask, original_image)
                
                # Add color label to candidates
                for candidate in candidates:
                    candidate.color = color
                
                all_candidates.extend(candidates)
            except Exception as e:
                print(f"Error detecting {color} signs: {e}")
        
        return all_candidates
    
    def draw_candidates(self, image: np.ndarray, 
                       candidates: List[SignCandidate],
                       thickness: int = 2,
                       color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw bounding boxes around detected candidates
        
        Args:
            image: Image to draw on
            candidates: List of candidates
            thickness: Box line thickness
            color: Box color (BGR), default is green
            
        Returns:
            Annotated image
        """
        if color is None:
            color = config.SIGN_BOX_COLOR
        
        result = image.copy()
        
        for candidate in candidates:
            cv2.rectangle(
                result,
                (candidate.x, candidate.y),
                (candidate.x + candidate.w, candidate.y + candidate.h),
                color,
                thickness
            )
            
            # Draw color label
            cv2.putText(
                result,
                candidate.color.upper(),
                (candidate.x, candidate.y - 5),
                config.FONT,
                0.5,
                color,
                1
            )
        
        return result


def main():
    """Test sign detection module"""
    print("Sign detection module loaded successfully")


if __name__ == "__main__":
    main()
