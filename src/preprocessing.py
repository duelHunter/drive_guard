"""
Preprocessing module for image enhancement and preparation
Includes: denoising, color space conversion, illumination compensation, edge detection
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ImagePreprocessor:
    """
    Handles preprocessing operations on video frames for robust detection
    """
    
    def __init__(self):
        """Initialize preprocessor with configuration parameters"""
        self.gaussian_kernel = config.GAUSSIAN_KERNEL_SIZE
        self.bilateral_d = config.BILATERAL_D
        self.bilateral_sigma_color = config.BILATERAL_SIGMA_COLOR
        self.bilateral_sigma_space = config.BILATERAL_SIGMA_SPACE
        self.clahe_clip_limit = config.CLAHE_CLIP_LIMIT
        self.clahe_tile_size = config.CLAHE_TILE_SIZE
        self.clahe_threshold = config.CLAHE_BRIGHTNESS_THRESHOLD
        
        # Initialize CLAHE
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=self.clahe_tile_size
        )
    
    def denoise_gaussian(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filtering for general smoothing
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.GaussianBlur(image, (self.gaussian_kernel, self.gaussian_kernel), 0)
    
    def denoise_bilateral(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filtering for edge-preserving smoothing
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        return cv2.bilateralFilter(
            image,
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        )
    
    def convert_to_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to HSV color space
        
        Args:
            image: BGR image
            
        Returns:
            HSV image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    def convert_to_lab(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to LAB color space
        
        Args:
            image: BGR image
            
        Returns:
            LAB image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    def convert_to_gray(self, image: np.ndarray) -> np.ndarray:
        """
        Convert BGR image to grayscale
        
        Args:
            image: BGR image
            
        Returns:
            Grayscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    def apply_clahe(self, image: np.ndarray, is_color: bool = False) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
        for illumination compensation
        
        Args:
            image: Input image (grayscale or color)
            is_color: True if input is color, will apply to V channel (HSV) or L channel (LAB)
            
        Returns:
            CLAHE-enhanced image
        """
        if is_color:
            # Convert to HSV and apply CLAHE to V channel
            hsv = self.convert_to_hsv(image)
            h, s, v = cv2.split(hsv)
            v = self.clahe.apply(v)
            hsv = cv2.merge([h, s, v])
            return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            # Apply CLAHE directly to grayscale
            return self.clahe.apply(image)
    
    def should_apply_clahe(self, image: np.ndarray) -> bool:
        """
        Determine if CLAHE should be applied based on average brightness
        
        Args:
            image: Input image
            
        Returns:
            True if CLAHE should be applied, False otherwise
        """
        gray = self.convert_to_gray(image)
        avg_brightness = np.mean(gray)
        return avg_brightness < self.clahe_threshold
    
    def detect_edges_canny(self, image: np.ndarray, 
                          threshold1: int = None, 
                          threshold2: int = None) -> np.ndarray:
        """
        Detect edges using Canny edge detector
        
        Args:
            image: Input image (grayscale)
            threshold1: Lower threshold (default from config)
            threshold2: Upper threshold (default from config)
            
        Returns:
            Edge map
        """
        if threshold1 is None:
            threshold1 = config.CANNY_THRESHOLD1
        if threshold2 is None:
            threshold2 = config.CANNY_THRESHOLD2
        
        return cv2.Canny(image, threshold1, threshold2)
    
    def preprocess_frame(self, frame: np.ndarray, 
                        denoise_method: str = 'gaussian',
                        apply_clahe: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for a frame
        
        Args:
            frame: Input BGR frame from video
            denoise_method: 'gaussian' or 'bilateral'
            apply_clahe: Whether to apply illumination compensation
            
        Returns:
            Tuple of (preprocessed frame, HSV frame)
        """
        # Step 1: Denoising
        if denoise_method == 'bilateral':
            denoised = self.denoise_bilateral(frame)
        else:
            denoised = self.denoise_gaussian(frame)
        
        # Step 2: Optional illumination compensation
        if apply_clahe and self.should_apply_clahe(frame):
            denoised = self.apply_clahe(denoised, is_color=True)
        
        # Step 3: Color space conversion to HSV
        hsv = self.convert_to_hsv(denoised)
        
        return denoised, hsv
    
    def extract_color_mask(self, hsv_frame: np.ndarray, 
                          color: str) -> np.ndarray:
        """
        Extract color mask from HSV frame
        
        Args:
            hsv_frame: HSV color frame
            color: 'red', 'blue', or 'yellow'
            
        Returns:
            Binary mask of the color
        """
        if color.lower() == 'red':
            # Red has hue wrap-around in HSV
            lower1 = np.array(config.RED_LOWER1)
            upper1 = np.array(config.RED_UPPER1)
            lower2 = np.array(config.RED_LOWER2)
            upper2 = np.array(config.RED_UPPER2)
            
            mask1 = cv2.inRange(hsv_frame, lower1, upper1)
            mask2 = cv2.inRange(hsv_frame, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
            
        elif color.lower() == 'blue':
            lower = np.array(config.BLUE_LOWER)
            upper = np.array(config.BLUE_UPPER)
            mask = cv2.inRange(hsv_frame, lower, upper)
            
        elif color.lower() == 'yellow':
            lower = np.array(config.YELLOW_LOWER)
            upper = np.array(config.YELLOW_UPPER)
            mask = cv2.inRange(hsv_frame, lower, upper)
        else:
            raise ValueError(f"Unknown color: {color}")
        
        return mask
    
    @staticmethod
    def apply_morphological_ops(mask: np.ndarray, 
                               operation: str = 'open',
                               kernel_size: int = None,
                               iterations: int = 1) -> np.ndarray:
        """
        Apply morphological operations to clean up masks
        
        Args:
            mask: Binary mask
            operation: 'open' (remove noise), 'close' (fill holes), or 'both'
            kernel_size: Kernel size for morphological operations
            iterations: Number of iterations
            
        Returns:
            Cleaned mask
        """
        if kernel_size is None:
            kernel_size = config.MORPH_KERNEL_SIZE
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        
        if operation == 'open':
            return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif operation == 'close':
            return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        elif operation == 'both':
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            return mask
        else:
            raise ValueError(f"Unknown operation: {operation}")


def main():
    """Test preprocessing module"""
    print("Preprocessing module loaded successfully")


if __name__ == "__main__":
    main()
