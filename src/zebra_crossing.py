"""
Optional: Zebra crossing detection using Hough line transform
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ZebraCrossingDetector:
    """Detects zebra crossings using edge and line detection"""
    
    def __init__(self):
        """Initialize detector"""
        self.hough_rho = config.HOUGH_RHO
        self.hough_theta = np.pi / 180 * config.HOUGH_THETA
        self.hough_threshold = config.HOUGH_THRESHOLD
        self.hough_min_length = config.HOUGH_MIN_LENGTH
        self.hough_max_gap = config.HOUGH_MAX_GAP
        self.line_angle_tolerance = config.LINE_ANGLE_TOLERANCE
        self.min_parallel_lines = config.MIN_PARALLEL_LINES
    
    def detect(self, image: np.ndarray, gray: Optional[np.ndarray] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect zebra crossings in image
        
        Args:
            image: Input BGR image
            gray: Pre-computed grayscale image (optional)
            
        Returns:
            List of detected crossing regions as bounding boxes
        """
        # Convert to grayscale
        if gray is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, config.CANNY_THRESHOLD1, config.CANNY_THRESHOLD2)
        
        # Apply Hough line transform
        lines = cv2.HoughLinesP(
            edges,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_length,
            maxLineGap=self.hough_max_gap
        )
        
        if lines is None:
            return []
        
        # Find parallel line groups
        parallel_groups = self._find_parallel_lines(lines)
        
        # Filter groups and create bounding boxes
        crossings = []
        for group in parallel_groups:
            if len(group) >= self.min_parallel_lines:
                bbox = self._lines_to_bbox(group)
                if bbox:
                    crossings.append(bbox)
        
        return crossings
    
    def _find_parallel_lines(self, lines: np.ndarray) -> List[List[np.ndarray]]:
        """
        Group lines by angle (find parallel groups)
        
        Args:
            lines: Array of detected lines
            
        Returns:
            List of parallel line groups
        """
        if lines.shape[0] == 0:
            return []
        
        # Calculate angles for each line
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            # Normalize to [0, 90]
            angle = angle % 180
            if angle > 90:
                angle = 180 - angle
            angles.append(angle)
        
        angles = np.array(angles)
        
        # Group lines by similar angle
        groups = []
        used = set()
        
        for i in range(len(angles)):
            if i in used:
                continue
            
            group = [lines[i]]
            used.add(i)
            
            for j in range(i + 1, len(angles)):
                if j in used:
                    continue
                
                # Check if angles are similar
                angle_diff = abs(angles[i] - angles[j])
                if angle_diff <= self.line_angle_tolerance or \
                   abs(angle_diff - 180) <= self.line_angle_tolerance:
                    group.append(lines[j])
                    used.add(j)
            
            if len(group) >= self.min_parallel_lines:
                groups.append(group)
        
        return groups
    
    def _lines_to_bbox(self, lines: List[np.ndarray]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Convert group of lines to bounding box
        
        Args:
            lines: List of line arrays
            
        Returns:
            Bounding box ((x1, y1), (x2, y2)) or None
        """
        if not lines:
            return None
        
        # Extract all points
        points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            points.extend([(x1, y1), (x2, y2)])
        
        points = np.array(points)
        
        # Get bounding box
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        
        # Check if bbox is reasonable
        width = x_max - x_min
        height = y_max - y_min
        
        if width < 10 or height < 10:
            return None
        
        return ((x_min, y_min), (x_max, y_max))
    
    def draw_crossings(self, image: np.ndarray, 
                      crossings: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> np.ndarray:
        """
        Draw detected zebra crossings on image
        
        Args:
            image: Input image
            crossings: List of crossing bounding boxes
            
        Returns:
            Annotated image
        """
        result = image.copy()
        
        for (x1, y1), (x2, y2) in crossings:
            cv2.rectangle(result, (x1, y1), (x2, y2), config.ZEBRA_BOX_COLOR, 2)
            cv2.putText(
                result,
                "ZEBRA",
                (x1, y1 - 5),
                config.FONT,
                0.6,
                config.ZEBRA_BOX_COLOR,
                2
            )
        
        return result


def main():
    """Test zebra crossing detection"""
    print("Zebra crossing detector loaded successfully")


if __name__ == "__main__":
    main()
