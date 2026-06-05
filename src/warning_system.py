"""
Warning system module - integrates detections and generates alerts
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class WarningSystem:
    """
    Manages warning logic, debouncing, and alert generation
    """
    
    def __init__(self):
        """Initialize warning system"""
        self.alert_queue = {}  # Track consecutive detections per class
        self.last_alert_frame = {}  # Track cooldown
        self.frame_counter = 0
    
    def assess_risk(self, detections: Dict[str, List[Tuple[int, int, int, int]]]) -> Dict[str, float]:
        """
        Assess risk level for each detection type
        
        Args:
            detections: Dictionary of detection lists {type: [(x, y, w, h), ...]}
            
        Returns:
            Risk scores for each type
        """
        risk_scores = {}
        
        for det_type, boxes in detections.items():
            if not boxes:
                risk_scores[det_type] = 0.0
                continue
            
            max_risk = 0.0
            
            for x, y, w, h in boxes:
                # Risk based on vertical position (lower = higher risk)
                y_center = y + h // 2
                if y_center > config.RISK_ZONE_LOWER_THRESHOLD:
                    y_risk = (y_center - config.RISK_ZONE_LOWER_THRESHOLD) / \
                             (config.FRAME_HEIGHT - config.RISK_ZONE_LOWER_THRESHOLD)
                else:
                    y_risk = 0.0
                
                # Risk based on horizontal position (center = higher risk)
                x_center = x + w // 2
                if config.RISK_ZONE_CENTER_START <= x_center <= \
                   config.RISK_ZONE_CENTER_START + config.RISK_ZONE_CENTER_WIDTH:
                    x_risk = 1.0
                else:
                    x_risk = 0.5
                
                # Risk based on size (larger = closer = higher risk)
                size_risk = min(1.0, (w * h) / (config.FRAME_WIDTH * config.FRAME_HEIGHT) * 10)
                
                # Combined risk
                risk = 0.4 * y_risk + 0.3 * x_risk + 0.3 * size_risk
                max_risk = max(max_risk, risk)
            
            risk_scores[det_type] = max_risk
        
        return risk_scores
    
    def should_alert(self, detection_type: str) -> bool:
        """
        Check if alert should be triggered (with debouncing)
        
        Args:
            detection_type: Type of detection (e.g., 'pedestrian', 'stop_sign')
            
        Returns:
            True if alert should be triggered
        """
        # Initialize if not present
        if detection_type not in self.alert_queue:
            self.alert_queue[detection_type] = 0
            self.last_alert_frame[detection_type] = -config.ALERT_COOLDOWN_FRAMES
        
        # Check cooldown
        frames_since_alert = self.frame_counter - self.last_alert_frame[detection_type]
        if frames_since_alert < config.ALERT_COOLDOWN_FRAMES:
            return False
        
        # Increment detection counter
        self.alert_queue[detection_type] += 1
        
        # Trigger alert if enough consecutive detections
        if self.alert_queue[detection_type] >= config.ALERT_DEBOUNCE_FRAMES:
            self.last_alert_frame[detection_type] = self.frame_counter
            self.alert_queue[detection_type] = 0
            return True
        
        return False
    
    def reset_detection_counter(self, detection_type: str):
        """
        Reset counter for a detection type (no longer detected)
        
        Args:
            detection_type: Type of detection
        """
        if detection_type in self.alert_queue:
            self.alert_queue[detection_type] = 0
    
    def update_frame(self):
        """Update frame counter"""
        self.frame_counter += 1
    
    def generate_alert_message(self, alerts: Dict[str, bool]) -> str:
        """
        Generate alert message text
        
        Args:
            alerts: Dictionary of {type: alert_triggered}
            
        Returns:
            Alert message string
        """
        messages = []
        
        if alerts.get('pedestrian'):
            messages.append("PEDESTRIAN DETECTED!")
        if alerts.get('stop_sign'):
            messages.append("STOP SIGN!")
        if alerts.get('speed_limit'):
            messages.append("SPEED LIMIT SIGN")
        if alerts.get('yield_sign'):
            messages.append("YIELD SIGN")
        if alerts.get('zebra_crossing'):
            messages.append("ZEBRA CROSSING!")
        
        return " | ".join(messages) if messages else ""


class Visualizer:
    """Handles visualization and annotation of detections"""
    
    @staticmethod
    def draw_detections(image: np.ndarray,
                       detections: Dict[str, List[Tuple[int, int, int, int]]],
                       risk_scores: Dict[str, float] = None) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: Dictionary of detection lists
            risk_scores: Optional risk scores for each type
            
        Returns:
            Annotated image
        """
        result = image.copy()
        
        # Colors for different detection types
        colors = {
            'pedestrian': config.PEDESTRIAN_BOX_COLOR,
            'pedestrian_on_road':  (0, 0, 255),    # Red   — danger
            'pedestrian_off_road': (0, 255, 0),    # Green — safe
            'stop_sign':     (0, 0, 255),           # Red
            'traffic_light': (0, 255, 255),         # Yellow
            'speed_limit':   (0, 165, 255),         # Orange
            'yield_sign':    (0, 255, 255),          # Yellow
            'zebra_crossing': config.ZEBRA_BOX_COLOR,
        }
        
        for det_type, boxes in detections.items():
            color = colors.get(det_type, config.SIGN_BOX_COLOR)
            
            for i, (x, y, w, h) in enumerate(boxes):
                # Draw bounding box
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
                
                # Draw label with risk score if available
                label = det_type.replace('_', ' ').upper()
                if risk_scores and det_type in risk_scores:
                    risk = risk_scores[det_type]
                    label += f" ({risk:.2f})"
                
                # Background for text
                text_size = cv2.getTextSize(label, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)[0]
                cv2.rectangle(
                    result,
                    (x, y - text_size[1] - 5),
                    (x + text_size[0], y),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    label,
                    (x, y - 5),
                    config.FONT,
                    config.FONT_SCALE,
                    config.FONT_COLOR,
                    config.FONT_THICKNESS
                )
        
        return result
    
    @staticmethod
    def draw_alert_message(image: np.ndarray, message: str, 
                          alert_triggered: bool = False) -> np.ndarray:
        """
        Draw alert message on image
        
        Args:
            image: Input image
            message: Message text
            alert_triggered: Whether an alert was triggered
            
        Returns:
            Image with message
        """
        if not message:
            return image
        
        result = image.copy()
        
        # Background color
        bg_color = (0, 0, 255) if alert_triggered else (0, 165, 255)  # Red if alert, orange otherwise
        
        # Text size
        text_size = cv2.getTextSize(message, config.FONT, 1.0, 2)[0]
        
        # Draw background
        margin = 10
        cv2.rectangle(
            result,
            (margin, margin),
            (margin + text_size[0] + 10, margin + text_size[1] + 10),
            bg_color,
            -1
        )
        
        # Draw text
        cv2.putText(
            result,
            message,
            (margin + 5, margin + text_size[1] + 5),
            config.FONT,
            1.0,
            (255, 255, 255),
            2
        )
        
        return result
    
    @staticmethod
    def draw_roi_zones(image: np.ndarray) -> np.ndarray:
        """
        Draw region of interest zones for debugging
        
        Args:
            image: Input image
            
        Returns:
            Image with ROI zones drawn
        """
        result = image.copy()
        
        # Draw risk zone
        cv2.rectangle(
            result,
            (config.RISK_ZONE_CENTER_START, config.RISK_ZONE_LOWER_THRESHOLD),
            (config.RISK_ZONE_CENTER_START + config.RISK_ZONE_CENTER_WIDTH, config.FRAME_HEIGHT),
            (0, 255, 0),
            2
        )
        cv2.putText(
            result,
            "Risk Zone",
            (config.RISK_ZONE_CENTER_START + 10, config.RISK_ZONE_LOWER_THRESHOLD + 30),
            config.FONT,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Draw pedestrian search region
        y1, y2 = config.PED_SEARCH_REGION[1], config.PED_SEARCH_REGION[3]
        cv2.rectangle(
            result,
            (0, y1),
            (config.FRAME_WIDTH, y2),
            (255, 0, 0),
            2
        )
        cv2.putText(
            result,
            "Pedestrian Search Zone",
            (10, y1 + 30),
            config.FONT,
            0.7,
            (255, 0, 0),
            2
        )
        
        return result


def main():
    """Test warning system module"""
    print("Warning system module loaded successfully")


if __name__ == "__main__":
    main()
