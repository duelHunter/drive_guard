"""
Optional: Kalman filter tracking for temporal stabilization
"""

import numpy as np
from typing import Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class KalmanFilterTracker:
    """Simple Kalman filter for tracking object centers"""
    
    def __init__(self, process_noise: float = None, measurement_noise: float = None):
        """
        Initialize Kalman filter
        
        Args:
            process_noise: Process noise covariance (default: config.KALMAN_PROCESS_NOISE)
            measurement_noise: Measurement noise covariance (default: config.KALMAN_MEASUREMENT_NOISE)
        """
        self.process_noise = process_noise or config.KALMAN_PROCESS_NOISE
        self.measurement_noise = measurement_noise or config.KALMAN_MEASUREMENT_NOISE
        
        # State: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)
        
        # State covariance
        self.P = np.eye(4) * 1.0
        
        # State transition matrix (constant velocity model)
        # [1 0 1 0]
        # [0 1 0 1]
        # [0 0 1 0]
        # [0 0 0 1]
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Measurement matrix (we measure position only)
        # [1 0 0 0]
        # [0 1 0 0]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process covariance
        self.Q = np.eye(4) * self.process_noise
        
        # Measurement covariance
        self.R = np.eye(2) * self.measurement_noise
        
        self.is_initialized = False
    
    def init(self, x: float, y: float):
        """
        Initialize filter with first measurement
        
        Args:
            x: X position
            y: Y position
        """
        self.state = np.array([x, y, 0, 0], dtype=np.float32)
        self.P = np.eye(4) * 1.0
        self.is_initialized = True
    
    def predict(self) -> Tuple[float, float]:
        """
        Predict next state
        
        Returns:
            Predicted (x, y) position
        """
        # x = F * x
        self.state = self.F @ self.state
        
        # P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return float(self.state[0]), float(self.state[1])
    
    def update(self, x: float, y: float):
        """
        Update filter with measurement
        
        Args:
            x: Measured X position
            y: Measured Y position
        """
        if not self.is_initialized:
            self.init(x, y)
            return
        
        # Measurement
        z = np.array([x, y], dtype=np.float32)
        
        # Innovation
        y_pred = self.H @ self.state
        y_innovation = z - y_pred
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y_innovation
        
        # Update covariance
        self.P = (np.eye(4) - K @ self.H) @ self.P
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current estimated position
        
        Returns:
            (x, y) position
        """
        return float(self.state[0]), float(self.state[1])
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get estimated velocity
        
        Returns:
            (vx, vy) velocity
        """
        return float(self.state[2]), float(self.state[3])


class MultiTracker:
    """Track multiple objects with Kalman filters"""
    
    def __init__(self, max_age: int = 30, distance_threshold: float = 50):
        """
        Initialize multi-object tracker
        
        Args:
            max_age: Maximum frames to keep track without detection
            distance_threshold: Max distance to associate detection with track
        """
        self.trackers = {}  # Dict[track_id] -> KalmanFilterTracker
        self.next_id = 0
        self.max_age = max_age
        self.age_counter = {}  # Dict[track_id] -> age
        self.distance_threshold = distance_threshold
    
    def update(self, detections: list) -> dict:
        """
        Update tracks with new detections
        
        Args:
            detections: List of (x, y, w, h) detections
            
        Returns:
            Dictionary of {track_id: (x, y, w, h)}
        """
        # Predict positions
        predictions = {}
        for track_id, tracker in self.trackers.items():
            x, y = tracker.predict()
            predictions[track_id] = (x, y)
        
        # Associate detections to tracks
        associated = set()
        updated_tracks = {}
        
        for det_idx, (det_x, det_y, det_w, det_h) in enumerate(detections):
            det_cx, det_cy = det_x + det_w // 2, det_y + det_h // 2
            
            best_track = None
            best_distance = self.distance_threshold
            
            for track_id, (pred_x, pred_y) in predictions.items():
                distance = np.sqrt((det_cx - pred_x) ** 2 + (det_cy - pred_y) ** 2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_track = track_id
            
            if best_track is not None:
                # Update track
                self.trackers[best_track].update(det_cx, det_cy)
                updated_tracks[best_track] = (det_x, det_y, det_w, det_h)
                associated.add(best_track)
                self.age_counter[best_track] = 0
            else:
                # Create new track
                tracker = KalmanFilterTracker()
                tracker.init(det_cx, det_cy)
                self.trackers[self.next_id] = tracker
                updated_tracks[self.next_id] = (det_x, det_y, det_w, det_h)
                self.age_counter[self.next_id] = 0
                self.next_id += 1
        
        # Age non-associated tracks
        to_remove = []
        for track_id in self.trackers:
            if track_id not in associated:
                self.age_counter[track_id] += 1
                if self.age_counter[track_id] > self.max_age:
                    to_remove.append(track_id)
        
        # Remove old tracks
        for track_id in to_remove:
            del self.trackers[track_id]
            del self.age_counter[track_id]
        
        return updated_tracks


def main():
    """Test tracking module"""
    print("Tracking module loaded successfully")


if __name__ == "__main__":
    main()
