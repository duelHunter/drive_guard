"""
Main pipeline for Vision-Based Vehicle Warning System
Integrates all modules: preprocessing, detection, classification, warning
"""

import cv2
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from preprocessing import ImagePreprocessor
from sign_detection import TrafficSignDetector
from sign_classification import HOGSignClassifier, TemplateSignClassifier
from pedestrian_detection import PedestrianDetection
from warning_system import WarningSystem, Visualizer
from zebra_crossing import ZebraCrossingDetector
from tracking import MultiTracker


class VehicleWarningPipeline:
    """
    Main pipeline for vehicle warning system
    """
    
    def __init__(self, enable_zebra: bool = True, enable_tracking: bool = True):
        """
        Initialize pipeline
        
        Args:
            enable_zebra: Enable zebra crossing detection
            enable_tracking: Enable tracking for temporal stability
        """
        print("Initializing Vehicle Warning System...")
        
        # Initialize modules
        self.preprocessor = ImagePreprocessor()
        self.sign_detector = TrafficSignDetector()
        self.sign_classifier = HOGSignClassifier()
        self.pedestrian_detector = PedestrianDetection()
        self.warning_system = WarningSystem()
        self.visualizer = Visualizer()
        
        # Optional modules
        self.zebra_detector = ZebraCrossingDetector() if enable_zebra else None
        self.pedestrian_tracker = MultiTracker() if enable_tracking else None
        self.sign_tracker = MultiTracker() if enable_tracking else None
        
        self.enable_zebra = enable_zebra
        self.enable_tracking = enable_tracking
        
        print("Pipeline initialized successfully!")
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame through the pipeline
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with detection results
        """
        results = {
            'frame': frame,
            'detections': {},
            'risk_scores': {},
            'alerts': {},
            'alert_message': ''
        }
        
        # Step 1: Preprocessing
        denoised, hsv = self.preprocessor.preprocess_frame(frame, apply_clahe=True)
        
        # Step 2: Detect and classify traffic signs
        sign_candidates = self.sign_detector.detect_all_signs(hsv, frame)
        classified_signs = {}
        
        for candidate in sign_candidates:
            sign_name, confidence = self.sign_classifier.classify(candidate.roi_normalized)
            if confidence > config.SVM_CONFIDENCE_THRESHOLD:
                if sign_name not in classified_signs:
                    classified_signs[sign_name] = []
                classified_signs[sign_name].append(
                    (candidate.x, candidate.y, candidate.w, candidate.h)
                )
        
        results['detections']['signs'] = classified_signs
        
        # Step 3: Detect pedestrians
        pedestrians = self.pedestrian_detector.detect(frame, use_roi=True)
        
        # Apply tracking if enabled
        if self.enable_tracking and self.pedestrian_tracker:
            tracked = self.pedestrian_tracker.update(pedestrians)
            results['detections']['pedestrian'] = list(tracked.values())
        else:
            results['detections']['pedestrian'] = pedestrians
        
        # Step 4: Detect zebra crossings (optional)
        if self.enable_zebra and self.zebra_detector:
            gray = self.preprocessor.convert_to_gray(denoised)
            crossings = self.zebra_detector.detect(frame, gray)
            if crossings:
                # Convert to (x, y, w, h) format
                zebra_boxes = []
                for (x1, y1), (x2, y2) in crossings:
                    zebra_boxes.append((x1, y1, x2 - x1, y2 - y1))
                results['detections']['zebra_crossing'] = zebra_boxes
        
        # Step 5: Assess risk and generate warnings
        flat_detections = self._flatten_detections(results['detections'])
        risk_scores = self.warning_system.assess_risk(flat_detections)
        results['risk_scores'] = risk_scores
        
        # Check for alerts
        for det_type in flat_detections:
            if flat_detections[det_type]:
                should_alert = self.warning_system.should_alert(det_type)
                results['alerts'][det_type] = should_alert
            else:
                self.warning_system.reset_detection_counter(det_type)
                results['alerts'][det_type] = False
        
        # Generate alert message
        alert_msg = self.warning_system.generate_alert_message(results['alerts'])
        results['alert_message'] = alert_msg
        
        # Update frame counter
        self.warning_system.update_frame()
        
        return results
    
    def _flatten_detections(self, nested_detections: Dict) -> Dict[str, List]:
        """
        Flatten nested detection dictionary
        
        Args:
            nested_detections: Nested detection dict
            
        Returns:
            Flattened detection dict
        """
        flat = {
            'pedestrian': nested_detections.get('pedestrian', []),
            'stop_sign': nested_detections.get('signs', {}).get('Stop', []),
            'speed_limit': nested_detections.get('signs', {}).get('Speed_Limit', []),
            'yield_sign': nested_detections.get('signs', {}).get('Yield', []),
            'zebra_crossing': nested_detections.get('zebra_crossing', [])
        }
        return flat
    
    def visualize_results(self, results: Dict) -> np.ndarray:
        """
        Visualize detection results on frame
        
        Args:
            results: Results from process_frame
            
        Returns:
            Annotated frame
        """
        frame = results['frame'].copy()
        
        # Draw detections
        flat_detections = self._flatten_detections(results['detections'])
        frame = self.visualizer.draw_detections(frame, flat_detections, results['risk_scores'])
        
        # Draw alert message
        if results['alert_message']:
            has_alert = any(results['alerts'].values())
            frame = self.visualizer.draw_alert_message(frame, results['alert_message'], has_alert)
        
        # Add FPS counter
        cv2.putText(frame, f"Frame: {self.warning_system.frame_counter}", 
                   (10, 30), config.FONT, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def process_video(self, video_path: str, output_path: str = None, display: bool = True):
        """
        Process entire video
        
        Args:
            video_path: Input video path
            output_path: Output video path (optional)
            display: Whether to display results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        # Setup output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Resize frame to target size
                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
                
                # Process frame
                results = self.process_frame(frame)
                
                # Visualize
                annotated_frame = self.visualize_results(results)
                
                # Display
                if display:
                    cv2.imshow("Vehicle Warning System", annotated_frame)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                
                # Write output
                if out:
                    # Resize back to original
                    output_frame = cv2.resize(annotated_frame, (width, height))
                    out.write(output_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"Video processing completed! Total frames: {frame_count}")
    
    def process_image(self, image_path: str) -> np.ndarray:
        """
        Process single image
        
        Args:
            image_path: Path to image
            
        Returns:
            Annotated image
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Cannot load image {image_path}")
            return None
        
        # Resize
        image = cv2.resize(image, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        
        # Process
        results = self.process_frame(image)
        
        # Visualize
        annotated = self.visualize_results(results)
        
        return annotated


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vehicle Warning System")
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--output', type=str, help='Path to output video/image')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--no-zebra', action='store_true', help='Disable zebra crossing detection')
    parser.add_argument('--no-tracking', action='store_true', help='Disable tracking')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = VehicleWarningPipeline(
        enable_zebra=not args.no_zebra,
        enable_tracking=not args.no_tracking
    )
    
    # Process input
    if args.video:
        pipeline.process_video(args.video, args.output, not args.no_display)
    elif args.image:
        result = pipeline.process_image(args.image)
        if result is not None:
            if args.output:
                cv2.imwrite(args.output, result)
            if not args.no_display:
                cv2.imshow("Result", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    else:
        print("Please provide --video or --image argument")
        parser.print_help()


if __name__ == "__main__":
    main()
