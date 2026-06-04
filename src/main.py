"""
Main pipeline for Vision-Based Vehicle Warning System
Integrates all modules: preprocessing, detection, classification, warning
"""

import cv2
import numpy as np
import sys
import os
import time
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

    def _draw_latency(self, frame: np.ndarray, latency_ms: float) -> np.ndarray:
        """Overlay latency and FPS on the frame."""
        fps = 1000.0 / latency_ms if latency_ms > 0 else 0
        text = f"Latency: {latency_ms:.1f} ms  |  FPS: {fps:.1f}"
        cv2.putText(frame, text, (10, 60), config.FONT, 0.7, (0, 255, 255), 2)
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
        
        # Process 1 frame per second — skip the rest
        process_interval = max(1, fps)
        processed_count = 0

        print(f"Processing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        print(f"Sampling: 1 frame every {process_interval} frames (1 per second)\n")

        # Setup output video
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 1, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            print(f"Output will be saved to: {output_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only process every Nth frame
                if frame_count % process_interval != 0:
                    frame_count += 1
                    continue

                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                # Process frame with latency measurement
                t0 = time.perf_counter()
                results = self.process_frame(frame)
                latency_ms = (time.perf_counter() - t0) * 1000

                processed_count += 1
                print(f"Second {processed_count:>4} (frame {frame_count:>5}) | Latency: {latency_ms:6.1f} ms | FPS: {1000/latency_ms:5.1f}")

                # Visualize
                annotated_frame = self.visualize_results(results)
                annotated_frame = self._draw_latency(annotated_frame, latency_ms)

                # Display
                if display:
                    cv2.imshow("Vehicle Warning System", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Write output
                if out:
                    out.write(annotated_frame)

                frame_count += 1
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            print(f"\nVideo processing completed! Frames read: {frame_count} | Frames processed: {processed_count}")
    
    def process_camera(self, camera_index: int = 0, output_path: str = None, display: bool = True):
        """
        Process live camera feed in real-time.

        Args:
            camera_index: Camera device index (0 = default webcam/dashcam)
            output_path: Optional path to save the output video
            display: Whether to show the live window
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_index}")
            return

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {camera_index} opened — {width}x{height}")
        print("Press 'q' to quit.\n")

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 15, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
            print(f"Recording to: {output_path}")

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to grab frame.")
                    break

                frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

                # Process with latency measurement
                t0 = time.perf_counter()
                results = self.process_frame(frame)
                latency_ms = (time.perf_counter() - t0) * 1000

                print(f"Frame {frame_count + 1:>5} | Latency: {latency_ms:6.1f} ms | FPS: {1000/latency_ms:5.1f}")

                annotated_frame = self.visualize_results(results)
                annotated_frame = self._draw_latency(annotated_frame, latency_ms)

                if display:
                    cv2.imshow("Vehicle Warning System — Live", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if out:
                    out.write(annotated_frame)

                frame_count += 1

        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print(f"\nCamera session ended. Total frames processed: {frame_count}")

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
        
        image = cv2.resize(image, (config.FRAME_WIDTH, config.FRAME_HEIGHT))

        t0 = time.perf_counter()
        results = self.process_frame(image)
        latency_ms = (time.perf_counter() - t0) * 1000

        print(f"Latency: {latency_ms:.1f} ms | FPS equivalent: {1000/latency_ms:.1f}")

        annotated = self.visualize_results(results)
        annotated = self._draw_latency(annotated, latency_ms)

        return annotated


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Vehicle Warning System")
    parser.add_argument('--video', type=str, help='Path to input video')
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--camera', type=int, nargs='?', const=0, metavar='INDEX',
                        help='Run on live camera feed (default index: 0)')
    parser.add_argument('--output', type=str, help='Path to output video/image')
    parser.add_argument('--no-display', action='store_true', help='Disable display window')
    parser.add_argument('--no-zebra', action='store_true', help='Disable zebra crossing detection')
    parser.add_argument('--no-tracking', action='store_true', help='Disable tracking')

    args = parser.parse_args()

    pipeline = VehicleWarningPipeline(
        enable_zebra=not args.no_zebra,
        enable_tracking=not args.no_tracking
    )

    if args.camera is not None:
        pipeline.process_camera(args.camera, args.output, not args.no_display)
    elif args.video:
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
        print("Please provide --video, --image, or --camera argument")
        parser.print_help()


if __name__ == "__main__":
    main()
