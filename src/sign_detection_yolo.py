"""
YOLO-based traffic sign detection module.
Detects traffic lights and stop signs using YOLOv8 (COCO pretrained).
Returns results in (x, y, w, h) format consistent with the rest of the pipeline.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


# COCO class IDs relevant to traffic scenes
TARGET_CLASSES = {
    9:  "traffic_light",
    11: "stop_sign",
}


class YOLOSignDetector:
    """
    Detects traffic signs using a YOLOv8 model.
    Uses the COCO-pretrained yolov8n.pt by default; swap for a custom
    fine-tuned model by passing a different model_path.
    """

    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = None):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.\n"
                "Install with:  pip install ultralytics"
            )

        self.confidence = confidence if confidence is not None else config.YOLO_SIGN_CONFIDENCE
        print(f"  Loading YOLO sign detector ({model_path})...")
        self.model = YOLO(model_path)
        print("  YOLO sign detector ready.")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run detection on a BGR frame.

        Returns a dict mapping class name → list of (x, y, w, h) boxes.
        Example: {"stop_sign": [(x, y, w, h), ...], "traffic_light": [...]}
        """
        results = self.model(frame, conf=self.confidence, verbose=False)

        detections: dict = {}

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in TARGET_CLASSES:
                    continue

                label = TARGET_CLASSES[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x, y, w, h = x1, y1, x2 - x1, y2 - y1

                detections.setdefault(label, []).append((x, y, w, h))

        return detections
