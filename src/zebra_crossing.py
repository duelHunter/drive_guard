"""
Zebra crossing detection using a YOLOv8 model fine-tuned for crosswalks.
Pretrained model source:
https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO (release v1.4.0)
"""

import cv2
import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

try:
    from ultralytics import YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False


class ZebraCrossingDetector:
    """Detects zebra/pedestrian crossings using a YOLOv8 crosswalk model"""

    def __init__(self, model_path: str = None, confidence: float = None):
        if not _YOLO_AVAILABLE:
            raise ImportError(
                "ultralytics is not installed.\n"
                "Install with:  pip install ultralytics"
            )

        model_path = model_path or config.ZEBRA_YOLO_MODEL
        self.confidence = confidence if confidence is not None else config.ZEBRA_YOLO_CONFIDENCE

        print(f"  Loading zebra crossing detector ({model_path})...")
        self.model = YOLO(model_path)
        print("  Zebra crossing detector ready.")

    def detect(self, image: np.ndarray) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect zebra crossings in image

        Args:
            image: Input BGR image

        Returns:
            List of detected crossing regions as ((x1, y1), (x2, y2)) bounding boxes
        """
        results = self.model(image, conf=self.confidence, verbose=False)

        crossings = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crossings.append(((x1, y1), (x2, y2)))

        return crossings

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
                (x1, max(15, y1 - 5)),
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
