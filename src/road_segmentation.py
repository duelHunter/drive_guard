"""
Road segmentation module using SegFormer (nvidia/segformer-b0-finetuned-cityscapes-1024-1024).
Identifies road pixels in dashcam frames and classifies pedestrian detections
as on-road (danger) or off-road (safe).
"""

import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    _DEPS_AVAILABLE = True
except ImportError:
    _DEPS_AVAILABLE = False


class RoadSegmenter:
    """
    Road segmentation using SegFormer pretrained on Cityscapes.
    Segments road pixels and checks whether detected pedestrians stand on the road.
    """

    MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
    ROAD_CLASS_ID = 0  # Cityscapes label index for 'road'

    def __init__(self):
        if not _DEPS_AVAILABLE:
            raise ImportError(
                "Missing dependencies for road segmentation.\n"
                "Install with:  pip install torch transformers"
            )

        print("  Loading SegFormer road segmentation model...")
        self.processor = SegformerImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.MODEL_NAME)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"  Road segmenter ready  [{self.device.upper()}]")

    def get_road_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a binary road mask from a BGR frame.
        Returns a uint8 mask: road pixels = 255, everything else = 0.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        inputs = self.processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        upsampled = torch.nn.functional.interpolate(
            logits,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

        pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        return (pred == self.ROAD_CLASS_ID).astype(np.uint8) * 255

    def is_on_road(self, box, road_mask: np.ndarray) -> bool:
        """
        Check whether the foot-point (bottom-centre) of a bounding box is on the road.
        box: (x, y, w, h)
        """
        x, y, w, h = box
        foot_x = int(x + w / 2)
        foot_y = int(y + h)

        h_mask, w_mask = road_mask.shape
        foot_x = max(0, min(foot_x, w_mask - 1))
        foot_y = max(0, min(foot_y, h_mask - 1))

        return bool(road_mask[foot_y, foot_x] > 0)

    def classify_pedestrians(self, boxes, road_mask: np.ndarray):
        """
        Split a list of pedestrian boxes into on-road and off-road groups.
        boxes: list of (x, y, w, h)
        Returns: (on_road, off_road) — two lists of (x, y, w, h)
        """
        on_road, off_road = [], []
        for box in boxes:
            (on_road if self.is_on_road(box, road_mask) else off_road).append(box)
        return on_road, off_road

    def create_overlay(self, frame: np.ndarray, road_mask: np.ndarray,
                       color=(255, 0, 0), alpha=0.35) -> np.ndarray:
        """
        Return a copy of frame with a semi-transparent coloured overlay on road pixels.
        Default colour is blue (BGR).
        """
        road_layer = np.full_like(frame, color)
        blended = cv2.addWeighted(frame, 1 - alpha, road_layer, alpha, 0)
        return np.where(road_mask[:, :, None] == 255, blended, frame)
