# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from ultralytics import YOLO
# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# # -----------------------------
# # Load YOLO pedestrian detector
# # -----------------------------
# yolo_model = YOLO("yolov8n.pt")   # COCO person class = 0


# # -----------------------------
# # Load road segmentation model
# # -----------------------------
# seg_model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"

# processor = SegformerImageProcessor.from_pretrained(seg_model_name)
# seg_model = SegformerForSemanticSegmentation.from_pretrained(seg_model_name)

# device = "cuda" if torch.cuda.is_available() else "cpu"
# seg_model.to(device)
# seg_model.eval()

# print(f"Running on: {device}")


# # -----------------------------
# # Cityscapes road label
# # -----------------------------
# ROAD_CLASS_ID = 0


# def get_road_mask(frame):
#     """
#     Generate road segmentation mask from input frame.
#     Returns binary mask where road pixels = 255.
#     """

#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     pil_image = Image.fromarray(rgb)

#     inputs = processor(images=pil_image, return_tensors="pt")
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = seg_model(**inputs)

#     logits = outputs.logits

#     # Resize segmentation result to original frame size
#     upsampled_logits = torch.nn.functional.interpolate(
#         logits,
#         size=frame.shape[:2],
#         mode="bilinear",
#         align_corners=False
#     )

#     pred = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

#     road_mask = (pred == ROAD_CLASS_ID).astype(np.uint8) * 255

#     return road_mask


# def is_box_on_road(box, road_mask):
#     """
#     Check whether the bottom-center point of a detected pedestrian is on the road.
#     """

#     x1, y1, x2, y2 = box

#     foot_x = int((x1 + x2) / 2)
#     foot_y = int(y2)

#     h, w = road_mask.shape

#     foot_x = max(0, min(foot_x, w - 1))
#     foot_y = max(0, min(foot_y, h - 1))

#     return road_mask[foot_y, foot_x] > 0


# def create_overlay(frame, road_mask):
#     """
#     Create transparent road overlay.
#     """

#     overlay = frame.copy()

#     road_color = np.zeros_like(frame)
#     road_color[:, :] = (255, 0, 0)  # Blue road overlay

#     alpha = 0.35
#     overlay = np.where(
#         road_mask[:, :, None] == 255,
#         cv2.addWeighted(frame, 1 - alpha, road_color, alpha, 0),
#         frame
#     )

#     return overlay


# # -----------------------------
# # Use webcam / dashcam / video
# # -----------------------------
# # For webcam:
# # cap = cv2.VideoCapture(0)

# # For video file, use this instead:
# cap = cv2.VideoCapture("data/videos/dashcam_footage_2.mp4")


# while True:
#     ret, frame = cap.read()

#     if not ret:
#         break

#     # Optional resize for better speed
#     frame = cv2.resize(frame, (960, 540))

#     # 1. Road segmentation
#     road_mask = get_road_mask(frame)

#     # 2. YOLO pedestrian detection
#     results = yolo_model(frame, classes=[0], conf=0.4, verbose=False)

#     # 3. Draw road overlay
#     output = create_overlay(frame, road_mask)

#     # 4. Draw pedestrians and warning logic
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             on_road = is_box_on_road((x1, y1, x2, y2), road_mask)

#             if on_road:
#                 label = f"WARNING: Person on road {conf:.2f}"
#                 color = (0, 0, 255)
#             else:
#                 label = f"Person {conf:.2f}"
#                 color = (0, 255, 0)

#             cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

#             cv2.putText(
#                 output,
#                 label,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 color,
#                 2
#             )

#     cv2.imshow("Road Segmentation + Pedestrian Warning", output)

#     if cv2.waitKey(1) & 0xFF == 27:
#         break


# cap.release()
# cv2.destroyAllWindows()