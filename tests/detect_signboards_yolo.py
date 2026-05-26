import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
# -----------------------------
# Default YOLO model
# COCO classes include:
# 9  = traffic light
# 11 = stop sign
model = YOLO("yolov8n.pt")

# For custom traffic sign model, replace above with:
# model = YOLO("best.pt")


# -----------------------------
# Input source
# -----------------------------
# Webcam / dashcam
cap = cv2.VideoCapture(0)

# For video file, use this instead:
# cap = cv2.VideoCapture("dashcam.mp4")


# COCO class names
TARGET_CLASSES = {
    9: "Traffic Light",
    11: "Stop Sign"
}


while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Optional resize for faster display
    frame = cv2.resize(frame, (960, 540))

    # Run YOLO detection
    results = model(frame, conf=0.35, verbose=False)

    for result in results:
        for box in result.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Detect only traffic-related signboard classes
            if cls_id in TARGET_CLASSES:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"{TARGET_CLASSES[cls_id]} {conf:.2f}"

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    cv2.imshow("Traffic Signboard Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()