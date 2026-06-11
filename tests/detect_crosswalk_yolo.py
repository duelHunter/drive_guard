import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLO model
model = YOLO("models/crosswalk_yolov8.onnx")

CONF_THRESHOLD = 0.35


# -----------------------------
# Input source
# -----------------------------
# Webcam / dashcam
# cap = cv2.VideoCapture(0)

# For video file, use this instead:
cap = cv2.VideoCapture("data/videos/dashcam_footage_3.mp4")

# YOLO inference is too slow to run on every frame —
# process only one frame per second of video.
video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
process_interval = max(1, int(video_fps))

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    if frame_count % process_interval != 0:
        continue

    # Optional resize for faster display
    frame = cv2.resize(frame, (960, 540))

    # Run YOLO detection
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)

    for result in results:
        for box in result.boxes:

            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"crosswalk {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("Crosswalk Detection (YOLOv8 ONNX)", frame)

    # Press ESC to exit
    if cv2.waitKey(1000) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
