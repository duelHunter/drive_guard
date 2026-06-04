import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt") 

# model = YOLO("yolov8x.pt") 

# # Webcam
# cap = cv2.VideoCapture(0)

# cap = cv2.VideoCapture("data/videos/dashcam_footage_3.mp4")
cap = cv2.VideoCapture("data/videos/output.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            cls = int(box.cls[0])

            # COCO class 0 = person
            if cls == 0:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    cv2.imshow("Pedestrian Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()