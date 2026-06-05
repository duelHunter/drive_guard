import cv2
import numpy as np


def detect_zebra_crossing(frame):
    """
    Detect zebra crossing using white stripe + line pattern.
    Returns processed frame and detection status.
    """

    output = frame.copy()

    h, w = frame.shape[:2]

    # 1. Use lower half of image as road ROI
    roi_y1 = int(h * 0.45)
    roi = frame[roi_y1:h, :]

    # 2. Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 3. Blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Threshold bright/white regions
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    # 5. Morphological close to connect white stripes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 6. Detect edges
    edges = cv2.Canny(morph, 50, 150)

    # 7. Detect lines using Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=60,
        maxLineGap=20
    )

    zebra_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            dx = x2 - x1
            dy = y2 - y1

            angle = np.degrees(np.arctan2(dy, dx))

            # Zebra stripes usually appear almost horizontal in dashcam view
            if -25 <= angle <= 25:
                length = np.sqrt(dx * dx + dy * dy)

                if length > 50:
                    zebra_lines.append((x1, y1, x2, y2))

                    cv2.line(
                        output,
                        (x1, y1 + roi_y1),
                        (x2, y2 + roi_y1),
                        (0, 255, 255),
                        2
                    )

    # 8. Decision logic
    zebra_detected = len(zebra_lines) >= 4

    if zebra_detected:
        cv2.putText(
            output,
            "ZEBRA CROSSING DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

    # Draw ROI line
    cv2.line(output, (0, roi_y1), (w, roi_y1), (255, 0, 0), 2)

    return output, zebra_detected


# -----------------------------
# Input source
# -----------------------------

# Webcam / dashcam
# cap = cv2.VideoCapture(0)

# For video file, use this instead:
cap = cv2.VideoCapture("data/videos/dashcam_footage_2.mp4")

# For image, use this instead:
# frame = cv2.imread("road.jpg")
# result, detected = detect_zebra_crossing(frame)
# print("Detected:", detected)
# cv2.imshow("Zebra Crossing Detection", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    result, detected = detect_zebra_crossing(frame)

    cv2.imshow("Zebra Crossing Detection", result)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()