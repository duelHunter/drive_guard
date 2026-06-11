import cv2


def detect_zebra_crossing(frame):
    """
    Detect a zebra / pedestrian crossing from a dashcam (vehicle-view) frame.

    Approach:
      1. Restrict the search to the road area (lower portion of the frame).
      2. Adaptive-threshold to isolate bright stripe regions
         (robust to shadows and changing illumination, unlike a fixed threshold).
      3. Morphological close/open to merge broken stripe segments and remove noise.
      4. Find contours and keep wide, short blobs (stripe-shaped: aspect ratio > 1.5).
      5. Cluster stripes that overlap horizontally and are vertically stacked —
         a real crossing produces several such stripes in a row.
      6. Draw a single bounding box around the strongest cluster, labeled "crosswalk".

    Returns:
        output: annotated BGR frame
        detected: bool
    """

    output = frame.copy()
    h, w = frame.shape[:2]

    # 1. Road ROI — lower portion of the frame
    roi_y1 = int(h * 0.45)
    roi = frame[roi_y1:h, :]

    # 2. Grayscale + blur
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Adaptive threshold — picks out stripes brighter than their local
    #    surroundings, so it works in both bright sun and shaded road sections.
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, -15
    )

    # 4. Morphology: close gaps within a stripe, remove small speckle noise
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, open_kernel)

    # 5. Find stripe-shaped contours (wide and short)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stripes = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)

        if area < 300 or cw < 40 or ch < 5:
            continue
        if cw / float(ch) < 1.5:
            continue

        stripes.append((x, y, cw, ch))

    # 6. Cluster stripes: same group if horizontally overlapping and
    #    vertically close (stacked stripes of one crosswalk)
    stripes.sort(key=lambda s: s[1])

    groups = []
    for stripe in stripes:
        x, y, cw, ch = stripe
        placed = False

        for group in groups:
            lx, ly, lcw, lch = group[-1]
            overlap = min(x + cw, lx + lcw) - max(x, lx)
            vertical_gap = y - (ly + lch)

            if overlap > 0 and vertical_gap < lch * 2:
                group.append(stripe)
                placed = True
                break

        if not placed:
            groups.append([stripe])

    # Pick the group with the most aligned stripes (strongest crosswalk evidence)
    best_group = None
    for group in groups:
        if len(group) >= 3 and (best_group is None or len(group) > len(best_group)):
            best_group = group

    zebra_detected = best_group is not None

    if zebra_detected:
        xs  = [s[0] for s in best_group]
        ys  = [s[1] for s in best_group]
        x2s = [s[0] + s[2] for s in best_group]
        y2s = [s[1] + s[3] for s in best_group]

        x_min, y_min = min(xs), min(ys) + roi_y1
        x_max, y_max = max(x2s), max(y2s) + roi_y1

        cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(
            output, "crosswalk", (x_min, max(15, y_min - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )

    # Draw ROI boundary for debugging
    cv2.line(output, (0, roi_y1), (w, roi_y1), (255, 0, 0), 1)

    return output, zebra_detected, morph


# -----------------------------
# Input source
# -----------------------------

# Webcam / dashcam
# cap = cv2.VideoCapture(0)

# For video file, use this instead:
cap = cv2.VideoCapture("data/videos/dashcam_footage_1.mp4")

# Match playback speed to the video's real frame rate
fps = cap.get(cv2.CAP_PROP_FPS) or 30
frame_delay_ms = int(1000 / fps)

# For a single image, use this instead:
# frame = cv2.imread("data/original.jpg")
# result, detected, mask = detect_zebra_crossing(frame)
# print("Detected:", detected)
# cv2.imshow("Zebra Crossing Detection", result)
# cv2.imshow("Mask", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# exit()


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize(frame, (960, 540))

    result, detected, mask = detect_zebra_crossing(frame)

    cv2.imshow("Zebra Crossing Detection", result)
    cv2.imshow("Mask", mask)  # debug view — tune thresholds against this

    if cv2.waitKey(frame_delay_ms) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
