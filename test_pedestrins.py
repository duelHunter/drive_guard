import cv2

# Load image
# image = cv2.imread("data/signboard.jpg")
# image = cv2.imread("data/signboard.png")
image = cv2.imread("data/peds5.jpg")

# Create HOG descriptor
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detect pedestrians
boxes, weights = hog.detectMultiScale(
    image,
    winStride=(4, 4),
    padding=(8, 8),
    scale=1.05
)

print(f"Detected {len(boxes)} pedestrians")

# Draw detections
for (x, y, w, h) in boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

display = cv2.resize(image, (800, 600))
cv2.imshow("Pedestrian Detection", display)
cv2.waitKey(0)
cv2.destroyAllWindows()