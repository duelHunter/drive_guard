"""
Configuration file for the Vision-Based Vehicle Warning System
"""

# ============================================================================
# VIDEO INPUT SETTINGS
# ============================================================================
VIDEO_FPS = 15  # Target processing frame rate
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
# FRAME_WIDTH = 1280
# FRAME_HEIGHT = 720

# Rolling buffer for temporal smoothing (in frames)
DETECTION_BUFFER_SIZE = 30  # ~2 seconds at 15 FPS

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

# Gaussian filtering kernel size (must be odd)
GAUSSIAN_KERNEL_SIZE = 5

# Bilateral filtering (for edge preservation)
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# CLAHE settings (for illumination compensation)
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
CLAHE_BRIGHTNESS_THRESHOLD = 100  # Apply CLAHE if avg brightness below this

# ============================================================================
# TRAFFIC SIGN DETECTION SETTINGS
# ============================================================================

# HSV color thresholds for traffic signs
# Red (two ranges due to hue wrap-around)
RED_LOWER1 = (0, 100, 100)
RED_UPPER1 = (10, 255, 255)
RED_LOWER2 = (170, 100, 100)
RED_UPPER2 = (180, 255, 255)

# Blue
BLUE_LOWER = (100, 100, 100)
BLUE_UPPER = (130, 255, 255)

# Yellow
YELLOW_LOWER = (20, 100, 100)
YELLOW_UPPER = (40, 255, 255)

# Morphological operations for sign candidate cleanup
MORPH_KERNEL_SIZE = 5  # Kernel size for opening/closing
MORPH_ITERATIONS = 1

# Geometric filtering for sign candidates
MIN_CONTOUR_AREA = 50  # Minimum area in pixels
MAX_CONTOUR_AREA = 5000  # Maximum area in pixels
MIN_ASPECT_RATIO = 0.5  # Width/Height ratio
MAX_ASPECT_RATIO = 2.5
MIN_EXTENT = 0.4  # Contour area / bounding box area
MAX_CIRCULARITY = 0.8  # For non-circular signs (4*pi*A/P^2)

# ROI normalization for sign classification
ROI_SIZE = 64  # 64x64 or 96x96

# ============================================================================
# SIGN CLASSIFICATION SETTINGS
# ============================================================================

# Template matching confidence threshold
TEMPLATE_THRESHOLD = 0.7

# HOG+SVM classification threshold
SVM_CONFIDENCE_THRESHOLD = 0.6

# HOG descriptor parameters
HOG_CELL_SIZE = 8
HOG_BLOCK_SIZE = 2  # blocks per side
HOG_NUM_BINS = 9

# Traffic signs to detect
SIGN_CLASSES = ['Stop', 'Speed_Limit', 'Pedestrian_Crossing', 'Yield', 'No_Entry']

# ============================================================================
# PEDESTRIAN DETECTION SETTINGS
# ============================================================================

# HOG parameters for pedestrian detection
PED_HOG_CELL_SIZE = 8
PED_HOG_BLOCK_SIZE = 2
PED_HOG_NUM_BINS = 9
PED_HOG_WINDOW_SIZE = (64, 128)  # Standard HOG window for pedestrians

# Detection scale parameters
PED_SCALE_FACTOR = 1.05  # Image pyramid scale
PED_MIN_SCALE = 0.5
PED_MAX_SCALE = 2.0

# NMS (Non-Maximum Suppression) parameters
NMS_OVERLAP_THRESHOLD = 0.3  # IoU threshold for merging overlapping boxes

# Pedestrian detection confidence threshold
PED_CONFIDENCE_THRESHOLD = 0.5

# Search region optimization (restrict to lower portion of frame)
# Detect pedestrians only in lower 2/3 of frame
PED_SEARCH_REGION = (0, int(FRAME_HEIGHT * 2/3), FRAME_WIDTH, FRAME_HEIGHT)

# ============================================================================
# ZEBRA CROSSING DETECTION SETTINGS (Optional)
# ============================================================================

# Canny edge detection
CANNY_THRESHOLD1 = 50
CANNY_THRESHOLD2 = 150

# Hough line transform
HOUGH_RHO = 1
HOUGH_THETA = 1  # degrees
HOUGH_THRESHOLD = 20  # Minimum votes
HOUGH_MIN_LENGTH = 50
HOUGH_MAX_GAP = 10

# Stripe periodicity parameters
MIN_PARALLEL_LINES = 3  # Minimum number of parallel stripes
LINE_ANGLE_TOLERANCE = 5  # degrees, for parallel detection

# ============================================================================
# KALMAN FILTER TRACKING (Optional)
# ============================================================================

# Process and measurement noise
KALMAN_PROCESS_NOISE = 0.01
KALMAN_MEASUREMENT_NOISE = 10.0

# ============================================================================
# WARNING SYSTEM SETTINGS
# ============================================================================

# Risk assessment based on position
# Detections in lower-center region are higher risk
RISK_ZONE_LOWER_THRESHOLD = int(FRAME_HEIGHT * 0.6)  # Bottom 40%
RISK_ZONE_CENTER_WIDTH = int(FRAME_WIDTH * 0.4)  # Center 40%
RISK_ZONE_CENTER_START = int(FRAME_WIDTH * 0.3)

# Debouncing: require N consecutive frames with detection
ALERT_DEBOUNCE_FRAMES = 2  # 2-3 frames
ALERT_COOLDOWN_FRAMES = 30  # Cooldown period between alerts (1-2 seconds at 15 FPS)

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Bounding box colors (BGR format)
SIGN_BOX_COLOR = (0, 255, 0)  # Green for signs
PEDESTRIAN_BOX_COLOR = (0, 0, 255)  # Red for pedestrians
ZEBRA_BOX_COLOR = (255, 0, 0)  # Cyan for zebra crossing

# Font settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2
FONT_COLOR = (255, 255, 255)  # White text

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Evaluation metrics
EVAL_IOU_THRESHOLD = 0.5  # IoU threshold for true positive
EVAL_CONFIDENCE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]  # For precision-recall curve

# ============================================================================
# AUDIO ALERTS
# ============================================================================

# Audio alert settings
ALERT_SOUND_PATH = 'assets/alert.wav'  # Path to alert sound
ALERT_VOLUME = 0.7

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FILE = 'output/system.log'

