# config.py
import sys
import os

# --- Version Check ---
MIN_PYTHON_VERSION = (3, 7)

# --- Paths and Filenames ---
DATABASE_NAME = 'posture_tracking.db' # Default, can be overridden by args
REPORTS_DIR = "reports"
MODELS_DIR = "models"
# Use forward slashes or raw string for default path
DEFAULT_VIDEO_PATH = r"C:\Users\surya\OneDrive\Desktop\Capstone Project\Basic Webpage\SafePosAI\input.mp4" # Example using raw string
YOLO_MODEL_NAME = 'yolov8n-pose.pt'

# --- Dependency Availability Flags (Set dynamically in main.py) ---
MP_AVAILABLE = False
YOLO_AVAILABLE = False

# --- Confidence Thresholds ---
YOLO_CONF_THRESHOLD = 0.5       # Person detection confidence
KEYPOINT_CONF_THRESHOLD = 0.5   # Using a keypoint for calculations/drawing
MP_POSE_CONFIDENCE = 0.6        # Min detection confidence for MediaPipe Pose on crops

# --- Tracker Parameters ---
TRACKER_CONF_THRESHOLD = 0.45   # Detection confidence threshold for tracker association
TRACKER_MATCH_THRESHOLD = 0.7   # IoU threshold for matching tracks
TRACKER_BUFFER = 30           # Frames to keep lost tracks before pruning

# --- Risk Calculation Parameters ---
BACK_BEND_MODERATE_THRESHOLD = 20
BACK_BEND_SEVERE_THRESHOLD = 60
NECK_BEND_THRESHOLD = 20
ARM_RAISE_THRESHOLD = 90
RISK_SCORE_NEUTRAL_MAX = 6
RISK_SCORE_STRAIN_MIN = 8

# --- Keypoint Definitions ---
# Map required calculation points (by name/concept) to YOLO (COCO) keypoint indices
KEYPOINT_MAP_FOR_RISK = {
    "NOSE": 0,
    "RIGHT_SHOULDER": 6,
    "RIGHT_ELBOW": 8,
    "RIGHT_HIP": 12,
    "RIGHT_KNEE": 14,
}
YOLO_INDICES_NEEDED = list(KEYPOINT_MAP_FOR_RISK.values())

# --- Skeleton Drawing (COCO Format) ---
COCO_SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 12), (5, 11), (6, 12),  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]
# MediaPipe connections will be handled by mp_drawing if used for visualization

# --- Visualization ---
STATUS_COLORS = {
    "Safe": (0, 255, 0),      # Green
    "Neutral": (0, 255, 255), # Yellow
    "Strain": (255, 0, 0),    # Red
    "Error": (0, 0, 0),        # Black
    "No Calc": (128, 128, 128) # Grey
}
# Add a color for fallback indication
STATUS_COLORS["Safe (YOLO)"] = (0, 180, 0)
STATUS_COLORS["Neutral (YOLO)"] = (50, 180, 180)
STATUS_COLORS["Strain (YOLO)"] = (180, 0, 0)


# --- Smoothing Factors ---
KEYPOINT_SMOOTHING_ALPHA = 0.5
LANDMARK_SMOOTHING_ALPHA = 0.5 # Can still be used if desired on MP results

# --- Reporting ---
MIN_FRAMES_FOR_SUMMARY = 10

# --- Plotting Styles ---
PLOT_STYLE = 'seaborn-v0_8-darkgrid'

# --- Simultaneous Processing Parameters ---
CROP_PADDING_FACTOR = 0.1 # Add 10% padding around bbox for MediaPipe crop

# --- Create Directories ---
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Python Version Check ---
def check_python_version():
    if sys.version_info < MIN_PYTHON_VERSION:
        print(f"Error: Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} or higher is required (You have {sys.version}).")
        sys.exit(1)

check_python_version()