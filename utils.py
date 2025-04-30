# utils.py
import numpy as np
import traceback

# Attempt to import mediapipe, but don't fail if it's not there initially
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MP_POSE_AVAILABLE = True
except ImportError:
    # print("Warning: MediaPipe not found in utils.py. Some functionality might be limited.")
    mp_pose = None
    MP_POSE_AVAILABLE = False

from config import (
    KEYPOINT_CONF_THRESHOLD, KEYPOINT_MAP_FOR_RISK,
    KEYPOINT_SMOOTHING_ALPHA, LANDMARK_SMOOTHING_ALPHA
)

# Simple class to mimic MediaPipe Landmark structure for mapped YOLO points
class MappedLandmark:
    """Represents a keypoint normalized and structured like a MediaPipe Landmark."""
    def __init__(self, x, y, z=0, visibility=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility # Store confidence here

# KeyPoint Smoother for YOLO keypoints (Exponential Moving Average)
class KeypointSmoother:
    """Applies Exponential Moving Average smoothing to keypoint coordinates."""
    def __init__(self, alpha=KEYPOINT_SMOOTHING_ALPHA):
        self.alpha = alpha
        self.prev_keypoints = None

    def smooth(self, keypoints_xyc): # Expects list/array of [x, y, confidence]
        if keypoints_xyc is None or len(keypoints_xyc) == 0:
            self.prev_keypoints = None
            return None

        keypoints_xyc = np.array(keypoints_xyc) # Ensure numpy array

        if self.prev_keypoints is None or self.prev_keypoints.shape != keypoints_xyc.shape:
            self.prev_keypoints = keypoints_xyc.copy()
            return self.prev_keypoints.tolist() # Return as list

        smoothed_keypoints = np.zeros_like(keypoints_xyc)

        for i, (x, y, conf) in enumerate(keypoints_xyc):
            prev_x, prev_y, prev_conf = self.prev_keypoints[i]
            # Apply EMA only if current point is confident and previous point exists
            if conf >= KEYPOINT_CONF_THRESHOLD and prev_conf >= 0: # Use >= 0 for prev_conf check
                new_x = self.alpha * x + (1 - self.alpha) * prev_x
                new_y = self.alpha * y + (1 - self.alpha) * prev_y
                new_conf = conf # Keep current confidence
            else: # If low confidence or no previous, use current point
                new_x, new_y, new_conf = x, y, conf

            smoothed_keypoints[i] = [new_x, new_y, new_conf]

        self.prev_keypoints = smoothed_keypoints.copy()
        return smoothed_keypoints.tolist() # Return as list

# (Optional) Landmark Smoother for MediaPipe results if extra smoothing is desired
class LandmarkSmoother:
    """Applies EMA smoothing to MediaPipe Landmark objects."""
    def __init__(self, alpha=LANDMARK_SMOOTHING_ALPHA):
        if not MP_POSE_AVAILABLE:
             raise ImportError("MediaPipe is required for LandmarkSmoother but not found.")
        self.alpha = alpha
        self.prev_landmarks_xyz = None # Store as list of (x, y, z) tuples

    def smooth(self, landmark_list): # Expects list of MediaPipe Landmark objects
        if not landmark_list:
            self.prev_landmarks_xyz = None
            return None

        current_landmarks_xyz = [(lm.x, lm.y, lm.z) for lm in landmark_list]

        if self.prev_landmarks_xyz is None or len(self.prev_landmarks_xyz) != len(current_landmarks_xyz):
            self.prev_landmarks_xyz = current_landmarks_xyz
            return landmark_list # Return original on first valid frame

        smoothed_landmarks = []
        new_prev_xyz = []
        for i, current_lm in enumerate(landmark_list):
            prev_x, prev_y, prev_z = self.prev_landmarks_xyz[i]
            curr_x, curr_y, curr_z = current_landmarks_xyz[i]

            new_x = self.alpha * curr_x + (1 - self.alpha) * prev_x
            new_y = self.alpha * curr_y + (1 - self.alpha) * prev_y
            new_z = self.alpha * curr_z + (1 - self.alpha) * prev_z

            # Create a new landmark object
            # Requires mediapipe to be imported
            smoothed_lm = mp.framework.formats.landmark_pb2.NormalizedLandmark(
                x=new_x, y=new_y, z=new_z,
                visibility=current_lm.visibility # Keep original visibility
            )
            smoothed_landmarks.append(smoothed_lm)
            new_prev_xyz.append((new_x, new_y, new_z))

        self.prev_landmarks_xyz = new_prev_xyz
        return smoothed_landmarks


def get_mediapipe_landmark_enum(landmark_name):
    """Safely gets the MediaPipe PoseLandmark enum value by name."""
    if not MP_POSE_AVAILABLE:
        # print(f"Warning: Cannot get MediaPipe landmark '{landmark_name}', MediaPipe not available.")
        return None
    try:
        return getattr(mp_pose.PoseLandmark, landmark_name)
    except AttributeError:
        # print(f"Warning: MediaPipe PoseLandmark '{landmark_name}' not found.")
        return None

def map_yolo_to_mediapipe(yolo_keypoints_xyc, width, height):
    """
    Maps YOLO keypoints (COCO format [x, y, conf]) to a MediaPipe-like landmark list.
    Returns a list of MappedLandmark objects or None if essential keypoints are missing/low confidence.
    """
    if not MP_POSE_AVAILABLE:
        # print("Warning: Cannot map to MediaPipe structure, MediaPipe not available.")
        # Return a basic structure or handle differently if needed without MP
        return None # Or potentially return a simplified dict if MP isn't strictly needed downstream

    if yolo_keypoints_xyc is None or len(yolo_keypoints_xyc) < 17:
        return None

    landmarks = [None] * 33  # MediaPipe has 33 landmarks

    required_landmarks_missing = False
    for landmark_name, yolo_idx in KEYPOINT_MAP_FOR_RISK.items():
        mp_landmark_enum = get_mediapipe_landmark_enum(landmark_name)
        if mp_landmark_enum is None:
            # print(f"Skipping mapping for {landmark_name} due to missing enum.")
            required_landmarks_missing = True # If a required landmark enum doesn't exist, fail mapping
            continue # Skip this landmark

        mp_idx = mp_landmark_enum.value

        try:
            if yolo_idx >= len(yolo_keypoints_xyc):
                # print(f"Warning: YOLO index {yolo_idx} for {landmark_name} out of bounds.")
                required_landmarks_missing = True
                continue

            x_pix, y_pix, conf = yolo_keypoints_xyc[yolo_idx]

            if conf >= KEYPOINT_CONF_THRESHOLD:
                norm_x = np.clip(x_pix / width, 0.0, 1.0)
                norm_y = np.clip(y_pix / height, 0.0, 1.0)
                landmarks[mp_idx] = MappedLandmark(norm_x, norm_y, visibility=conf)
            else:
                # print(f"Debug: Low confidence ({conf:.2f}) for {landmark_name} (YOLO idx {yolo_idx}).")
                required_landmarks_missing = True # Mark as missing if below threshold

        except (IndexError, ValueError, TypeError) as e:
            # print(f"Warning: Error processing {landmark_name} (YOLO idx {yolo_idx}): {e}")
            required_landmarks_missing = True
            continue

    if required_landmarks_missing:
        # print("Debug: One or more required landmarks were missing or had low confidence.")
        return None

    return landmarks


def calculate_angle(p1_xy, p2_xy, p3_xy):
    """
    Calculates the angle (in degrees) between three 2D points p1, p2, p3 (angle at p2).
    Returns 0.0 if points are collinear or calculation fails.
    """
    # Input validation
    if p1_xy is None or p2_xy is None or p3_xy is None:
        # print("Warning: Received None for point coordinates in calculate_angle.")
        return 0.0

    try:
        p1 = np.array(p1_xy, dtype=float)
        p2 = np.array(p2_xy, dtype=float)
        p3 = np.array(p3_xy, dtype=float)

        # Check for identical points which would lead to zero vectors
        if np.array_equal(p1, p2) or np.array_equal(p3, p2):
            # print("Warning: Identical points provided to calculate_angle (p1=p2 or p3=p2).")
            return 0.0 # Or handle as 180 if appropriate for the context

        vector1 = p1 - p2
        vector2 = p3 - p2

        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            # This case should be caught by the identical point check above, but kept as safeguard
            # print("Warning: Zero vector encountered in angle calculation.")
            return 0.0

        dot_product = np.dot(vector1, vector2)
        # Prevent floating point errors from causing domain errors in arccos
        cosine_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)

        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    except Exception as e:
        print(f"Error calculating angle between {p1_xy}, {p2_xy}, {p3_xy}: {e}")
        print(traceback.format_exc())
        return 0.0 # Return neutral angle on error