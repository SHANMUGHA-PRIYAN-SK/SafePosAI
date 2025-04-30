# risk_assessment.py
import numpy as np
import traceback

# Attempt import, check availability
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MP_POSE_AVAILABLE = True
except ImportError:
    mp_pose = None
    MP_POSE_AVAILABLE = False

from utils import calculate_angle, MappedLandmark, get_mediapipe_landmark_enum

from config import (
    BACK_BEND_MODERATE_THRESHOLD, BACK_BEND_SEVERE_THRESHOLD,
    NECK_BEND_THRESHOLD, ARM_RAISE_THRESHOLD,
    RISK_SCORE_NEUTRAL_MAX, RISK_SCORE_STRAIN_MIN, STATUS_COLORS,
    KEYPOINT_MAP_FOR_RISK # Use the name-to-index map
)


def calculate_msd_risk(landmarks):
    """
    Calculates MSD risk score, status, and angles based on landmarks.
    Accepts a list where indices correspond to MediaPipe PoseLandmark values,
    containing either MediaPipe Landmark objects or MappedLandmark objects.

    Returns: status (str), color (tuple), score (int), back_angle, neck_angle, arm_angle
    Returns None for angles if calculation fails or landmarks are missing.
    """
    if not landmarks: # Basic check if landmarks list is provided
        return "Error", STATUS_COLORS["Error"], -1, None, None, None
    if not MP_POSE_AVAILABLE: # Cannot proceed without MP landmark definitions
         print("Error: MediaPipe not available, cannot perform risk calculation based on its landmarks.")
         return "Error", STATUS_COLORS["Error"], -1, None, None, None

    try:
        # Helper to get landmark coordinates safely from the list
        def get_coords(landmark_enum):
            if landmark_enum is None: return None # Handle if enum wasn't found
            lm_index = landmark_enum.value
            if lm_index < 0 or lm_index >= len(landmarks): return None # Index out of bounds

            lm = landmarks[lm_index]
            # Check if landmark exists and has sufficient visibility/confidence
            if lm and hasattr(lm, 'visibility') and lm.visibility >= 0.1:
                return np.array([lm.x, lm.y])
            return None

        # Get required landmark enums using the defined names
        required_enums = {name: get_mediapipe_landmark_enum(name) for name in KEYPOINT_MAP_FOR_RISK.keys()}

        # Extract coordinates using the enums
        coords = {name: get_coords(enum) for name, enum in required_enums.items()}

        # Check if all necessary coordinates were extracted
        if any(v is None for v in coords.values()):
            # print("Debug: Missing essential landmark coordinates for risk calculation.")
            return "No Calc", STATUS_COLORS["No Calc"], -1, None, None, None

        # --- Calculate Angles ---
        # Use the extracted coordinates by name
        shoulder = coords["RIGHT_SHOULDER"]
        hip = coords["RIGHT_HIP"]
        knee = coords["RIGHT_KNEE"]
        neck_base = shoulder # Approximate neck base as shoulder
        head = coords["NOSE"] # Use nose as head indicator
        elbow = coords["RIGHT_ELBOW"]

        back_angle_raw = calculate_angle(shoulder, hip, knee)
        # Angle interpretation: Measures angle inside the body at the hip.
        # Straight back relative to legs is ~180. Bending forward decreases this angle.
        # Deviation from straight (180) is a measure of bend.
        back_bend_angle = abs(180 - back_angle_raw)

        neck_angle_raw = calculate_angle(head, neck_base, hip)
        # Angle interpretation: Angle at the shoulder between head-shoulder and hip-shoulder.
        # Smaller angle means head is more forward relative to torso line.
        neck_bend_angle = neck_angle_raw

        arm_angle_raw = calculate_angle(elbow, shoulder, hip)
        # Angle interpretation: Angle at the shoulder between elbow-shoulder and hip-shoulder.
        # Larger angle means arm is raised further away from the torso side.
        arm_raise_angle = arm_angle_raw

        # --- Scoring Logic ---
        score = 1 # Base score for neutral

        # Back Score
        if back_bend_angle > BACK_BEND_SEVERE_THRESHOLD: score += 4
        elif back_bend_angle > BACK_BEND_MODERATE_THRESHOLD: score += 2

        # Neck Score
        if neck_bend_angle > NECK_BEND_THRESHOLD: score += 2 # Check if this interpretation matches RULA/REBA intent

        # Arm Score
        if arm_raise_angle > ARM_RAISE_THRESHOLD: score += 2

        # --- Determine Status and Color ---
        if score <= RISK_SCORE_NEUTRAL_MAX:
            status = "Safe"
        elif score < RISK_SCORE_STRAIN_MIN:
            status = "Neutral"
        else:
            status = "Strain"

        color = STATUS_COLORS[status]

        # Return the calculated *bend/raise* angles for reporting clarity
        return status, color, score, back_bend_angle, neck_bend_angle, arm_raise_angle

    except Exception as e:
        print(f"Error calculating MSD risk: {e}")
        print(traceback.format_exc())
        return "Error", STATUS_COLORS["Error"], -1, None, None, None


def get_posture_suggestions(status, back_angle, neck_angle, arm_angle):
    """Generates posture correction suggestions based on angles and status."""
    suggestions = []
    if status == "Error" or status == "No Calc":
        return ["Posture could not be reliably assessed."]
    # Check if angles are valid numbers
    if back_angle is None or neck_angle is None or arm_angle is None:
         return ["Angle calculation failed, cannot provide suggestions."]


    if back_angle > BACK_BEND_SEVERE_THRESHOLD:
        suggestions.append(f"Severe back bend ({back_angle:.0f}°). Aim for < {BACK_BEND_MODERATE_THRESHOLD}°. Straighten back significantly.")
    elif back_angle > BACK_BEND_MODERATE_THRESHOLD:
        suggestions.append(f"Moderate back bend ({back_angle:.0f}°). Keep back straighter (< {BACK_BEND_MODERATE_THRESHOLD}°).")

    if neck_angle > NECK_BEND_THRESHOLD:
        suggestions.append(f"Neck bent forward ({neck_angle:.0f}°). Align head with spine (< {NECK_BEND_THRESHOLD}°).")

    if arm_angle > ARM_RAISE_THRESHOLD:
        suggestions.append(f"Arm raised high ({arm_angle:.0f}°). Lower arm (< {ARM_RAISE_THRESHOLD}°).")

    if status == "Strain" and not suggestions:
         suggestions.append("High risk posture detected (check overall position).")
    elif status == "Neutral" and not suggestions:
        suggestions.append("Posture acceptable, but monitor for prolonged duration.")
    elif status == "Safe":
        suggestions.append("Good posture maintained.")

    # Default if no specific issues but status indicates risk
    if not suggestions and status != "Safe":
        suggestions.append("Review overall posture for potential risks.")
    elif not suggestions and status == "Safe": # Ensure safe always gets a positive message
         suggestions.append("Good posture maintained.")


    return suggestions


def get_ergonomic_remedies(safe_percent, neutral_percent, strain_percent):
    """Generates ergonomic remedies based on overall posture distribution."""
    remedies = []
    # Use slightly adjusted thresholds for broader advice categories
    if strain_percent > 40: # High risk threshold
        remedies.append("High Strain (>40%): Critical risk. Immediate action needed.")
        remedies.append("-> Use mechanical aids (hoists, lifts) for heavy materials (e.g., formwork).")
        remedies.append("-> Implement mandatory frequent breaks (e.g., 5-10 min every hour) with stretching.")
        remedies.append("-> Re-evaluate task design: Adjust work heights, eliminate unnecessary lifting.")
        remedies.append("-> Consider job rotation to less strenuous tasks.")
        remedies.append("-> Provide/enforce use of appropriate PPE (e.g., back support - with training).")
    elif strain_percent > 20: # Moderate risk threshold
        remedies.append("Moderate Strain (>20%): Significant risk. Action recommended.")
        remedies.append("-> Ensure adjustable work platforms/scaffolding are used effectively.")
        remedies.append("-> Encourage team lifting and use of dollies/carts.")
        remedies.append("-> Schedule regular micro-breaks (1-2 min every 30 min) for posture change.")
        remedies.append("-> Provide training on proper body mechanics for specific tasks.")
        remedies.append("-> Ensure ergonomic tools are available and maintained.")
    elif neutral_percent + strain_percent > 60: # High amount of non-safe postures
        remedies.append("Frequent Non-Safe Postures: Monitor closely.")
        remedies.append("-> Check for static load issues (holding awkward positions).")
        remedies.append("-> Provide anti-fatigue mats or cushioned knee pads where applicable.")
        remedies.append("-> Promote awareness of posture variation throughout the shift.")
    else: # High percentage of Safe
        remedies.append("Good Ergonomic Practice: Mostly safe postures observed.")
        remedies.append("-> Continue promoting safe work habits and proper tool use.")
        remedies.append("-> Regularly review work processes for continuous improvement.")
        remedies.append("-> Encourage pre-shift warm-ups and cool-down stretches.")

    # Add a general recommendation applicable to most construction scenarios
    remedies.append("-> Ensure adequate lighting and clear walkways on site.")

    return remedies