# video_processor.py
import os
import sys
import time
import cv2
import numpy as np
import traceback
import sqlite3
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
import tkinter as tk # Import tkinter for GUI check

# --- Configuration ---
from config import (
    YOLO_MODEL_NAME, YOLO_CONF_THRESHOLD, KEYPOINT_CONF_THRESHOLD,
    COCO_SKELETON_CONNECTIONS, STATUS_COLORS, DATABASE_NAME,
    MIN_FRAMES_FOR_SUMMARY # Import summary threshold
)

# --- Utilities ---
from utils import (
    map_yolo_to_mediapipe, KeypointSmoother, LandmarkSmoother, MappedLandmark
)
from tracker import CustomTracker
from risk_assessment import calculate_msd_risk, get_posture_suggestions
from database import init_database
from gui import PostureGUI # Import GUI class
from reporting import generate_reports # Import the main reporting function

# --- Conditional Imports ---
try:
    from ultralytics import YOLO
    import supervision as sv
    from supervision.draw.color import ColorPalette
    from supervision.detection.annotate import BoxAnnotator, LabelAnnotator
    from supervision.detection.core import Detections
    from supervision.geometry.core import Position # For label position
    YOLO_INSTALLED = True
except ImportError:
    YOLO_INSTALLED = False
    YOLO = None
    sv = None
    # Define dummy classes if needed for type hinting or structure
    BoxAnnotator = LabelAnnotator = ColorPalette = Detections = Position = None
    print("Warning: YOLO/Supervision not installed. YOLO processing disabled.")

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    MP_INSTALLED = True
except ImportError:
    MP_INSTALLED = False
    mp_pose = None
    mp_drawing = None
    print("Warning: MediaPipe not installed. MediaPipe processing disabled.")


def process_video(video_path, ground_truth=None, use_yolo=True, db_name=DATABASE_NAME, show_gui=True):
    """
    Processes the video for posture analysis using YOLO or MediaPipe fallback.

    Args:
        video_path (str): Path to the video file.
        ground_truth (dict, optional): Dictionary with ground truth data per frame.
        use_yolo (bool): Flag to attempt using YOLO if available.
        db_name (str): Name of the database file.
        show_gui (bool): Whether to display the Tkinter GUI.

    Returns:
        dict: Summary tracking data including session info, reports, and metrics.
              Returns None if processing fails critically.
    """
    print(f"--- Starting Video Processing: {Path(video_path).name} ---")
    # Determine effective processing mode based on availability and preference
    effective_use_yolo = use_yolo and YOLO_INSTALLED
    use_mediapipe_fallback = not effective_use_yolo and MP_INSTALLED

    if not effective_use_yolo and not use_mediapipe_fallback:
        print("Error: Cannot process video. Neither YOLO/Supervision nor MediaPipe is available/enabled.")
        return None

    print(f"Processing Mode: {'YOLO' if effective_use_yolo else 'MediaPipe'}")

    # --- Initialization ---
    conn = init_database(db_name)
    if conn is None:
        print("Error: Database initialization failed. Exiting.")
        return None
    cursor = conn.cursor()
    session_id = None
    root = None
    gui = None
    cap = None

    # Data storage for the session
    person_data = defaultdict(lambda: {
        "risk_scores": deque(), "back_angles": deque(), "neck_angles": deque(),
        "arm_angles": deque(), "statuses": deque(), "frames": [], "timestamps": [],
        # Initialize smoother based on the *actually used* method
        "smoother": KeypointSmoother() if effective_use_yolo else (LandmarkSmoother() if use_mediapipe_fallback else None),
        "last_seen_frame": 0
    })
    detected_persons_in_session = set()

    # Validation metrics storage (collected during processing)
    validation_metrics_live = {
        'total_frames_processed': 0, 'frames_with_detections': 0,
        'frames_with_risk_calculation': 0, 'risk_predictions': [],
        'risk_ground_truth': [], 'angle_errors': {'back': [], 'neck': [], 'arm': []},
        'keypoint_precision': [], 'processing_times': [],
        'avg_keypoint_confidence': []
    }

    try:
        # --- Session Setup ---
        session_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO person_sessions (session_timestamp, video_path) VALUES (?, ?)',
                       (session_timestamp, str(Path(video_path).resolve()))) # Store absolute path
        session_id = cursor.lastrowid
        conn.commit()
        print(f"Created session ID: {session_id}")

        # --- Video Capture ---
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error: Could not open video file: {video_path}")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video Info: {frame_width}x{frame_height} @ {fps:.2f} FPS")

        # --- Model & Tool Initialization ---
        model = None
        tracker = None
        box_annotator = None
        label_annotator = None
        pose_model_mp = None # MediaPipe model instance

        if effective_use_yolo:
            try:
                model = YOLO(YOLO_MODEL_NAME)
                tracker = CustomTracker()
                box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.4)
                # Use default color palette or customize
                color_palette = sv.ColorPalette.default()
                label_annotator = sv.LabelAnnotator(text_scale=0.4, text_thickness=1,
                                                    text_position=sv.Position.TOP_LEFT, color=color_palette)
                print("YOLO and Supervision components initialized.")
            except Exception as e:
                print(f"Error initializing YOLO/Supervision components: {e}")
                print(traceback.format_exc())
                if not use_mediapipe_fallback: # If no fallback, raise error
                     raise RuntimeError("YOLO initialization failed and MediaPipe fallback is not available.") from e
                print("Falling back to MediaPipe.")
                effective_use_yolo = False # Force fallback

        if not effective_use_yolo and use_mediapipe_fallback:
             try:
                 pose_model_mp = mp_pose.Pose(
                     min_detection_confidence=0.6, # Adjust confidence as needed
                     min_tracking_confidence=0.6
                 )
                 print("MediaPipe Pose model initialized.")
             except Exception as e:
                  raise RuntimeError(f"Failed to initialize MediaPipe Pose model: {e}") from e


        # --- GUI Setup ---
        if show_gui:
            try:
                # Run GUI in the main thread (can cause blocking)
                # For non-blocking GUI, would need threading/multiprocessing
                root = tk.Tk()
                gui = PostureGUI(root)
                root.update() # Initial draw
                print("GUI Initialized.")
            except Exception as e:
                print(f"Warning: Failed to initialize GUI: {e}. Running without GUI.")
                gui = None
                if root: root.destroy() # Clean up if partially created
                root = None


        # --- Main Processing Loop ---
        frame_count = 0
        while True:
            if gui and gui.is_closing:
                print("GUI closed, stopping analysis.")
                break

            frame_start_time = time.time()

            try:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached or read error.")
                    break

                frame_count += 1
                validation_metrics_live['total_frames_processed'] += 1
                current_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                annotated_frame = frame.copy()
                persons_in_frame_data = [] # Data for GUI update
                processed_persons_in_frame = set() # Track IDs processed this frame

                # --- Pose Estimation ---
                if effective_use_yolo:
                    # Process with YOLO
                    yolo_results = model(frame, verbose=False)[0] # Process frame, get first result object

                    # Convert results to Supervision Detections
                    try:
                        detections = Detections.from_ultralytics(yolo_results)
                    except Exception as e_conv:
                        print(f"Warning: Failed sv.Detections.from_ultralytics: {e_conv}. Trying manual.")
                        # Manual fallback conversion (ensure correct parsing of yolo_results.boxes)
                        try:
                           boxes_data = yolo_results.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, class_id]
                           if boxes_data.shape[1] >= 6:
                                detections = Detections(
                                    xyxy=boxes_data[:, :4],
                                    confidence=boxes_data[:, 4],
                                    class_id=boxes_data[:, 5].astype(int)
                                )
                           else: # Handle cases with missing class_id
                                detections = Detections(
                                    xyxy=boxes_data[:,:4],
                                    confidence=boxes_data[:,4]
                                )
                                # Assign default class_id if missing
                                detections.class_id = np.zeros(len(detections), dtype=int)
                        except Exception as e_manual:
                            print(f"Error: Failed manual conversion of YOLO results: {e_manual}")
                            detections = Detections.empty() # Continue with empty


                    # Filter for persons (class_id 0 in COCO) and confidence
                    person_detections = detections[(detections.class_id == 0) & (detections.confidence >= YOLO_CONF_THRESHOLD)]

                    if len(person_detections) > 0:
                        validation_metrics_live['frames_with_detections'] += 1

                        # Update tracker
                        try:
                            tracked_detections = tracker.update(detections=person_detections)
                        except Exception as e_track:
                            print(f"Error during tracker update: {e_track}")
                            print(traceback.format_exc())
                            tracked_detections = Detections.empty() # Handle tracker failure


                        # Extract keypoints for ALL original person detections
                        yolo_keypoints_all = None
                        try:
                            if hasattr(yolo_results, 'keypoints') and yolo_results.keypoints is not None:
                                 # Check common attributes for keypoint data
                                 if hasattr(yolo_results.keypoints, 'data') and yolo_results.keypoints.data is not None:
                                     yolo_keypoints_all = yolo_results.keypoints.data.cpu().numpy() # Shape (n_dets, 17, 3) [x,y,conf]
                                 elif hasattr(yolo_results.keypoints, 'xy') and yolo_results.keypoints.xy is not None:
                                     # Handle format with separate xy and conf if needed
                                     kp_xy = yolo_results.keypoints.xy.cpu().numpy()
                                     kp_conf = None
                                     if hasattr(yolo_results.keypoints, 'conf') and yolo_results.keypoints.conf is not None:
                                         kp_conf = yolo_results.keypoints.conf.cpu().numpy().reshape(kp_xy.shape[0], kp_xy.shape[1], 1)
                                     else: # Default confidence if not available
                                         kp_conf = np.ones((kp_xy.shape[0], kp_xy.shape[1], 1))
                                     yolo_keypoints_all = np.concatenate([kp_xy, kp_conf], axis=2)

                            if yolo_keypoints_all is None:
                                 print(f"Warning: Frame {frame_count}: Could not extract keypoints from YOLO results.")

                        except Exception as e_kp:
                             print(f"Error extracting keypoints: {e_kp}")
                             yolo_keypoints_all = None


                        labels = [] # For supervision annotation

                        # Match tracked detections back to original detections to get keypoints
                        # Using simple greedy matching based on IoU for association
                        matches = {} # track_idx -> det_idx
                        used_dets = set()
                        if len(tracked_detections) > 0 and len(person_detections) > 0:
                             for t_idx in range(len(tracked_detections)):
                                 best_iou = -1
                                 best_d_idx = -1
                                 for d_idx in range(len(person_detections)):
                                     if d_idx in used_dets: continue
                                     # Ensure tracker._iou exists and handles potential errors
                                     try:
                                         iou = tracker._iou(tracked_detections.xyxy[t_idx], person_detections.xyxy[d_idx])
                                     except Exception as iou_err:
                                         # print(f"Warning: IoU calculation error: {iou_err}")
                                         iou = 0.0 # Treat as no overlap on error

                                     if iou >= tracker.match_thresh and iou > best_iou:
                                         best_iou = iou
                                         best_d_idx = d_idx
                                 if best_d_idx != -1:
                                     matches[t_idx] = best_d_idx
                                     used_dets.add(best_d_idx)


                        # Process each *tracked* person
                        for track_idx in range(len(tracked_detections)):
                            try: # Add try-except around each person's processing
                                tracker_id = int(tracked_detections.tracker_id[track_idx])
                                bbox = tracked_detections.xyxy[track_idx]
                                processed_persons_in_frame.add(tracker_id)
                                detected_persons_in_session.add(tracker_id)

                                # Get the corresponding original detection index
                                original_det_idx = matches.get(track_idx, -1) # Get matched original index

                                keypoints_xyc_raw = None
                                if original_det_idx != -1 and yolo_keypoints_all is not None and original_det_idx < len(yolo_keypoints_all):
                                    keypoints_xyc_raw = yolo_keypoints_all[original_det_idx]

                                # Initialize defaults
                                status, color, score = "No Calc", STATUS_COLORS["No Calc"], -1
                                back_angle, neck_angle, arm_angle = None, None, None
                                suggestions = ["Ensure clear view for keypoint detection."]
                                label = f"P:{tracker_id} No Keypoints"

                                if keypoints_xyc_raw is not None:
                                    # Smooth keypoints
                                    smoother = person_data[tracker_id]["smoother"]
                                    smoothed_keypoints_xyc = smoother.smooth(keypoints_xyc_raw) if smoother else keypoints_xyc_raw

                                    if smoothed_keypoints_xyc: # Check if smoothing returned valid data
                                        # Map to MediaPipe structure
                                        mapped_landmarks = map_yolo_to_mediapipe(smoothed_keypoints_xyc, frame_width, frame_height)

                                        if mapped_landmarks:
                                            # Calculate risk
                                            status, color, score, back_angle, neck_angle, arm_angle = calculate_msd_risk(mapped_landmarks)

                                            if status != "Error" and status != "No Calc":
                                                validation_metrics_live['frames_with_risk_calculation'] += 1
                                                # Store data
                                                p_data = person_data[tracker_id]
                                                p_data["risk_scores"].append(score)
                                                p_data["statuses"].append(status)
                                                p_data["back_angles"].append(back_angle)
                                                p_data["neck_angles"].append(neck_angle)
                                                p_data["arm_angles"].append(arm_angle)
                                                p_data["frames"].append(frame_count)
                                                p_data["timestamps"].append(current_timestamp_str)
                                                p_data["last_seen_frame"] = frame_count

                                                # Add data to DB
                                                cursor.execute('''
                                                    INSERT INTO posture_data (session_id, person_tracker_id, frame_number, timestamp, risk_score, status, back_angle, neck_angle, arm_angle)
                                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                                    (session_id, tracker_id, frame_count, current_timestamp_str, score, status, back_angle, neck_angle, arm_angle))
                                                # Commit periodically
                                                if frame_count % 50 == 0: conn.commit()

                                                # Get suggestions
                                                suggestions = get_posture_suggestions(status, back_angle, neck_angle, arm_angle)

                                                # --- Validation data collection ---
                                                if ground_truth and str(frame_count) in ground_truth:
                                                    gt_frame = ground_truth[str(frame_count)]
                                                    risk_map = {"Safe": 0, "Neutral": 1, "Strain": 2, "Error": -1, "No Calc": -1}
                                                    if 'risk_level' in gt_frame:
                                                        validation_metrics_live['risk_ground_truth'].append(gt_frame['risk_level'])
                                                        validation_metrics_live['risk_predictions'].append(risk_map.get(status, -1))
                                                    # Add angle error calculation if GT provides angles
                                                    # if 'back_angle' in gt_frame and back_angle is not None:
                                                    #     validation_metrics_live['angle_errors']['back'].append(abs(back_angle - gt_frame['back_angle']))
                                                    # ... etc for neck, arm ...

                                                # Avg confidence of used keypoints
                                                used_kpt_conf = [c for x, y, c in smoothed_keypoints_xyc if c >= KEYPOINT_CONF_THRESHOLD]
                                                if used_kpt_conf:
                                                     validation_metrics_live['avg_keypoint_confidence'].append(np.mean(used_kpt_conf))
                                                # --- End Validation ---


                                            # Draw Skeleton Manually
                                            for pt1_idx, pt2_idx in COCO_SKELETON_CONNECTIONS:
                                                # Check bounds for smoothed_keypoints_xyc
                                                if pt1_idx < len(smoothed_keypoints_xyc) and pt2_idx < len(smoothed_keypoints_xyc):
                                                    pt1_x, pt1_y, pt1_conf = smoothed_keypoints_xyc[pt1_idx]
                                                    pt2_x, pt2_y, pt2_conf = smoothed_keypoints_xyc[pt2_idx]
                                                    # Draw connection only if both points are confident enough
                                                    if pt1_conf >= KEYPOINT_CONF_THRESHOLD and pt2_conf >= KEYPOINT_CONF_THRESHOLD:
                                                        # Use integer coordinates for drawing
                                                        cv2.line(annotated_frame, (int(pt1_x), int(pt1_y)), (int(pt2_x), int(pt2_y)), color, 1) # Thinner line
                                            # Optional: Draw keypoints
                                            # for x, y, conf in smoothed_keypoints_xyc:
                                            #     if conf >= KEYPOINT_CONF_THRESHOLD:
                                            #         cv2.circle(annotated_frame, (int(x), int(y)), 2, color, -1)

                                        else: # mapped_landmarks is None
                                            status, color, score = "No Calc", STATUS_COLORS["No Calc"], -1
                                            suggestions = ["Insufficient keypoints for analysis."]

                                    else: # Smoothed keypoints were None
                                         status, color, score = "No Calc", STATUS_COLORS["No Calc"], -1
                                         suggestions = ["Keypoint smoothing failed."]

                                else: # Raw keypoints were None
                                     pass # Defaults already set

                                label = f"P:{tracker_id} {status} ({score if score != -1 else 'N/A'})"


                                # Store data for GUI update for this person
                                persons_in_frame_data.append({
                                    "id": tracker_id, "status": status, "score": score,
                                    "suggestions": suggestions, "bbox": bbox, "color": color
                                })
                                labels.append(label) # Add label for Supervision annotator

                            except Exception as person_err:
                                 print(f"Error processing person {tracker_id} in frame {frame_count}: {person_err}")
                                 print(traceback.format_exc())
                                 # Add a placeholder label if processing failed for this person
                                 labels.append(f"P:{tracker_id} Error")
                                 continue # Skip to next person


                        # Annotate frame with boxes and labels using Supervision
                        if tracked_detections and labels:
                             # Ensure labels list length matches tracked_detections length
                             if len(labels) == len(tracked_detections):
                                try:
                                    annotated_frame = box_annotator.annotate(
                                        scene=annotated_frame, detections=tracked_detections
                                    )
                                    annotated_frame = label_annotator.annotate(
                                        scene=annotated_frame, detections=tracked_detections, labels=labels
                                    )
                                except Exception as e_annotate:
                                    print(f"Warning: Error during Supervision annotation: {e_annotate}")
                             else:
                                  print(f"Warning: Mismatch between labels ({len(labels)}) and tracked detections ({len(tracked_detections)}) in frame {frame_count}. Skipping label annotation.")
                                  # Annotate only boxes if labels mismatch
                                  try:
                                       annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                                  except Exception as e_box_annotate:
                                       print(f"Warning: Error during Supervision box annotation: {e_box_annotate}")


                # --- MediaPipe Fallback Processing ---
                elif use_mediapipe_fallback and pose_model_mp:
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_rgb.flags.writeable = False # Performance opt
                    results = pose_model_mp.process(image_rgb)
                    image_rgb.flags.writeable = True

                    if results.pose_landmarks:
                        validation_metrics_live['frames_with_detections'] += 1
                        # Simple single-person tracking for MediaPipe
                        person_id = 1 # Assume person 1
                        processed_persons_in_frame.add(person_id)
                        detected_persons_in_session.add(person_id)

                        # Optional extra smoothing
                        landmarks_to_use = results.pose_landmarks.landmark
                        smoother = person_data[person_id]["smoother"]
                        if smoother:
                             smoothed_landmarks = smoother.smooth(landmarks_to_use)
                             if smoothed_landmarks: # Check if smoothing was successful
                                  landmarks_to_use = smoothed_landmarks

                        # Calculate risk (pass the list/iterable of landmarks)
                        status, color, score, back_angle, neck_angle, arm_angle = calculate_msd_risk(landmarks_to_use)

                        if status != "Error" and status != "No Calc":
                            validation_metrics_live['frames_with_risk_calculation'] += 1
                            # Store data
                            p_data = person_data[person_id]
                            p_data["risk_scores"].append(score)
                            p_data["statuses"].append(status)
                            p_data["back_angles"].append(back_angle)
                            p_data["neck_angles"].append(neck_angle)
                            p_data["arm_angles"].append(arm_angle)
                            p_data["frames"].append(frame_count)
                            p_data["timestamps"].append(current_timestamp_str)
                            p_data["last_seen_frame"] = frame_count

                            # Add to DB
                            cursor.execute('''
                                INSERT INTO posture_data (session_id, person_tracker_id, frame_number, timestamp, risk_score, status, back_angle, neck_angle, arm_angle)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                (session_id, person_id, frame_count, current_timestamp_str, score, status, back_angle, neck_angle, arm_angle))
                            if frame_count % 50 == 0: conn.commit()

                            suggestions = get_posture_suggestions(status, back_angle, neck_angle, arm_angle)

                            # Draw MediaPipe landmarks
                            if mp_drawing:
                                mp_drawing.draw_landmarks(
                                    annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=1)
                                )
                            cv2.putText(annotated_frame, f"P:{person_id} {status} ({score})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                            # Store data for GUI
                            persons_in_frame_data.append({
                                "id": person_id, "status": status, "score": score,
                                "suggestions": suggestions, "bbox": None, "color": color
                            })
                        else:
                             # Handle No Calc / Error case for MediaPipe
                             cv2.putText(annotated_frame, f"P:{person_id} {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                             persons_in_frame_data.append({
                                "id": person_id, "status": status, "score": score,
                                "suggestions": ["Could not calculate risk."], "bbox": None, "color": color
                            })

                # --- Update GUI ---
                if gui and not gui.is_closing:
                    highest_risk_person_data = None
                    if persons_in_frame_data:
                         try:
                              # Find person with highest score in the current frame
                              highest_risk_person_data = max(persons_in_frame_data, key=lambda p: p.get('score', -1))
                         except ValueError: # Handle empty list case
                              highest_risk_person_data = None

                    if highest_risk_person_data:
                        gui.update(
                            highest_risk_person_data["id"],
                            highest_risk_person_data["status"],
                            highest_risk_person_data["score"],
                            highest_risk_person_data["suggestions"]
                        )
                    else:
                         # Update GUI to show N/A if no persons processed or found
                         gui.update(None, "N/A", "N/A", [])

                # --- Display Frame ---
                # Add frame number and detected persons count to the frame
                cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(annotated_frame, f"Persons: {len(processed_persons_in_frame)}", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                # Display the annotated frame
                cv2.imshow('MSD Posture Analysis', annotated_frame)

                # --- Frame Timing & Exit ---
                frame_end_time = time.time()
                processing_time = frame_end_time - frame_start_time
                validation_metrics_live['processing_times'].append(processing_time)
                # waitKey(1) allows frame display and checks for key press without significant delay
                wait_time = 1

                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'):
                    print("'q' pressed, stopping analysis.")
                    break
                # Check GUI status again after waitKey, as it might be closed during the wait
                if gui and gui.is_closing:
                    print("GUI closed detected after waitKey, stopping.")
                    break

            except Exception as loop_err:
                print(f"\n!!! Error during processing loop (Frame {frame_count}) !!!")
                print(f"Error: {loop_err}")
                print(traceback.format_exc())
                # Option: break or continue
                user_choice = input("An error occurred. Type 'c' to continue, 'q' to quit: ").lower()
                if user_choice == 'q':
                    break
                print("Attempting to continue...")
                time.sleep(0.1) # Small pause before next frame attempt

        # --- End of Processing Loop ---
        print("Video processing loop finished.")

    except (IOError, RuntimeError, Exception) as main_err:
        print("\n!!! Critical Error during video processing setup or execution !!!")
        print(f"Error: {main_err}")
        print(traceback.format_exc())
        # Ensure cleanup happens in finally block
        session_id = None # Mark session as potentially incomplete/failed
        final_results = None # Indicate failure
    finally:
        # --- Cleanup ---
        print("Cleaning up resources...")
        if cap is not None and cap.isOpened():
            cap.release()
            print("Video capture released.")
        cv2.destroyAllWindows()
        print("OpenCV windows destroyed.")
        # Ensure GUI is properly handled if it exists
        if root and show_gui: # Check if root was created
             try:
                 # Check if the GUI object exists and hasn't been closed yet
                 if gui and not gui.is_closing:
                     print("Attempting to close GUI...")
                     gui.on_closing() # Use the closing method
                 elif not gui: # If GUI object creation failed but root exists
                      root.destroy()
                      print("Destroyed Tkinter root.")
             except Exception as e_gui_close:
                  print(f"Error during final GUI close/destroy: {e_gui_close}")
        # Close DB connection
        if conn:
            try:
                conn.commit() # Final commit
                conn.close()
                print("Database connection closed.")
            except Exception as db_close_err:
                 print(f"Error closing database connection: {db_close_err}")

    # --- Post-Processing: Reporting (only if session was successfully created) ---
    if session_id is not None:
        # Call the reporting function from reporting.py
        final_results = generate_reports(
            session_id=session_id,
            session_timestamp=session_timestamp, # Use timestamp from start of processing
            video_path=video_path,
            person_data=person_data, # Pass the collected raw data
            validation_data=validation_metrics_live, # Pass collected metrics
            ground_truth_data=ground_truth,
            db_name=db_name
        )
    else:
         print("Error: Session ID not available. Skipping report generation.")
         final_results = None


    print("--- Processing Complete ---")
    return final_results # Return the dictionary generated by generate_reports or None