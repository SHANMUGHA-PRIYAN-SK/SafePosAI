# tracker.py
import numpy as np

# Attempt to import supervision, handle failure gracefully if needed elsewhere
try:
    import supervision as sv
    from supervision.detection.core import Detections # Explicit import
    SV_AVAILABLE = True
except ImportError:
    # print("Warning: Supervision library not found in tracker.py.")
    SV_AVAILABLE = False
    # Define dummy classes if SV is essential for this module to load
    class Detections: # Dummy class
        @staticmethod
        def empty(): return None # Or some representation of empty
    sv = None # Assign None to sv if not imported


from config import TRACKER_CONF_THRESHOLD, TRACKER_BUFFER, TRACKER_MATCH_THRESHOLD

class CustomTracker:
    """
    A simple custom tracker based on IoU matching and track buffering.
    Manages track IDs for detected objects over frames.
    Requires Supervision's Detections object for input/output.
    """
    def __init__(self, track_thresh=TRACKER_CONF_THRESHOLD, track_buffer=TRACKER_BUFFER, match_thresh=TRACKER_MATCH_THRESHOLD):
        if not SV_AVAILABLE:
            raise ImportError("Supervision library is required for CustomTracker but not found.")

        self.track_thresh = track_thresh # Min detection confidence to consider
        self.track_buffer = track_buffer # Max frames to keep inactive track
        self.match_thresh = match_thresh # Min IoU to match detection to track
        self.tracks = {}  # {track_id: {box: [], score: float, last_update_frame: int, inactive_count: int, start_frame: int, class_id: int}}
        self.next_id = 1
        self.frame_count = 0

    def _iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        if inter_area == 0:
            return 0.0

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        if union_area <= 0: # Handle cases of zero area boxes
             return 0.0

        return inter_area / union_area

    def update(self, detections: Detections):
        """Updates tracks with new detections for the current frame."""
        self.frame_count += 1

        # Ensure input is a Supervision Detections object
        if not isinstance(detections, Detections):
             # Handle cases where input might be None or empty list gracefully
             if detections is None or len(detections) == 0:
                  detections = Detections.empty()
             else:
                  # Attempt conversion if possible, otherwise raise error
                  try:
                       # Example: If input was just boxes, confidences
                       # detections = Detections(xyxy=..., confidence=...)
                       raise TypeError(f"CustomTracker.update expects a supervision.Detections object, got {type(detections)}")
                  except Exception as e:
                       raise TypeError(f"Invalid input to CustomTracker.update: {e}")


        current_boxes = detections.xyxy
        current_scores = detections.confidence
        # Handle optional class_id, default to 0 if not present
        current_class_ids = detections.class_id if hasattr(detections, 'class_id') and detections.class_id is not None else np.zeros(len(detections), dtype=int)


        # --- Track Management ---
        updated_tracks = {}
        matched_det_indices = set()
        active_track_ids = list(self.tracks.keys()) # Process existing tracks

        # 1. Try to match existing active tracks with current detections
        for track_id in active_track_ids:
            track = self.tracks[track_id]
            if track['inactive_count'] >= self.track_buffer: # Skip tracks inactive for too long
                 continue

            best_match_iou = -1.0 # Use -1 to ensure any valid IoU is higher
            best_match_idx = -1

            for i in range(len(detections)):
                if i in matched_det_indices: continue # Skip already matched detections

                # Optional: Class ID check - only match if class IDs are the same
                # if 'class_id' in track and current_class_ids is not None:
                #     if track['class_id'] != current_class_ids[i]:
                #         continue

                iou = self._iou(track['box'], current_boxes[i])

                # Match if IoU is above threshold and better than previous matches for this track
                if iou >= self.match_thresh and iou > best_match_iou:
                    best_match_iou = iou
                    best_match_idx = i

            if best_match_idx != -1:
                # Update track with the matched detection
                track['box'] = current_boxes[best_match_idx]
                track['score'] = current_scores[best_match_idx]
                track['last_update_frame'] = self.frame_count
                track['inactive_count'] = 0 # Reset inactivity
                track['class_id'] = current_class_ids[best_match_idx] # Update class ID
                updated_tracks[track_id] = track
                matched_det_indices.add(best_match_idx)
            else:
                # No match found for this track in the current frame
                track['inactive_count'] += 1
                # Keep the track in updated_tracks for now, will prune later
                updated_tracks[track_id] = track


        # 2. Create new tracks for unmatched high-confidence detections
        for i in range(len(detections)):
            if i not in matched_det_indices and current_scores[i] >= self.track_thresh:
                new_track = {
                    'box': current_boxes[i],
                    'score': current_scores[i],
                    'start_frame': self.frame_count,
                    'last_update_frame': self.frame_count,
                    'inactive_count': 0,
                    'class_id': current_class_ids[i] # Store class ID
                }
                updated_tracks[self.next_id] = new_track
                self.next_id += 1

        # 3. Prune tracks that have been inactive for too long
        self.tracks = {tid: t for tid, t in updated_tracks.items() if t['inactive_count'] < self.track_buffer}


        # --- Prepare Output Detections ---
        # Return detections only for tracks that are currently active (inactive_count == 0)
        # and were updated in this frame (or potentially recently if needed for smoothing?)
        # Let's stick to tracks updated *this frame* for clarity.
        output_boxes = []
        output_scores = []
        output_track_ids = []
        output_class_ids = []

        for track_id, track in self.tracks.items():
             # Only output tracks updated in this frame
             if track['last_update_frame'] == self.frame_count and track['inactive_count'] == 0:
                output_boxes.append(track['box'])
                output_scores.append(track['score'])
                output_track_ids.append(track_id)
                output_class_ids.append(track['class_id'])

        if not output_boxes:
            return Detections.empty()

        # Ensure all outputs are numpy arrays
        return Detections(
            xyxy=np.array(output_boxes),
            confidence=np.array(output_scores),
            tracker_id=np.array(output_track_ids),
            class_id=np.array(output_class_ids, dtype=int) # Ensure class_id is int
        )