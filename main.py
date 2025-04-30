# main.py
import time
import argparse
import sys
import json
import traceback
from pathlib import Path
import matplotlib.pyplot as plt # For final cleanup

# Import necessary functions and classes from other modules
from config import DEFAULT_VIDEO_PATH, DATABASE_NAME, REPORTS_DIR, MODELS_DIR
from video_processor import process_video
from database import get_tracking_data

# --- Dependency Checks (Informational) ---
# Actual availability is checked within the modules that use them.
try:
    import mediapipe
    print("Info: MediaPipe found.")
except ImportError:
    print("Info: MediaPipe library not found.")

try:
    import ultralytics
    import supervision
    print("Info: Ultralytics and Supervision found.")
except ImportError:
    print("Info: Ultralytics or Supervision library not found. YOLO processing may be unavailable.")


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(
        description='MSD Posture Analysis Tool using Computer Vision.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    parser.add_argument('--video', type=str, default=None,
                        help=f'Path to input video file. If omitted, tries "{DEFAULT_VIDEO_PATH}" in script dir.')
    parser.add_argument('--ground-truth', type=str, default=None,
                        help='Path to ground truth JSON file (optional) for validation.')
    parser.add_argument('--use-mediapipe', action='store_true',
                        help='Force using MediaPipe for pose estimation, even if YOLO is available.')
    parser.add_argument('--db-name', type=str, default=DATABASE_NAME,
                        help='Name of the SQLite database file.')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run analysis without displaying the real-time GUI window.')
    args = parser.parse_args()

    # --- Determine Video Path ---
    video_path_arg = args.video
    if video_path_arg is None:
        # Try default path relative to script location
        script_dir = Path(__file__).parent.resolve() # Get absolute path of script dir
        default_path_check = script_dir / DEFAULT_VIDEO_PATH
        if default_path_check.is_file():
            video_path_arg = str(default_path_check)
            print(f"Using default video path: {video_path_arg}")
        else:
            print(f"Error: No video path provided via --video and default '{DEFAULT_VIDEO_PATH}' not found in script directory ({script_dir}).")
            sys.exit(1) # Exit if no video source found
    else:
        # Validate provided path
        video_path_arg_path = Path(video_path_arg)
        if not video_path_arg_path.is_file():
            print(f"Error: Video file not found at specified path: '{video_path_arg}'")
            sys.exit(1)
        # Use resolved absolute path
        video_path_arg = str(video_path_arg_path.resolve())


    # --- Load Ground Truth ---
    ground_truth_data = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth).resolve()
        if gt_path.is_file():
            try:
                with open(gt_path, 'r') as f:
                    # Load JSON and ensure frame keys are strings for lookup consistency
                    raw_gt = json.load(f)
                    ground_truth_data = {str(k): v for k, v in raw_gt.items()}
                    print(f"Successfully loaded ground truth data from: {gt_path}")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format in ground truth file: {gt_path}")
            except Exception as e:
                print(f"Error loading ground truth file {gt_path}: {e}")
        else:
            print(f"Warning: Ground truth file not found: {gt_path}")


    # --- Print Configuration ---
    print(f"\n--- Configuration ---")
    print(f"Video Source: {video_path_arg}")
    print(f"Database: {args.db_name}")
    # Processing mode determined inside process_video based on availability and args.use_mediapipe
    print(f"Force MediaPipe: {args.use_mediapipe}")
    print(f"Show GUI: {not args.no_gui}")
    print(f"Ground Truth File: {args.ground_truth if ground_truth_data else 'Not Provided / Failed to Load'}")
    print(f"Reports Directory: {REPORTS_DIR}")
    print(f"--------------------\n")


    # --- Run Video Processing ---
    final_results = None
    start_time_main = time.time()
    try:
        final_results = process_video(
            video_path=video_path_arg,
            ground_truth=ground_truth_data,
            use_yolo=(not args.use_mediapipe), # Try YOLO unless mediapipe is forced
            db_name=args.db_name,
            show_gui=(not args.no_gui)
        )

        if final_results:
            print("\n--- Analysis Summary ---")
            # Print key results nicely
            print(f"Session ID: {final_results.get('session_id')}")
            print(f"Timestamp: {final_results.get('timestamp')}")
            print(f"Detected Persons (with summary): {final_results.get('detected_persons')}")

            consolidated = final_results.get('consolidated_data', {})
            print("\nConsolidated Results:")
            print(f"  Overall Safe: {consolidated.get('overall_safe_percent', 0):.1f}%")
            print(f"  Overall Neutral: {consolidated.get('overall_neutral_percent', 0):.1f}%")
            print(f"  Overall Strain: {consolidated.get('overall_strain_percent', 0):.1f}%")

            validation = final_results.get('validation_metrics', {})
            print("\nValidation Metrics:")
            print(f"  Detection Rate: {validation.get('detection_rate', 0)*100:.1f}%")
            print(f"  Risk Calc Rate: {validation.get('risk_calculation_rate', 0)*100:.1f}%")
            print(f"  Avg FPS: {validation.get('effective_fps', 0):.1f}")
            if 'gt_accuracy' in validation:
                 print(f"  GT Accuracy: {validation['gt_accuracy']:.3f}")
                 print(f"  GT F1-Score: {validation.get('gt_f1_score', 0):.3f}")

            report_paths = final_results.get('report_paths', {})
            print("\nGenerated Reports:")
            if report_paths:
                for report_type, path in report_paths.items():
                     print(f"  {report_type.replace('_', ' ').title()}: {path}")
            else:
                 print("  No report paths found in results.")


            # Example: Retrieve and print historical data for the first detected person
            if final_results.get('detected_persons'):
                first_person_id = final_results['detected_persons'][0]
                print(f"\n--- Historical Data for Person {first_person_id} (Last 5 Sessions) ---")
                history = get_tracking_data(person_id=first_person_id, db_name=args.db_name)
                if history:
                    for record in history[:5]: # Print last 5 sessions
                        print(f"  Session {record['session_id']} ({record['session_timestamp']}): "
                              f"AvgScore={record.get('avg_risk_score', 0):.1f}, "
                              f"Strain%={record.get('strain_percent', 0):.1f}, "
                              f"Frames={record.get('total_frames', 0)}")
                else:
                    print("  No historical data found in database.")

        else:
            print("\nError: Video processing did not return results or failed.")

    except Exception as e:
        print("\n!!! An unhandled error occurred in the main execution flow !!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        # Ensure plots are closed if any were left open due to errors
        plt.close('all')
        end_time_main = time.time()
        print(f"\nTotal execution time: {end_time_main - start_time_main:.2f} seconds.")
        print("Analysis finished.")


if __name__ == "__main__":
    main()