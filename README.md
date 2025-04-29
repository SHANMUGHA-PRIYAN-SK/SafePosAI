# SafePosAI - MSD Posture Analysis Tool

## Description

SafePosAI is a Python-based application designed for analyzing human posture from video feeds to assess the risk of Musculoskeletal Disorders (MSDs), particularly relevant in occupational settings like construction sites involving formwork. It uses computer vision techniques to detect individuals, estimate their body keypoints, calculate relevant joint angles, and classify posture into risk categories (Safe, Neutral, Strain).

The tool processes video files, tracks multiple individuals, analyzes their posture frame-by-frame, provides real-time feedback (optional), and generates comprehensive reports including visualizations and ergonomic recommendations.

## Purpose

The primary goal of SafePosAI is to identify and quantify potentially harmful postures that can lead to MSDs over time. By analyzing posture data, it aims to:

*   **Raise awareness:** Highlight risky postures and their frequency for individuals.
*   **Provide actionable insights:** Offer specific ergonomic remedies and suggestions tailored to the observed postures and overall strain levels.
*   **Enable tracking:** Store historical data to monitor posture trends and the effectiveness of interventions over time.
*   **Improve workplace safety:** Reduce the risk of MSDs by providing data-driven feedback for workers and safety managers.

## Features

*   **Pose Estimation:** Utilizes YOLOv8-Pose for keypoint detection (falls back to MediaPipe Pose if YOLO/Supervision libraries are unavailable).
*   **Multi-Person Tracking:** Employs a custom tracker to maintain consistent IDs for individuals throughout the video.
*   **MSD Risk Assessment:** Calculates risk scores based on back, neck, and arm angles using a RULA-inspired model.
*   **Posture Classification:** Categorizes posture as "Safe," "Neutral," or "Strain."
*   **Real-time Feedback (Optional):** Displays status and basic suggestions in a simple Tkinter GUI during processing.
*   **Data Persistence:** Stores session details, frame-by-frame posture data, and summary statistics in an SQLite database (`posture_tracking.db`).
*   **Comprehensive Reporting:**
    *   Generates detailed PDF reports with overall analysis, individual summaries, key findings, and ergonomic remedies.
    *   Creates multiple visualizations (saved as PNG files in a `reports/` directory):
        *   Consolidated risk score comparisons (bar chart).
        *   Overall posture status distribution (pie chart).
        *   Status distribution per person (stacked bar chart).
        *   Risk score trends over time (line chart).
        *   Average angle comparisons (bar charts for back, neck, arm).
        *   Posture radar profiles per person.
        *   Risk status timeline visualization.
        *   Risk distribution over time (stacked area chart).
        *   Individual angle trends over time (line chart).
    *   Exports raw and processed data to CSV files (`reports/consolidated_data_*.csv`, `reports/person_*_data_*.csv`).
*   **Historical Analysis:** Can retrieve and visualize posture trends across multiple sessions for an individual.
*   **Validation Metrics (Optional):** Includes capabilities for calculating accuracy, precision, recall, F1-score, angle errors, keypoint precision, and processing time if ground truth data is provided.

## Dependencies

The project requires Python 3 and the following libraries:

```
flask==2.0.1
Werkzeug==2.0.1
opencv-python-headless==4.5.5.64
mediapipe==0.10.5
ultralytics==8.1.32
supervision==0.11.1
numpy==1.24.3
matplotlib==3.7.1
pandas==2.0.1
seaborn==0.12.2
fpdf==1.7.2
python-dotenv==0.21.1
scikit-learn==1.2.2
torch==2.1.0
```

See `requirements.txt` for the full list.

## Setup / Installation

1.  **Clone the repository or download the files.**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate    # On Windows
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing `torch` might require specific commands depending on your system and CUDA version if you have a GPU. Refer to the official PyTorch website ([https://pytorch.org/](https://pytorch.org/)) for instructions.*
4.  **Download YOLOv8-Pose model:** The script attempts to download `yolov8n-pose.pt` automatically if it's not present. If this fails, you might need to download it manually from the Ultralytics repository.

## How to Run

1.  **Prepare your input video:** Place the video file you want to analyze (e.g., `input.mp4`) in a location accessible by the script.
2.  **Modify the video path:** Open `pose.py` and update the `video_path` variable in the `if __name__ == "__main__":` block to point to your video file.
    ```python
    # Example near the end of the file:
    if __name__ == "__main__":
        # ... other code ...
        video_path = r"path/to/your/video.mp4" # UPDATE THIS LINE
        print(f"Processing video: {video_path}")
        ground_truth = None # Set this if you have ground truth data
        tracking_data = process_video(video_path, ground_truth)
        # ... rest of the code ...
    ```
3.  **Run the script:**
    ```bash
    python pose.py
    ```
4.  **Observe the process:**
    *   A window titled "MSD Posture Detection" will appear, showing the video feed with detected persons, bounding boxes, keypoints (skeleton), and posture status/score.
    *   A separate Tkinter window titled "MSD Posture Analysis Dashboard" may show real-time status and suggestions for the person with the highest risk score in the current frame.
    *   Console output will show processing information and final summaries.
5.  **Review the outputs:**
    *   After processing, check the `reports/` directory for generated PDF reports, CSV data files, and PNG image files containing the various analysis charts.
    *   An SQLite database file `posture_tracking.db` will be created or updated in the same directory as the script.
    *   Summary dashboards (Tkinter windows) might pop up at the end showing consolidated results or individual person summaries.

## Database Schema (`posture_tracking.db`)

*   **`person_sessions`**: Stores information about each processing session.
    *   `id` (INTEGER PRIMARY KEY): Unique session ID.
    *   `person_id` (INTEGER): The primary person ID associated with the session (updated during processing).
    *   `session_timestamp` (TEXT): When the analysis session started.
    *   `video_path` (TEXT): Path to the processed video file.
*   **`posture_data`**: Stores frame-level posture details.
    *   `id` (INTEGER PRIMARY KEY): Unique record ID.
    *   `session_id` (INTEGER): Foreign key linking to `person_sessions`.
    *   `person_id` (INTEGER): ID of the detected person in that frame.
    *   `frame_number` (INTEGER): Frame number in the video.
    *   `risk_score` (INTEGER): Calculated MSD risk score.
    *   `status` (TEXT): Posture status ("Safe", "Neutral", "Strain").
    *   `back_angle` (REAL): Calculated back angle.
    *   `neck_angle` (REAL): Calculated neck angle.
    *   `arm_angle` (REAL): Calculated arm angle.
    *   `timestamp` (TEXT): Precise timestamp for the frame.
*   **`stress_summary`**: Stores summary statistics for each person per session.
    *   `id` (INTEGER PRIMARY KEY): Unique summary record ID.
    *   `session_id` (INTEGER): Foreign key linking to `person_sessions`.
    *   `person_id` (INTEGER): ID of the person this summary belongs to.
    *   `safe_percent` (REAL): Percentage of frames in "Safe" posture.
    *   `neutral_percent` (REAL): Percentage of frames in "Neutral" posture.
    *   `strain_percent` (REAL): Percentage of frames in "Strain" posture.
    *   `avg_risk_score` (REAL): Average risk score over the session.
    *   `timestamp` (TEXT): When the summary was generated.

## Future Enhancements

*   Integrate with live camera feeds.
*   Implement more sophisticated tracking algorithms.
*   Refine the MSD risk scoring model with more complex biomechanical factors.
*   Develop a web-based dashboard for visualization and reporting instead of relying solely on Matplotlib/Tkinter pop-ups and static files.
*   Add user management and configuration options.
*   Improve error handling and robustness. 
