# reporting.py
import os
import traceback
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for saving files
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import sqlite3

# Import functions/constants needed for reporting
from risk_assessment import get_ergonomic_remedies
from config import (
    REPORTS_DIR, PLOT_STYLE, STATUS_COLORS, DATABASE_NAME,
    BACK_BEND_MODERATE_THRESHOLD, BACK_BEND_SEVERE_THRESHOLD,
    NECK_BEND_THRESHOLD, ARM_RAISE_THRESHOLD,
    RISK_SCORE_NEUTRAL_MAX, RISK_SCORE_STRAIN_MIN
)

# Import metrics if ground truth comparison is done here
try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not found. Ground truth comparison metrics will be unavailable.")
    # Define dummy functions or skip calculations if needed
    accuracy_score = precision_score = recall_score = f1_score = confusion_matrix = lambda *args, **kwargs: None


def generate_plots(valid_persons_summary, session_id, report_prefix):
    """Generates and saves consolidated analysis plots."""
    print("Generating consolidated visualizations...")
    plot_paths = {} # Dictionary to store paths of generated plots

    if not valid_persons_summary:
        print("Skipping plot generation - no valid person data.")
        return plot_paths

    plt.style.use(PLOT_STYLE)
    person_ids = list(valid_persons_summary.keys())
    num_persons = len(person_ids)
    x_labels_p = [f"P{pid}" for pid in person_ids] # Labels for person-based plots

    # --- Plot 1: Risk Score & Status Distribution Comparison ---
    try:
        fig1, axes1 = plt.subplots(2, 2, figsize=(15, 12))
        fig1.suptitle(f'Consolidated Analysis - Session {session_id}', fontsize=16)

        # Avg Risk Score Bar Chart
        avg_scores = [valid_persons_summary[pid]["avg_risk_score"] for pid in person_ids]
        colors_risk = ['red' if s >= RISK_SCORE_STRAIN_MIN else 'gold' if s > RISK_SCORE_NEUTRAL_MAX else 'green' for s in avg_scores]
        bars = axes1[0, 0].bar(x_labels_p, avg_scores, color=colors_risk)
        axes1[0, 0].set_title("Average Risk Score per Person")
        axes1[0, 0].set_ylabel("Avg. Score")
        axes1[0, 0].bar_label(bars, fmt='%.1f') # Add labels to bars
        axes1[0, 0].tick_params(axis='x', rotation=45)

        # Overall Status Pie Chart
        total_safe = sum(p["data"]["statuses"].count("Safe") for p in valid_persons_summary.values())
        total_neutral = sum(p["data"]["statuses"].count("Neutral") for p in valid_persons_summary.values())
        total_strain = sum(p["data"]["statuses"].count("Strain") for p in valid_persons_summary.values())
        pie_sizes = [total_safe, total_neutral, total_strain]
        pie_labels = [f'{stat} ({count})' for stat, count in zip(['Safe', 'Neutral', 'Strain'], pie_sizes)]
        # Convert RGB tuples (0-255) to matplotlib format (0-1)
        pie_colors_mpl = [(c[0]/255, c[1]/255, c[2]/255) for c in STATUS_COLORS.values() if c in [(0,255,0), (0,255,255), (255,0,0)]] # Ensure order matches labels
        axes1[0, 1].pie(pie_sizes, labels=pie_labels, colors=pie_colors_mpl, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
        axes1[0, 1].set_title("Overall Posture Status Distribution (All Persons)")
        axes1[0, 1].axis('equal')

        # Status Percentage Stacked Bar Chart
        safe_p = np.array([valid_persons_summary[pid]["safe_percent"] for pid in person_ids])
        neut_p = np.array([valid_persons_summary[pid]["neutral_percent"] for pid in person_ids])
        stra_p = np.array([valid_persons_summary[pid]["strain_percent"] for pid in person_ids])
        axes1[1, 0].bar(x_labels_p, safe_p, label='Safe', color=(0,1,0,0.7))
        axes1[1, 0].bar(x_labels_p, neut_p, bottom=safe_p, label='Neutral', color=(1,1,0,0.7))
        axes1[1, 0].bar(x_labels_p, stra_p, bottom=safe_p + neut_p, label='Strain', color=(1,0,0,0.7))
        axes1[1, 0].set_title("Status Distribution per Person")
        axes1[1, 0].set_ylabel("Percentage (%)")
        axes1[1, 0].legend()
        axes1[1, 0].set_ylim(0, 105) # Extend ylim slightly
        axes1[1, 0].tick_params(axis='x', rotation=45)

        # Risk Score Trend Line Chart
        max_frames = 0
        for pid in person_ids:
            frames = valid_persons_summary[pid]["data"]["frames"]
            scores = valid_persons_summary[pid]["data"]["risk_scores"]
            if frames: # Check if frames list is not empty
                 axes1[1, 1].plot(frames, scores, marker='.', linestyle='-', label=f"P{pid}")
                 max_frames = max(max_frames, max(frames) if frames else 0)
        axes1[1, 1].set_title("Risk Score Trends Over Time")
        axes1[1, 1].set_xlabel("Frame Number")
        axes1[1, 1].set_ylabel("Risk Score")
        if max_frames > 0: axes1[1, 1].set_xlim(left=0, right=max_frames * 1.05)
        axes1[1, 1].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
        plot_path1 = Path(REPORTS_DIR) / f"{report_prefix}_consolidated_analysis.png"
        plt.savefig(plot_path1)
        plt.close(fig1) # Close the figure
        plot_paths['consolidated_analysis'] = plot_path1
        print(f"Saved consolidated analysis plot: {plot_path1}")

    except Exception as plot_err:
        print(f"Error generating consolidated plot 1: {plot_err}")
        print(traceback.format_exc())
        plt.close('all')

    # --- Plot 2: Angle Comparison & Radar Charts ---
    try:
        # Determine layout: 1 row for bars, then rows for radar charts (3 per row)
        num_radar_rows = (num_persons + 2) // 3
        fig2_rows = 1 + num_radar_rows
        fig2, axes2 = plt.subplots(fig2_rows, 3, figsize=(18, 5 * fig2_rows), squeeze=False) # Ensure axes2 is always 2D
        fig2.suptitle(f'Angle Analysis & Profiles - Session {session_id}', fontsize=16)

        # Average Angle Bar Charts (Top Row)
        avg_backs = [valid_persons_summary[pid]["avg_back_angle"] for pid in person_ids]
        avg_necks = [valid_persons_summary[pid]["avg_neck_angle"] for pid in person_ids]
        avg_arms = [valid_persons_summary[pid]["avg_arm_angle"] for pid in person_ids]

        axes2[0, 0].bar(x_labels_p, avg_backs, color='skyblue')
        axes2[0, 0].set_title('Avg. Back Bend Angle')
        axes2[0, 0].set_ylabel('Degrees')
        axes2[0, 0].axhline(BACK_BEND_MODERATE_THRESHOLD, color='orange', linestyle='--', lw=1, label=f'Mod ({BACK_BEND_MODERATE_THRESHOLD}°)')
        axes2[0, 0].axhline(BACK_BEND_SEVERE_THRESHOLD, color='red', linestyle='--', lw=1, label=f'Sev ({BACK_BEND_SEVERE_THRESHOLD}°)')
        axes2[0, 0].legend(fontsize='small')
        axes2[0, 0].tick_params(axis='x', rotation=45)

        axes2[0, 1].bar(x_labels_p, avg_necks, color='lightgreen')
        axes2[0, 1].set_title('Avg. Neck Bend Angle')
        axes2[0, 1].set_ylabel('Degrees')
        axes2[0, 1].axhline(NECK_BEND_THRESHOLD, color='red', linestyle='--', lw=1, label=f'Risk ({NECK_BEND_THRESHOLD}°)')
        axes2[0, 1].legend(fontsize='small')
        axes2[0, 1].tick_params(axis='x', rotation=45)

        axes2[0, 2].bar(x_labels_p, avg_arms, color='lightcoral')
        axes2[0, 2].set_title('Avg. Arm Raise Angle')
        axes2[0, 2].set_ylabel('Degrees')
        axes2[0, 2].axhline(ARM_RAISE_THRESHOLD, color='red', linestyle='--', lw=1, label=f'Risk ({ARM_RAISE_THRESHOLD}°)')
        axes2[0, 2].legend(fontsize='small')
        axes2[0, 2].tick_params(axis='x', rotation=45)

        # Radar Charts (Subsequent Rows)
        categories = ['Back Bend', 'Neck Bend', 'Arm Raise', 'Avg Risk']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1] # Close the loop

        radar_axes = [] # Collect axes used for radar plots
        for i, pid in enumerate(person_ids):
            row_idx = 1 + (i // 3)
            col_idx = i % 3
            # Create polar subplot explicitly
            ax_radar = fig2.add_subplot(fig2_rows, 3, 1 + (row_idx * 3) + col_idx, polar=True)
            radar_axes.append(ax_radar)

            # Normalize values (example normalization, adjust ranges if needed)
            norm_back = min(valid_persons_summary[pid]["avg_back_angle"] / 90, 1) # Max expected bend 90?
            norm_neck = min(valid_persons_summary[pid]["avg_neck_angle"] / 60, 1) # Max expected bend 60?
            norm_arm = min(valid_persons_summary[pid]["avg_arm_angle"] / 180, 1) # Max arm angle 180
            norm_risk = min(valid_persons_summary[pid]["avg_risk_score"] / 10, 1) # Assuming max score ~10

            data_radar = np.array([norm_back, norm_neck, norm_arm, norm_risk])
            data_radar = np.concatenate((data_radar, [data_radar[0]])) # Close the loop

            ax_radar.plot(angles, data_radar, linewidth=1.5, linestyle='solid', label=f"P{pid}")
            ax_radar.fill(angles, data_radar, alpha=0.4)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(categories, size=8)
            ax_radar.set_yticks(np.linspace(0, 1, 5)) # Radial ticks 0 to 1
            ax_radar.set_yticklabels([]) # Hide radial labels
            ax_radar.set_title(f"P{pid} Profile", size=10, pad=15) # Add padding
            ax_radar.grid(True)

        # Hide unused subplots in the radar rows
        total_radar_plots = num_persons
        total_radar_axes = num_radar_rows * 3
        for i in range(total_radar_plots, total_radar_axes):
             row_idx = 1 + (i // 3)
             col_idx = i % 3
             # Check if the axis exists before trying to hide it
             if row_idx < axes2.shape[0] and col_idx < axes2.shape[1]:
                  # Check if it's not one of the top row axes
                  if row_idx > 0:
                       # Check if it's not an already created polar axis
                       is_polar = False
                       for r_ax in radar_axes:
                           # This comparison might be tricky, rely on index
                           if fig2.axes.index(r_ax) == 1 + (row_idx * 3) + col_idx:
                               is_polar = True
                               break
                       if not is_polar:
                            axes2[row_idx, col_idx].set_visible(False)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_path2 = Path(REPORTS_DIR) / f"{report_prefix}_angle_analysis.png"
        plt.savefig(plot_path2)
        plt.close(fig2)
        plot_paths['angle_analysis'] = plot_path2
        print(f"Saved angle analysis plot: {plot_path2}")

    except Exception as plot_err:
        print(f"Error generating angle/radar plot 2: {plot_err}")
        print(traceback.format_exc())
        plt.close('all')

    # --- Plot 3: Validation Metrics (if available) ---
    # This plot is generated after metrics calculation later

    return plot_paths


def generate_csv_report(valid_persons_summary, report_prefix):
    """Generates a consolidated CSV file with frame-by-frame data."""
    print("Generating consolidated CSV report...")
    if not valid_persons_summary:
        print("Skipping CSV report - no valid person data.")
        return None

    all_person_data_list = []
    for pid, summary in valid_persons_summary.items():
        # Access the raw data stored within the summary
        data = summary.get("data", {})
        if not data or not data.get("frames"): # Check if data exists and is not empty
            print(f"Warning: No raw data found for Person {pid} in summary.")
            continue

        try:
            # Ensure all lists have the same length, pad if necessary (though ideally they should match)
            num_frames = len(data["frames"])
            df_person = pd.DataFrame({
                "Person_ID": pid,
                "Frame": data["frames"],
                "Timestamp": data.get("timestamps", [None]*num_frames)[:num_frames], # Handle potential missing keys
                "Status": data.get("statuses", [None]*num_frames)[:num_frames],
                "Risk_Score": data.get("risk_scores", [None]*num_frames)[:num_frames],
                "Back_Angle": data.get("back_angles", [None]*num_frames)[:num_frames],
                "Neck_Angle": data.get("neck_angles", [None]*num_frames)[:num_frames],
                "Arm_Angle": data.get("arm_angles", [None]*num_frames)[:num_frames],
            })
            all_person_data_list.append(df_person)
        except Exception as df_err:
             print(f"Error creating DataFrame for Person {pid}: {df_err}")


    if all_person_data_list:
        try:
            consolidated_df = pd.concat(all_person_data_list, ignore_index=True)
            csv_path = Path(REPORTS_DIR) / f"{report_prefix}_consolidated_data.csv"
            consolidated_df.to_csv(csv_path, index=False, float_format='%.2f') # Format floats
            print(f"Saved consolidated data CSV: {csv_path}")
            return csv_path
        except Exception as csv_err:
            print(f"Error saving consolidated CSV: {csv_err}")
            print(traceback.format_exc())
    else:
        print("No valid dataframes created for CSV report.")

    return None


def generate_pdf_report(session_id, session_timestamp, video_path,
                        valid_persons_summary, overall_summary,
                        validation_metrics, plot_paths, report_prefix):
    """Generates a comprehensive PDF report."""
    print("Generating PDF report...")
    if not valid_persons_summary:
        print("Skipping PDF report - no valid person data.")
        return None

    pdf_path = Path(REPORTS_DIR) / f"{report_prefix}_report.pdf"
    num_persons = len(valid_persons_summary)

    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # --- Header ---
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "MSD Posture Analysis Report", ln=True, align="C")
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 7, f"Session ID: {session_id}", ln=True, align="C")
        pdf.cell(0, 7, f"Video File: {Path(video_path).name}", ln=True, align="C")
        pdf.cell(0, 7, f"Analysis Timestamp: {session_timestamp}", ln=True, align="C")
        pdf.cell(0, 7, f"Persons Analyzed (Sufficient Data): {num_persons}", ln=True, align="C")
        pdf.ln(5)

        # --- Overall Summary Section ---
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Overall Session Summary", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 7, f"- Safe Postures: {overall_summary['safe_percent']:.1f}%", ln=True)
        pdf.cell(0, 7, f"- Neutral Postures: {overall_summary['neutral_percent']:.1f}%", ln=True)
        pdf.cell(0, 7, f"- Strain Postures: {overall_summary['strain_percent']:.1f}%", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Overall Ergonomic Recommendations:", ln=True)
        pdf.set_font("Arial", "", 10)
        if overall_summary['remedies']:
            for remedy in overall_summary['remedies']:
                 pdf.multi_cell(0, 5, f"{remedy}", ln=True) # Use multi_cell for wrapping
        else:
             pdf.cell(0, 5, "No specific remedies generated based on overall distribution.", ln=True)
        pdf.ln(5)

        # --- Consolidated Plots ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Consolidated Analysis Visuals", ln=True)
        pdf.ln(5)
        for plot_key, plot_file in plot_paths.items():
             if plot_file and plot_file.exists():
                  try:
                       # Calculate image width to fit page (A4 width ~ 210mm, margins 15mm*2)
                       available_width = pdf.w - 2 * pdf.l_margin
                       pdf.image(str(plot_file), x=pdf.l_margin, w=available_width)
                       pdf.ln(5) # Add space after image
                  except Exception as img_err:
                       pdf.set_font("Arial", "I", 10)
                       pdf.cell(0, 8, f"(Error embedding image '{plot_key}': {img_err})", ln=True)
             else:
                  pdf.set_font("Arial", "I", 10)
                  pdf.cell(0, 8, f"({plot_key} plot not found or not generated)", ln=True)
        pdf.ln(5)


        # --- Individual Person Summaries ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Individual Person Details", ln=True)
        pdf.ln(5)

        for pid, summary in valid_persons_summary.items():
            pdf.set_font("Arial", "B", 12)
            # Use Cell for better control over line breaks if needed, or multi_cell
            pdf.cell(0, 8, f"Person ID: {pid}", ln=True)
            pdf.set_font("Arial", "", 10)
            pdf.cell(0, 6, f"- Data Points (Frames): {summary['total_frames']}", ln=True)
            pdf.cell(0, 6, f"- Avg Risk Score: {summary['avg_risk_score']:.2f}", ln=True)
            pdf.cell(0, 6, f"- Status %: Safe({summary['safe_percent']:.1f}%) / Neutral({summary['neutral_percent']:.1f}%) / Strain({summary['strain_percent']:.1f}%)", ln=True)
            pdf.cell(0, 6, f"- Avg Angles (°): Back({summary['avg_back_angle']:.1f}) / Neck({summary['avg_neck_angle']:.1f}) / Arm({summary['avg_arm_angle']:.1f})", ln=True)
            pdf.ln(4)
            # Add separator
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)


        # --- Validation Metrics Section ---
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Model Performance & Validation", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.ln(3)

        # Processing Stats
        pdf.cell(0, 7, f"Total Frames Processed: {validation_metrics.get('total_frames_processed', 'N/A')}", ln=True)
        det_rate = validation_metrics.get('detection_rate', 0) * 100
        pdf.cell(0, 7, f"Person Detection Rate: {det_rate:.1f}%", ln=True)
        risk_calc_rate = validation_metrics.get('risk_calculation_rate', 0) * 100
        pdf.cell(0, 7, f"Risk Calculation Rate (given detection): {risk_calc_rate:.1f}%", ln=True)
        avg_conf = validation_metrics.get('avg_keypoint_confidence', 0)
        pdf.cell(0, 7, f"Avg. Confidence of Used Keypoints: {avg_conf:.3f}", ln=True)
        avg_proc_time_ms = validation_metrics.get('avg_processing_time_ms', 0)
        eff_fps = validation_metrics.get('effective_fps', 0)
        pdf.cell(0, 7, f"Avg. Processing Time per Frame: {avg_proc_time_ms:.1f} ms ({eff_fps:.1f} FPS)", ln=True)
        pdf.ln(5)

        # Ground Truth Comparison (if available)
        if validation_metrics.get('gt_accuracy') is not None:
             pdf.set_font("Arial", "B", 12)
             pdf.cell(0, 8, "Ground Truth Comparison (Risk Classification):", ln=True)
             pdf.set_font("Arial", "", 11)
             pdf.cell(0, 7, f"- Accuracy: {validation_metrics['gt_accuracy']:.3f}", ln=True)
             pdf.cell(0, 7, f"- Precision (Weighted): {validation_metrics.get('gt_precision', 'N/A'):.3f}", ln=True)
             pdf.cell(0, 7, f"- Recall (Weighted): {validation_metrics.get('gt_recall', 'N/A'):.3f}", ln=True)
             pdf.cell(0, 7, f"- F1-Score (Weighted): {validation_metrics.get('gt_f1_score', 'N/A'):.3f}", ln=True)
             pdf.ln(3)

             # Embed Confusion Matrix Plot if generated
             cm_plot_path = plot_paths.get('confusion_matrix')
             if cm_plot_path and cm_plot_path.exists():
                  try:
                       # Embed smaller CM image
                       img_w = 80 # Width in mm
                       img_x = (pdf.w - img_w) / 2 # Center the image
                       pdf.image(str(cm_plot_path), x=img_x, w=img_w)
                       pdf.ln(3)
                  except Exception as img_err:
                       pdf.set_font("Arial", "I", 10)
                       pdf.cell(0, 7, f"(Error embedding confusion matrix image: {img_err})", ln=True)
             else:
                  pdf.set_font("Arial", "I", 10)
                  pdf.cell(0, 7, "(Confusion matrix plot not generated or found)", ln=True)
        else:
             pdf.set_font("Arial", "I", 11)
             pdf.cell(0, 7, "(No ground truth data provided or metrics calculation failed)", ln=True)


        # --- Save PDF ---
        pdf.output(str(pdf_path))
        print(f"Saved PDF report: {pdf_path}")
        return pdf_path

    except Exception as pdf_err:
        print(f"Error generating PDF report: {pdf_err}")
        print(traceback.format_exc())
        return None


def calculate_validation_metrics(validation_data, ground_truth_data):
    """Calculates performance metrics based on collected data and ground truth."""
    metrics = {}
    print("\nCalculating final validation metrics...")

    # --- Basic Processing Metrics ---
    total_processed = validation_data.get('total_frames_processed', 0)
    metrics['total_frames_processed'] = total_processed
    if total_processed > 0:
        frames_w_det = validation_data.get('frames_with_detections', 0)
        metrics['frames_with_detections'] = frames_w_det
        metrics['detection_rate'] = frames_w_det / total_processed

        frames_w_risk = validation_data.get('frames_with_risk_calculation', 0)
        metrics['frames_with_risk_calculation'] = frames_w_risk
        if frames_w_det > 0:
            metrics['risk_calculation_rate'] = frames_w_risk / frames_w_det
        else:
            metrics['risk_calculation_rate'] = 0
    else:
        metrics['detection_rate'] = 0
        metrics['risk_calculation_rate'] = 0

    if validation_data.get('avg_keypoint_confidence'):
        metrics['avg_keypoint_confidence'] = np.mean(validation_data['avg_keypoint_confidence'])
    else:
        metrics['avg_keypoint_confidence'] = 0

    if validation_data.get('processing_times'):
        avg_time = np.mean(validation_data['processing_times'])
        metrics['avg_processing_time_ms'] = avg_time * 1000
        metrics['effective_fps'] = 1.0 / avg_time if avg_time > 0 else 0
    else:
        metrics['avg_processing_time_ms'] = 0
        metrics['effective_fps'] = 0

    # --- Ground Truth Comparison Metrics ---
    plot_paths = {} # Store paths for validation plots
    report_prefix = validation_data.get("report_prefix", "validation") # Get prefix if passed

    if ground_truth_data and SKLEARN_AVAILABLE:
        print("Comparing predictions with ground truth...")
        gt_risk = validation_data.get('risk_ground_truth', [])
        pred_risk = validation_data.get('risk_predictions', [])

        if gt_risk and pred_risk:
            min_len = min(len(gt_risk), len(pred_risk))
            if min_len > 0:
                gt_labels = gt_risk[:min_len]
                pred_labels = pred_risk[:min_len]

                try:
                    metrics['gt_accuracy'] = accuracy_score(gt_labels, pred_labels)
                    metrics['gt_precision'] = precision_score(gt_labels, pred_labels, average='weighted', zero_division=0)
                    metrics['gt_recall'] = recall_score(gt_labels, pred_labels, average='weighted', zero_division=0)
                    metrics['gt_f1_score'] = f1_score(gt_labels, pred_labels, average='weighted', zero_division=0)

                    print(f"  GT Accuracy: {metrics['gt_accuracy']:.3f}")
                    print(f"  GT F1 (Weighted): {metrics['gt_f1_score']:.3f}")

                    # Generate Confusion Matrix Plot
                    cm = confusion_matrix(gt_labels, pred_labels, labels=[0, 1, 2]) # Ensure labels are ordered
                    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                xticklabels=['Safe', 'Neutral', 'Strain'],
                                yticklabels=['Safe', 'Neutral', 'Strain'])
                    ax_cm.set_xlabel('Predicted Label')
                    ax_cm.set_ylabel('True Label')
                    ax_cm.set_title('Risk Classification Confusion Matrix')
                    plt.tight_layout()
                    cm_plot_path = Path(REPORTS_DIR) / f"{report_prefix}_confusion_matrix.png"
                    plt.savefig(cm_plot_path)
                    plt.close(fig_cm)
                    plot_paths['confusion_matrix'] = cm_plot_path
                    print(f"  Saved confusion matrix plot: {cm_plot_path}")

                except Exception as metric_err:
                    print(f"  Error calculating/plotting GT metrics: {metric_err}")
            else:
                print("  No matching ground truth/predictions found for comparison.")
        else:
            print("  Ground truth or prediction lists are empty.")
    elif not SKLEARN_AVAILABLE:
        print("  Skipping ground truth comparison (scikit-learn not available).")
    else:
        print("  Skipping ground truth comparison (no ground truth data provided).")

    # Add plot paths to metrics dict for use in PDF report
    metrics['plot_paths'] = plot_paths
    return metrics


def generate_reports(session_id, session_timestamp, video_path, person_data,
                     validation_data, ground_truth_data, db_name=DATABASE_NAME):
    """
    Orchestrates the generation of all reports (plots, CSV, PDF)
    after video processing is complete.
    """
    print("\n--- Starting Post-Processing and Reporting ---")
    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_prefix = f"session_{session_id}_{timestamp_suffix}"
    validation_data["report_prefix"] = report_prefix # Pass prefix for plot filenames

    # --- Connect to DB to save summaries ---
    conn = None
    valid_persons_summary = {} # Store summaries of persons meeting criteria
    detected_person_ids = list(person_data.keys())

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        for person_id in detected_person_ids:
            data = person_data[person_id]
            total_valid_frames = len(data.get("statuses", []))

            if total_valid_frames < MIN_FRAMES_FOR_SUMMARY:
                print(f"Person {person_id}: Insufficient data ({total_valid_frames} frames), skipping summary.")
                continue

            # Calculate summary stats
            safe_count = data["statuses"].count("Safe")
            neutral_count = data["statuses"].count("Neutral")
            strain_count = data["statuses"].count("Strain")

            safe_perc = (safe_count / total_valid_frames) * 100
            neutral_perc = (neutral_count / total_valid_frames) * 100
            strain_perc = (strain_count / total_valid_frames) * 100

            avg_risk = np.mean(data["risk_scores"]) if data["risk_scores"] else 0
            # Calculate average angles, handling None values
            avg_back = np.mean([a for a in data.get("back_angles", []) if a is not None]) if data.get("back_angles") else 0
            avg_neck = np.mean([a for a in data.get("neck_angles", []) if a is not None]) if data.get("neck_angles") else 0
            avg_arm = np.mean([a for a in data.get("arm_angles", []) if a is not None]) if data.get("arm_angles") else 0

            summary_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Use INSERT OR REPLACE to handle potential reruns or updates
                cursor.execute('''
                    INSERT OR REPLACE INTO stress_summary
                        (session_id, person_tracker_id, timestamp, total_frames,
                         safe_percent, neutral_percent, strain_percent, avg_risk_score,
                         avg_back_angle, avg_neck_angle, avg_arm_angle)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session_id, person_id, summary_timestamp, total_valid_frames,
                      safe_perc, neutral_perc, strain_perc, avg_risk,
                      avg_back, avg_neck, avg_arm))
                conn.commit()

                # Store summary along with raw data reference for plotting
                valid_persons_summary[person_id] = {
                    "total_frames": total_valid_frames,
                    "safe_percent": safe_perc, "neutral_percent": neutral_perc, "strain_percent": strain_perc,
                    "avg_risk_score": avg_risk, "avg_back_angle": avg_back,
                    "avg_neck_angle": avg_neck, "avg_arm_angle": avg_arm,
                    "data": data # Keep raw data reference
                }
                # print(f"Person {person_id}: Summary calculated and saved/updated.")

            except sqlite3.Error as db_err:
                 print(f"Error saving summary for person {person_id} to database: {db_err}")
                 conn.rollback() # Rollback failed transaction
            except Exception as e:
                 print(f"Unexpected error calculating/saving summary for person {person_id}: {e}")
                 print(traceback.format_exc())

    except sqlite3.Error as e:
        print(f"Database connection error during summary saving: {e}")
    finally:
        if conn:
            conn.close()
            # print("Database connection closed after summary saving.")


    if not valid_persons_summary:
        print("No persons met the criteria for summary reporting.")
        # Calculate only validation metrics if no summaries generated
        final_metrics = calculate_validation_metrics(validation_data, ground_truth_data)
        return { # Return partial results
            'session_id': session_id, 'timestamp': session_timestamp,
            'detected_persons': [], 'person_data': {}, 'consolidated_data': {},
            'validation_metrics': final_metrics, 'report_paths': {}
        }


    # --- Generate Plots ---
    plot_paths = generate_plots(valid_persons_summary, session_id, report_prefix)

    # --- Generate CSV ---
    csv_path = generate_csv_report(valid_persons_summary, report_prefix)
    if csv_path: plot_paths['csv_report'] = csv_path # Add CSV path if generated

    # --- Calculate Overall Summary & Remedies ---
    overall_safe_perc = 0
    overall_neutral_perc = 0
    overall_strain_perc = 0
    total_frames_all_valid = sum(p["total_frames"] for p in valid_persons_summary.values())

    if total_frames_all_valid > 0:
        overall_safe_perc = sum(p["safe_percent"] * p["total_frames"] for p in valid_persons_summary.values()) / total_frames_all_valid
        overall_neutral_perc = sum(p["neutral_percent"] * p["total_frames"] for p in valid_persons_summary.values()) / total_frames_all_valid
        overall_strain_perc = sum(p["strain_percent"] * p["total_frames"] for p in valid_persons_summary.values()) / total_frames_all_valid

    overall_remedies = get_ergonomic_remedies(overall_safe_perc, overall_neutral_perc, overall_strain_perc)
    overall_summary = {
        'safe_percent': overall_safe_perc,
        'neutral_percent': overall_neutral_perc,
        'strain_percent': overall_strain_perc,
        'remedies': overall_remedies
    }
    print("\n--- Overall Session Summary ---")
    print(f"Safe: {overall_safe_perc:.1f}% | Neutral: {overall_neutral_perc:.1f}% | Strain: {overall_strain_perc:.1f}%")
    # print("Overall Remedies:", overall_remedies)


    # --- Calculate Final Validation Metrics (includes GT comparison if applicable) ---
    final_metrics = calculate_validation_metrics(validation_data, ground_truth_data)
    # Add any validation plots (like confusion matrix) to plot_paths
    plot_paths.update(final_metrics.pop('plot_paths', {}))


    # --- Generate PDF Report ---
    pdf_path = generate_pdf_report(session_id, session_timestamp, video_path,
                                   valid_persons_summary, overall_summary,
                                   final_metrics, plot_paths, report_prefix)
    if pdf_path: plot_paths['pdf_report'] = pdf_path # Add PDF path if generated


    # --- Prepare Final Return Data ---
    # Exclude the raw 'data' field from the returned person_data summary
    person_data_summary_only = {
        pid: {k: v for k, v in summary.items() if k != 'data'}
        for pid, summary in valid_persons_summary.items()
    }

    final_tracking_data = {
        'session_id': session_id,
        'timestamp': session_timestamp,
        'detected_persons': list(valid_persons_summary.keys()),
        'person_data': person_data_summary_only,
        'consolidated_data': {
            'overall_safe_percent': overall_safe_perc,
            'overall_neutral_percent': overall_neutral_perc,
            'overall_strain_percent': overall_strain_perc,
            'person_count': len(valid_persons_summary),
            'remedies': overall_remedies
        },
        'validation_metrics': final_metrics,
        'report_paths': {k: str(v) for k, v in plot_paths.items()} # Convert Path objects to strings
    }

    return final_tracking_data