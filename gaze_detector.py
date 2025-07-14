import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
import json
import os
import subprocess
import sys

# ─── Helpers ─────────────────────────────────────────────────────────────

def extract_euler_angles(R):
    pitch = np.arcsin(-R[2, 0])
    if abs(np.cos(pitch)) > 1e-6:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        roll = 0
    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def sector_3x3(x, y, w, h):
    col = 0 if x < w/3 else (1 if x < 2*w/3 else 2)
    row = 0 if y < h/3 else (1 if y < 2*h/3 else 2)
    return row*3 + col + 1

def head_sector_3x3(yaw, pitch):
    col = 0 if yaw < -15 else (1 if yaw < 15 else 2)
    row = 0 if pitch < -10 else (1 if pitch < 10 else 2)
    return row*3 + col + 1

def compute_arrow_raw(lm, w, h):
    L_IRIS = range(468, 473)
    R_IRIS = range(473, 478)
    left_iris = np.array([np.mean([lm[i].x for i in L_IRIS]) * w,
                          np.mean([lm[i].y for i in L_IRIS]) * h])
    right_iris = np.array([np.mean([lm[i].x for i in R_IRIS]) * w,
                           np.mean([lm[i].y for i in R_IRIS]) * h])
    left_center = (np.array([lm[133].x, lm[133].y]) + np.array([lm[33].x, lm[33].y])) / 2 * [w, h]
    right_center = (np.array([lm[362].x, lm[362].y]) + np.array([lm[263].x, lm[263].y])) / 2 * [w, h]
    vec = ((left_iris - left_center) + (right_iris - right_center)) / 2
    eye_mid = (left_center + right_center) / 2
    magnitude = np.linalg.norm(vec)
    if magnitude < 1e-6:
        vec = np.array([1.0, 0.0])  # Default to a small horizontal vector if zero
    else:
        vec = vec / magnitude  # Normalize to unit vector
    return vec * magnitude, eye_mid

# ─── Calibration Mode ────────────────────────────────────────────────────

def calibrate_gaze(calib_path: Path, samples: int = 20):
    """Run gaze calibration and save calibration data"""
    try:
        # Get the be directory path
        be_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Ensure the calibration file path is relative to the be directory
        calib_filename = calib_path.name
        calib_relative_path = calib_filename
        
        print(f"Running calibration:")
        print(f"  Calibration file: {calib_path} -> {calib_relative_path}")
        print(f"  Working directory: {be_dir}")
        
        # Run the calibration using the existing script
        result = subprocess.run([
            sys.executable, "mobile_gaze_v8.py",
            "--mode", "calibrate",
            "--calib", calib_relative_path,
            "--samples", str(samples)
        ], capture_output=True, text=True, cwd=be_dir)
        
        if result.returncode == 0:
            return True, "Calibration completed successfully"
        else:
            return False, f"Calibration failed: {result.stderr}"
    except Exception as e:
        return False, f"Calibration error: {str(e)}"

def process_gaze_detection(webcam_folder, output_folder, calib_path):
    """Process webcam frames with gaze detection"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Check if gaze log already exists
        gaze_log_path = os.path.join(os.path.dirname(output_folder), "gaze_log.json")
        if os.path.exists(gaze_log_path):
            # Load existing gaze data to check if it's valid
            try:
                with open(gaze_log_path, 'r') as f:
                    gaze_data = json.load(f)
                if gaze_data and len(gaze_data) > 0:
                    return True, f"Gaze detection already completed. Found {len(gaze_data)} existing frames."
            except (json.JSONDecodeError, FileNotFoundError):
                # If the file is corrupted or empty, we'll reprocess
                pass
        
        # Get the be directory path
        be_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Ensure the calibration file path is relative to the be directory
        calib_filename = calib_path.name
        calib_relative_path = calib_filename
        
        # Convert paths to be relative to the be directory
        # When running from root, webcam_folder is "./frames/webcam" and output_folder is "./frames/gaze_output"
        # We need to make them relative to the be directory
        webcam_relative = os.path.relpath(webcam_folder, be_dir)
        output_relative = os.path.relpath(output_folder, be_dir)
        
        print(f"Processing gaze detection:")
        print(f"  Webcam folder: {webcam_folder} -> {webcam_relative}")
        print(f"  Output folder: {output_folder} -> {output_relative}")
        print(f"  Calibration: {calib_path} -> {calib_relative_path}")
        print(f"  Working directory: {be_dir}")
        
        # Run the gaze processing using the existing script
        result = subprocess.run([
            sys.executable, "mobile_gaze_v8.py",
            "--mode", "process",
            "--images", webcam_relative,
            "--out", output_relative,
            "--calib", calib_relative_path
        ], capture_output=True, text=True, cwd=be_dir)
        
        if result.returncode == 0:
            # Load the gaze log - it's written one directory up from the output folder
            gaze_log_path = os.path.join(os.path.dirname(output_folder), "gaze_log.json")
            if os.path.exists(gaze_log_path):
                with open(gaze_log_path, 'r') as f:
                    gaze_data = json.load(f)
                return True, f"Gaze detection completed. Processed {len(gaze_data)} frames."
            else:
                return False, "Gaze detection completed but no log file found."
        else:
            return False, f"Gaze detection failed: {result.stderr}"
    except Exception as e:
        return False, f"Gaze detection error: {str(e)}"

def integrate_gaze_with_combined_log(combined_log_path, gaze_log_path):
    """Integrate gaze detection results with the combined log"""
    try:
        # Load combined log
        with open(combined_log_path, 'r') as f:
            combined_log = json.load(f)
        
        # Load gaze log - it's written one directory up from the gaze_output folder
        if not os.path.exists(gaze_log_path):
            # Try the correct location (one directory up from gaze_output)
            gaze_output_dir = os.path.dirname(gaze_log_path)
            correct_gaze_log_path = os.path.join(os.path.dirname(gaze_output_dir), "gaze_log.json")
            if os.path.exists(correct_gaze_log_path):
                gaze_log_path = correct_gaze_log_path
        
        with open(gaze_log_path, 'r') as f:
            gaze_data = json.load(f)
        
        # Create a mapping from filename to gaze data
        gaze_map = {entry['file']: entry for entry in gaze_data}
        
        # Integrate gaze data into webcam entries
        for webcam_entry in combined_log.get('webcam', []):
            filename = webcam_entry['file']
            if filename in gaze_map:
                gaze_entry = gaze_map[filename]
                webcam_entry.update({
                    'gaze_yaw': gaze_entry.get('yaw'),
                    'gaze_pitch': gaze_entry.get('pitch'),
                    'gaze_sector': gaze_entry.get('sector'),
                    'gaze_iris_sector': gaze_entry.get('iris_sector'),
                    'gaze_head_sector': gaze_entry.get('head_sector'),
                    'gaze_engaged': gaze_entry.get('engaged')
                })
        
        # Save updated combined log
        with open(combined_log_path, 'w') as f:
            json.dump(combined_log, f, indent=2)
        
        return True, f"Integrated gaze data for {len(gaze_map)} frames"
    except Exception as e:
        return False, f"Error integrating gaze data: {str(e)}"

def check_calibration_exists(calib_path):
    """Check if calibration file exists"""
    return os.path.exists(calib_path) 