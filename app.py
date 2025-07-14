import streamlit as st
import json
import os
import io
from PIL import Image, ImageDraw
from datetime import datetime, timedelta
import time
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
from record import start_recording, stop_recording, is_recording_active
from response_time import update_combined_log_with_responses
from emotion_detector import analyze_emotions_in_recording, get_dominant_emotion
from gaze_detector import calibrate_gaze, process_gaze_detection, integrate_gaze_with_combined_log, check_calibration_exists
from Final_Report import get_summary, get_ollama_summary
from frame_analysis import run_frame_analysis

# Check if frames directory exists (will be updated dynamically)
frames_exist = os.path.exists("./frames")

# Load JSON data with error handling
data = {"webcam": [], "screen": [], "keyboard": [], "mouse": []}
if frames_exist and os.path.exists("./frames/combined_log.json"):
    try:
        with open("./frames/combined_log.json", "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        st.warning("Could not load existing log file. Starting with empty data.")

webcam_entries = data.get("webcam", [])
screen_entries = data.get("screen", [])
keyboard_entries = data.get("keyboard", [])
mouse_entries = data.get("mouse", [])

# Parse times into datetime for sorting
for entry in webcam_entries:
    entry["dt"] = datetime.fromisoformat(entry["time"])
for entry in screen_entries:
    entry["dt"] = datetime.fromisoformat(entry["time"])
for entry in keyboard_entries:
    entry["dt"] = datetime.fromisoformat(entry["time"])
for entry in mouse_entries:
    entry["dt"] = datetime.fromisoformat(entry["time"])

# Build a combined sorted timeline of timestamps
combined_times = sorted(set([e["dt"] for e in webcam_entries] + [e["dt"] for e in screen_entries]))

# Build synced frame mapping
synced_frames = []
latest_webcam = None
latest_screen = None


for t in combined_times:
    for w in webcam_entries:
        if w["dt"] <= t:
            latest_webcam = w
        else:
            break
    for s in screen_entries:
        if s["dt"] <= t:
            latest_screen = s
        else:
            break
    if latest_webcam and latest_screen:
        synced_frames.append({
            "time": t,
            "webcam_file": os.path.join("frames/webcam", latest_webcam["file"]),
            "screen_file": os.path.join("frames/screen", latest_screen["file"])
        })

def add_click_highlight(img, click_position, button_type="left"):
    """Add a highlight circle to the image at the click position"""
    try:
        # Create a copy of the image to avoid modifying the original
        img_copy = img.copy()
        draw = ImageDraw.Draw(img_copy)
        
        # Circle properties
        x, y = click_position
        radius = 15
        outline_color = (255, 0, 0) if button_type == "left" else (0, 255, 0)  # Red for left, Green for right
        
        # Draw circle
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                    outline=outline_color, width=3)
        
        # Add button type text
        draw.text((x+radius+5, y-radius), button_type.upper(), fill=outline_color)
        
        return img_copy
    except Exception as e:
        st.error(f"Error highlighting image: {e}")
        return img

def create_emotion_trend_chart(webcam_entries, title="Emotion Trends Over Time"):
    """
    Create a line chart showing emotion probabilities over time
    
    Parameters:
    - webcam_entries: List of webcam entries with emotion data
    - title: Chart title
    
    Returns:
    - plotly.graph_objects.Figure: The emotion trend chart
    """
    if not webcam_entries:
        return None
    
    # Filter entries with emotion data
    emotion_entries = [entry for entry in webcam_entries if "emotion" in entry]
    
    if not emotion_entries:
        return None
    
    # Create DataFrame for plotting
    data = []
    for entry in emotion_entries:
        frame_num = entry.get("frame", 0)
        time_str = entry.get("time", "")
        
        # Convert time string to datetime for x-axis
        try:
            timestamp = datetime.fromisoformat(time_str)
        except:
            timestamp = frame_num  # Fallback to frame number
        
        for emotion, prob_str in entry["emotion"].items():
            try:
                prob = float(prob_str)
                data.append({
                    "Frame": frame_num,
                    "Time": timestamp,
                    "Emotion": emotion.title(),
                    "Probability": prob,
                    "Percentage": prob
                })
            except ValueError:
                continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    # Create the chart
    fig = go.Figure()
    
    # Color map for emotions
    emotion_colors = {
        "Angry": "#FF6B6B",
        "Disgust": "#4ECDC4", 
        "Fear": "#45B7D1",
        "Happy": "#96CEB4",
        "Sad": "#FFEAA7",
        "Surprise": "#DDA0DD",
        "Neutral": "#A8A8A8"
    }
    
    # Add line for each emotion
    for emotion in df["Emotion"].unique():
        emotion_data = df[df["Emotion"] == emotion]
        color = emotion_colors.get(emotion, "#000000")
        
        fig.add_trace(go.Scatter(
            x=emotion_data["Time"],
            y=emotion_data["Percentage"],
            mode="lines+markers",
            name=emotion,
            line=dict(color=color, width=2),
            marker=dict(size=4),
            hovertemplate="<b>%{fullData.name}</b><br>" +
                        "Time: %{x}<br>" +
                        "Probability: %{y:.1f}<br>" +
                        "<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Emotion Probability",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400,
        showlegend=True
    )
    
    return fig

def create_emotion_summary_chart(webcam_entries, title="Emotion Distribution"):
    """
    Create a bar chart showing overall emotion distribution
    
    Parameters:
    - webcam_entries: List of webcam entries with emotion data
    - title: Chart title
    
    Returns:
    - plotly.graph_objects.Figure: The emotion summary chart
    """
    if not webcam_entries:
        return None
    
    # Filter entries with emotion data
    emotion_entries = [entry for entry in webcam_entries if "emotion" in entry]
    
    if not emotion_entries:
        return None
    
    # Count dominant emotions
    emotion_counts = {}
    total_frames = len(emotion_entries)
    
    for entry in emotion_entries:
        dominant_emotion = get_dominant_emotion(entry["emotion"])
        emotion_counts[dominant_emotion] = emotion_counts.get(dominant_emotion, 0) + 1
    
    if not emotion_counts:
        return None
    
    # Create DataFrame
    data = []
    for emotion, count in emotion_counts.items():
        percentage = (count / total_frames) * 100
        data.append({
            "Emotion": emotion.title(),
            "Count": count,
            "Percentage": percentage
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("Count", ascending=False)
    
    # Color map for emotions
    emotion_colors = {
        "Angry": "#FF6B6B",
        "Disgust": "#4ECDC4", 
        "Fear": "#45B7D1",
        "Happy": "#96CEB4",
        "Sad": "#FFEAA7",
        "Surprise": "#DDA0DD",
        "Neutral": "#A8A8A8"
    }
    
    # Create the chart
    fig = go.Figure(data=[
        go.Bar(
            x=df["Emotion"],
            y=df["Percentage"],
            text=[f"{row['Count']} frames ({row['Percentage']:.1f}%)" for _, row in df.iterrows()],
            textposition="auto",
            marker_color=[emotion_colors.get(emotion, "#000000") for emotion in df["Emotion"]]
        )
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Emotion",
        yaxis_title="Percentage of Frames (%)",
        height=400,
        showlegend=False
    )
    
    return fig

def load_comparison_data(recording_path):
    """Load data from a comparison recording"""
    try:
        # Load JSON data
        log_file = os.path.join(recording_path, "combined_log.json")
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                data = json.load(f)
        else:
            return None
        
        # Parse times into datetime for sorting
        webcam_entries = data.get("webcam", [])
        screen_entries = data.get("screen", [])
        keyboard_entries = data.get("keyboard", [])
        mouse_entries = data.get("mouse", [])
        
        for entry in webcam_entries:
            entry["dt"] = datetime.fromisoformat(entry["time"])
        for entry in screen_entries:
            entry["dt"] = datetime.fromisoformat(entry["time"])
        for entry in keyboard_entries:
            entry["dt"] = datetime.fromisoformat(entry["time"])
        for entry in mouse_entries:
            entry["dt"] = datetime.fromisoformat(entry["time"])
        
        # Build a combined sorted timeline of timestamps
        combined_times = sorted(set([e["dt"] for e in webcam_entries] + [e["dt"] for e in screen_entries]))
        
        # Build synced frame mapping with validation
        synced_frames = []
        latest_webcam = None
        latest_screen = None
        
        for t in combined_times:
            for w in webcam_entries:
                if w["dt"] <= t:
                    latest_webcam = w
                else:
                    break
            for s in screen_entries:
                if s["dt"] <= t:
                    latest_screen = s
                else:
                    break
            if latest_webcam and latest_screen:
                # Validate file paths exist
                webcam_file = os.path.join(recording_path, "webcam", latest_webcam["file"])
                screen_file = os.path.join(recording_path, "screen", latest_screen["file"])
                
                # Only add frame if both files exist
                if os.path.exists(webcam_file) and os.path.exists(screen_file):
                    synced_frames.append({
                        "time": t,
                        "webcam_file": webcam_file,
                        "screen_file": screen_file
                    })
        
        return {
            "data": data,
            "synced_frames": synced_frames,
            "webcam_entries": webcam_entries,
            "screen_entries": screen_entries,
            "keyboard_entries": keyboard_entries,
            "mouse_entries": mouse_entries
        }
    except Exception as e:
        st.error(f"Error loading comparison data: {e}")
        return None

# Set page layout to wide mode to reduce padding
st.set_page_config(layout="wide", page_title="chAI - Human Action Analysis")

# Custom CSS to reduce spacing
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stButton > button {
        height: 2.5rem;
        font-size: 0.9rem;
    }
    .stSlider {
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    .stDataFrame {
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("Checking Human Actions Intelligently - chAI")

# Initialize session state for calibration
if 'show_calibration' not in st.session_state:
    st.session_state.show_calibration = False
if 'calibration_status' not in st.session_state:
    st.session_state.calibration_status = "Not calibrated"

webcam_dir = "./frames/webcam"
screen_dir = "./frames/screen"

# Calibration section
st.markdown("### 🎯 Gaze Calibration")
calib_col1, calib_col2, calib_col3 = st.columns([1, 1, 2])

with calib_col1:
    if st.button("🎯 Calibrate Gaze"):
        st.session_state.show_calibration = True

with calib_col2:
    # Check if calibration file exists - it's saved in the be directory
    calib_path = Path("calibration.json")
    if check_calibration_exists(calib_path):
        st.session_state.calibration_status = "✅ Calibrated"
    else:
        st.session_state.calibration_status = "❌ Not calibrated"

with calib_col3:
    st.write(f"**Status:** {st.session_state.calibration_status}")

# Calibration dialog
if st.session_state.show_calibration:
    st.divider()
    st.warning("🎯 **Gaze Calibration Mode**")
    st.write("This will open a full-screen calibration window. Follow the instructions:")
    st.write("1. Look at the center point when prompted")
    st.write("2. Look at each of the 9 grid points when they appear")
    st.write("3. Look at the 5 arrow calibration points")
    st.write("4. Press SPACE to continue or wait 10 seconds at each point")
    st.write("5. Press ESC to exit early")
    
    calib_dialog_col1, calib_dialog_col2 = st.columns([1, 1])
    
    with calib_dialog_col1:
        samples = st.slider("Calibration samples per point", 10, 50, 20, 5)
        if st.button("🚀 Start Calibration"):
            with st.spinner("Running gaze calibration..."):
                success, message = calibrate_gaze(calib_path, samples)
                if success:
                    st.success(message)
                    st.session_state.calibration_status = "✅ Calibrated"
                    st.session_state.show_calibration = False
                    st.rerun()
                else:
                    st.error(message)
    
    with calib_dialog_col2:
        if st.button("❌ Cancel Calibration"):
            st.session_state.show_calibration = False
            st.rerun()

# Initialize session state for recording management
if 'recording_name' not in st.session_state:
    st.session_state.recording_name = ""
if 'show_rename_dialog' not in st.session_state:
    st.session_state.show_rename_dialog = False
if 'show_recording_selector' not in st.session_state:
    st.session_state.show_recording_selector = False
if 'selected_recording' not in st.session_state:
    st.session_state.selected_recording = ""
if 'show_comparison' not in st.session_state:
    st.session_state.show_comparison = False
if 'comparison_recording' not in st.session_state:
    st.session_state.comparison_recording = ""
if 'comparison_data' not in st.session_state:
    st.session_state.comparison_data = {"webcam": [], "screen": [], "keyboard": [], "mouse": []}
if 'comparison_synced_frames' not in st.session_state:
    st.session_state.comparison_synced_frames = []
if 'comparison_frame_idx' not in st.session_state:
    st.session_state.comparison_frame_idx = 0
if 'comparison_is_playing' not in st.session_state:
    st.session_state.comparison_is_playing = False
if 'comparison_last_update' not in st.session_state:
    st.session_state.comparison_last_update = time.time()

# Initialize session state for play/pause and recording
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_frame_idx' not in st.session_state:
    st.session_state.current_frame_idx = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'recording_status' not in st.session_state:
    st.session_state.recording_status = "Not Recording"
if 'show_playback' not in st.session_state:
    st.session_state.show_playback = False
if 'frame_analysis_results' not in st.session_state:
    st.session_state.frame_analysis_results = None
if 'frame_summary' not in st.session_state:
    st.session_state.frame_summary = None

# Recording controls section
st.markdown("### Recording Controls")
recording_col1, recording_col2, recording_col3 = st.columns([1, 1, 2])

with recording_col1:
    if st.button("🎬 Start Recording"):
        # Check if there's an existing recording with actual data
        has_existing_data = False
        if frames_exist:
            # Check if there are actual files in the recording directories
            webcam_files = os.listdir(webcam_dir) if os.path.exists(webcam_dir) else []
            screen_files = os.listdir(screen_dir) if os.path.exists(screen_dir) else []
            log_exists = os.path.exists("./frames/combined_log.json")
            
            # Consider it an existing recording only if there are actual files
            has_existing_data = len(webcam_files) > 0 or len(screen_files) > 0 or log_exists
        
        if has_existing_data:
            # Ask user to rename existing recording
            st.session_state.show_rename_dialog = True
        else:
            # Start new recording
            success, message = start_recording()
            if success:
                st.session_state.recording_status = "Recording..."
                st.success(message)
            else:
                st.error(message)

with recording_col2:
    if st.button("⏹️ Stop Recording"):
        success, message = stop_recording()
        if success:
            st.session_state.recording_status = "Not Recording"
            st.success(message)
            # Reload data after recording
            st.rerun()
        else:
            st.error(message)

with recording_col3:
    st.write(f"**Status:** {st.session_state.recording_status}")

# Rename dialog for existing recordings
if st.session_state.show_rename_dialog:
    st.divider()
    st.warning("⚠️ Existing recording found!")
    st.write("Please provide a name for the existing recording to save it before starting a new one.")
    
    rename_col1, rename_col2, rename_col3 = st.columns([2, 1, 1])
    
    with rename_col1:
        new_name = st.text_input("Recording name:", value=st.session_state.recording_name, 
                                placeholder="e.g., recording_2024_01_15")
    
    with rename_col2:
        if st.button("💾 Save & Start New"):
            if new_name.strip():
                # Rename existing frames directory
                try:
                    import shutil
                    from datetime import datetime
                    
                    # Create backup directory with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_name = f"{new_name}_{timestamp}"
                    backup_path = f"./recordings/{backup_name}"
                    
                    # Create recordings directory if it doesn't exist
                    os.makedirs("./recordings", exist_ok=True)
                    
                    # Move existing frames to backup
                    shutil.move("./frames", backup_path)
                    
                    st.success(f"✅ Existing recording saved as '{backup_name}'")
                    st.session_state.show_rename_dialog = False
                    st.session_state.recording_name = ""
                    
                    # Start new recording
                    success, message = start_recording()
                    if success:
                        st.session_state.recording_status = "Recording..."
                        st.success(message)
                    else:
                        st.error(message)
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error saving recording: {e}")
            else:
                st.error("Please provide a name for the recording")
    
    with rename_col3:
        if st.button("❌ Cancel"):
            st.session_state.show_rename_dialog = False
            st.session_state.recording_name = ""
            st.rerun()

# Recording selector dialog
if st.session_state.show_recording_selector:
    st.divider()
    st.header("📁 Select Existing Recording")
    
    # Get list of available recordings
    recordings_dir = "./recordings"
    available_recordings = []
    
    if os.path.exists(recordings_dir):
        for item in os.listdir(recordings_dir):
            item_path = os.path.join(recordings_dir, item)
            if os.path.isdir(item_path):
                # Check if it has the required structure
                log_file = os.path.join(item_path, "combined_log.json")
                if os.path.exists(log_file):
                    available_recordings.append(item)
    
    if available_recordings:
        st.write("Choose a recording to process:")
        
        # Sort recordings by creation time (newest first)
        available_recordings.sort(key=lambda x: os.path.getctime(os.path.join(recordings_dir, x)), reverse=True)
        
        for recording in available_recordings:
            recording_path = os.path.join(recordings_dir, recording)
            creation_time = datetime.fromtimestamp(os.path.getctime(recording_path))
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{recording}**")
                st.write(f"Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                if st.button("📂 Load", key=f"load_{recording}"):
                    # Copy the selected recording to frames directory
                    try:
                        import shutil
                        
                        # Remove existing frames directory if it exists
                        if os.path.exists("./frames"):
                            shutil.rmtree("./frames")
                        
                        # Copy selected recording to frames directory
                        shutil.copytree(recording_path, "./frames")
                        
                        st.success(f"✅ Loaded recording: {recording}")
                        st.session_state.show_recording_selector = False
                        st.session_state.show_playback = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error loading recording: {e}")
            
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{recording}"):
                    try:
                        import shutil
                        shutil.rmtree(recording_path)
                        st.success(f"✅ Deleted recording: {recording}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error deleting recording: {e}")
    else:
        st.warning("No saved recordings found.")
    
    if st.button("❌ Cancel"):
        st.session_state.show_recording_selector = False
        st.rerun()

# Comparison selector dialog
if st.session_state.show_comparison:
    st.divider()
    st.header("📁 Select Recording to Compare")
    
    # Get list of available recordings
    recordings_dir = "./recordings"
    available_recordings = []
    
    if os.path.exists(recordings_dir):
        for item in os.listdir(recordings_dir):
            item_path = os.path.join(recordings_dir, item)
            if os.path.isdir(item_path):
                # Check if it has the required structure
                log_file = os.path.join(item_path, "combined_log.json")
                if os.path.exists(log_file):
                    available_recordings.append(item)
    
    if available_recordings:
        st.write("Choose a recording to compare with the current one:")
        
        # Sort recordings by creation time (newest first)
        available_recordings.sort(key=lambda x: os.path.getctime(os.path.join(recordings_dir, x)), reverse=True)
        
        for recording in available_recordings:
            recording_path = os.path.join(recordings_dir, recording)
            creation_time = datetime.fromtimestamp(os.path.getctime(recording_path))
            
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{recording}**")
                st.write(f"Created: {creation_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            with col2:
                if st.button("📊 Compare", key=f"compare_{recording}"):
                    # Load comparison data
                    comparison_data = load_comparison_data(recording_path)
                    if comparison_data:
                        st.session_state.comparison_data = comparison_data["data"]
                        st.session_state.comparison_synced_frames = comparison_data["synced_frames"]
                        st.session_state.comparison_recording = recording
                        st.session_state.comparison_frame_idx = 0
                        st.session_state.show_comparison = False
                        st.success(f"✅ Loaded comparison recording: {recording}")
                        st.rerun()
                    else:
                        st.error(f"❌ Error loading comparison recording: {recording}")
            
            with col3:
                if st.button("❌ Cancel", key=f"cancel_compare_{recording}"):
                    st.session_state.show_comparison = False
                    st.rerun()
    else:
        st.warning("No saved recordings found for comparison.")
    
    if st.button("❌ Cancel Comparison"):
        st.session_state.show_comparison = False
        st.rerun()

# Check if recording is active
if is_recording_active():
    st.session_state.recording_status = "Recording..."
    st.info("🔄 Recording is currently active...")

st.divider()

# Post-processing section (hidden during comparison)
if not st.session_state.show_comparison:
    st.markdown("### Post-Processing")
    post_col1, post_col2 = st.columns([1, 3])

    with post_col1:
        if st.button("🔍 Process Recording"):
            # Check if there's a current recording or existing recordings to choose from
            current_frames_exist = os.path.exists("./frames")
            recordings_exist = os.path.exists("./recordings")
            
            if not current_frames_exist and recordings_exist:
                # Show recording selector
                st.session_state.show_recording_selector = True
            else:
                # Process the recording with response time analysis and emotion detection
                if current_frames_exist and os.path.exists("./frames/combined_log.json"):
                    try:
                        # Create a progress container for multiple steps
                        progress_container = st.container()
                        
                        with progress_container:
                            st.write("🔍 Processing recording...")
                            
                            # Step 1: Gaze detection (if calibrated)
                            calib_path = Path("calibration.json")
                            if check_calibration_exists(calib_path):
                                with st.spinner("👁️ Running gaze detection..."):
                                    gaze_success, gaze_message = process_gaze_detection(
                                        "./frames/webcam",
                                        "./frames/gaze_output",
                                        calib_path
                                    )
                                    if gaze_success:
                                        # Integrate gaze data with combined log
                                        gaze_log_path = "./frames/gaze_output/gaze_log.json"
                                        if os.path.exists(gaze_log_path):
                                            integrate_success, integrate_message = integrate_gaze_with_combined_log(
                                                "./frames/combined_log.json",
                                                gaze_log_path
                                            )
                                            if integrate_success:
                                                st.success(f"✅ {gaze_message} {integrate_message}")
                                                
                                                # Generate gaze heatmap after successful gaze processing
                                                with st.spinner("🔥 Generating gaze heatmap..."):
                                                    try:
                                                        import subprocess
                                                        import sys
                                                        
                                                        # Run the gaze heatmap script
                                                        result = subprocess.run([
                                                            sys.executable, 
                                                            "gaze_heatmap.py"
                                                        ], capture_output=True, text=True, cwd=".")
                                                        
                                                        if result.returncode == 0:
                                                            st.success("✅ Gaze heatmap generated successfully!")
                                                            st.info("📊 Heatmap saved as './frames/gaze_heatmap.png'")
                                                        else:
                                                            st.warning(f"⚠️ Heatmap generation had issues: {result.stderr}")
                                                    except Exception as e:
                                                        st.warning(f"⚠️ Could not generate heatmap: {e}")
                                            else:
                                                st.warning(f"⚠️ {gaze_message} but {integrate_message}")
                                        else:
                                            st.warning(f"⚠️ {gaze_message}")
                                    else:
                                        st.warning(f"⚠️ Gaze detection failed: {gaze_message}")
                            else:
                                st.info("ℹ️ Skipping gaze detection - no calibration found. Run calibration first.")
                            
                            # Step 2: Response time analysis
                            with st.spinner("📊 Analyzing response times..."):
                                responses = update_combined_log_with_responses(
                                    "./frames/combined_log.json",
                                    "./frames",
                                    "./frames/combined_log.json",  # Update the same file
                                    similarity_threshold=0.8
                                )
                                
                                if responses:
                                    st.success(f"✅ Response time analysis complete! Found {len(responses)} click responses.")
                                else:
                                    st.warning("⚠️ No click responses found. Make sure you have mouse clicks in your recording.")
                            
                            # Step 4: Emotion detection
                            with st.spinner("😊 Analyzing emotions in webcam frames..."):
                                emotion_result = analyze_emotions_in_recording(
                                    "./frames/combined_log.json",
                                    "./frames/webcam/"
                                )
                                
                                if emotion_result:
                                    # Count frames with emotion data
                                    emotion_frames = sum(1 for entry in emotion_result.get("webcam", []) 
                                                       if "emotion" in entry)
                                    total_frames = len(emotion_result.get("webcam", []))
                                    st.success(f"✅ Emotion analysis complete! {emotion_frames}/{total_frames} frames have emotion data.")
                                    
                                    # Calculate emotion statistics
                                    if emotion_frames > 0:
                                        emotion_counts = {}
                                        for entry in emotion_result.get("webcam", []):
                                            if "emotion" in entry:
                                                dominant = get_dominant_emotion(entry["emotion"])
                                                emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
                                        
                                        if emotion_counts:
                                            # Show top emotions
                                            top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                                            emotion_summary = " | ".join([
                                                f"{emotion.title()}: {count} frames"
                                                for emotion, count in top_emotions
                                            ])
                                            st.info(f"📊 **Emotion Summary:** {emotion_summary}")
                                    
                                    # Show processing efficiency info
                                    if emotion_frames < total_frames:
                                        skipped_frames = total_frames - emotion_frames
                                        st.info(f"\U0001F4A1 **Efficiency:** Skipped {skipped_frames} frames that already had emotion data.")
                                else:
                                    st.warning("\u26A0\uFE0F Emotion analysis failed or no webcam frames found.")
                            # --- Final Report Generation ---
                            if (not st.session_state.show_comparison and 
                                not st.session_state.comparison_is_playing and 
                                not st.session_state.comparison_recording):
                                try:
                                    gaze_freq, max_emotion, number_of_mouse_clicks, number_of_keyboard_events = get_summary('frames')
                                    with st.spinner("🤖 Generating AI summary..."):
                                        ollama_summary = get_ollama_summary(gaze_freq, max_emotion, number_of_mouse_clicks, number_of_keyboard_events)
                                    st.session_state.ollama_summary = ollama_summary
                                    st.markdown('### 📝 Final Report Summary')
                                    st.info(ollama_summary)
                                except Exception as e:
                                    st.warning(f"Could not generate final report: {e}")
                        
                        st.session_state.show_playback = True
                    except Exception as e:
                        st.error(f"❌ Error during processing: {e}")
                        st.session_state.show_playback = True
                else:
                    st.warning("⚠️ No recording data found. Please record some data first.")
                    st.session_state.show_playback = True
            st.rerun()

        # Separate button for frame analysis
        if st.button("🖼️ Analyze Frame Interactions"):
            if os.path.exists("./frames") and os.path.exists("./frames/combined_log.json"):
                try:
                    with st.spinner("🖼️ Analyzing frame interactions..."):
                        # Check if frame analysis already exists
                        with open("./frames/combined_log.json", "r") as f:
                            current_log = json.load(f)
                        
                        # Check if responses already have analysis
                        responses = current_log.get('responses', [])
                        has_existing_analysis = any('analysis' in response for response in responses)
                        
                        if has_existing_analysis:
                            st.success("✅ Frame analysis already exists! Loading existing results.")
                            # Load existing analysis results
                            st.session_state.frame_analysis_results = current_log
                            st.session_state.frame_summary = current_log.get('session_analysis', {}).get('summary', 'No summary available')
                        else:
                            # Run frame analysis
                            frame_analysis_result, frame_summary = run_frame_analysis(
                                "./frames/combined_log.json",
                                "./frames/screen"
                            )
                            
                            if frame_analysis_result:
                                st.success(f"✅ Frame analysis complete! Analyzed {len(frame_analysis_result.get('responses', []))} interactions.")
                                # Store the frame analysis results for display
                                st.session_state.frame_analysis_results = frame_analysis_result
                                st.session_state.frame_summary = frame_summary
                            else:
                                st.warning(f"⚠️ Frame analysis failed: {frame_summary}")
                        
                        st.session_state.show_playback = True
                except Exception as e:
                    st.error(f"❌ Error during frame analysis: {e}")
            else:
                st.warning("⚠️ No recording data found. Please record some data first.")
            st.rerun()

    with post_col2:
        # Create two columns for stats and heatmap
        stats_col, heatmap_col = st.columns([2, 1])
        
        with stats_col:
            # Check frames existence dynamically
            current_frames_exist = os.path.exists("./frames")
            if not current_frames_exist:
                if os.path.exists("./recordings"):
                    st.write("📊 **Available Frames:** 0 (Select from existing recordings)")
                else:
                    st.write("📊 **Available Frames:** 0 (No recordings found)")
            elif synced_frames:
                st.write(f"📊 **Available Frames:** {len(synced_frames)}")
                st.write(f"⌨️ **Keyboard Events:** {len(keyboard_entries)}")
                st.write(f"🖱️ **Mouse Events:** {len(mouse_entries)}")
            else:
                st.write("📊 **Available Frames:** 0 (Record some data first)")
        
        with heatmap_col:
            # Compact gaze heatmap generation
            gaze_output_exists = os.path.exists("./frames/gaze_log.json")
            
            if gaze_output_exists:
                if st.button("🔥 Generate Heatmap"):
                    with st.spinner("Generating..."):
                        try:
                            import subprocess
                            import sys
                            result = subprocess.run([sys.executable, "gaze_heatmap.py"], 
                                                 capture_output=True, text=True, cwd=".")
                            if result.returncode == 0:
                                st.success("✅ Generated!")
                                heatmap_path = "./frames/gaze_heatmap.png"
                                if os.path.exists(heatmap_path):
                                    st.image(heatmap_path, use_container_width=True)
                            else:
                                st.error("❌ Failed")
                        except Exception as e:
                            st.error("❌ Error")
                
                if os.path.exists("./frames/gaze_heatmap.png"):
                    st.caption("✅ Heatmap exists")
                else:
                    st.caption("ℹ️ Click to generate")
            else:
                st.caption("⚠️ No gaze data")

# Playback section (only shown when post-processing is clicked)
if st.session_state.show_playback:
    st.markdown("### Playback Controls")
    
    # Reload data dynamically to ensure we have the latest
    current_frames_exist = os.path.exists("./frames")
    current_data = {"webcam": [], "screen": [], "keyboard": [], "mouse": []}
    
    # Compact debug information
    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.caption(f"Frames: {current_frames_exist}")
    with debug_col2:
        if current_frames_exist and os.path.exists("./frames/combined_log.json"):
            try:
                with open("./frames/combined_log.json", "r") as f:
                    current_data = json.load(f)
                st.caption(f"W:{len(current_data.get('webcam', []))} S:{len(current_data.get('screen', []))} K:{len(current_data.get('keyboard', []))} M:{len(current_data.get('mouse', []))}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                st.caption("Log error")
        else:
            st.caption("No log file")
    
    # Parse current data
    current_webcam_entries = current_data.get("webcam", [])
    current_screen_entries = current_data.get("screen", [])
    current_keyboard_entries = current_data.get("keyboard", [])
    current_mouse_entries = current_data.get("mouse", [])
    
    # Parse times into datetime for sorting
    for entry in current_webcam_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    for entry in current_screen_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    for entry in current_keyboard_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    for entry in current_mouse_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    
    # Build current synced frames
    current_combined_times = sorted(set([e["dt"] for e in current_webcam_entries] + [e["dt"] for e in current_screen_entries]))
    
    current_synced_frames = []
    latest_webcam = None
    latest_screen = None
    
    for t in current_combined_times:
        for w in current_webcam_entries:
            if w["dt"] <= t:
                latest_webcam = w
            else:
                break
        for s in current_screen_entries:
            if s["dt"] <= t:
                latest_screen = s
            else:
                break
        if latest_webcam and latest_screen:
            # Check if files actually exist
            webcam_file = os.path.join("frames/webcam", latest_webcam["file"])
            screen_file = os.path.join("frames/screen", latest_screen["file"])
            
            if os.path.exists(webcam_file) and os.path.exists(screen_file):
                current_synced_frames.append({
                    "time": t,
                    "webcam_file": webcam_file,
                    "screen_file": screen_file
                })
    
    # Compact frame info
    frame_info_col1, frame_info_col2 = st.columns(2)
    with frame_info_col1:
        st.caption(f"Combined timestamps: {len(current_combined_times)}")
    with frame_info_col2:
        st.caption(f"Synced frames: {len(current_synced_frames)}")
    
    # Add comparison option
    if current_synced_frames and os.path.exists("./recordings"):
        comparison_col1, comparison_col2 = st.columns([1, 3])
        
        with comparison_col1:
            if st.button("🔍 Compare with Previous Recording"):
                st.session_state.show_comparison = True
                st.rerun()
        
        with comparison_col2:
            if st.session_state.show_comparison:
                st.caption("✅ Comparison mode active")
            else:
                st.caption("Compare with previous recording")
    
    if current_synced_frames and not st.session_state.show_comparison and not st.session_state.comparison_recording and not st.session_state.comparison_synced_frames:
        # Compact playback controls
        controls_col1, controls_col2, controls_col3, controls_col4 = st.columns([1, 2, 1, 2])

        with controls_col1:
            play_pause_text = "⏸️" if st.session_state.is_playing else "▶️"
            if st.button(play_pause_text, key="play_pause"):
                st.session_state.is_playing = not st.session_state.is_playing
                st.session_state.last_update = time.time()

        with controls_col2:
            # Frame slider
            if len(current_synced_frames) > 0:
                frame_idx = st.slider("Frame", 0, len(current_synced_frames) - 1, st.session_state.current_frame_idx, 1)
                st.session_state.current_frame_idx = frame_idx
            else:
                st.caption("No frames")

        with controls_col3:
            if st.button("⏹️"):
                st.session_state.is_playing = False
                st.session_state.current_frame_idx = 0

        with controls_col4:
            # Ensure frame index is within bounds
            if st.session_state.current_frame_idx >= len(current_synced_frames):
                st.session_state.current_frame_idx = len(current_synced_frames) - 1
            current_frame = current_synced_frames[st.session_state.current_frame_idx]
            st.caption(f"Frame {st.session_state.current_frame_idx + 1}/{len(current_synced_frames)}")

        # Find events from the past 2-3 seconds for context
        current_time = current_frame["time"]
        time_window = timedelta(seconds=3)  # Show events from past 3 seconds
        
        # Find keyboard events in the time window
        recent_keyboard_events = []
        for event in current_keyboard_entries:
            if current_time - time_window <= event["dt"] <= current_time:
                recent_keyboard_events.append(event)
        
        # Find mouse events in the time window
        recent_mouse_events = []
        for event in current_mouse_entries:
            if current_time - time_window <= event["dt"] <= current_time:
                recent_mouse_events.append(event)

        # Video frames with better proportions
        col1, col2 = st.columns([1, 2])

        with col1:
            try:
                # Check if gaze output exists and use it instead of webcam
                gaze_output_file = current_frame["webcam_file"].replace("/webcam/", "/gaze_output/")
                if os.path.exists(gaze_output_file):
                    # Use gaze output frame (with annotations)
                    st.image(Image.open(gaze_output_file), caption="Webcam (Gaze Annotated)", use_container_width=True)
                elif os.path.exists(current_frame["webcam_file"]):
                    # Fall back to original webcam frame
                    st.image(Image.open(current_frame["webcam_file"]), caption="Webcam", use_container_width=True)
                else:
                    st.caption("Webcam file not found")
                
                # Display emotion information if available
                current_webcam_entry = None
                for entry in current_webcam_entries:
                    if entry["file"] == os.path.basename(current_frame["webcam_file"]):
                        current_webcam_entry = entry
                        break
                
                if current_webcam_entry and "emotion" in current_webcam_entry:
                    emotion_data = current_webcam_entry["emotion"]
                    dominant_emotion = get_dominant_emotion(emotion_data)
                    
                    # Create emotion display with emojis
                    emotion_emojis = {
                        "angry": "😠",
                        "disgust": "🤢", 
                        "fear": "😨",
                        "happy": "😊",
                        "sad": "😢",
                        "surprise": "😲",
                        "neutral": "😐"
                    }
                    
                    # Get dominant emotion probability
                    dominant_prob = float(emotion_data.get(dominant_emotion, 0))
                    
                    # Create a more prominent emotion display
                    emotion_col1, emotion_col2 = st.columns([1, 2])
                    
                    with emotion_col1:
                        # Large emotion emoji and name
                        st.markdown(f"## {emotion_emojis.get(dominant_emotion, '😐')}")
                        st.markdown(f"**{dominant_emotion.title()}**")
                        st.markdown(f"**{dominant_prob:.1f}%**")
                    
                    with emotion_col2:
                        # Show top 2 emotions as a compact list
                        emotions_sorted = sorted(emotion_data.items(), key=lambda x: float(x[1]), reverse=True)
                        top_emotions = emotions_sorted[:2]
                        
                        for emotion, prob in top_emotions:
                            prob_val = float(prob)
                            emoji = emotion_emojis.get(emotion, '😐')
                            st.caption(f"{emoji} {emotion.title()}: {prob_val:.1f}%")
                
                # Display gaze information if available
                if current_webcam_entry and "gaze_engaged" in current_webcam_entry:
                    gaze_engaged = current_webcam_entry.get("gaze_engaged", False)
                    if gaze_engaged:
                        gaze_yaw = current_webcam_entry.get("gaze_yaw")
                        gaze_pitch = current_webcam_entry.get("gaze_pitch")
                        gaze_sector = current_webcam_entry.get("gaze_sector")
                        
                        st.divider()
                        st.markdown("### 👁️ Gaze Data")
                        
                        gaze_col1, gaze_col2 = st.columns(2)
                        
                        with gaze_col1:
                            if gaze_yaw is not None:
                                st.caption(f"**Yaw:** {gaze_yaw:.1f}°")
                            if gaze_pitch is not None:
                                st.caption(f"**Pitch:** {gaze_pitch:.1f}°")
                        
                        with gaze_col2:
                            if gaze_sector is not None:
                                st.caption(f"**Sector:** {gaze_sector}")
                            st.caption(f"**Engaged:** {'✅' if gaze_engaged else '❌'}")
                    else:
                        st.caption("👁️ **Gaze:** Not engaged")
            except Exception as e:
                st.caption("Webcam error")

        with col2:
            try:
                if os.path.exists(current_frame["screen_file"]):
                    # Check if we should highlight a click on the screen
                    screen_img = Image.open(current_frame["screen_file"])
                    # Find the most recent mouse press event for highlighting
                    recent_mouse_press = None
                    for event in reversed(recent_mouse_events):
                        if event["pressed"]:
                            recent_mouse_press = event
                            break
                    
                    if recent_mouse_press:
                        # Highlight the click position
                        button_type = recent_mouse_press["button"].split(".")[-1].lower()
                        screen_img = add_click_highlight(
                            screen_img, 
                            recent_mouse_press["position"], 
                            button_type
                        )
                    
                    st.image(screen_img, caption="Screen", use_container_width=True)
                else:
                    st.caption("Screen file not found")
            except Exception as e:
                st.caption("Screen error")

        # Recent interaction events display - compact version
        interaction_col1, interaction_col2 = st.columns(2)
        
        with interaction_col1:
            st.caption("**⌨️ Recent Keys:**")
            if recent_keyboard_events:
                # Sort by time (most recent first)
                recent_keyboard_events.sort(key=lambda x: x["dt"], reverse=True)
                # Show only the last 5 keyboard events
                for event in recent_keyboard_events[:5]:
                    time_diff = (current_time - event["dt"]).total_seconds()
                    event_type = "🔴" if event["event"] == "press" else "🟢"
                    st.caption(f"{event_type} `{event['key']}` ({time_diff:.1f}s)")
            else:
                st.caption("*No keyboard events*")
        
        with interaction_col2:
            st.caption("**🖱️ Recent Clicks:**")
            if recent_mouse_events:
                # Sort by time (most recent first)
                recent_mouse_events.sort(key=lambda x: x["dt"], reverse=True)
                # Show only the last 3 mouse events
                for event in recent_mouse_events[:3]:
                    time_diff = (current_time - event["dt"]).total_seconds()
                    action = "🔴" if event["pressed"] else "🟢"
                    button = event["button"].split(".")[-1]
                    pos = event["position"]
                    
                    # Check if this click has response time data
                    response_info = ""
                    if event["pressed"] and event["button"] == "Button.left":
                        # Load response data if available
                        try:
                            with open("./frames/combined_log.json", "r") as f:
                                current_data = json.load(f)
                            
                            if "responses" in current_data:
                                for response in current_data["responses"]:
                                    if response["click_time"] == event["time"]:
                                        response_info = f" ⚡ {response['response_time_ms']:.1f}ms"
                                        break
                        except:
                            pass
                    
                    st.caption(f"{action} {button} ({pos[0]}, {pos[1]}) ({time_diff:.1f}s){response_info}")
            else:
                st.caption("*No mouse events*")
        
        # Emotion charts section
        if any("emotion" in entry for entry in current_webcam_entries):
            st.divider()
            st.markdown("### 📊 Emotion Analysis")
            
            # Emotion trend chart
            trend_fig = create_emotion_trend_chart(current_webcam_entries, "Emotion Probabilities Over Time")
            if trend_fig:
                st.plotly_chart(trend_fig, use_container_width=True)
            else:
                st.warning("No emotion data available for trend chart.")
        else:
            st.warning("No recorded frames found. Start recording to capture webcam and screen data.")

# Auto-play functionality
if st.session_state.is_playing and st.session_state.show_playback:
    # Reload current data for auto-play
    current_frames_exist = os.path.exists("./frames")
    current_data = {"webcam": [], "screen": [], "keyboard": [], "mouse": []}
    
    if current_frames_exist and os.path.exists("./frames/combined_log.json"):
        try:
            with open("./frames/combined_log.json", "r") as f:
                current_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            pass
    
    # Build current synced frames for auto-play
    current_webcam_entries = current_data.get("webcam", [])
    current_screen_entries = current_data.get("screen", [])
    
    # Parse times into datetime for sorting
    for entry in current_webcam_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    for entry in current_screen_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    
    # Build current synced frames
    current_combined_times = sorted(set([e["dt"] for e in current_webcam_entries] + [e["dt"] for e in current_screen_entries]))
    current_synced_frames = []
    latest_webcam = None
    latest_screen = None
    
    for t in current_combined_times:
        for w in current_webcam_entries:
            if w["dt"] <= t:
                latest_webcam = w
            else:
                break
        for s in current_screen_entries:
            if s["dt"] <= t:
                latest_screen = s
            else:
                break
        if latest_webcam and latest_screen:
            current_synced_frames.append({
                "time": t,
                "webcam_file": os.path.join("frames/webcam", latest_webcam["file"]),
                "screen_file": os.path.join("frames/screen", latest_screen["file"])
            })
    
    current_time = time.time()
    if current_time - st.session_state.last_update >= 0.04:  # 25 FPS (faster)
        if st.session_state.current_frame_idx < len(current_synced_frames) - 1:
            st.session_state.current_frame_idx += 1
            st.session_state.last_update = current_time
        else:
            st.session_state.is_playing = False

        # Auto-refresh when playing
        if st.session_state.is_playing:
            st.rerun()

# Frame Analysis Results Display
if st.session_state.show_playback and hasattr(st.session_state, 'frame_analysis_results') and st.session_state.frame_analysis_results:
    # Check if there are actual analysis results
    responses = st.session_state.frame_analysis_results.get('responses', [])
    has_analysis_results = any('analysis' in response for response in responses)
    
    if has_analysis_results:
        st.divider()
        st.markdown("### 🖼️ Frame Analysis Results")
        
        # Add option to re-run frame analysis
        if st.button("🔄 Re-run Frame Analysis"):
            try:
                # Clear existing analysis from the log file
                with open("./frames/combined_log.json", "r") as f:
                    current_log = json.load(f)
                
                # Remove analysis from responses
                for response in current_log.get('responses', []):
                    if 'analysis' in response:
                        del response['analysis']
                
                # Remove session analysis
                if 'session_analysis' in current_log:
                    del current_log['session_analysis']
                
                # Save the cleaned log
                with open("./frames/combined_log.json", "w") as f:
                    json.dump(current_log, f, indent=2)
                
                # Clear any analysis data from the log file
                # (The analysis data is now stored in the same file, so we don't need to remove a separate file)
                
                # Clear session state
                if hasattr(st.session_state, 'frame_analysis_results'):
                    st.session_state.frame_analysis_results = None
                if hasattr(st.session_state, 'frame_summary'):
                    st.session_state.frame_summary = None
                
                st.success("✅ Cleared existing frame analysis. Click '🖼️ Analyze Frame Interactions' to generate new analysis.")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error clearing analysis: {e}")
        
        # Display session summary if available
        if hasattr(st.session_state, 'frame_summary') and st.session_state.frame_summary:
            st.markdown("#### 📝 Session Summary")
            st.info(st.session_state.frame_summary)
        
        # Display interaction summaries as a table
        if responses and any('analysis' in response for response in responses):
            st.markdown("#### 📊 Interaction Summaries")
            
            # Create a table of interaction summaries
            interaction_data = []
            for i, response in enumerate(responses):
                if 'analysis' in response and 'interaction_summary' in response['analysis']:
                    # Extract click time for reference
                    click_time = response.get('click_time', 'Unknown')
                    
                    # Get response time if available
                    response_time = response.get('response_time_ms', 'N/A')
                    if response_time != 'N/A':
                        response_time = f"{response_time:.1f}ms"
                    
                    # Get the interaction summary
                    summary = response['analysis']['interaction_summary']
                    
                    # Truncate summary if too long for table display
                    if len(summary) > 200:
                        summary = summary[:200] + "..."
                    
                    interaction_data.append({
                        "Interaction #": i + 1,
                        "Click Time": click_time,
                        "Response Time": response_time,
                        "Summary": summary
                    })
            
            if interaction_data:
                # Create a DataFrame for better table display
                import pandas as pd
                df = pd.DataFrame(interaction_data)
                st.dataframe(df, use_container_width=True)
                
                # Add expandable details for each interaction
                st.markdown("#### 🔍 Detailed Analysis")
                for i, response in enumerate(responses):
                    if 'analysis' in response:
                        with st.expander(f"Interaction {i + 1} - {response.get('click_time', 'Unknown')}"):
                            analysis = response['analysis']
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Before State:**")
                                st.write(analysis.get('before_description', 'No description available'))
                            
                            with col2:
                                st.markdown("**After State:**")
                                st.write(analysis.get('after_description', 'No description available'))
                            
                            st.markdown("**Full Interaction Summary:**")
                            st.write(analysis.get('interaction_summary', 'No summary available'))
            else:
                st.warning("No interaction analysis data available.")
        else:
            st.info("No frame analysis results found. Run frame analysis first.")
    else:
        st.info("No frame analysis results available. Process recording to generate analysis.")

# Side-by-side comparison display (independent of main playback)
if st.session_state.comparison_recording and st.session_state.comparison_synced_frames:
    st.divider()
    st.header(f"📊 Side-by-Side Comparison: Current vs {st.session_state.comparison_recording}")
    
    # Comparison controls with separate sliders
    comp_col1, comp_col2, comp_col3, comp_col4 = st.columns([1, 1, 1, 1])
    
    # Ensure frame indices are within bounds BEFORE rendering sliders
    if st.session_state.current_frame_idx >= len(synced_frames):
        st.session_state.current_frame_idx = len(synced_frames) - 1
    if st.session_state.comparison_frame_idx >= len(st.session_state.comparison_synced_frames):
        st.session_state.comparison_frame_idx = len(st.session_state.comparison_synced_frames) - 1
    
    with comp_col1:
        if st.button("⏸️ Pause" if st.session_state.comparison_is_playing else "▶️ Play", key="comp_play_pause"):
            st.session_state.comparison_is_playing = not st.session_state.comparison_is_playing
            st.session_state.comparison_last_update = time.time()
    
    with comp_col2:
        # Frame slider for current recording
        if len(synced_frames) > 0:
            frame_idx = st.slider("Current Frame", 0, len(synced_frames) - 1, st.session_state.current_frame_idx, 1, key="current_slider")
            # Only update if user manually changed the slider (not during auto-play)
            if frame_idx != st.session_state.current_frame_idx:
                st.session_state.current_frame_idx = frame_idx
        else:
            st.warning("No frames available in current recording")
    
    with comp_col3:
        # Frame slider for comparison recording
        if len(st.session_state.comparison_synced_frames) > 0:
            comp_frame_idx = st.slider("Comparison Frame", 0, len(st.session_state.comparison_synced_frames) - 1, st.session_state.comparison_frame_idx, 1, key="comparison_slider")
            # Only update if user manually changed the slider (not during auto-play)
            if comp_frame_idx != st.session_state.comparison_frame_idx:
                st.session_state.comparison_frame_idx = comp_frame_idx
        else:
            st.warning("No frames available in comparison recording")
    
    with comp_col4:
        if st.button("⏹️ Stop", key="comp_stop"):
            st.session_state.comparison_is_playing = False
            st.session_state.current_frame_idx = 0
            st.session_state.comparison_frame_idx = 0
    
    # Auto-play for comparison (advances both sliders independently)
    if st.session_state.comparison_is_playing:
        current_time = time.time()
        if current_time - st.session_state.comparison_last_update >= 0.04:  # 25 FPS (faster)
            # Advance current recording frame
            if st.session_state.current_frame_idx < len(synced_frames) - 1:
                st.session_state.current_frame_idx += 1
            
            # Advance comparison recording frame
            if st.session_state.comparison_frame_idx < len(st.session_state.comparison_synced_frames) - 1:
                st.session_state.comparison_frame_idx += 1
            
            # Stop playing if both recordings have reached the end
            if (st.session_state.current_frame_idx >= len(synced_frames) - 1 and 
                st.session_state.comparison_frame_idx >= len(st.session_state.comparison_synced_frames) - 1):
                st.session_state.comparison_is_playing = False
            else:
                st.session_state.comparison_last_update = current_time
    
    # Get current frames for both recordings with bounds checking
    if len(synced_frames) > 0 and len(st.session_state.comparison_synced_frames) > 0:
        current_frame = synced_frames[st.session_state.current_frame_idx]
        comparison_frame = st.session_state.comparison_synced_frames[st.session_state.comparison_frame_idx]
    else:
        st.error("No valid frames found for comparison")
        st.stop()
    
    # Get comparison data
    comp_data = st.session_state.comparison_data
    comp_keyboard_entries = comp_data.get("keyboard", [])
    comp_mouse_entries = comp_data.get("mouse", [])
    
    # Parse comparison data times
    for entry in comp_keyboard_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    for entry in comp_mouse_entries:
        entry["dt"] = datetime.fromisoformat(entry["time"])
    
    # Find events for comparison
    current_time = current_frame["time"]
    comp_current_time = comparison_frame["time"]
    time_window = timedelta(seconds=3)
    
    # Current recording events
    recent_keyboard_events = []
    recent_mouse_events = []
    for event in keyboard_entries:
        if current_time - time_window <= event["dt"] <= current_time:
            recent_keyboard_events.append(event)
    for event in mouse_entries:
        if current_time - time_window <= event["dt"] <= current_time:
            recent_mouse_events.append(event)
    
    # Comparison recording events
    comp_recent_keyboard_events = []
    comp_recent_mouse_events = []
    for event in comp_keyboard_entries:
        if comp_current_time - time_window <= event["dt"] <= comp_current_time:
            comp_recent_keyboard_events.append(event)
    for event in comp_mouse_entries:
        if comp_current_time - time_window <= event["dt"] <= comp_current_time:
            comp_recent_mouse_events.append(event)
    
    # Side-by-side video frames
    st.write("**Video Comparison**")
    comp_video_col1, comp_video_col2 = st.columns(2)
    
    with comp_video_col1:
        st.write("**Current Recording**")
        # Current recording frames
        curr_col1, curr_col2 = st.columns([1, 2])
        
        with curr_col1:
            try:
                # Check if gaze output exists for current recording
                curr_gaze_output_file = current_frame["webcam_file"].replace("/webcam/", "/gaze_output/")
                if os.path.exists(curr_gaze_output_file):
                    # Use gaze output frame (with annotations)
                    with open(curr_gaze_output_file, "rb") as f:
                        st.image(f.read(), caption="Webcam (Gaze Annotated)", use_container_width=True)
                elif os.path.exists(current_frame["webcam_file"]):
                    # Fall back to original webcam frame
                    with open(current_frame["webcam_file"], "rb") as f:
                        st.image(f.read(), caption="Webcam", use_container_width=True)
                else:
                    st.error(f"Webcam file not found: {current_frame['webcam_file']}")
                
                # Display emotion information for current recording
                current_webcam_entry = None
                for entry in webcam_entries:
                    if entry["file"] == os.path.basename(current_frame["webcam_file"]):
                        current_webcam_entry = entry
                        break
                    
                    if current_webcam_entry and "emotion" in current_webcam_entry:
                        emotion_data = current_webcam_entry["emotion"]
                        dominant_emotion = get_dominant_emotion(emotion_data)
                        
                        # Create emotion display with emojis
                        emotion_emojis = {
                            "angry": "😠",
                            "disgust": "🤢", 
                            "fear": "😨",
                            "happy": "😊",
                            "sad": "😢",
                            "surprise": "😲",
                            "neutral": "😐"
                        }
                        
                        # Get dominant emotion probability
                        dominant_prob = float(emotion_data.get(dominant_emotion, 0))
                        
                        # Compact emotion display for comparison
                        st.caption(f"**{emotion_emojis.get(dominant_emotion, '😐')} {dominant_emotion.title()}: {dominant_prob:.1f}%**")
                        
                        # Show top 2 emotions for compact display
                        emotions_sorted = sorted(emotion_data.items(), key=lambda x: float(x[1]), reverse=True)
                        top_emotions = emotions_sorted[:2]
                        
                        for emotion, prob in top_emotions[1:]:  # Skip the first one as it's already shown
                            prob_val = float(prob)
                            emoji = emotion_emojis.get(emotion, '😐')
                            st.caption(f"{emoji} {emotion.title()}: {prob_val:.1f}%")
                    
                    # Display gaze information for current recording
                    if current_webcam_entry and "gaze_engaged" in current_webcam_entry:
                        gaze_engaged = current_webcam_entry.get("gaze_engaged", False)
                        if gaze_engaged:
                            gaze_yaw = current_webcam_entry.get("gaze_yaw")
                            gaze_pitch = current_webcam_entry.get("gaze_pitch")
                            gaze_sector = current_webcam_entry.get("gaze_sector")
                            
                            st.caption(f"👁️ **Gaze:** Yaw: {gaze_yaw:.1f}°, Pitch: {gaze_pitch:.1f}°, Sector: {gaze_sector}")
                        else:
                            st.caption("👁️ **Gaze:** Not engaged")
                else:
                    st.error(f"Webcam file not found: {current_frame['webcam_file']}")
            except Exception as e:
                st.error(f"Error loading webcam image: {e}")
        
        with curr_col2:
            try:
                if os.path.exists(current_frame["screen_file"]):
                    # Use file path directly instead of Streamlit's media storage
                    with open(current_frame["screen_file"], "rb") as f:
                        screen_data = f.read()
                    
                    # Create PIL image for highlighting
                    screen_img = Image.open(io.BytesIO(screen_data))
                    
                    # Highlight clicks for current recording
                    recent_mouse_press = None
                    for event in reversed(recent_mouse_events):
                        if event["pressed"]:
                            recent_mouse_press = event
                            break
                    
                    if recent_mouse_press:
                        button_type = recent_mouse_press["button"].split(".")[-1].lower()
                        screen_img = add_click_highlight(
                            screen_img, 
                            recent_mouse_press["position"], 
                            button_type
                        )
                    
                    # Convert back to bytes for display
                    img_byte_arr = io.BytesIO()
                    screen_img.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    st.image(img_byte_arr.read(), caption="Screen", use_container_width=True)
                else:
                    st.error(f"Screen file not found: {current_frame['screen_file']}")
            except Exception as e:
                st.error(f"Error loading screen image: {e}")
    
    with comp_video_col2:
        st.write(f"**{st.session_state.comparison_recording}**")
        # Comparison recording frames
        comp_col1, comp_col2 = st.columns([1, 2])
        
        with comp_col1:
            try:
                # Check if gaze output exists for comparison recording
                comp_gaze_output_file = comparison_frame["webcam_file"].replace("/webcam/", "/gaze_output/")
                if os.path.exists(comp_gaze_output_file):
                    # Use gaze output frame (with annotations)
                    with open(comp_gaze_output_file, "rb") as f:
                        st.image(f.read(), caption="Webcam (Gaze Annotated)", use_container_width=True)
                elif os.path.exists(comparison_frame["webcam_file"]):
                    # Fall back to original webcam frame
                    with open(comparison_frame["webcam_file"], "rb") as f:
                        st.image(f.read(), caption="Webcam", use_container_width=True)
                else:
                    st.error(f"Webcam file not found: {comparison_frame['webcam_file']}")
                
                # Display emotion information for comparison recording
                comp_webcam_entry = None
                for entry in st.session_state.comparison_data.get("webcam", []):
                    if entry["file"] == os.path.basename(comparison_frame["webcam_file"]):
                        comp_webcam_entry = entry
                        break
                    
                    if comp_webcam_entry and "emotion" in comp_webcam_entry:
                        emotion_data = comp_webcam_entry["emotion"]
                        dominant_emotion = get_dominant_emotion(emotion_data)
                        
                        # Create emotion display with emojis
                        emotion_emojis = {
                            "angry": "😠",
                            "disgust": "🤢", 
                            "fear": "😨",
                            "happy": "😊",
                            "sad": "😢",
                            "surprise": "😲",
                            "neutral": "😐"
                        }
                        
                        # Get dominant emotion probability
                        dominant_prob = float(emotion_data.get(dominant_emotion, 0))
                        
                        # Compact emotion display for comparison
                        st.caption(f"**{emotion_emojis.get(dominant_emotion, '😐')} {dominant_emotion.title()}: {dominant_prob:.1f}%**")
                        
                        # Show top 2 emotions for compact display
                        emotions_sorted = sorted(emotion_data.items(), key=lambda x: float(x[1]), reverse=True)
                        top_emotions = emotions_sorted[:2]
                        
                        for emotion, prob in top_emotions[1:]:  # Skip the first one as it's already shown
                            prob_val = float(prob)
                            emoji = emotion_emojis.get(emotion, '😐')
                            st.caption(f"{emoji} {emotion.title()}: {prob_val:.1f}%")
                    
                    # Display gaze information for comparison recording
                    if comp_webcam_entry and "gaze_engaged" in comp_webcam_entry:
                        gaze_engaged = comp_webcam_entry.get("gaze_engaged", False)
                        if gaze_engaged:
                            gaze_yaw = comp_webcam_entry.get("gaze_yaw")
                            gaze_pitch = comp_webcam_entry.get("gaze_pitch")
                            gaze_sector = comp_webcam_entry.get("gaze_sector")
                            
                            st.caption(f"👁️ **Gaze:** Yaw: {gaze_yaw:.1f}°, Pitch: {gaze_pitch:.1f}°, Sector: {gaze_sector}")
                        else:
                            st.caption("👁️ **Gaze:** Not engaged")
                else:
                    st.error(f"Webcam file not found: {comparison_frame['webcam_file']}")
            except Exception as e:
                st.error(f"Error loading webcam image: {e}")
        
        with comp_col2:
            try:
                if os.path.exists(comparison_frame["screen_file"]):
                    # Use file path directly instead of Streamlit's media storage
                    with open(comparison_frame["screen_file"], "rb") as f:
                        screen_data = f.read()
                    
                    # Create PIL image for highlighting
                    screen_img = Image.open(io.BytesIO(screen_data))
                    
                    # Highlight clicks for comparison recording
                    comp_recent_mouse_press = None
                    for event in reversed(comp_recent_mouse_events):
                        if event["pressed"]:
                            comp_recent_mouse_press = event
                            break
                    
                    if comp_recent_mouse_press:
                        button_type = comp_recent_mouse_press["button"].split(".")[-1].lower()
                        screen_img = add_click_highlight(
                            screen_img, 
                            comp_recent_mouse_press["position"], 
                            button_type
                        )
                    
                    # Convert back to bytes for display
                    img_byte_arr = io.BytesIO()
                    screen_img.save(img_byte_arr, format='JPEG')
                    img_byte_arr.seek(0)
                    
                    st.image(img_byte_arr.read(), caption="Screen", use_container_width=True)
                else:
                    st.error(f"Screen file not found: {comparison_frame['screen_file']}")
            except Exception as e:
                st.error(f"Error loading screen image: {e}")
    
    # Side-by-side interaction comparison
    st.write("**Interaction Comparison**")
    comp_interaction_col1, comp_interaction_col2 = st.columns(2)
    
    with comp_interaction_col1:
        st.markdown("**Current Recording - Recent Keys:**")
        if recent_keyboard_events:
            recent_keyboard_events.sort(key=lambda x: x["dt"], reverse=True)
            for event in recent_keyboard_events[:5]:
                time_diff = (current_time - event["dt"]).total_seconds()
                event_type = "🔴" if event["event"] == "press" else "🟢"
                st.markdown(f"{event_type} `{event['key']}` ({time_diff:.1f}s)")
        else:
            st.markdown("*No keyboard events*")
        
        st.markdown("**Current Recording - Recent Clicks:**")
        if recent_mouse_events:
            recent_mouse_events.sort(key=lambda x: x["dt"], reverse=True)
            for event in recent_mouse_events[:3]:
                time_diff = (current_time - event["dt"]).total_seconds()
                action = "🔴" if event["pressed"] else "🟢"
                button = event["button"].split(".")[-1]
                pos = event["position"]
                
                # Check if this click has response time data
                response_info = ""
                if event["pressed"] and event["button"] == "Button.left":
                    try:
                        with open("./frames/combined_log.json", "r") as f:
                            current_data = json.load(f)
                        
                        if "responses" in current_data:
                            for response in current_data["responses"]:
                                if response["click_time"] == event["time"]:
                                    response_info = f" ⚡ {response['response_time_ms']:.1f}ms"
                                    break
                    except:
                        pass
                
                st.markdown(f"{action} {button} ({pos[0]}, {pos[1]}) ({time_diff:.1f}s){response_info}")
        else:
            st.markdown("*No mouse events*")
    
    with comp_interaction_col2:
        st.markdown(f"**{st.session_state.comparison_recording} - Recent Keys:**")
        if comp_recent_keyboard_events:
            comp_recent_keyboard_events.sort(key=lambda x: x["dt"], reverse=True)
            for event in comp_recent_keyboard_events[:5]:
                time_diff = (comp_current_time - event["dt"]).total_seconds()
                event_type = "🔴" if event["event"] == "press" else "🟢"
                st.markdown(f"{event_type} `{event['key']}` ({time_diff:.1f}s)")
        else:
            st.markdown("*No keyboard events*")
        
        st.markdown(f"**{st.session_state.comparison_recording} - Recent Clicks:**")
        if comp_recent_mouse_events:
            comp_recent_mouse_events.sort(key=lambda x: x["dt"], reverse=True)
            for event in comp_recent_mouse_events[:3]:
                time_diff = (comp_current_time - event["dt"]).total_seconds()
                action = "🔴" if event["pressed"] else "🟢"
                button = event["button"].split(".")[-1]
                pos = event["position"]
                
                # Check if this click has response time data
                response_info = ""
                if event["pressed"] and event["button"] == "Button.left":
                    try:
                        # Use comparison data for response times
                        if "responses" in st.session_state.comparison_data:
                            for response in st.session_state.comparison_data["responses"]:
                                if response["click_time"] == event["time"]:
                                    response_info = f" ⚡ {response['response_time_ms']:.1f}ms"
                                    break
                    except:
                        pass
                
                st.markdown(f"{action} {button} ({pos[0]}, {pos[1]}) ({time_diff:.1f}s){response_info}")
        else:
            st.markdown("*No mouse events*")
    
    # Side-by-side detailed frame analysis comparison
    st.divider()
    st.markdown("### 🖼️ Detailed Frame Analysis Comparison")
    
    # Check for frame analysis results in both recordings
    current_has_frame_analysis = False
    comp_has_frame_analysis = False
    
    # Check current recording for frame analysis
    try:
        with open("./frames/combined_log.json", "r") as f:
            current_log = json.load(f)
        current_responses = current_log.get('responses', [])
        current_has_frame_analysis = any('analysis' in response for response in current_responses)
    except:
        pass
    
    # Check comparison recording for frame analysis
    try:
        comp_responses = st.session_state.comparison_data.get('responses', [])
        comp_has_frame_analysis = any('analysis' in response for response in comp_responses)
    except:
        pass
    
    if current_has_frame_analysis or comp_has_frame_analysis:
        # Side-by-side frame analysis summaries
        comp_frame_analysis_col1, comp_frame_analysis_col2 = st.columns(2)
        
        with comp_frame_analysis_col1:
            st.markdown("**Current Recording - Frame Analysis**")
            if current_has_frame_analysis:
                # Display session summary if available
                current_session_summary = current_log.get('session_analysis', {}).get('summary', 'No summary available')
                if current_session_summary and current_session_summary != 'No summary available':
                    st.markdown("**Session Summary:**")
                    st.info(current_session_summary)
                
                # Display interaction summaries as a table
                if current_responses and any('analysis' in response for response in current_responses):
                    st.markdown("**Interaction Summaries:**")
                    
                    # Create a table of interaction summaries
                    current_interaction_data = []
                    for i, response in enumerate(current_responses):
                        if 'analysis' in response and 'interaction_summary' in response['analysis']:
                            # Extract click time for reference
                            click_time = response.get('click_time', 'Unknown')
                            
                            # Get response time if available
                            response_time = response.get('response_time_ms', 'N/A')
                            if response_time != 'N/A':
                                response_time = f"{response_time:.1f}ms"
                            
                            # Get the interaction summary
                            summary = response['analysis']['interaction_summary']
                            
                            # Truncate summary if too long for table display
                            if len(summary) > 150:
                                summary = summary[:150] + "..."
                            
                            current_interaction_data.append({
                                "Interaction #": i + 1,
                                "Click Time": click_time,
                                "Response Time": response_time,
                                "Summary": summary
                            })
                    
                    if current_interaction_data:
                        # Create a DataFrame for better table display
                        import pandas as pd
                        current_df = pd.DataFrame(current_interaction_data)
                        st.dataframe(current_df, use_container_width=True)
                        
                        # Add expandable details for each interaction
                        st.markdown("**Detailed Analysis:**")
                        for i, response in enumerate(current_responses):
                            if 'analysis' in response:
                                with st.expander(f"Interaction {i + 1} - {response.get('click_time', 'Unknown')}"):
                                    analysis = response['analysis']
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Before State:**")
                                        st.write(analysis.get('before_description', 'No description available'))
                                    
                                    with col2:
                                        st.markdown("**After State:**")
                                        st.write(analysis.get('after_description', 'No description available'))
                                    
                                    st.markdown("**Full Interaction Summary:**")
                                    st.write(analysis.get('interaction_summary', 'No summary available'))
                    else:
                        st.warning("No interaction analysis data available.")
                else:
                    st.info("No frame analysis results found.")
            else:
                st.info("No frame analysis available for current recording.")
        
        with comp_frame_analysis_col2:
            st.markdown(f"**{st.session_state.comparison_recording} - Frame Analysis**")
            if comp_has_frame_analysis:
                # Display session summary if available
                comp_session_summary = st.session_state.comparison_data.get('session_analysis', {}).get('summary', 'No summary available')
                if comp_session_summary and comp_session_summary != 'No summary available':
                    st.markdown("**Session Summary:**")
                    st.info(comp_session_summary)
                
                # Display interaction summaries as a table
                if comp_responses and any('analysis' in response for response in comp_responses):
                    st.markdown("**Interaction Summaries:**")
                    
                    # Create a table of interaction summaries
                    comp_interaction_data = []
                    for i, response in enumerate(comp_responses):
                        if 'analysis' in response and 'interaction_summary' in response['analysis']:
                            # Extract click time for reference
                            click_time = response.get('click_time', 'Unknown')
                            
                            # Get response time if available
                            response_time = response.get('response_time_ms', 'N/A')
                            if response_time != 'N/A':
                                response_time = f"{response_time:.1f}ms"
                            
                            # Get the interaction summary
                            summary = response['analysis']['interaction_summary']
                            
                            # Truncate summary if too long for table display
                            if len(summary) > 150:
                                summary = summary[:150] + "..."
                            
                            comp_interaction_data.append({
                                "Interaction #": i + 1,
                                "Click Time": click_time,
                                "Response Time": response_time,
                                "Summary": summary
                            })
                    
                    if comp_interaction_data:
                        # Create a DataFrame for better table display
                        import pandas as pd
                        comp_df = pd.DataFrame(comp_interaction_data)
                        st.dataframe(comp_df, use_container_width=True)
                        
                        # Add expandable details for each interaction
                        st.markdown("**Detailed Analysis:**")
                        for i, response in enumerate(comp_responses):
                            if 'analysis' in response:
                                with st.expander(f"Interaction {i + 1} - {response.get('click_time', 'Unknown')}"):
                                    analysis = response['analysis']
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**Before State:**")
                                        st.write(analysis.get('before_description', 'No description available'))
                                    
                                    with col2:
                                        st.markdown("**After State:**")
                                        st.write(analysis.get('after_description', 'No description available'))
                                    
                                    st.markdown("**Full Interaction Summary:**")
                                    st.write(analysis.get('interaction_summary', 'No summary available'))
                    else:
                        st.warning("No interaction analysis data available.")
                else:
                    st.info("No frame analysis results found.")
            else:
                st.info("No frame analysis available for comparison recording.")
    else:
        st.info("No frame analysis results available for either recording. Process recordings to generate analysis.")
    
    # Emotion comparison charts
    current_has_emotions = any("emotion" in entry for entry in webcam_entries)
    comp_has_emotions = any("emotion" in entry for entry in st.session_state.comparison_data.get("webcam", []))
    
    if current_has_emotions or comp_has_emotions:
        st.divider()
        st.markdown("### 📊 Emotion Analysis Comparison")
        
        # Side-by-side emotion trend charts
        comp_trend_col1, comp_trend_col2 = st.columns(2)
        
        with comp_trend_col1:
            st.markdown("**Current Recording - Emotion Trends**")
            current_trend_fig = create_emotion_trend_chart(webcam_entries, "Current Recording")
            if current_trend_fig:
                st.plotly_chart(current_trend_fig, use_container_width=True)
            else:
                st.warning("No emotion data available for current recording.")
        
        with comp_trend_col2:
            st.markdown(f"**{st.session_state.comparison_recording} - Emotion Trends**")
            comp_trend_fig = create_emotion_trend_chart(
                st.session_state.comparison_data.get("webcam", []), 
                st.session_state.comparison_recording
            )
            if comp_trend_fig:
                st.plotly_chart(comp_trend_fig, use_container_width=True)
            else:
                st.warning("No emotion data available for comparison recording.")
    
    # Gaze heatmap comparison
    current_heatmap_exists = os.path.exists("./frames/gaze_heatmap.png")
    comp_heatmap_exists = os.path.exists(f"./recordings/{st.session_state.comparison_recording}/gaze_heatmap.png")
    
    if current_heatmap_exists or comp_heatmap_exists:
        st.divider()
        st.markdown("### 🔥 Gaze Heatmap Comparison")
        
        # Side-by-side heatmap display
        heatmap_col1, heatmap_col2 = st.columns(2)
        
        with heatmap_col1:
            st.markdown("**Current Recording - Gaze Heatmap**")
            if current_heatmap_exists:
                st.image("./frames/gaze_heatmap.png", caption="Current Recording Heatmap", use_container_width=True)
            else:
                st.info("ℹ️ No heatmap available for current recording")
                if st.button("🔥 Generate Current Heatmap", key="gen_current_heatmap"):
                    with st.spinner("Generating current heatmap..."):
                        try:
                            import subprocess
                            import sys
                            result = subprocess.run([sys.executable, "gaze_heatmap.py"], 
                                                 capture_output=True, text=True, cwd=".")
                            if result.returncode == 0:
                                st.success("✅ Generated!")
                                st.rerun()
                            else:
                                st.error("❌ Failed")
                        except Exception as e:
                            st.error("❌ Error")
        
        with heatmap_col2:
            st.markdown(f"**{st.session_state.comparison_recording} - Gaze Heatmap**")
            if comp_heatmap_exists:
                st.image(f"./recordings/{st.session_state.comparison_recording}/gaze_heatmap.png", 
                        caption=f"{st.session_state.comparison_recording} Heatmap", use_container_width=True)
            else:
                st.info("ℹ️ No heatmap available for comparison recording")
                if st.button("🔥 Generate Comparison Heatmap", key="gen_comp_heatmap"):
                    with st.spinner("Generating comparison heatmap..."):
                        try:
                            import subprocess
                            import sys
                            # Change to comparison recording directory and generate heatmap
                            comp_dir = f"./recordings/{st.session_state.comparison_recording}"
                            result = subprocess.run([sys.executable, "../gaze_heatmap.py"], 
                                                 capture_output=True, text=True, cwd=comp_dir)
                            if result.returncode == 0:
                                st.success("✅ Generated!")
                                st.rerun()
                            else:
                                st.error("❌ Failed")
                        except Exception as e:
                            st.error("❌ Error")
    
    # Auto-refresh for comparison
    if st.session_state.comparison_is_playing:
        st.rerun()
