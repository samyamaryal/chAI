import cv2
import os
import json
import time
from datetime import datetime
import threading
import mss
import numpy as np
from pynput import keyboard, mouse

# Configuration
output_base = "frames"
webcam_folder = os.path.join(output_base, "webcam")
screen_folder = os.path.join(output_base, "screen")
log_file = "./frames/combined_log.json"

# Setup
os.makedirs(webcam_folder, exist_ok=True)
os.makedirs(screen_folder, exist_ok=True)

# Global recording state
recording_threads = []
recording_listeners = []
is_recording = False
shared_data = {}
data_lock = threading.Lock()

def start_recording():
    """Start recording webcam, screen, keyboard, and mouse events"""
    global is_recording, recording_threads, recording_listeners, shared_data
    
    print("Starting recording process...")
    
    if is_recording:
        print("Already recording")
        return False, "Already recording"
    
    # Initialize shared data
    with data_lock:
        shared_data = {
            "log_data": {
                "webcam": [],
                "screen": [],
                "keyboard": [],
                "mouse": []
            },
            "running": True
        }
    
    print("Shared data initialized")
    
    # Webcam capture function
    def capture_webcam():
        print("Starting webcam capture...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        frame_count = 0
        last_time = time.time()

        try:
            while shared_data["running"]:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read webcam frame")
                    break

                now = time.time()
                fps = 1 / (now - last_time) if frame_count > 0 else 0.0
                last_time = now

                filename = f"webcam_{frame_count:06d}.jpg"
                path = os.path.join(webcam_folder, filename)
                
                # Ensure webcam directory exists
                os.makedirs(webcam_folder, exist_ok=True)
                
                success = cv2.imwrite(path, frame)
                if success:
                    print(f"Saved webcam frame: {filename}")
                else:
                    print(f"Failed to save webcam frame: {filename}")

                timestamp = datetime.now().isoformat()
                with data_lock:
                    shared_data["log_data"]["webcam"].append({
                        "frame": frame_count,
                        "time": timestamp,
                        "file": filename,
                        "fps": round(fps, 2)
                    })

                frame_count += 1
                time.sleep(1 / 30)
        except Exception as e:
            print(f"Error in webcam capture: {e}")
        finally:
            cap.release()
            print(f"Webcam capture stopped. Total frames: {frame_count}")

    # Screen capture function
    def capture_screen():
        print("Starting screen capture...")
        sct = mss.mss()
        monitor = sct.monitors[1]

        frame_count = 0
        last_time = time.time()

        try:
            while shared_data["running"]:
                img = sct.grab(monitor)
                frame = np.array(img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                now = time.time()
                fps = 1 / (now - last_time) if frame_count > 0 else 0.0
                last_time = now

                filename = f"screen_{frame_count:06d}.jpg"
                path = os.path.join(screen_folder, filename)
                
                # Ensure screen directory exists
                os.makedirs(screen_folder, exist_ok=True)
                
                success = cv2.imwrite(path, frame)
                if success:
                    print(f"Saved screen frame: {filename}")
                else:
                    print(f"Failed to save screen frame: {filename}")

                timestamp = datetime.now().isoformat()
                with data_lock:
                    shared_data["log_data"]["screen"].append({
                        "frame": frame_count,
                        "time": timestamp,
                        "file": filename,
                        "fps": round(fps, 2)
                    })

                frame_count += 1
                time.sleep(1 / 60)
        except Exception as e:
            print(f"Error in screen capture: {e}")
        finally:
            sct.close()
            print(f"Screen capture stopped. Total frames: {frame_count}")

    # Keyboard event handlers
    def on_key_press(key):
        try:
            with data_lock:
                shared_data["log_data"]["keyboard"].append({
                    "event": "press",
                    "key": str(key.char),
                    "time": datetime.now().isoformat()
                })
        except AttributeError:
            with data_lock:
                shared_data["log_data"]["keyboard"].append({
                    "event": "press",
                    "key": str(key),
                    "time": datetime.now().isoformat()
                })

    def on_key_release(key):
        with data_lock:
            shared_data["log_data"]["keyboard"].append({
                "event": "release",
                "key": str(key),
                "time": datetime.now().isoformat()
            })

    # Mouse click event handler only
    def on_click(x, y, button, pressed):
        with data_lock:
            shared_data["log_data"]["mouse"].append({
                "event": "click",
                "button": str(button),
                "pressed": pressed,
                "position": [x, y],
                "time": datetime.now().isoformat()
            })

    # Create threads and listeners
    webcam_thread = threading.Thread(target=capture_webcam)
    screen_thread = threading.Thread(target=capture_screen)
    keyboard_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click)
    
    # Store references for cleanup
    recording_threads = [webcam_thread, screen_thread]
    recording_listeners = [keyboard_listener, mouse_listener]
    
    print("Threads and listeners created")
    
    # Start recording
    try:
        print("Starting threads...")
        webcam_thread.start()
        screen_thread.start()
        keyboard_listener.start()
        mouse_listener.start()
        
        is_recording = True
        print("Recording started successfully")
        return True, "Recording started successfully"
        
    except Exception as e:
        print(f"Failed to start recording: {e}")
        # Cleanup on error
        with data_lock:
            shared_data["running"] = False
        for thread in recording_threads:
            if thread.is_alive():
                thread.join(timeout=1)
        for listener in recording_listeners:
            listener.stop()
        recording_threads = []
        recording_listeners = []
        return False, f"Failed to start recording: {str(e)}"

def stop_recording():
    """Stop recording and save log data"""
    global is_recording, recording_threads, recording_listeners, shared_data
    
    print("Stopping recording...")
    
    if not is_recording:
        print("Not currently recording")
        return False, "Not currently recording"
    
    try:
        # Stop recording
        with data_lock:
            shared_data["running"] = False
        is_recording = False
        print("Recording flag set to False")
        
        # Stop threads
        print("Stopping threads...")
        for thread in recording_threads:
            if thread.is_alive():
                thread.join(timeout=2)
        
        # Stop listeners
        print("Stopping listeners...")
        for listener in recording_listeners:
            listener.stop()
        
        # Save log data
        with data_lock:
            log_data = shared_data["log_data"].copy()
        
        print(f"Log data summary:")
        print(f"- Webcam entries: {len(log_data.get('webcam', []))}")
        print(f"- Screen entries: {len(log_data.get('screen', []))}")
        print(f"- Keyboard entries: {len(log_data.get('keyboard', []))}")
        print(f"- Mouse entries: {len(log_data.get('mouse', []))}")
        
        # Ensure frames directory exists
        os.makedirs(output_base, exist_ok=True)
        
        # Save log data with error handling
        try:
            with open(log_file, "w") as f:
                json.dump(log_data, f, indent=4)
            print(f"Log file saved to: {log_file}")
        except Exception as e:
            print(f"Warning: Could not save log file: {e}")
            # Continue even if log save fails
        
        # Clear references
        recording_threads = []
        recording_listeners = []
        
        print("Recording stopped successfully")
        return True, "Recording stopped and log saved"
        
    except Exception as e:
        print(f"Failed to stop recording: {e}")
        return False, f"Failed to stop recording: {str(e)}"

def is_recording_active():
    """Check if recording is currently active"""
    return is_recording

# For standalone execution
if __name__ == "__main__":
    print("Starting recording... Press Ctrl+C to stop.")
    success, message = start_recording()
    print(message)
    
    try:
        while is_recording:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping recording...")
        success, message = stop_recording()
        print(message)
