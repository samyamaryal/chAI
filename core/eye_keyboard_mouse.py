# core/main.py
import json
import os
import time
from datetime import datetime
from threading import Thread
from pynput import mouse, keyboard
import cv2
import numpy as np
import torch
from qai_hub_models.models.facemap_3dmm.app import FaceMap_3DMMApp
from qai_hub_models.models.facemap_3dmm.model import FaceMap_3DMM

# Setup base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, '..', 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Shared data structures
events = []
face_data = []
running = True

# --- Mouse Tracking ---
def on_click(x, y, button, pressed):
    if pressed:
        events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "mouse_click",
            "button": str(button),
            "position": {"x": x, "y": y}
        })

# --- Keyboard Tracking ---
def on_press(key):
    try:
        events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "key_press",
            "key": key.char
        })
    except AttributeError:
        events.append({
            "timestamp": datetime.now().isoformat(),
            "type": "key_press",
            "key": str(key)
        })

# --- Webcam with FaceMap 3DMM model ---
def run_webcam():
    global running
    cap = cv2.VideoCapture(0)

    model = FaceMap_3DMM.from_pretrained()
    app = FaceMap_3DMMApp(model)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    start_time = time.time()

    while running and (time.time() - start_time < 30):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        annotated_img = frame.copy()

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            x0, x1, y0, y1 = x, x + w, y, y + h

            with torch.no_grad():
                landmarks, annotated_img = app.landmark_prediction(frame, x0, x1, y0, y1)

            landmarks = landmarks.detach().cpu().numpy().flatten().tolist()

            face_data.append({
                "timestamp": datetime.now().isoformat(),
                "landmarks": landmarks
            })

            for i in range(0, len(landmarks), 2):
                x_norm, y_norm = landmarks[i], landmarks[i + 1]
                x_px, y_px = int(x_norm * frame.shape[1]), int(y_norm * frame.shape[0])
                cv2.circle(annotated_img, (x_px, y_px), 2, (0, 255, 0), -1)

        cv2.imshow('Webcam (Press Q to quit)', annotated_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Start everything ---
try:
    print("Tracking started for 30 seconds or until 'q' is pressed in webcam window...")
    webcam_thread = Thread(target=run_webcam)
    webcam_thread.start()

    mouse_listener = mouse.Listener(on_click=on_click)
    keyboard_listener = keyboard.Listener(on_press=on_press)
    mouse_listener.start()
    keyboard_listener.start()

    time.sleep(30)
    running = False

    mouse_listener.stop()
    keyboard_listener.stop()
    webcam_thread.join()
    mouse_listener.join()
    keyboard_listener.join()

except KeyboardInterrupt:
    print("Interrupted manually.")
    running = False

except Exception as e:
    print(f"Error: {e}")

finally:
    print("Tracking stopped. Saving logs...")
    mouse_log_path = os.path.join(LOGS_DIR, "mouse_keyboard_log.json")
    face_log_path = os.path.join(LOGS_DIR, "face_log.json")

    with open(mouse_log_path, "w") as f:
        json.dump(events, f, indent=2)

    with open(face_log_path, "w") as f:
        json.dump(face_data, f, indent=2)

    print(f"Logs saved to {LOGS_DIR}")
