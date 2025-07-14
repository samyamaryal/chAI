import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import json
import math
import argparse
import time

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

def compensate_head_rotation(v2d, R):
    gaze3 = np.array([v2d[0], v2d[1], 1.0])
    corr = R.T.dot(gaze3)
    if corr[2] == 0:
        return v2d
    return corr[:2] / corr[2]

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

def calibrate(calib_path: Path, samples: int = 20):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam for calibration")
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from webcam")
    h, w = frame.shape[:2]

    mesh = mp.solutions.face_mesh.FaceMesh(
        refine_landmarks=True, max_num_faces=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    calibration = {'baseline': None, 'means': {}, 'arrow': None}

    cv2.namedWindow("Calibrate", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibrate", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Sector calib: look center + SPACE or wait 10s")
    start = time.time()
    while True:
        ret, live = cap.read()
        if not ret: continue
        disp = live.copy(); cv2.circle(disp, (w//2,h//2), 20, (0,0,255), -1)
        cv2.imshow("Calibrate", disp)
        key = cv2.waitKey(1)
        if key==27: break
        if key==32 or time.time()-start>10: break
    base_samples = []
    while len(base_samples)<samples:
        ret, fr = cap.read()
        if not ret: continue
        rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        if not res.multi_face_landmarks: continue
        lm = res.multi_face_landmarks[0].landmark
        li = np.array([np.mean([lm[i].x for i in range(468,473)])*w,
                       np.mean([lm[i].y for i in range(468,473)])*h])
        ri = np.array([np.mean([lm[i].x for i in range(473,478)])*w,
                       np.mean([lm[i].y for i in range(473,478)])*h])
        imu = (li+ri)/2.0
        base_samples.append(imu - np.array([w/2,h/2]))
    calibration['baseline'] = np.mean(base_samples,axis=0).tolist()
    print("Sector baseline:", calibration['baseline'])
    for s in range(1,10):
        row,col = divmod(s-1,3)
        tx,ty = int((col+0.5)*w/3), int((row+0.5)*h/3)
        print(f"Sector {s}: look + SPACE or wait 10s")
        start=time.time()
        while True:
            ret,live=cap.read()
            if not ret: continue
            disp=live.copy(); cv2.circle(disp,(tx,ty),20,(0,0,255),-1)
            cv2.imshow("Calibrate",disp); key=cv2.waitKey(1)
            if key==27: break
            if key==32 or time.time()-start>10: break
        samps=[]
        while len(samps)<samples:
            ret,fr=cap.read()
            if not ret: continue
            rgb=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
            res=mesh.process(rgb)
            if not res.multi_face_landmarks: continue
            lm=res.multi_face_landmarks[0].landmark
            li=np.array([np.mean([lm[i].x for i in range(468,473)])*w,
                         np.mean([lm[i].y for i in range(468,473)])*h])
            ri=np.array([np.mean([lm[i].x for i in range(473,478)])*w,
                         np.mean([lm[i].y for i in range(473,478)])*h])
            imu=(li+ri)/2.0
            samp=imu - np.array([w/2,h/2]) - np.array(calibration['baseline'])
            samps.append(samp)
        calibration['means'][str(s)] = np.mean(samps,axis=0).tolist()
        print(f"Sector {s} mean:", calibration['means'][str(s)])

    targets = {
        'center': (w//2, h//2), 'right': (int(0.9*w), h//2),
        'left': (int(0.1*w), h//2), 'up': (w//2, int(0.1*h)),
        'down': (w//2, int(0.9*h))
    }
    arrow_data = {}
    for name, (tx, ty) in targets.items():
        print(f"Arrow calib {name}: look + SPACE or wait 10s")
        start = time.time()
        while True:
            ret, live = cap.read()
            if not ret: continue
            disp = live.copy()
            cv2.circle(disp, (tx, ty), 20, (0, 0, 255), -1)
            cv2.imshow("Calibrate", disp)
            key = cv2.waitKey(1)
            if key == 27: break
            if key == 32 or time.time() - start > 10: break
        vecs = []
        while len(vecs) < samples:
            ret, fr = cap.read()
            if not ret: continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)
            if not res.multi_face_landmarks: continue
            lm = res.multi_face_landmarks[0].landmark
            raw, _ = compute_arrow_raw(lm, w, h)
            vecs.append(raw)
        arrow_data[name] = np.mean(vecs, axis=0).tolist()
        print(f"{name} raw:", arrow_data[name])
    bs = np.array(arrow_data['center'])
    sx = 1.0 / ((arrow_data['right'][0] - bs[0]) or 1e-6)
    sy = -1.0 / ((arrow_data['up'][1] - bs[1]) or 1e-6)
    calibration['arrow'] = {'baseline': bs.tolist(), 'scale_x': sx, 'scale_y': sy}
    print("Arrow calib saved")

    cap.release()
    cv2.destroyAllWindows()
    with open(calib_path,'w') as f: json.dump(calibration,f,indent=2)
    print("Calibration written to",calib_path)

prev_end = None

def process_images(images_dir, out_dir, calib_path):
    global prev_end
    data = json.loads(Path(calib_path).read_text())
    baseline = np.array(data['baseline'])
    cal_means = {int(k): np.array(v) for k, v in data['means'].items()}
    arrow_cal = data['arrow']
    bs = np.array(arrow_cal['baseline']); sx = arrow_cal['scale_x']; sy = arrow_cal['scale_y']

    mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1, min_detection_confidence=0.5)
    records = []
    for img in tqdm(sorted(Path(images_dir).iterdir()), desc="Processing"):
        if img.suffix.lower() not in {'.jpg', '.jpeg', '.png'}: continue
        frm = cv2.imread(str(img)); h, w = frm.shape[:2]
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        res = mesh.process(rgb)
        yaw = pitch = sector = iris_sector = head_sector = None
        engaged = False
        if res.multi_face_landmarks:
            engaged = True
            lm = res.multi_face_landmarks[0].landmark
            le = np.array([lm[33].x, lm[33].y, lm[33].z]); re = np.array([lm[263].x, lm[263].y, lm[263].z]); nt = np.array([lm[1].x, lm[1].y, lm[1].z])
            xax = re - le
            if np.linalg.norm(xax) > 1e-6:
                mid = (le + re) / 2; zax = nt - mid; yax = np.cross(zax, xax)
                xax /= np.linalg.norm(xax); yax /= np.linalg.norm(yax); zax /= np.linalg.norm(zax)
                R = np.column_stack((xax, yax, zax)); yaw, pitch, _ = extract_euler_angles(R)
            else: R = np.eye(3)
            li = np.array([np.mean([lm[i].x for i in range(468, 473)]) * w,
                           np.mean([lm[i].y for i in range(468, 473)]) * h])
            ri = np.array([np.mean([lm[i].x for i in range(473, 478)]) * w,
                           np.mean([lm[i].y for i in range(473, 478)]) * h])
            imu = (li + ri) / 2
            iris_sector = sector_3x3(imu[0], imu[1], w, h)
            raw_eye = imu - np.array([w/2, h/2]) - baseline
            dists = {s: np.linalg.norm(raw_eye - cal_means[s]) for s in cal_means}
            sector = min(dists, key=dists.get)
            if yaw is not None and pitch is not None:
                head_sector = head_sector_3x3(yaw, pitch)
            raw, eye_mid = compute_arrow_raw(lm, w, h)
            norm = raw - bs
            dir_x = norm[0] * sx
            dir_y = norm[1] * sy
            length = 0.5 * min(w, h)  # Adjusted length for visibility
            raw_x = eye_mid[0] + dir_x * length
            raw_y = eye_mid[1] + dir_y * length
            raw_x = min(max(raw_x, 0), w-1)
            raw_y = min(max(raw_y, 0), h-1)
            new_end = (int(raw_x), int(raw_y))
            alpha = 0.6
            if prev_end is None:
                sm = new_end
            else:
                sm = (int(alpha * new_end[0] + (1-alpha) * prev_end[0]),
                      int(alpha * new_end[1] + (1-alpha) * prev_end[1]))
            prev_end = sm
            # Draw visualizations
            for i in [1, 2]:
                cv2.line(frm, (i*w//3, 0), (i*w//3, h), (255, 255, 255), 1)
                cv2.line(frm, (0, int((i*h)/3)), (w, int((i*h)/3)), (255, 255, 255), 1)
            cv2.arrowedLine(frm, tuple(eye_mid.astype(int)), sm, (0, 0, 255), 2, tipLength=0.2)
            cv2.circle(frm, tuple(eye_mid.astype(int)), 4, (255, 0, 0), -1)

        cv2.imwrite(str(Path(out_dir) / img.name), frm)
        records.append({
            "file": img.name,
            "yaw": None if yaw is None else float(yaw),
            "pitch": None if pitch is None else float(pitch),
            "sector": sector,
            "iris_sector": iris_sector,
            "head_sector": head_sector,
            "engaged": engaged
        })
    with open(Path(out_dir) / "../gaze_log.json", "w") as f: json.dump(records, f, indent=2)
    print(f"Done : {out_dir} & gaze_log.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["calibrate", "process"], required=True)
    p.add_argument("--images", help="Input images folder (process mode)")
    p.add_argument("--out", help="Output folder for annotated frames")
    p.add_argument("--calib", default="calibration.json", help="Path to calibration JSON")
    p.add_argument("--samples", type=int, default=50, help="Samples per point")
    args = p.parse_args()
    if args.mode == "calibrate": calibrate(Path(args.calib), args.samples)
    else:
        if not args.images or not args.out: p.error("--images and --out are required in process mode")
        process_images(args.images, args.out, args.calib)