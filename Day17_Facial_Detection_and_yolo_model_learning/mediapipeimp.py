"""
MediaPipe Face Mesh demo with Real vs Fake smile classification + blink counting:
- Draw mesh + bounding box per face
- Detect blink using EAR (eye aspect ratio) and keep a blink_count
- Detect smile using mouth-corner spread & mouth-open heuristics
- Classify smile as REAL (teeth visible) or FAKE (no teeth + eyes squinted)
- Detect face movement across frames
- Save labeled screenshots to Output/{blinks, smiles, real_smiles, fake_smiles, motion}
- Show per-face Blink counts on screen
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from math import hypot

# ------------ CONFIG -------------
OUTPUT_DIR = "Output"
BLINK_DIR = os.path.join(OUTPUT_DIR, "blinks")
SMILE_DIR = os.path.join(OUTPUT_DIR, "smiles")
REAL_SMILE_DIR = os.path.join(OUTPUT_DIR, "real_smiles")
FAKE_SMILE_DIR = os.path.join(OUTPUT_DIR, "fake_smiles")
MOTION_DIR = os.path.join(OUTPUT_DIR, "motion")
os.makedirs(BLINK_DIR, exist_ok=True)
os.makedirs(SMILE_DIR, exist_ok=True)
os.makedirs(REAL_SMILE_DIR, exist_ok=True)
os.makedirs(FAKE_SMILE_DIR, exist_ok=True)
os.makedirs(MOTION_DIR, exist_ok=True)

# face mesh & drawing
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
try:
    mp_drawing_styles = mp.solutions.drawing_styles
    HAVE_DRAWING_STYLES = True
except Exception:
    mp_drawing_styles = None
    HAVE_DRAWING_STYLES = False

# Landmark indices for EAR (MediaPipe indices)
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]   # right eye: p1,p2,p3,p4,p5,p6
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]  # left eye

# Mouth landmarks (use some common outer mouth indexes to build ROI)
MOUTH_LANDMARKS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 78, 95, 88]
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

# thresholds / tuning (tweak for your webcam/lighting)
EAR_THRESH = 0.21             # if EAR below => eye closed (blink)
EAR_CONSEC_FRAMES = 3         # consecutive frames to count a blink
SMILE_WIDTH_RATIO = 0.45      # mouth_width / face_box_width
SMILE_OPEN_RATIO = 0.28       # mouth_open / mouth_width to avoid wide-open mouth
MOVE_RATIO = 0.08             # fraction of face box width considered "moved"
SAVE_BLOCK_FRAMES = 12        # block repeated saves for same face/event (frames)

# Real vs Fake smile parameters
TEETH_V_THRESHOLD = 150       # HSV V brightness threshold for "bright" pixels (teeth)
TEETH_S_THRESHOLD = 100       # HSV S max for low-saturation (teeth)
TEETH_RATIO_THRESHOLD = 0.12  # fraction of mouth pixels required to consider teeth visible
SQUINT_EAR_THRESH = 0.25      # EAR less than this considered "squinted" (not fully closed)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ------------ HELPERS -------------
def now_ts():
    return time.strftime("%Y%m%d_%H%M%S")

def save_labeled_frame(frame, label, subfolder, color=(0,255,0)):
    """Draw label onto a copy and save to folder."""
    out = frame.copy()
    cv2.putText(out, label, (50,50), FONT, 1.2, color, 3, cv2.LINE_AA)
    filename = os.path.join(subfolder, f"{label.replace(' ','_')}_{now_ts()}.jpg")
    cv2.imwrite(filename, out)
    print(f"[SAVED] {filename}")

def to_pixel_coords(landmark, image_w, image_h):
    return int(landmark.x * image_w), int(landmark.y * image_h)

def euclid(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(landmarks, image_w, image_h, idxs):
    # idxs = [p1,p2,p3,p4,p5,p6]
    try:
        p1 = to_pixel_coords(landmarks[idxs[0]], image_w, image_h)
        p2 = to_pixel_coords(landmarks[idxs[1]], image_w, image_h)
        p3 = to_pixel_coords(landmarks[idxs[2]], image_w, image_h)
        p4 = to_pixel_coords(landmarks[idxs[3]], image_w, image_h)
        p5 = to_pixel_coords(landmarks[idxs[4]], image_w, image_h)
        p6 = to_pixel_coords(landmarks[idxs[5]], image_w, image_h)
    except Exception:
        return 0.0
    A = euclid(p2, p6)
    B = euclid(p3, p5)
    C = euclid(p1, p4)
    if C == 0:
        return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_has_teeth_from_landmarks(frame, landmarks, image_w, image_h, face_bbox):
    """
    Extract a mouth ROI using landmark indices, convert to HSV and detect
    bright low-saturation pixels that likely correspond to teeth.
    Returns True if teeth ratio >= TEETH_RATIO_THRESHOLD.
    """
    coords = []
    for idx in MOUTH_LANDMARKS:
        try:
            coords.append(to_pixel_coords(landmarks[idx], image_w, image_h))
        except Exception:
            continue
    if not coords:
        return False

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    x_min = max(0, min(xs) - 2)
    x_max = min(image_w, max(xs) + 2)
    y_min = max(0, min(ys) - 2)
    y_max = min(image_h, max(ys) + 2)

    bx, by, bw, bh = face_bbox
    pad_x = max(4, int(bw * 0.03))
    pad_y = max(3, int(bh * 0.02))
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(image_w, x_max + pad_x)
    y2 = min(image_h, y_max + pad_y)

    if x2 - x1 <= 2 or y2 - y1 <= 2:
        return False

    mouth_roi = frame[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return False

    hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    bright_mask = (v_ch >= TEETH_V_THRESHOLD) & (s_ch <= TEETH_S_THRESHOLD)
    bright_count = np.count_nonzero(bright_mask)
    total = mouth_roi.shape[0] * mouth_roi.shape[1]
    ratio = float(bright_count) / float(total) if total > 0 else 0.0

    return ratio >= TEETH_RATIO_THRESHOLD

# ------------ MAIN -------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=4,
                               refine_landmarks=False,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        faces_state = {}   # id -> state dict
        next_face_id = 0
        frame_idx = 0

        print("Starting. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror for webcam
            frame_idx += 1
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = face_mesh.process(img_rgb)

            detections = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    pts = []
                    xs = []
                    ys = []
                    for lm in face_landmarks.landmark:
                        x_px = int(lm.x * w)
                        y_px = int(lm.y * h)
                        pts.append((x_px, y_px))
                        xs.append(x_px)
                        ys.append(y_px)
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox_w = x_max - x_min
                    bbox_h = y_max - y_min
                    center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
                    detections.append({
                        'landmark_obj': face_landmarks,
                        'landmarks': face_landmarks.landmark,
                        'pts': pts,
                        'bbox': (x_min, y_min, bbox_w, bbox_h),
                        'center': center
                    })

            # --- simple tracker by center ---
            matched = set()
            updated_states = {}
            for det in detections:
                cx, cy = det['center']
                bx, by, bw, bh = det['bbox']
                best_id = None
                best_dist = None
                match_thresh = max(20, bw * 0.6)
                for fid, st in faces_state.items():
                    prev_c = st['center']
                    d = euclid(prev_c, (cx, cy))
                    if d <= match_thresh and (best_dist is None or d < best_dist):
                        best_dist = d
                        best_id = fid

                if best_id is None:
                    fid = next_face_id
                    next_face_id += 1
                    st = {
                        'id': fid,
                        'center': (cx, cy),
                        'bbox': (bx, by, bw, bh),
                        'last_seen': frame_idx,
                        'blink_counter': 0,
                        'smile_counter': 0,
                        'blink_count': 0,                # <-- added blink count
                        'last_saved_event_frame': 0,
                    }
                else:
                    fid = best_id
                    st = faces_state[fid]
                    st['center'] = (cx, cy)
                    st['bbox'] = (bx, by, bw, bh)
                    st['last_seen'] = frame_idx

                st['det'] = det
                updated_states[fid] = st
                matched.add(fid)

            # keep unmatched recently seen faces briefly
            TIMEOUT = 12
            for fid, st in list(faces_state.items()):
                if fid not in matched and (frame_idx - st.get('last_seen', 0) <= TIMEOUT):
                    updated_states[fid] = st

            faces_state = updated_states

            # --- analyze each tracked face ---
            for fid, st in faces_state.items():
                if 'det' not in st:
                    continue
                det = st['det']
                landmark_obj = det['landmark_obj']
                landmarks = det['landmarks']
                bx, by, bw, bh = st['bbox']
                cx, cy = st['center']

                # draw bbox & ID
                cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0,255,0), 1)
                cv2.putText(frame, f"ID:{fid}", (bx, by - 8), FONT, 0.6, (0,255,255), 2)

                # draw face mesh overlay
                try:
                    if HAVE_DRAWING_STYLES:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=landmark_obj,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    else:
                        mp_drawing.draw_landmarks(
                            image=frame,
                            landmark_list=landmark_obj,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,200), thickness=1, circle_radius=1)
                        )
                except Exception:
                    pass

                # --- BLINK detection (EAR) ---
                ear_right = eye_aspect_ratio(landmarks, w, h, RIGHT_EYE_IDX)
                ear_left = eye_aspect_ratio(landmarks, w, h, LEFT_EYE_IDX)
                ear = (ear_left + ear_right) / 2.0

                blink_detected = False
                if ear < EAR_THRESH:
                    st['blink_counter'] = st.get('blink_counter', 0) + 1
                else:
                    if st.get('blink_counter', 0) >= EAR_CONSEC_FRAMES:
                        blink_detected = True
                    st['blink_counter'] = 0

                # increment persistent blink count when a blink is detected
                if blink_detected:
                    st['blink_count'] = st.get('blink_count', 0) + 1

                # --- SMILE detection using mouth landmarks ---
                try:
                    p_left = to_pixel_coords(landmarks[MOUTH_LEFT], w, h)
                    p_right = to_pixel_coords(landmarks[MOUTH_RIGHT], w, h)
                    p_top = to_pixel_coords(landmarks[MOUTH_TOP], w, h)
                    p_bottom = to_pixel_coords(landmarks[MOUTH_BOTTOM], w, h)
                except Exception:
                    st['smile_counter'] = 0
                    p_left = p_right = p_top = p_bottom = (0,0)

                mouth_width = euclid(p_left, p_right)
                mouth_open = euclid(p_top, p_bottom)
                mouth_width_ratio = mouth_width / float(bw + 1e-6)
                mouth_open_ratio = mouth_open / float(mouth_width + 1e-6)

                smile_detected = False
                if mouth_width_ratio >= SMILE_WIDTH_RATIO and mouth_open_ratio < SMILE_OPEN_RATIO:
                    st['smile_counter'] = st.get('smile_counter', 0) + 1
                else:
                    st['smile_counter'] = 0

                if st.get('smile_counter', 0) >= 3:
                    smile_detected = True
                    st['smile_counter'] = 0

                # --- MOVEMENT detection ---
                moved = False
                prev_center = st.get('center_prev', None)
                if prev_center is not None:
                    dist_moved = euclid(prev_center, (cx, cy))
                else:
                    dist_moved = 0.0
                move_thresh = bw * MOVE_RATIO
                if dist_moved > move_thresh and (frame_idx - st.get('last_saved_event_frame', 0) > SAVE_BLOCK_FRAMES):
                    moved = True

                # --- If smile detected, classify real vs fake ---
                if smile_detected and (frame_idx - st.get('last_saved_event_frame', 0) > SAVE_BLOCK_FRAMES):
                    try:
                        teeth_visible = mouth_has_teeth_from_landmarks(frame, landmarks, w, h, (bx, by, bw, bh))
                    except Exception:
                        teeth_visible = False

                    eyes_squinted = (ear < SQUINT_EAR_THRESH)

                    if teeth_visible:
                        label = f"REAL_SMILE_ID{fid}"
                        save_labeled_frame(frame, label, REAL_SMILE_DIR, color=(0,255,0))
                        save_labeled_frame(frame, f"SMILE_ID{fid}", SMILE_DIR, color=(0,255,0))
                        st['last_saved_event_frame'] = frame_idx
                        cv2.putText(frame, "REAL_SMILE", (bx, by + bh + 30), FONT, 0.8, (0,255,0), 2)
                    elif (not teeth_visible) and eyes_squinted:
                        label = f"FAKE_SMILE_ID{fid}"
                        save_labeled_frame(frame, label, FAKE_SMILE_DIR, color=(0,165,255))
                        save_labeled_frame(frame, f"SMILE_ID{fid}", SMILE_DIR, color=(0,165,255))
                        st['last_saved_event_frame'] = frame_idx
                        cv2.putText(frame, "FAKE_SMILE", (bx, by + bh + 30), FONT, 0.8, (0,165,255), 2)
                    else:
                        label = f"REAL_SMILE_ID{fid}"
                        save_labeled_frame(frame, label, REAL_SMILE_DIR, color=(0,255,0))
                        save_labeled_frame(frame, f"SMILE_ID{fid}", SMILE_DIR, color=(0,255,0))
                        st['last_saved_event_frame'] = frame_idx
                        cv2.putText(frame, "REAL_SMILE", (bx, by + bh + 30), FONT, 0.8, (0,255,0), 2)

                # --- Save blink or moved events ---
                if blink_detected and (frame_idx - st.get('last_saved_event_frame', 0) > SAVE_BLOCK_FRAMES):
                    label = f"BLINK_ID{fid}"
                    save_labeled_frame(frame, label, BLINK_DIR, color=(0,0,255))  # red
                    st['last_saved_event_frame'] = frame_idx
                    cv2.putText(frame, "BLINK", (bx, by + bh + 30), FONT, 0.8, (0,0,255), 2)
                else:
                    # even if save-blocked, still show BLINK text briefly on detection (optional)
                    if blink_detected:
                        cv2.putText(frame, "BLINK", (bx, by + bh + 30), FONT, 0.8, (0,0,255), 2)

                if moved and (frame_idx - st.get('last_saved_event_frame', 0) > SAVE_BLOCK_FRAMES):
                    label = f"MOVED_ID{fid}"
                    save_labeled_frame(frame, label, MOTION_DIR, color=(0,255,0))
                    st['last_saved_event_frame'] = frame_idx
                    cv2.putText(frame, "MOVED", (bx, by + bh + 30), FONT, 0.8, (0,255,0), 2)

                # debug overlays and blink count display
                cv2.putText(frame, f"EAR:{ear:.2f}", (bx, by + bh + 10), FONT, 0.5, (255,255,255), 1)
                cv2.putText(frame, f"MWR:{mouth_width_ratio:.2f}", (bx, by + bh + 22), FONT, 0.5, (255,255,255), 1)
                # show blink count
                cv2.putText(frame, f"Blinks:{st.get('blink_count', 0)}", (bx + bw - 100, by - 8), FONT, 0.6, (255,200,0), 2)

                st['center_prev'] = (cx, cy)

            # show final frame
            cv2.imshow("MediaPipe FaceMesh - real vs fake smile + blink count", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()