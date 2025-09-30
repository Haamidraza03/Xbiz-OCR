import cv2
import numpy as np
import time
import os

# --- 1. CONFIGURATION AND CONSTANTS ---
CASCADE_FACE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
CASCADE_EYE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"
CASCADE_SMILE_PATH = cv2.data.haarcascades + "haarcascade_smile.xml"

# Base thresholds (you can tweak these)
BASE_MOVE_THRESH = 15               # base pixel threshold, scaled by face width
SMILE_MIN_WIDTH_RATIO = 0.58        # min width ratio to consider a smile candidate
BLINK_BASE_FRAMES = 6               # base frames to confirm a blink (adapted by face size)
SMILE_CONFIRM_FRAMES = 5            # consecutive frames to confirm a smile
MAX_MATCH_DIST_FACTOR = 0.5         # matching distance factor relative to face width
REFERENCE_FACE_WIDTH = 200.0

# Teeth / fake smile detection tuning
TEETH_V_THRESHOLD = 200             # V channel brightness threshold to count bright pixels (0-255)
TEETH_S_THRESHOLD = 160              # S channel maximum to consider 'white-ish' pixels
TEETH_RATIO_THRESHOLD = 0.25        # fraction of mouth pixels required to consider teeth visible (12%)
EYE_SQUINT_HEIGHT_RATIO = 0.32      # average eye height < this * face_width => likely squinted

# Colors (BGR)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)

# Output setup
OUTPUT_DIR = "Output"
BLINK_DIR = os.path.join(OUTPUT_DIR, "blinks")
SMILE_DIR = os.path.join(OUTPUT_DIR, "smiles")           # original smile folder (keeps backups)
REAL_SMILE_DIR = os.path.join(OUTPUT_DIR, "real_smiles")
FAKE_SMILE_DIR = os.path.join(OUTPUT_DIR, "fake_smiles")
MOTION_DIR = os.path.join(OUTPUT_DIR, "motion")
os.makedirs(BLINK_DIR, exist_ok=True)
os.makedirs(SMILE_DIR, exist_ok=True)
os.makedirs(REAL_SMILE_DIR, exist_ok=True)
os.makedirs(FAKE_SMILE_DIR, exist_ok=True)
os.makedirs(MOTION_DIR, exist_ok=True)


# --- 2. HELPER FUNCTIONS ---
def euclidean_dist(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def timestamp_str():
    return time.strftime("%Y%m%d_%H%M%S")


def save_capture(frame, label, subfolder):
    """Save full-frame capture with label text into the chosen subfolder."""
    save_frame = frame.copy()
    ts = timestamp_str()
    filename = os.path.join(subfolder, f"{label.replace(' ', '_')}_{ts}.jpg")
    cv2.putText(save_frame, label.upper(), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLOR_RED, 3, cv2.LINE_AA)
    cv2.imwrite(filename, save_frame)
    print(f"[SAVE] {label} -> {filename}")


def mouth_has_teeth(roi_color, sx, sy, sw, sh, fh):
    """
    Inspect the mouth ROI to estimate if teeth are visible.
    roi_color is the face-colored ROI (size fh x fw). sx,sy are relative to the smile_roi,
    where smile_roi starts at vertical offset fh//2 inside the face ROI.
    """
    # compute actual mouth box coordinates inside roi_color
    mouth_y1 = (fh // 2) + sy
    mouth_y2 = mouth_y1 + sh
    mouth_x1 = sx
    mouth_x2 = sx + sw

    # boundary checks
    h, w, _ = roi_color.shape
    mouth_y1 = max(0, min(h, mouth_y1))
    mouth_y2 = max(0, min(h, mouth_y2))
    mouth_x1 = max(0, min(w, mouth_x1))
    mouth_x2 = max(0, min(w, mouth_x2))

    if mouth_y2 - mouth_y1 <= 2 or mouth_x2 - mouth_x1 <= 2:
        return False  # too small to evaluate

    mouth_roi = roi_color[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    if mouth_roi.size == 0:
        return False

    # convert to HSV and look for bright, low-saturation pixels (likely teeth)
    hsv = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    bright_mask = (v_ch >= TEETH_V_THRESHOLD) & (s_ch <= TEETH_S_THRESHOLD)
    bright_count = np.count_nonzero(bright_mask)
    total = mouth_roi.shape[0] * mouth_roi.shape[1]
    ratio = float(bright_count) / float(total) if total > 0 else 0.0

    # Debugging print (comment out if noisy)
    # print(f"Teeth ratio: {ratio:.3f} (bright {bright_count} / total {total})")

    return ratio >= TEETH_RATIO_THRESHOLD


# --- 3. MAIN EXECUTION: multi-face tracking + improved detection ---
def main():
    # Load cascades
    face_cascade = cv2.CascadeClassifier(CASCADE_FACE_PATH)
    eye_cascade = cv2.CascadeClassifier(CASCADE_EYE_PATH)
    smile_cascade = cv2.CascadeClassifier(CASCADE_SMILE_PATH)

    if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
        print("ERROR: Could not load one or more Haar cascades. Check the paths.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    # tracker: id -> state dict
    faces_state = {}
    next_face_id = 0
    frame_idx = 0

    print("Starting multi-face Haar detector. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from camera.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w, _ = frame.shape
        frame_idx += 1

        # detect faces (returns list of (x,y,w,h))
        detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        # Build list of detected face centers
        detections = []
        for (fx, fy, fw, fh) in detected:
            cx = fx + fw // 2
            cy = fy + fh // 2
            detections.append({'rect': (fx, fy, fw, fh), 'center': (cx, cy)})

        # --- Simple tracker: match detections to existing faces by nearest center ---
        matched = set()
        updated_states = {}

        # For each detection, find nearest existing face within a reasonable distance
        for det in detections:
            fx, fy, fw, fh = det['rect']
            center = det['center']

            best_id = None
            best_dist = None
            # dynamic match distance threshold relative to face width
            match_threshold = fw * (MAX_MATCH_DIST_FACTOR)

            for fid, state in faces_state.items():
                prev_center = state['center']
                d = euclidean_dist(center, prev_center)
                if d <= match_threshold and (best_dist is None or d < best_dist):
                    best_dist = d
                    best_id = fid

            if best_id is None:
                # create new face state
                fid = next_face_id
                next_face_id += 1
                state = {
                    'id': fid,
                    'center': center,
                    'rect': (fx, fy, fw, fh),
                    'last_seen': frame_idx,
                    'blink_counter': 0,
                    'blink_block_until': 0,
                    'smile_counter': 0,
                    'smile_block_until': 0,
                    'last_saved_motion': 0,
                }
            else:
                # update existing state
                state = faces_state[best_id]
                state['center'] = center
                state['rect'] = (fx, fy, fw, fh)
                state['last_seen'] = frame_idx

            updated_states[state['id']] = state
            matched.add(state['id'])

        # any previous faces not matched: keep them for a few frames (brief timeout)
        TIMEOUT_FRAMES = 12
        for fid, state in list(faces_state.items()):
            if fid not in matched:
                # if it was seen recently, keep it, else drop
                if frame_idx - state.get('last_seen', 0) <= TIMEOUT_FRAMES:
                    updated_states[fid] = state
                # otherwise, it will be removed automatically

        faces_state = updated_states

        # --- Process each tracked face for blink/smile/motion ---
        for fid, state in faces_state.items():
            fx, fy, fw, fh = state['rect']
            cx, cy = state['center']

            # draw face rectangle and ID
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), COLOR_GREEN, 1)
            cv2.putText(frame, f"ID:{fid}", (fx, fy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_YELLOW, 2)

            # Define ROIs
            roi_gray = gray[fy:fy + fh, fx:fx + fw]
            roi_color = frame[fy:fy + fh, fx:fx + fw]

            # --- Motion detection (scale threshold by face width) ---
            prev_center = state.get('center_prev', None)
            if prev_center is not None:
                dist_moved = euclidean_dist((cx, cy), prev_center)
            else:
                dist_moved = 0.0
            # dynamic threshold: larger faces (close) require larger pixel movement to be "motion"
            dynamic_move_thresh = BASE_MOVE_THRESH * (fw / REFERENCE_FACE_WIDTH)
            is_moved = dist_moved > dynamic_move_thresh
            if is_moved and (frame_idx - state.get('last_saved_motion', 0) > 10):
                # save motion capture
                label = f"motion_id{fid}"
                save_capture(frame, label, MOTION_DIR)
                state['last_saved_motion'] = frame_idx

            # --- Eye detection + adaptive blink logic ---
            # Use top half for eyes
            eye_roi_gray = roi_gray[: max(1, fh // 2), :]
            eyes = []
            try:
                # adjust scaleFactor depending on face size: smaller faces need a lower scale step
                eye_scale = 1.1 if fw >= 100 else 1.05
                eyes = eye_cascade.detectMultiScale(eye_roi_gray, scaleFactor=eye_scale, minNeighbors=8, minSize=(8, 8))
            except Exception:
                eyes = []

            # adaptive blink threshold: far faces (small fw) need bigger consecutive frame requirement to avoid noise
            scale_factor_for_size = REFERENCE_FACE_WIDTH / float(max(fw, 20))
            blink_frames_needed = int(round(BLINK_BASE_FRAMES * scale_factor_for_size))
            blink_frames_needed = max(2, min(blink_frames_needed, 10))  # clamp

            # blink logic: when eyes become undetected for consecutive frames -> confirm blink
            if len(eyes) < 1:
                state['blink_counter'] = state.get('blink_counter', 0) + 1
            else:
                # detected eyes -> reset counter so future blinks can be detected
                state['blink_counter'] = 0

            is_blinking = False
            if state['blink_counter'] >= blink_frames_needed and frame_idx > state.get('blink_block_until', 0):
                is_blinking = True
                # record block period to avoid multiple saves for one blink
                state['blink_block_until'] = frame_idx
                state['blink_counter'] = 0
                # save capture
                label = f"blink_id{fid}"
                save_capture(frame, label, BLINK_DIR)

            # Draw eyes (if any) inside ROI
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), COLOR_BLUE, 1)

            # --- Smile detection + adaptive confirmation + real/fake classification ---
            smile_roi_gray = roi_gray[max(1, fh // 2):, :]
            smiles = []
            try:
                # for smile cascade, tuning scaleFactor/minNeighbors to balance far/near detection
                smile_scale = 1.6 if fw >= 120 else 1.3
                mn = 18 if fw >= 120 else 12
                smiles = smile_cascade.detectMultiScale(smile_roi_gray, scaleFactor=smile_scale, minNeighbors=mn, minSize=(20, 20))
            except Exception:
                smiles = []

            # Determine if any smile candidate meets width-ratio condition
            smile_detected_this_frame = False
            chosen_smile = None
            for (sx, sy, sw, sh) in smiles:
                if (sw / float(fw)) >= SMILE_MIN_WIDTH_RATIO:
                    smile_detected_this_frame = True
                    chosen_smile = (sx, sy, sw, sh)
                    # draw rectangle (sy offset because smile_roi starts at fh//2)
                    cv2.rectangle(roi_color, (sx, sy + fh // 2), (sx + sw, sy + fh // 2 + sh), COLOR_GREEN, 1)
                    break

            if smile_detected_this_frame:
                state['smile_counter'] = state.get('smile_counter', 0) + 1
            else:
                state['smile_counter'] = 0

            # adapt smile confirm frames by distance (far faces need more frames)
            smile_frames_needed = int(round(SMILE_CONFIRM_FRAMES * (REFERENCE_FACE_WIDTH / float(max(fw, 20)))) )
            smile_frames_needed = max(2, min(smile_frames_needed, 8))

            # When smile is confirmed, classify as real/fake and save to appropriate folder
            if state['smile_counter'] >= smile_frames_needed and frame_idx > state.get('smile_block_until', 0):
                # classify real vs fake
                is_teeth_open = False
                is_eyes_squinted = False

                if chosen_smile is not None:
                    sx, sy, sw, sh = chosen_smile
                    try:
                        is_teeth_open = mouth_has_teeth(roi_color, sx, sy, sw, sh, fh)
                    except Exception:
                        is_teeth_open = False

                # eyes squinted logic: if eyes not detected (len==0) or average eye height very small
                if len(eyes) < 1:
                    is_eyes_squinted = True
                else:
                    # compute avg eye height and compare with face width ratio
                    avg_eye_h = float(np.mean([eh for (_ex, _ey, _ew, eh) in eyes])) if len(eyes) > 0 else 0.0
                    if avg_eye_h < (fw * EYE_SQUINT_HEIGHT_RATIO):
                        is_eyes_squinted = True

                # decide classification
                if is_teeth_open:
                    label = f"real_smile_id{fid}"
                    save_capture(frame, label, REAL_SMILE_DIR)
                    # also save a copy to the original SMILE_DIR for legacy reasons
                    save_capture(frame, f"smile_id{fid}", SMILE_DIR)
                    state['smile_block_until'] = frame_idx
                    state['smile_counter'] = 0
                    classification_text = "REAL_SMILE"
                elif (not is_teeth_open) and is_eyes_squinted:
                    label = f"fake_smile_id{fid}"
                    save_capture(frame, label, FAKE_SMILE_DIR)
                    # keep copy in SMILE_DIR as well
                    save_capture(frame, f"smile_id{fid}", SMILE_DIR)
                    state['smile_block_until'] = frame_idx
                    state['smile_counter'] = 0
                    classification_text = "FAKE_SMILE"
                else:
                    # default/fallback treat as real if ambiguous (no teeth but not squinted)
                    label = f"real_smile_id{fid}"
                    save_capture(frame, label, REAL_SMILE_DIR)
                    save_capture(frame, f"smile_id{fid}", SMILE_DIR)
                    state['smile_block_until'] = frame_idx
                    state['smile_counter'] = 0
                    classification_text = "REAL_SMILE"

                # annotate the frame near the face with classification
                cv2.putText(frame, classification_text, (fx, fy + fh + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_GREEN, 2)

            # --- Put status text near face (existing statuses) ---
            status_text = []
            if is_blinking:
                status_text.append("BLINK")
            # If recently blocked as smile, we show SMILED (old logic)
            if state.get('smile_block_until', 0) >= frame_idx:
                status_text.append("SMILED")
            if is_moved:
                status_text.append("MOTION")

            if status_text:
                cv2.putText(frame, "|".join(status_text), (fx, fy + fh + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_RED, 2)

            # keep previous center for next frame motion calc
            state['center_prev'] = (cx, cy)

        # Display the frame
        cv2.imshow("Multi-Face Event Detector (Haar) - Real vs Fake Smiles", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
