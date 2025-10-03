import cv2
import mediapipe as mp
import os
import time
from math import hypot
from ultralytics import YOLO

# ---------- CONFIG ----------
OUTPUT_DIR = "Output"
BLINK_DIR = os.path.join(OUTPUT_DIR, "blinks")
SMILE_DIR = os.path.join(OUTPUT_DIR, "smiles")
MOTION_DIR = os.path.join(OUTPUT_DIR, "motion")
FAKE_FACE_DIR = os.path.join(OUTPUT_DIR, "fake_faces")
for d in (BLINK_DIR, SMILE_DIR, MOTION_DIR, FAKE_FACE_DIR):
    os.makedirs(d, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
try:
    mp_drawing_styles = mp.solutions.drawing_styles
    HAVE_DRAWING_STYLES = True
except Exception:
    mp_drawing_styles = None
    HAVE_DRAWING_STYLES = False

RIGHT_EYE_IDX = [33,160,158,133,153,144]
LEFT_EYE_IDX  = [362,385,387,263,373,380]
MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM = 61, 291, 13, 14

EAR_THRESH = 0.27
EAR_CONSEC_FRAMES = 3
SMILE_WIDTH_RATIO = 0.49
SMILE_OPEN_RATIO = 0.28
MOVE_RATIO = 0.05
SAVE_BLOCK_FRAMES = 12

SMALL_FACE_PIX = 140
SMALL_FACE_MOUTH_PIX_RATIO = 0.30

FACE_SCALE = 0.6
YOLO_SKIP_FRAMES = 6

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ---------- HELPERS ----------
def now_ts(): return time.strftime("%Y%m%d_%H%M%S")

def save_labeled_frame(frame, label, subfolder, color=(0,255,0)):
    out = frame.copy()
    cv2.putText(out, label, (50,50), FONT, 1.0, color, 2, cv2.LINE_AA)
    fn = os.path.join(subfolder, f"{label.replace(' ','_')}_{now_ts()}.jpg")
    cv2.imwrite(fn, out)

def to_pixel(landmark, w, h): return int(landmark.x * w), int(landmark.y * h)
def dist(a,b): return hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(lms, w, h, idxs):
    try:
        p1 = to_pixel(lms[idxs[0]], w, h); p2 = to_pixel(lms[idxs[1]], w, h)
        p3 = to_pixel(lms[idxs[2]], w, h); p4 = to_pixel(lms[idxs[3]], w, h)
        p5 = to_pixel(lms[idxs[4]], w, h); p6 = to_pixel(lms[idxs[5]], w, h)
    except Exception:
        return 0.0
    A = dist(p2,p6); B = dist(p3,p5); C = dist(p1,p4)
    return 0.0 if C == 0 else (A + B) / (2.0 * C)

# ---------- MAIN ----------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera"); return

    model = YOLO("yolov8n.pt")  # YOLOv8-nano (COCO)
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4,
                               refine_landmarks=False, min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        faces_state = {}     # fid -> state
        next_id = 0
        frame_idx = 0
        prev_screen_rects = []

        print("Starting. press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            frame_idx += 1
            h, w = frame.shape[:2]

            # small frame for inference
            sw, sh = max(1,int(w*FACE_SCALE)), max(1,int(h*FACE_SCALE))
            frame_small = cv2.resize(frame, (sw, sh))
            img_rgb_small = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

            # YOLO (detect screens) every N frames
            if frame_idx % YOLO_SKIP_FRAMES == 0:
                screen_rects = []
                for r in model(frame_small):
                    for box in r.boxes:
                        cls_id = int(box.cls)
                        if cls_id in (62,63,67):  # tv, laptop, cell phone
                            x1,y1,x2,y2 = map(int, box.xyxy[0].cpu().numpy())
                            sx = int((x1 / sw) * w); sy = int((y1 / sh) * h)
                            swid = int(((x2-x1) / sw) * w); shei = int(((y2-y1) / sh) * h)
                            screen_rects.append((sx,sy,swid,shei))
                prev_screen_rects = screen_rects
            else:
                screen_rects = prev_screen_rects

            # draw screens
            for sx,sy,swid,shei in screen_rects:
                cv2.rectangle(frame,(sx,sy),(sx+swid,sy+shei),(0,0,255),2)

            # FaceMesh on small image (map normalized landmarks to original size)
            results = face_mesh.process(img_rgb_small)
            detections = []
            if results.multi_face_landmarks:
                for face_lms in results.multi_face_landmarks:
                    xs = []; ys = []
                    for lm in face_lms.landmark:
                        x_px = int(lm.x * w); y_px = int(lm.y * h)
                        xs.append(x_px); ys.append(y_px)
                    x_min,x_max = min(xs), max(xs); y_min,y_max = min(ys), max(ys)
                    bw = x_max - x_min; bh = y_max - y_min
                    center = ( (x_min+x_max)//2, (y_min+y_max)//2 )
                    detections.append({'landmark_obj': face_lms, 'landmarks': face_lms.landmark,
                                       'bbox':(x_min,y_min,bw,bh), 'center':center})

            # tracker: center-based matching
            matched = set(); updated = {}
            for det in detections:
                cx,cy = det['center']; bx,by,bw,bh = det['bbox']
                best_id = None; best_dist = None; match_thresh = max(20, bw*0.6)
                for fid,st in faces_state.items():
                    d = dist(st['center'], (cx,cy))
                    if d <= match_thresh and (best_dist is None or d < best_dist):
                        best_dist = d; best_id = fid
                if best_id is None:
                    fid = next_id; next_id += 1
                    st = {'id':fid, 'center':(cx,cy), 'bbox':(bx,by,bw,bh),
                          'last_seen':frame_idx, 'blink_counter':0, 'smile_counter':0,
                          'blink_count':0, 'last_saved_event_frame':0, 'is_fake_face':False}
                else:
                    fid = best_id; st = faces_state[fid]
                    st['center']=(cx,cy); st['bbox']=(bx,by,bw,bh); st['last_seen']=frame_idx
                st['det']=det; updated[fid]=st; matched.add(fid)

            # keep recently lost tracks
            TIMEOUT = 12
            for fid,st in list(faces_state.items()):
                if fid not in matched and (frame_idx - st.get('last_seen',0) <= TIMEOUT):
                    updated[fid]=st
            faces_state = updated

            # fake-face detection: face fully inside any detected screen rect
            for fid,st in faces_state.items():
                if 'det' not in st: continue
                bx,by,bw,bh = st['bbox']
                st['is_fake_face'] = any(bx >= sx and by >= sy and bx + bw <= sx + sw and by + bh <= sy + sh
                                         for sx,sy,sw,sh in screen_rects)

            # analyze tracked faces
            for fid,st in faces_state.items():
                if 'det' not in st: continue
                landmarks = st['det']['landmarks']; bx,by,bw,bh = st['bbox']; cx,cy = st['center']

                # color and label for fake face
                if st.get('is_fake_face'):
                    color = (255,0,0)  # blue
                    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh), color, 2)
                    cv2.putText(frame, f"ID:{fid}", (bx,by-8), FONT, 0.6, (0,255,255), 2)
                    # label the blue bounding box with "FAKE FACE"
                    cv2.putText(frame, "FAKE_FACE", (bx, by + bh + 28), FONT, 0.7, color, 2)
                else:
                    color = (0,255,0)  # green
                    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh), color, 2)
                    cv2.putText(frame, f"ID:{fid}", (bx,by-8), FONT, 0.6, (0,255,255), 2)

                # mesh
                try:
                    if HAVE_DRAWING_STYLES:
                        mp_drawing.draw_landmarks(image=frame, landmark_list=st['det']['landmark_obj'],
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    else:
                        mp_drawing.draw_landmarks(image=frame, landmark_list=st['det']['landmark_obj'],
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
                            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,200), thickness=1, circle_radius=1))
                except Exception:
                    pass

                # save fake face occasionally
                if st['is_fake_face'] and (frame_idx - st.get('last_saved_event_frame',0) > SAVE_BLOCK_FRAMES):
                    save_labeled_frame(frame, f"FAKE_FACE_ID{fid}", FAKE_FACE_DIR, color=color)
                    st['last_saved_event_frame'] = frame_idx

                # movement (early)
                prev_center = st.get('center_prev')
                moved = False
                if prev_center is not None:
                    if dist(prev_center, (cx,cy)) > bw * MOVE_RATIO and (frame_idx - st.get('last_saved_event_frame',0) > SAVE_BLOCK_FRAMES):
                        moved = True

                # blink (EAR)
                ear = (eye_aspect_ratio(landmarks, w, h, RIGHT_EYE_IDX) + eye_aspect_ratio(landmarks, w, h, LEFT_EYE_IDX)) / 2.0
                blink_detected = False
                if ear < EAR_THRESH:
                    st['blink_counter'] = st.get('blink_counter',0) + 1
                else:
                    if st.get('blink_counter',0) >= EAR_CONSEC_FRAMES and not moved:
                        blink_detected = True
                    st['blink_counter'] = 0
                if blink_detected:
                    st['blink_count'] = st.get('blink_count',0) + 1

                # smile (mouth geometry)
                try:
                    p_left = to_pixel(landmarks[MOUTH_LEFT], w, h)
                    p_right = to_pixel(landmarks[MOUTH_RIGHT], w, h)
                    p_top = to_pixel(landmarks[MOUTH_TOP], w, h)
                    p_bottom = to_pixel(landmarks[MOUTH_BOTTOM], w, h)
                except Exception:
                    p_left = p_right = p_top = p_bottom = (0,0)

                mouth_w = dist(p_left, p_right)
                mouth_open = dist(p_top, p_bottom)
                mouth_w_ratio = mouth_w / float(bw + 1e-6)
                mouth_open_ratio = mouth_open / float(mouth_w + 1e-6)

                small_mode = (bw < SMALL_FACE_PIX)
                smile_detected = False
                if small_mode:
                    req_px = max(14, int(bw * SMALL_FACE_MOUTH_PIX_RATIO))
                    if mouth_w >= req_px and mouth_open_ratio < SMILE_OPEN_RATIO:
                        st['smile_counter'] = st.get('smile_counter',0) + 1
                    else:
                        st['smile_counter'] = 0
                else:
                    if mouth_w_ratio >= SMILE_WIDTH_RATIO and mouth_open_ratio < SMILE_OPEN_RATIO:
                        st['smile_counter'] = st.get('smile_counter',0) + 1
                    else:
                        st['smile_counter'] = 0

                if st.get('smile_counter',0) >= 3:
                    smile_detected = True
                    st['smile_counter'] = 0

                # save events
                if smile_detected and (frame_idx - st.get('last_saved_event_frame',0) > SAVE_BLOCK_FRAMES):
                    # If smile on a fake face -> save to FAKE_SMILE_DIR (and also keep a copy in SMILE_DIR)
                    if st.get('is_fake_face'):
                        save_labeled_frame(frame, f"FAKE_SMILE_ID{fid}", SMILE_DIR, color=(0,0,255))
                    else:
                        save_labeled_frame(frame, f"SMILE_ID{fid}", SMILE_DIR, color=(0,255,0))
                        cv2.putText(frame, "SMILE", (bx, by + bh + 30), FONT, 0.8, (0,255,0), 2)
                    st['last_saved_event_frame'] = frame_idx

                if blink_detected and (frame_idx - st.get('last_saved_event_frame',0) > SAVE_BLOCK_FRAMES):
                    save_labeled_frame(frame, f"BLINK_ID{fid}", BLINK_DIR, color=(0,0,255))
                    st['last_saved_event_frame'] = frame_idx
                    cv2.putText(frame, "BLINK", (bx, by + bh + 30), FONT, 0.8, (0,0,255), 2)
                elif blink_detected:
                    cv2.putText(frame, "BLINK", (bx, by + bh + 30), FONT, 0.8, (0,0,255), 2)

                if moved and (frame_idx - st.get('last_saved_event_frame',0) > SAVE_BLOCK_FRAMES):
                    save_labeled_frame(frame, f"MOVED_ID{fid}", MOTION_DIR, color=(0,255,0))
                    st['last_saved_event_frame'] = frame_idx
                    cv2.putText(frame, "MOVED", (bx, by + bh + 30), FONT, 0.8, (0,255,0), 2)

                # overlays
                cv2.putText(frame, f"EAR:{ear:.2f}", (bx, by + bh + 10), FONT, 0.5, (255,255,255), 2)
                cv2.putText(frame, f"MWR:{mouth_w_ratio:.2f}", (bx, by + bh + 22), FONT, 0.5, (255,255,255), 2)

                st['center_prev'] = (cx,cy)

            cv2.imshow("FaceMesh - blink/smile/fakeface", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
