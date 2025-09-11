# ================== 로그 억제(선택) ==================
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# ================== 기본 임포트 ==================
import cv2
import mediapipe as mp
import numpy as np
import time
import argparse, json, sys
from collections import deque

# ================== CLI ==================
parser = argparse.ArgumentParser(description="Front-only posture & blink (BB with debug/JSONL)")
parser.add_argument("--video", type=str, default=None, help="입력 영상 경로(없으면 웹캠 0번)")
parser.add_argument("--json_out", type=str, default=None, help="프레임별 결과(JSONL) 저장 경로")
parser.add_argument("--no_display", action="store_true", help="창 표시 없이 실행")
parser.add_argument("--max_frames", type=int, default=0, help="0=제한 없음")
args, _ = parser.parse_known_args()

# ================== 기본 설정 (필요시 조정) ==================
BASE_EAR_THRESHOLD = 0.20
BLINK_THRESHOLD_PER_WIN = 20
BLINK_WINDOW_SEC = 300
MIN_CLOSED_FRAMES = 2
MIN_OPEN_FRAMES = 1
Y_THR_RATIO_BROW  = 0.06
Y_THR_RATIO_EYE   = 0.05
Y_THR_RATIO_CHEEK = 0.06
X_THR_RATIO_MID   = 0.05
Y_THR_RATIO_SHOULDER = 0.06
X_THR_RATIO_NOSE     = 0.05
MIN_PX_THR = 2.0
EMA_ALPHA  = 0.2
PROC_W, PROC_H = 640, 360
FRAME_SKIP_N   = 2

# ================== 상태값 ==================
blink_count = 0
eye_closed = False
win_start = time.time()
flip_view = False
ema_vals = {}
consecutive_closed = 0
consecutive_open = 0
left_eye_closed = False
right_eye_closed = False
ear_history = deque(maxlen=100)
ear_baseline = None
frame_idx_infer = 0
last_face_landmarks = None
last_pose_landmarks = None

# ================== MediaPipe ==================
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

# ================== 필요한 FaceMesh 인덱스 ==================
NEEDED_IDXS = {
    "L_EYE_OUTER": 33, "R_EYE_OUTER": 263,
    "L_EYE_INNER": 133, "R_EYE_INNER": 362,
    "BROW_L": 105, "BROW_R": 334,
    "CHEEK_L": 50,  "CHEEK_R": 280,
    "TOP_C": 10,    "BOT_C": 152,
    "LE_1": 33, "LE_2": 159, "LE_3": 158, "LE_5": 153, "LE_6": 145, "LE_4": 133,
    "RE_1": 263, "RE_2": 386, "RE_3": 385, "RE_5": 380, "RE_6": 374, "RE_4": 362,
}

# ================== 보조 함수 ==================
def ema(key, value, alpha=EMA_ALPHA):
    if value is None: return None
    if key not in ema_vals:
        ema_vals[key] = value
    else:
        ema_vals[key] = alpha * value + (1 - alpha) * ema_vals[key]
    return ema_vals[key]

def safe_dist(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))

def adaptive_thresh(ipd, ratio):
    if ipd is None or ipd <= 1:
        return max(MIN_PX_THR, 60.0 * ratio)
    return max(MIN_PX_THR, ipd * ratio)

def compute_ear_from_points(p1,p2,p3,p5,p6,p4):
    den = 2.0 * np.linalg.norm(p1-p4)
    if den < 1e-6:
        return None
    ear = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / den
    return float(np.clip(ear, 0.0, 1.0))

def update_ear_baseline(ear_avg):
    global ear_baseline, ear_history
    if ear_avg is not None:
        ear_history.append(ear_avg)
        if len(ear_history) >= 30:
            ear_baseline = np.percentile(ear_history, 75)

def get_dynamic_threshold(base_threshold, ipd_px):
    global ear_baseline
    ear_thr = base_threshold
    if ipd_px is not None and ipd_px > 1:
        ear_thr = np.clip(base_threshold * (ipd_px / 60.0), 0.15, 0.28)
    if ear_baseline is not None:
        dynamic_threshold = max(0.17, ear_baseline * 0.7)
        ear_thr = min(ear_thr, dynamic_threshold)
    return ear_thr

def detect_blink_improved(ear_l, ear_r, ear_thr):
    global consecutive_closed, consecutive_open, eye_closed, blink_count
    global left_eye_closed, right_eye_closed
    if ear_l is None or ear_r is None:
        return
    left_currently_closed = ear_l < ear_thr
    right_currently_closed = ear_r < ear_thr
    any_eye_closed = (left_currently_closed or right_currently_closed)
    if any_eye_closed:
        consecutive_closed += 1
        consecutive_open = 0
        if consecutive_closed >= MIN_CLOSED_FRAMES and not eye_closed:
            eye_closed = True
    else:
        consecutive_open += 1
        consecutive_closed = 0
        if consecutive_open >= MIN_OPEN_FRAMES and eye_closed:
            blink_count += 1
            eye_closed = False
    left_eye_closed = left_currently_closed
    right_eye_closed = right_currently_closed

def draw_marker(img, pt, color, r=4, filled=True):
    if pt is None: return
    cv2.circle(img, (int(pt[0]), int(pt[1])), r, color, -1 if filled else 2, cv2.LINE_AA)

def draw_line(img, p1, p2, color, thick=2):
    if p1 is None or p2 is None: return
    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thick, cv2.LINE_AA)

def put_text(img, text, org, color, scale=0.7, thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def draw_panel(img, x, y, w, h, alpha=0.35):
    overlay = img.copy()
    cv2.rectangle(overlay, (x,y), (x+w, y+h), (20,20,20), -1)
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

def pick_points(flm, W, H):
    P = {}
    for k, idx in NEEDED_IDXS.items():
        l = flm[idx]
        P[k] = np.array([l.x * W, l.y * H], dtype=np.float32)
    return P

# ================== 입력 소스 열기 (FFMPEG 우선) ==================
if args.video is not None:
    cap = cv2.VideoCapture(args.video, cv2.CAP_FFMPEG)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# --- 디버그: 소스/속성 출력 ---
print(">>> OPENED:", cap.isOpened())
print(">>> SRC:", args.video or "camera0")
print(">>> FPS:", cap.get(cv2.CAP_PROP_FPS), " COUNT:", cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print(f"[ERR] Cannot open input: {args.video or 'camera 0'}", file=sys.stderr)
    sys.exit(2)

# ================== JSONL 로그 준비 ==================
f_out = None
if args.json_out:
    try:
        f_out = open(args.json_out, "w", encoding="utf-8")
        print(">>> JSONL ready:", args.json_out)
    except Exception as e:
        print("[ERR] cannot open JSONL for write:", e, file=sys.stderr)
        f_out = None

t0 = time.time()
frame_idx_json = 0
_prev_blink_for_event = 0

# ================== Solutions 초기화 ==================
mp_face_obj = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    refine_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_pose_obj = mp.solutions.pose.Pose(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

fps_deque = deque(maxlen=20)
last_time = time.time()

# ================== 메인 루프 ==================
___first_read_flag = True  # 첫 프레임 디버그용
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        if ___first_read_flag:
            print(">>> READ FAIL (EOF or cannot decode)")
        break
    if ___first_read_flag:
        print(">>> READ OK: first frame captured")
        ___first_read_flag = False

    if flip_view:
        frame = cv2.flip(frame, 1)

    h, w = frame.shape[:2]
    proc = cv2.resize(frame, (PROC_W, PROC_H), interpolation=cv2.INTER_LINEAR)
    proc_rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
    proc_rgb.flags.writeable = False

    run_inference = (frame_idx_infer % FRAME_SKIP_N == 0)
    frame_idx_infer += 1

    if run_inference:
        results_face = mp_face_obj.process(proc_rgb)
        results_pose = mp_pose_obj.process(proc_rgb)
        last_face_landmarks = results_face.multi_face_landmarks[0].landmark if results_face.multi_face_landmarks else None
        last_pose_landmarks = results_pose.pose_landmarks.landmark if results_pose.pose_landmarks else None
    else:
        class _Dummy: pass
        results_face = _Dummy(); results_pose = _Dummy()
        results_face.multi_face_landmarks = [_Dummy()] if last_face_landmarks is not None else None
        if results_face.multi_face_landmarks:
            results_face.multi_face_landmarks[0].landmark = last_face_landmarks
        results_pose.pose_landmarks = _Dummy() if last_pose_landmarks is not None else None
        if results_pose.pose_landmarks:
            results_pose.pose_landmarks.landmark = last_pose_landmarks

    proc_rgb.flags.writeable = True
    image = frame.copy()

    GREEN=(0,210,0); RED=(0,0,230); YEL=(0,220,220)
    WHITE=(230,230,230); GRAY=(160,160,160); CYAN=(200,255,255)
    BLUE=(190,160,0); ORANGE=(0,140,255)

    ipd_px = None
    head_level_face = None
    shoulders_level = None
    nose_aligned = None

    # ---------- FaceMesh ----------
    if results_face.multi_face_landmarks:
        flm = results_face.multi_face_landmarks[0].landmark
        P = pick_points(flm, w, h)
        L_eye_outer, R_eye_outer = P["L_EYE_OUTER"], P["R_EYE_OUTER"]
        L_eye_inner, R_eye_inner = P["L_EYE_INNER"], P["R_EYE_INNER"]
        brow_L, brow_R = P["BROW_L"], P["BROW_R"]
        cheek_L, cheek_R = P["CHEEK_L"], P["CHEEK_R"]
        top_c, bot_c = P["TOP_C"], P["BOT_C"]

        ipd_px = safe_dist(L_eye_outer, R_eye_outer)
        thr_brow  = adaptive_thresh(ipd_px, Y_THR_RATIO_BROW)
        thr_eye   = adaptive_thresh(ipd_px, Y_THR_RATIO_EYE)
        thr_cheek = adaptive_thresh(ipd_px, Y_THR_RATIO_CHEEK)
        thr_mid   = adaptive_thresh(ipd_px, X_THR_RATIO_MID)

        dy_brow  = abs(brow_L[1] - brow_R[1]);   dy_brow_s  = ema("dy_brow", dy_brow)
        L_eye_c  = (L_eye_outer + L_eye_inner)/2.0
        R_eye_c  = (R_eye_outer + R_eye_inner)/2.0
        dy_eye   = abs(L_eye_c[1] - R_eye_c[1]);  dy_eye_s   = ema("dy_eye", dy_eye)
        dy_cheek = abs(cheek_L[1] - cheek_R[1]);  dy_cheek_s = ema("dy_cheek", dy_cheek)
        dx_mid   = abs(top_c[0] - bot_c[0]);      dx_mid_s   = ema("dx_mid", dx_mid)

        brow_ok  = dy_brow_s  is not None and dy_brow_s  <= thr_brow
        eye_ok   = dy_eye_s   is not None and dy_eye_s   <= thr_eye
        cheek_ok = dy_cheek_s is not None and dy_cheek_s <= thr_cheek
        mid_ok   = dx_mid_s   is not None and dx_mid_s   <= thr_mid
        head_level_face = (brow_ok or eye_ok or cheek_ok or mid_ok)

        # 시각화
        draw_line(image, brow_L, brow_R, GREEN if brow_ok else RED, 3)
        draw_marker(image, brow_L, BLUE, 4); draw_marker(image, brow_R, BLUE, 4)
        put_text(image, f"Brows dy={dy_brow_s:.1f}/{thr_brow:.1f}px", (30, 80),
                 GREEN if brow_ok else RED, 0.65, 2)
        draw_line(image, L_eye_c, R_eye_c, GREEN if eye_ok else RED, 3)
        draw_marker(image, L_eye_c, CYAN, 4); draw_marker(image, R_eye_c, CYAN, 4)
        put_text(image, f"Eyes  dy={dy_eye_s:.1f}/{thr_eye:.1f}px", (30, 110),
                 GREEN if eye_ok else RED, 0.65, 2)
        draw_line(image, cheek_L, cheek_R, GREEN if cheek_ok else RED, 3)
        draw_marker(image, cheek_L, ORANGE, 4); draw_marker(image, cheek_R, ORANGE, 4)
        put_text(image, f"Cheek dy={dy_cheek_s:.1f}/{thr_cheek:.1f}px", (30, 140),
                 GREEN if cheek_ok else RED, 0.65, 2)
        draw_line(image, top_c, bot_c, GREEN if mid_ok else RED, 3)
        draw_marker(image, top_c, YEL, 5); draw_marker(image, bot_c, YEL, 5)
        put_text(image, f"Mid   dx={dx_mid_s:.1f}/{thr_mid:.1f}px", (30, 170),
                 GREEN if mid_ok else RED, 0.65, 2)

        # EAR/깜빡임
        le = [P["LE_1"], P["LE_2"], P["LE_3"], P["LE_5"], P["LE_6"], P["LE_4"]]
        re = [P["RE_1"], P["RE_2"], P["RE_3"], P["RE_5"], P["RE_6"], P["RE_4"]]
        ear_l = compute_ear_from_points(*le)
        ear_r = compute_ear_from_points(*re)
        if ear_l is not None and ear_r is not None:
            ear_avg = (ear_l + ear_r) / 2.0
            update_ear_baseline(ear_avg)
            ear_thr = get_dynamic_threshold(BASE_EAR_THRESHOLD, ipd_px)
            detect_blink_improved(ear_l, ear_r, ear_thr)

    # ---------- Pose ----------
    if hasattr(results_pose, "pose_landmarks") and results_pose.pose_landmarks:
        lm = results_pose.pose_landmarks.landmark
        def get_xy(idx):
            L = lm[idx]
            return np.array([L.x * w, L.y * h], dtype=np.float32)
        L_sh = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        R_sh = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        nose = get_xy(mp_pose.PoseLandmark.NOSE.value)
        thr_shoulder = adaptive_thresh(ipd_px, Y_THR_RATIO_SHOULDER)
        thr_nose     = adaptive_thresh(ipd_px, X_THR_RATIO_NOSE)
        dy_sh = abs(L_sh[1] - R_sh[1]); dy_sh_s = ema("dy_shoulder", dy_sh)
        shoulders_level = (dy_sh_s is not None and dy_sh_s <= thr_shoulder)
        draw_line(image, L_sh, R_sh, (0,210,0) if shoulders_level else (0,0,230), 3)
        draw_marker(image, L_sh, (255,120,120), 5); draw_marker(image, R_sh, (255,120,120), 5)
        put_text(image, f"Shoulders dy={dy_sh_s:.1f}/{thr_shoulder:.1f}px",
                 (30, 210), (0,210,0) if shoulders_level else (0,0,230), 0.65, 2)
        center_sh = (L_sh + R_sh) / 2.0
        dx_nc = abs(nose[0] - center_sh[0]); dx_nc_s = ema("dx_nose_center", dx_nc)
        nose_aligned = (dx_nc_s is not None and dx_nc_s <= thr_nose)
        up = np.array([center_sh[0], max(0, center_sh[1]-100)])
        dn = np.array([center_sh[0], min(h-1, center_sh[1]+100)])
        draw_line(image, up, dn, (180,180,255) if nose_aligned else (120,120,255), 2)
        draw_marker(image, center_sh, (200,200,255), 5)
        draw_marker(image, nose, (255,255,0), 5)
        put_text(image, f"Nose   dx={dx_nc_s:.1f}/{thr_nose:.1f}px",
                 (30, 240), (0,210,0) if nose_aligned else (0,0,230), 0.65, 2)

    # ---------- 타이틀 / 표시 ----------
    any_ok = None
    if 'head_level_face' in locals() or ('shoulders_level' in locals() and 'nose_aligned' in locals()):
        face_ok = bool(locals().get('head_level_face')) if 'head_level_face' in locals() else False
        body_ok = (locals().get('shoulders_level') and locals().get('nose_aligned')) if ('shoulders_level' in locals() and 'nose_aligned' in locals()) else False
        any_ok = (face_ok or body_ok)
    title = "Head Level" if any_ok else "Head Tilted" if any_ok is not None else "Detecting..."
    put_text(image, title, (30, 40), (0,200,0) if any_ok else ((0,0,230) if any_ok is not None else (200,200,200)), 1.0, 2)

    # 깜빡임/시간/FPS 표시(간단)
    now = time.time()
    dt = now - last_time
    last_time = now
    if dt > 0:
        fps_deque.append(1.0/dt)
        fps = np.mean(fps_deque)
        put_text(image, f"FPS: {fps:.1f}", (w - 130, h - 20), (200,200,200), 0.8, 2)

    # ---------- JSONL 기록 ----------
    if f_out:
        if frame_idx_json == 0:
            print(">>> JSONL write started:", args.json_out)
        rec = {
            "idx": frame_idx_json + 1,
            "ts": time.time() - t0,
            "title": title,
            "blink_count": int(blink_count),
            "blink_event": int(blink_count > _prev_blink_for_event)
        }
        f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    frame_idx_json += 1
    _prev_blink_for_event = blink_count

    # ---------- 디스플레이 / 키 처리 ----------
    if not args.no_display:
        cv2.imshow("Front Posture - BB", image)
        key = cv2.waitKey(1) & 0xFF
    else:
        key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        flip_view = not flip_view
    elif key == ord('r'):
        blink_count = 0; win_start = time.time(); ema_vals.clear()
        consecutive_closed = 0; consecutive_open = 0
        eye_closed = False; left_eye_closed = False; right_eye_closed = False
        ear_history.clear(); ear_baseline = None

    if args.max_frames > 0 and frame_idx_json >= args.max_frames:
        break

# ================== 종료 처리 ==================
if f_out:
    f_out.close()
    print(">>> JSONL closed:", args.json_out)
cap.release()
cv2.destroyAllWindows()
