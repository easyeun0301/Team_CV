import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# ================== 설정값 ==================
BASE_EAR_THRESHOLD = 0.20
BLINK_THRESHOLD_PER_WIN = 20
BLINK_WINDOW_SEC = 300

# FaceMesh 정면 수평 판단 (IPD 비례 가변 임계값)
Y_THR_RATIO_BROW  = 0.06   # 눈썹 Δy
Y_THR_RATIO_EYE   = 0.05   # 눈 Δy
Y_THR_RATIO_CHEEK = 0.06   # 광대 Δy
X_THR_RATIO_MID   = 0.05   # 중앙선 Δx

# Pose 기반
Y_THR_RATIO_SHOULDER = 0.06  # 어깨 Δy
X_THR_RATIO_NOSE     = 0.05  # 코-어깨중앙 Δx

MIN_PX_THR = 2.0
EMA_ALPHA  = 0.2

# ================== 상태값 ==================
blink_count = 0
eye_closed = False
win_start = time.time()
flip_view = False
ema_vals = {}

# ================== MediaPipe ==================
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

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
        return max(MIN_PX_THR, 60.0 * ratio)  # IPD 미검출 시 60px 가정
    return max(MIN_PX_THR, ipd * ratio)

def compute_ear(flm, w, h, left=True):
    def fxy(i):
        return np.array([flm[i].x * w, flm[i].y * h], dtype=np.float32)
    if left:
        p1,p2,p3,p5,p6,p4 = fxy(33), fxy(159), fxy(158), fxy(153), fxy(145), fxy(133)
    else:
        p1,p2,p3,p5,p6,p4 = fxy(263), fxy(386), fxy(385), fxy(380), fxy(374), fxy(362)
    den = 2.0 * np.linalg.norm(p1-p4)
    if den < 1e-6: return None
    ear = (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / den
    return float(np.clip(ear, 0.0, 1.0))

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

# ================== 메인 ==================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    fps_deque = deque(maxlen=20)
    last_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        if flip_view: frame = cv2.flip(frame, 1)

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results_face = face_mesh.process(rgb)
        results_pose = pose.process(rgb)
        rgb.flags.writeable = True

        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # 색상 팔레트
        GREEN = (0, 210, 0)
        RED   = (0, 0, 230)
        YEL   = (0, 220, 220)
        WHITE = (230, 230, 230)
        GRAY  = (160, 160, 160)
        CYAN  = (200, 255, 255)
        BLUE  = (190, 160, 0)
        ORANGE= (0, 140, 255)

        ipd_px = None
        head_level_face = None
        shoulders_level = None
        nose_aligned    = None

        # ================== FaceMesh: 눈썹/눈/광대/중앙선 ==================
        if results_face.multi_face_landmarks:
            flm = results_face.multi_face_landmarks[0].landmark
            def p(i): return np.array([flm[i].x * w, flm[i].y * h], dtype=np.float32)

            # 포인트
            L_eye_outer, R_eye_outer = p(33), p(263)
            L_eye_inner, R_eye_inner = p(133), p(362)
            brow_L, brow_R           = p(105), p(334)
            cheek_L, cheek_R         = p(50),  p(280)
            top_c, bot_c             = p(10),  p(152)

            # IPD
            ipd_px = safe_dist(L_eye_outer, R_eye_outer)

            # 임계값
            thr_brow  = adaptive_thresh(ipd_px, Y_THR_RATIO_BROW)
            thr_eye   = adaptive_thresh(ipd_px, Y_THR_RATIO_EYE)
            thr_cheek = adaptive_thresh(ipd_px, Y_THR_RATIO_CHEEK)
            thr_mid   = adaptive_thresh(ipd_px, X_THR_RATIO_MID)

            # 지표 계산 + EMA
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

            # 보조선/마커/라벨
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
            ear_l = compute_ear(flm, w, h, left=True)
            ear_r = compute_ear(flm, w, h, left=False)
            ear = None
            if ear_l is not None and ear_r is not None:
                ear = ema("ear", (ear_l + ear_r)/2.0)

            ear_thr = BASE_EAR_THRESHOLD
            if ipd_px is not None and ipd_px > 1:
                ear_thr = np.clip(BASE_EAR_THRESHOLD * (ipd_px / 60.0), 0.15, 0.28)

            if ear is not None:
                if ear < ear_thr:
                    if not eye_closed:
                        eye_closed = True
                else:
                    if eye_closed:
                        blink_count += 1
                        eye_closed = False

        # ================== Pose: 어깨 높낮이, 코-어깨중앙 ==================
        if results_pose.pose_landmarks:
            lm = results_pose.pose_landmarks.landmark
            def get_xy(idx): return np.array([lm[idx].x*w, lm[idx].y*h], dtype=np.float32)

            L_sh = get_xy(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            R_sh = get_xy(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            nose = get_xy(mp_pose.PoseLandmark.NOSE.value)

            # 임계값
            thr_shoulder = adaptive_thresh(ipd_px, Y_THR_RATIO_SHOULDER)
            thr_nose     = adaptive_thresh(ipd_px, X_THR_RATIO_NOSE)

            # 어깨 높낮이
            dy_sh = abs(L_sh[1] - R_sh[1]); dy_sh_s = ema("dy_shoulder", dy_sh)
            shoulders_level = (dy_sh_s is not None and dy_sh_s <= thr_shoulder)

            draw_line(image, L_sh, R_sh, GREEN if shoulders_level else RED, 3)
            draw_marker(image, L_sh, (255,120,120), 5); draw_marker(image, R_sh, (255,120,120), 5)
            put_text(image, f"Shoulders dy={dy_sh_s:.1f}/{thr_shoulder:.1f}px",
                     (30, 210), GREEN if shoulders_level else RED, 0.65, 2)

            # 코-어깨중앙 정렬 (수직 이등분선 위?)
            center_sh = (L_sh + R_sh) / 2.0
            dx_nc = abs(nose[0] - center_sh[0]); dx_nc_s = ema("dx_nose_center", dx_nc)
            nose_aligned = (dx_nc_s is not None and dx_nc_s <= thr_nose)

            # 이등분선 시각화: 어깨 중앙에서 위/아래로 수직선
            up = np.array([center_sh[0], max(0, center_sh[1]-100)])
            dn = np.array([center_sh[0], min(h-1, center_sh[1]+100)])
            draw_line(image, up, dn, (180,180,255) if nose_aligned else (120,120,255), 2)
            draw_marker(image, center_sh, (200,200,255), 5)
            draw_marker(image, nose, (255,255,0), 5)
            put_text(image, f"Nose   dx={dx_nc_s:.1f}/{thr_nose:.1f}px",
                     (30, 240), GREEN if nose_aligned else RED, 0.65, 2)

        # ================== 상단 타이틀 ==================
        # 최종 헤드 레벨 판단(얼굴 OR 어깨/코 기준 중 하나라도 OK면 OK로 볼지, 모두 OK여야 OK로 볼지 선택)
        # 여기서는 "둘 중 하나라도 OK면 OK"로 표시 (원하면 AND 로 바꾸세요)
        any_ok = None
        if head_level_face is not None or (shoulders_level is not None and nose_aligned is not None):
            face_ok = bool(head_level_face) if head_level_face is not None else False
            body_ok = (shoulders_level and nose_aligned) if (shoulders_level is not None and nose_aligned is not None) else False
            any_ok = (face_ok or body_ok)

        title = "Head Level" if any_ok else "Head Tilted" if any_ok is not None else "Detecting..."
        title_color = (0, 200, 0) if any_ok else ((0, 0, 230) if any_ok is not None else (200,200,200))
        put_text(image, title, (30, 40), title_color, 1.0, 2)

        # ================== 우측 패널 ==================
        panel_w = 330
        image = draw_panel(image, w - panel_w - 20, 20, panel_w, 185, 0.35)
        px, py = w - panel_w - 10, 42
        put_text(image, "Summary", (px, py), WHITE, 0.85, 2); py += 28
        put_text(image, "Green=within threshold", (px, py), (180,255,180), 0.58, 2); py += 22
        put_text(image, "Red  =tilted / misaligned", (px, py), (150,150,255), 0.58, 2); py += 22
        put_text(image, "Keys: [f] flip  [r] reset  [q] quit", (px, py), GRAY, 0.58, 2); py += 22

        # ================== 깜빡임 / FPS ==================
        now = time.time()
        elapsed = now - win_start
        if elapsed >= BLINK_WINDOW_SEC:
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {blink_count}/{BLINK_THRESHOLD_PER_WIN} in {int(elapsed)}s"
            put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else RED, 0.75, 2)
            blink_count = 0
            win_start = now
        else:
            remain = BLINK_WINDOW_SEC - int(elapsed)
            pass_flag = (blink_count >= BLINK_THRESHOLD_PER_WIN)
            msg = f"Blink {blink_count}/{BLINK_THRESHOLD_PER_WIN} (remain {remain}s)"
            put_text(image, msg, (30, h - 60), (0, 220, 0) if pass_flag else (255, 255, 0), 0.75, 2)

        dt = now - last_time
        last_time = now
        if dt > 0:
            fps_deque.append(1.0/dt)
            fps = np.mean(fps_deque)
            put_text(image, f"FPS: {fps:.1f}", (w - 130, h - 20), (200,200,200), 0.8, 2)

        # 하단 도움말
        put_text(image, "Front-only | FaceMesh + Pose fusion", (30, h - 25), GRAY, 0.65, 2)

        cv2.imshow("Front Posture - Head Level (FaceMesh+Pose)", image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            flip_view = not flip_view
        elif key == ord('r'):
            blink_count = 0
            win_start = time.time()
            ema_vals.clear()

cap.release()
cv2.destroyAllWindows()
