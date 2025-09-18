import cv2
import argparse
import logging
import sys
import os
import numpy as np
import mediapipe as mp
from collections import deque
from spinepose import SpinePoseEstimator
import threading
import queue
import time
import cProfile
import pstats
import platform

# ───────────────────────────────────────────────────────────────────
# 상위 디렉토리를 path에 추가 (server 폴더 접근용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import server
# ───────────────────────────────────────────────────────────────────

# 로깅 설정: 오류 및 경고만 출력하도록 설정
logging.basicConfig(level=logging.ERROR)
logging.getLogger('aioice').setLevel(logging.WARNING)
logging.getLogger('absl').disabled = True

# ───────────────────────────────────────────────────────────────────
# 동적 환경 최적화 (로컬 CPU/코어/스레드에 맞춤)
# ───────────────────────────────────────────────────────────────────
def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

psutil = _safe_import("psutil")
cpuinfo = _safe_import("cpuinfo")

def detect_cpu_env():
    logical = os.cpu_count() or 4
    physical = None
    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    if physical is None:
        physical = logical

    # 프로세스 affinity로 제한된 코어 수(가능하면)
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        try:
            affinity = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if affinity is None and psutil:
        try:
            p = psutil.Process()
            if hasattr(p, "cpu_affinity"):
                affinity = len(p.cpu_affinity())
        except Exception:
            pass
    usable = affinity or physical

    # 이름/플래그(선택)
    name = platform.processor() or platform.machine()
    if cpuinfo:
        try:
            info = cpuinfo.get_cpu_info()
            name = info.get("brand_raw") or name
        except Exception:
            pass

    return {
        "cpu_name": name,
        "physical": int(physical),
        "logical": int(logical),
        "usable": int(usable),
    }

def apply_thread_tuning(env):
    """OpenCV=1, BLAS/OMP 스레드 수 통일"""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    n = max(1, min(env["usable"], 8))  # 노트북 과열 방지 상한 8
    for k in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
    ]:
        os.environ[k] = str(n)

    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    return n

def recommend_period_ms(env):
    phys = env["physical"]
    if phys <= 4:
        return 110
    elif phys <= 6:
        return 100
    else:
        return 90

# ───────────────────────────────────────────────────────────────────
# 프로파일러 유틸(옵션)
# ───────────────────────────────────────────────────────────────────
def run_profiler():
    cProfile.run('main()', 'profile_results')

def analyze_profile():
    p = pstats.Stats('profile_results')
    p.strip_dirs().sort_stats('time').print_stats(20)

# 스티키(깜빡임 방지)
STICKY_MS = 500
last_good_results = {"sp_kpts": [], "sp_scores": [], "roi": None}
last_update_ms = 0.0

# ───────────────────────────────────────────────────────────────────
# AI 모델 초기화 - 보편적 최적화
# ───────────────────────────────────────────────────────────────────
try:
    spine_est = SpinePoseEstimator(mode="small", device="cpu")
except Exception as e:
    print(f"SpinePoseEstimator 로드 실패: {e}. 프로그램을 종료합니다.")
    sys.exit(1)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
PL = mp_pose.PoseLandmark

# 표시 파라미터
WIN_W, WIN_H = 720, 1440
SPINE_SCORE_TH = 0.1

# 성능 측정 변수 (latency 히스토리)
blazepose_latency_hist = deque(maxlen=30)
spinepose_latency_hist = deque(maxlen=30)
end_to_end_latency_hist = deque(maxlen=30)
display_latency_hist = deque(maxlen=30)

# 프레임 버퍼링
frame_queue = queue.Queue(maxsize=1)
result_queue = queue.Queue(maxsize=1)

# ───────────────────────────────────────────────────────────────────
# 유틸 함수들
# ───────────────────────────────────────────────────────────────────
def lm_to_px_dict(res_lm, w, h):
    d = {}
    if not res_lm: return d
    for p in PL:
        lm = res_lm.landmark[p.value]
        d[p.name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return d

def spinepose_infer_any(est, img_bgr, bboxes=None):
    if est is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        out = est(img_rgb, bboxes) if bboxes is not None else est(img_rgb)
    except Exception as e:
        print(f"SpinePose inference error: {e}")
        return [], []
    kpts_xy, scores = None, None
    try:
        if isinstance(out, dict):
            kpts_xy = out.get("keypoints") or out.get("kpts_xy")
            scores = out.get("scores")
        elif isinstance(out, (list, tuple)):
            if out and isinstance(out[0], np.ndarray): kpts_xy = out[0]
            if len(out) > 1 and isinstance(out[1], np.ndarray): scores = out[1]
        elif hasattr(out, "shape"):
            kpts_xy = out
        if kpts_xy is None:
            return [], []
        kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
        if kpts_xy.size == 0: return [], []
        if scores is None:
            scores = np.ones((len(kpts_xy),), dtype=np.float32)
        else:
            scores = np.asarray(scores, dtype=np.float32).reshape(-1)
            if scores.shape[0] != kpts_xy.shape[0]:
                scores = np.ones((len(kpts_xy),), dtype=np.float32)
        return kpts_xy, scores
    except Exception as e:
        print(f"SpinePose output processing error: {e}")
        return [], []

class SpineTracker:
    def __init__(self, history_size=3):
        self.history = deque(maxlen=history_size)
    def add_detection(self, spine_map):
        self.history.append(spine_map.copy())
    def get_smoothed_spine_map(self, current_spine_map):
        if len(self.history) < 2:
            return current_spine_map
        smoothed_map = {}
        for name in current_spine_map.keys():
            coords_history, scores_history = [], []
            for hist_map in self.history:
                if name in hist_map:
                    coords_history.append(hist_map[name]['coord'])
                    scores_history.append(hist_map[name]['score'])
            if len(coords_history) >= 2:
                weights = np.linspace(0.2, 1.0, len(coords_history))
                weights /= weights.sum()
                avg_x = np.average([c[0] for c in coords_history], weights=weights)
                avg_y = np.average([c[1] for c in coords_history], weights=weights)
                avg_score = np.average(scores_history, weights=weights)
                smoothed_map[name] = {
                    'coord': (avg_x, avg_y),
                    'score': avg_score,
                    'index': current_spine_map[name]['index']
                }
            else:
                smoothed_map[name] = current_spine_map[name]
        return smoothed_map

spine_tracker = SpineTracker(history_size=3)

def get_spine_keypoint_indices():
    return list(range(24, 33))

def filter_spine_keypoints(all_kpts, all_scores, spine_only=False):
    if not spine_only or not all_kpts:
        return all_kpts, all_scores
    return all_kpts, all_scores

def detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th=0.1):
    if not sp_kpts or len(sp_kpts) < 3:
        return {}
    valid_points = []
    for i, (x, y) in enumerate(sp_kpts):
        if i < len(sp_scores) and sp_scores[i] >= score_th:
            valid_points.append((i, x, y, sp_scores[i]))
    if len(valid_points) < 3:
        return {}
    valid_points.sort(key=lambda p: p[2])
    center_x = np.median([p[1] for p in valid_points])
    spine_candidates, tolerance = [], 45
    for idx, x, y, score in valid_points:
        if abs(x - center_x) <= tolerance:
            spine_candidates.append((idx, x, y, score))
    if len(spine_candidates) < 3:
        return {}
    spine_map = {}
    spine_candidates.sort(key=lambda p: p[2])
    if len(spine_candidates) >= 6:
        spine_names = ["C7", "T3", "T8", "T12", "L1", "L5"]
        indices = np.linspace(0, len(spine_candidates)-1, len(spine_names), dtype=int)
        for i, name in enumerate(spine_names):
            idx = indices[i]
            candidate = spine_candidates[idx]
            spine_map[name] = {'index': candidate[0], 'coord': (candidate[1], candidate[2]), 'score': candidate[3]}
    else:
        key_names = ["C7", "T8", "L5"]
        spine_candidates_scored = sorted(spine_candidates, key=lambda p: p[3], reverse=True)
        for i, name in enumerate(key_names[:len(spine_candidates_scored)]):
            candidate = spine_candidates_scored[i]
            spine_map[name] = {'index': candidate[0], 'coord': (candidate[1], candidate[2]), 'score': candidate[3]}
    return spine_map

def extract_spine_keypoints(sp_kpts, sp_scores, score_th=0.1):
    spine_map = detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th)
    if spine_map:
        spine_tracker.add_detection(spine_map)
        spine_map = spine_tracker.get_smoothed_spine_map(spine_map)
    return spine_map

def compute_spine_cva(spine_map):
    cervical_pairs = [('C7', 'T8'), ('C7', 'T3')]
    for top, bottom in cervical_pairs:
        if top in spine_map and bottom in spine_map:
            tx, ty = spine_map[top]['coord']
            bx, by = spine_map[bottom]['coord']
            dx, dy = tx - bx, ty - by
            if abs(dy) > 10:
                angle = abs(np.degrees(np.arctan2(dx, dy)))
                return min(angle, 180 - angle)
    return None

def make_side_roi_from_mp(lm_px, w, h, margin=0.10, square_pad=True, pad_ratio=0.10):
    """MediaPipe 결과로 ROI 생성 + (옵션) 정사각 패딩"""
    def get(name):
        v = lm_px.get(name)
        return v if (v and v[2] > 0.4) else None

    sh = [get("RIGHT_SHOULDER"), get("LEFT_SHOULDER")]
    sh = [p for p in sh if p]
    if not sh:
        return (0, 0, w, h)

    sx = sum(p[0] for p in sh)/len(sh)
    sy = sum(p[1] for p in sh)/len(sh)

    hips = [get("RIGHT_HIP"), get("LEFT_HIP")]
    hips = [p for p in hips if p]
    if hips:
        hy = sum(p[1] for p in hips)/len(hips)
        torso_h = abs(hy - sy)
    else:
        torso_h = 120  # fallback

    cx, cy = sx, sy + 0.25*torso_h
    H = torso_h * 2.2
    W = H * 0.8
    W *= (1+margin); H *= (1+margin)

    if square_pad:
        side = max(W, H) * (1.0 + pad_ratio)
        W, H = side, side

    x1 = int(max(0, cx - W/2)); y1 = int(max(0, cy - H/2))
    x2 = int(min(w-1, cx + W/2)); y2 = int(min(h-1, cy + H/2))
    return (x1, y1, x2, y2)

def calculate_forward_head_posture(spine_coords):
    if len(spine_coords) < 3:
        return None
    try:
        neck_idx = len(spine_coords) // 4
        shoulder_idx = len(spine_coords) // 2
        if neck_idx >= len(spine_coords) or shoulder_idx >= len(spine_coords):
            return None
        neck_point = spine_coords[neck_idx]
        shoulder_point = spine_coords[shoulder_idx]
        forward_distance = neck_point[0] - shoulder_point[0]
        vertical_distance = abs(neck_point[1] - shoulder_point[1])
        if vertical_distance > 20:
            forward_angle = np.degrees(np.arctan(abs(forward_distance) / vertical_distance))
            return forward_angle
        return None
    except:
        return None

def calculate_spinal_curvature(spine_coords):
    if len(spine_coords) < 4:
        return None
    try:
        upper_idx = len(spine_coords) // 4
        middle_idx = len(spine_coords) // 2
        lower_idx = 3 * len(spine_coords) // 4
        if lower_idx >= len(spine_coords):
            return None
        upper = spine_coords[upper_idx]
        middle = spine_coords[middle_idx]
        lower = spine_coords[lower_idx]
        dx1, dy1 = middle[0] - upper[0], middle[1] - upper[1]
        dx2, dy2 = lower[0] - middle[0], lower[1] - middle[1]
        if abs(dy1) > 10 and abs(dy2) > 10:
            angle1 = np.degrees(np.arctan2(dx1, dy1))
            angle2 = np.degrees(np.arctan2(dx2, dy2))
            curvature = abs(angle2 - angle1)
            return min(curvature, 180 - curvature)
        return None
    except:
        return None

def visualize_spine_analysis(img, sp_kpts, sp_scores, spine_only=None):
    """키포인트/선을 그리고, 지표 텍스트는 그리지 않음(고정 크기 표시는 콜백에서)"""
    global spine_only_mode
    if spine_only is None:
        spine_only = spine_only_mode
    if not sp_kpts or len(sp_kpts) == 0:
        return img

    spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]
    if spine_only:
        spine_coords = []
        for i in spine_indices:
            if i < len(sp_kpts) and i < len(sp_scores):
                kpt = sp_kpts[i]
                score = sp_scores[i]
                if score > 0.2:
                    x, y = int(kpt[0]), int(kpt[1])
                    cv2.circle(img, (x, y), 6, (255, 255, 0), -1)
                    cv2.circle(img, (x, y), 8, (0, 0, 255), 2)
                    spine_coords.append((x, y))
        if len(spine_coords) >= 2:
            spine_coords.sort(key=lambda p: p[1])
            cv2.polylines(img, [np.array(spine_coords, np.int32)], False, (0, 255, 255), 3)
    else:
        for i, (kpt, score) in enumerate(zip(sp_kpts, sp_scores)):
            if score > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img, (x, y), 3, (255, 255, 0), -1)
    return img

# ───────────────────────────────────────────────────────────────────
# 프레임 콜백: 항상 최신 프레임만 큐에 보관(덮어쓰기)
# ───────────────────────────────────────────────────────────────────
async def process_frame_callback(img):
    global last_results, spine_only_mode, last_good_results, last_update_ms

    h, w = img.shape[:2]

    # 입력 큐 최신 프레임 1장만 유지 (덮어쓰기)
    try:
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put_nowait((img.copy(), w, h))
    except Exception as e:
        print(f"Frame queue error: {e}")

    # 결과 큐: 최신만 반영
    try:
        while not result_queue.empty():
            last_results = result_queue.get_nowait()
    except queue.Empty:
        pass

    # ── sticky 적용 ──
    now_ms = time.perf_counter() * 1000.0
    use_results = last_results
    has_valid = bool(last_results.get("sp_kpts")) and len(last_results["sp_kpts"]) >= 3
    if not has_valid and (now_ms - last_update_ms) <= STICKY_MS:
        use_results = last_good_results
        has_valid = bool(use_results.get("sp_kpts")) and len(use_results["sp_kpts"]) >= 3

    if has_valid:
        try:
            img = visualize_spine_analysis(img, use_results["sp_kpts"], use_results["sp_scores"])
        except Exception as e:
            cv2.putText(img, f"Analysis error: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        kpt_count = len(use_results.get("sp_kpts") or [])
        mode_text = "[SPINE-ONLY]" if spine_only_mode else "[ALL KEYPOINTS]"
        cv2.putText(img, f"SpinePose: {kpt_count} points (need ≥3) {mode_text}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # ---------- 고정 크기 HUD: resize 이후 disp에 표기 ----------
    disp = cv2.resize(img, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)

    # FPS/latency 계산
    def avg_ms(hist):
        return (np.mean(hist) * 1000.0) if hist else 0.0
    bp_ms  = avg_ms(blazepose_latency_hist)
    sp_ms  = avg_ms(spinepose_latency_hist)
    e2e_ms = avg_ms(end_to_end_latency_hist)

    fps_bp  = (1000.0 / bp_ms)  if bp_ms  > 0 else 0.0
    fps_sp  = (1000.0 / sp_ms)  if sp_ms  > 0 else 0.0
    fps_e2e = (1000.0 / e2e_ms) if e2e_ms > 0 else 0.0

    # 수신/디코드→표시 지연(ms) (server.py가 제공)
    now = time.perf_counter()
    rx_ms  = (now - getattr(server, 'last_recv_ts', 0.0))   * 1000.0
    dec_ms = (now - getattr(server, 'last_decode_ts', 0.0)) * 1000.0

    # 통일된 텍스트 스타일
    font_scale = 0.5
    thickness = 1
    line_height = 25
    
    # 성능 지표들 (작은 크기로 통일)
    cv2.putText(disp, f"BlazePose: {fps_bp:.1f} FPS / {bp_ms:.1f} ms", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
    cv2.putText(disp, f"SpinePose: {fps_sp:.1f} FPS / {sp_ms:.1f} ms", (10, 90 + line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
    cv2.putText(disp, f"End-to-End: {fps_e2e:.1f} FPS / {e2e_ms:.1f} ms", (10, 90 + 2*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,255), thickness, cv2.LINE_AA)
    cv2.putText(disp, f"Net->Disp: {rx_ms:5.1f} ms  Dec->Disp: {dec_ms:5.1f} ms", (10, 90 + 3*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,255,0), thickness, cv2.LINE_AA)

    # 모드 표시 - 작고 일관된 크기
    mode_text = "SPINE-ONLY" if spine_only_mode else "ALL KEYPOINTS"
    cv2.putText(disp, f"Mode: {mode_text}", (10, 90 + 4*line_height),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Spine 분석 지표들 (spine_only 모드일 때만, 작은 크기로)
    if spine_only_mode and has_valid:
        try:
            spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]
            spine_coords = []
            for i in spine_indices:
                if i < len(use_results["sp_kpts"]) and i < len(use_results["sp_scores"]):
                    kpt = use_results["sp_kpts"][i]
                    score = use_results["sp_scores"][i]
                    if score > 0.2:
                        spine_coords.append((int(kpt[0]), int(kpt[1])))
            
            if len(spine_coords) >= 2:
                spine_coords.sort(key=lambda p: p[1])
                
                forward_head = calculate_forward_head_posture(spine_coords)
                if forward_head is not None:
                    color = (0, 0, 255) if forward_head > 20 else (0, 255, 0)
                    cv2.putText(disp, f"Forward Head: {forward_head:.1f}deg", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
                
                spinal_curve = calculate_spinal_curvature(spine_coords)
                if spinal_curve is not None:
                    color = (0, 0, 255) if spinal_curve > 25 else (0, 255, 0)
                    cv2.putText(disp, f"Spinal Curve: {spinal_curve:.1f}deg", (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        except Exception:
            pass

    cv2.imshow("SpinePose Analysis (Optimized)", disp)
    # -------------------------------------------------------------

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        try: frame_queue.put_nowait(None)
        except: pass
        print("프로그램 종료")
        exit(0)
    elif k == ord('s'):
        spine_only_mode = not spine_only_mode
        print(f"모드 변경: {'SPINE-ONLY' if spine_only_mode else 'ALL KEYPOINTS'}")

    return img

# ───────────────────────────────────────────────────────────────────
# 추론 워커(별도 스레드)
# ───────────────────────────────────────────────────────────────────
_last_roi = None  # 안정화된 최종 ROI 전역 상태

def _smooth_roi(prev, new, alpha=0.7, max_scale_step=0.10, frame_w=None, frame_h=None):
    if new is None:
        return prev
    if prev is None:
        x1, y1, x2, y2 = new
        if frame_w is not None and frame_h is not None:
            x1 = max(0, min(x1, frame_w-2)); y1 = max(0, min(y1, frame_h-2))
            x2 = max(x1+1, min(x2, frame_w-1)); y2 = max(y1+1, min(y2, frame_h-1))
        return (x1, y1, x2, y2)

    px1, py1, px2, py2 = prev
    nx1, ny1, nx2, ny2 = new
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    nw, nh = max(1, nx2 - nx1), max(1, ny2 - ny1)

    def clamp_len(new_len, prev_len):
        up = prev_len * (1.0 + max_scale_step)
        dn = prev_len * (1.0 - max_scale_step)
        return max(min(new_len, up), dn)
    cw = clamp_len(nw, pw)
    ch = clamp_len(nh, ph)

    pcx, pcy = px1 + pw/2.0, py1 + ph/2.0
    ncx, ncy = nx1 + nw/2.0, ny1 + nh/2.0
    cx = alpha*pcx + (1.0-alpha)*ncx
    cy = alpha*pcy + (1.0-alpha)*ncy

    x1 = int(round(cx - cw/2.0)); y1 = int(round(cy - ch/2.0))
    x2 = int(round(cx + cw/2.0)); y2 = int(round(cy + ch/2.0))

    if frame_w is not None and frame_h is not None:
        x1 = max(0, min(x1, frame_w-2)); y1 = max(0, min(y1, frame_h-2))
        x2 = max(x1+1, min(x2, frame_w-1)); y2 = max(y1+1, min(y2, frame_h-1))
    return (x1, y1, x2, y2)

TARGET_PERIOD_MS = 100  # main()에서 환경 기반으로 갱신

def inference_worker():
    global blazepose_latency_hist, spinepose_latency_hist, end_to_end_latency_hist
    global _last_roi, last_good_results, last_update_ms
    next_spine_ts = 0.0  # ms

    while True:
        try:
            start_time = time.perf_counter()
            frame_data = frame_queue.get(timeout=1.0)
            if frame_data is None:
                break
            img, w, h = frame_data

            # BlazePose (매 프레임)
            bp_t0 = time.perf_counter()
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            bp_t1 = time.perf_counter()
            blazepose_latency_hist.append(bp_t1 - bp_t0)

            # 원시 ROI 계산
            if res.pose_landmarks:
                lm_px = lm_to_px_dict(res.pose_landmarks, w, h)
                raw_roi = make_side_roi_from_mp(lm_px, w, h, margin=0.10, square_pad=True, pad_ratio=0.10)
            else:
                raw_roi = _last_roi if _last_roi is not None else (int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))

            # ROI 안정화
            _last_roi = _smooth_roi(_last_roi, raw_roi, alpha=0.7, max_scale_step=0.10, frame_w=w, frame_h=h)
            x1, y1, x2, y2 = _last_roi

            # SpinePose (시간 간격)
            now_ms = time.perf_counter() * 1000.0
            sp_kpts, sp_scores = [], []
            if now_ms >= next_spine_ts:
                next_spine_ts = now_ms + TARGET_PERIOD_MS
                sp_t0 = time.perf_counter()
                bbox = [[x1, y1, x2, y2]]
                sp_kpts, sp_scores = spinepose_infer_any(spine_est, img, bboxes=bbox)
                sp_t1 = time.perf_counter()
                spinepose_latency_hist.append(sp_t1 - sp_t0)

            end_time = time.perf_counter()
            end_to_end_latency_hist.append(end_time - start_time)

            # 최신 결과만 유지
            result = {
                "sp_kpts": [(int(x), int(y)) for x, y in sp_kpts] if sp_kpts is not None else [],
                "sp_scores": sp_scores if sp_scores is not None else [],
                "roi": _last_roi
            }
            try:
                if result_queue.full():
                    result_queue.get_nowait()
                result_queue.put_nowait(result)
            except queue.Full:
                pass

            # 스티키 캐시 갱신(유효 결과일 때)
            try:
                if result["sp_kpts"] and len(result["sp_kpts"]) >= 3:
                    last_good_results.update(result)
                    last_update_ms = time.perf_counter() * 1000.0
            except Exception:
                pass

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Inference worker error: {e}")

# ───────────────────────────────────────────────────────────────────
# 글로벌 상태
# ───────────────────────────────────────────────────────────────────
last_results = {"sp_kpts": [], "sp_scores": [], "roi": None}
spine_only_mode = True

inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

# ───────────────────────────────────────────────────────────────────
# 메인
# ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="SpinePose Analysis WebRTC Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--model_size", default="small", choices=["small", "medium", "large", "xlarge"],
                        help="SpinePose model size")
    parser.add_argument("--spine-only", action="store_true",
                        help="Start with spine-only mode (toggle with 's' key)")
    parser.add_argument("--period_ms", type=int, default=None,
                        help="(옵션) SpinePose 호출 간격(ms). 미지정 시 자동 추천값 사용.")
    args = parser.parse_args()

    global spine_only_mode, spine_est, TARGET_PERIOD_MS
    spine_only_mode = args.spine_only

    # 환경 감지 & 스레드/주기 자동 최적화
    env = detect_cpu_env()
    tuned_threads = apply_thread_tuning(env)
    auto_period = recommend_period_ms(env)
    TARGET_PERIOD_MS = args.period_ms if args.period_ms is not None else auto_period

    print(f"===== SpinePose Analysis System (환경 적응형 최적화) =====")
    print(f"CPU: {env['cpu_name']} | physical={env['physical']} logical={env['logical']} usable={env['usable']}")
    print(f"스레드 설정: OpenCV=1, OMP류={tuned_threads}  | SpinePose 주기: {TARGET_PERIOD_MS} ms")
    print(f"Model: {args.model_size} | BlazePose(매 프레임 ROI) + SpinePose(시간 간격)")
    print(f"ROI 마진: 0.10 (정사각 패딩)")
    print(f"서버 시작: http://{args.host}:{args.port}")
    print(f"[Controls] q: 종료 | s: spine-only 토글")
    print(f"spine_only 모드: {'활성화' if spine_only_mode else '비활성화'}")

    try:
        spine_est = SpinePoseEstimator(mode=args.model_size, device="cpu")
        print(f"✓ SpinePose mode={args.model_size} 모델 로드 완료")
    except Exception as e:
        print(f"⚠ SpinePose 로드 실패: {e}")
        print("기본 모델로 재시도...")
        try:
            spine_est = SpinePoseEstimator(device="cpu")
            print("✓ SpinePose 기본 모델 로드 완료")
        except Exception as e2:
            print(f"✗ SpinePose 완전 실패: {e2}")
            return

    cv2.namedWindow("SpinePose Analysis (Optimized)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SpinePose Analysis (Optimized)", WIN_W, WIN_H)

    server.set_frame_callback(process_frame_callback)

    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    finally:
        try:
            frame_queue.put_nowait(None)
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()