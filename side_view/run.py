# --- auto (re)exec under local .venv ---
import os, sys
import subprocess
import shutil
from math import degrees, acos
import math


# 현재 파일의 절대 경로
HERE = os.path.dirname(os.path.abspath(__file__))
# 가상 환경의 Python 실행 파일 경로
VENV_PYTHON = os.path.join(HERE, ".venv", "Scripts", "python.exe") if os.name == "nt" else os.path.join(HERE, ".venv", "bin", "python")

# 현재 실행 중인 Python이 가상 환경의 Python이 아니면서,
# 가상 환경의 Python 실행 파일이 존재할 경우
if os.path.exists(VENV_PYTHON) and os.path.normcase(sys.executable) != os.path.normcase(VENV_PYTHON):
    print("Not in venv. Restarting with venv Python...")
    
    args = [VENV_PYTHON] + sys.argv
    
    try:
        subprocess.run(args, check=True, cwd=HERE)
        sys.exit(0)
    except FileNotFoundError:
        print(f"Error: Venv Python executable not found at {VENV_PYTHON}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error restarting with venv: {e}", file=sys.stderr)
        sys.exit(e.returncode)
# ---------------------------------------

import cv2, json, asyncio, logging, ssl, pathlib, socket, re, argparse, glob, contextlib
import numpy as np
import mediapipe as mp
from collections import deque
from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
from spinepose import SpinePoseEstimator

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

logging.basicConfig(level=logging.INFO)

# MediaPipe 초기화 및 SpinePose 초기화
spine_est = SpinePoseEstimator(device="cpu")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
PL = mp_pose.PoseLandmark

# ========= Utils: IP 탐지 & 인증서 탐색 =========
IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

def get_lan_ip() -> str:
    """현재 PC의 사설 LAN IP를 추정합니다. 실패 시 '127.0.0.1' 반환."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def find_cert_pair_for_ip(ip: str):
    """인증서 찾기"""
    ov_cert = os.environ.get("OVERRIDE_CERT")
    ov_key  = os.environ.get("OVERRIDE_KEY")
    if ov_cert and ov_key and pathlib.Path(ov_cert).exists() and pathlib.Path(ov_key).exists():
        return pathlib.Path(ov_cert), pathlib.Path(ov_key)

    # 정확 일치
    c = pathlib.Path(f"{ip}.pem"); k = pathlib.Path(f"{ip}-key.pem")
    if c.exists() and k.exists(): return c, k

    # IP+N 패턴
    for cp in glob.glob(f"{ip}+*.pem"):
        kp = pathlib.Path(cp.replace(".pem", "-key.pem"))
        if pathlib.Path(cp).exists() and kp.exists():
            return pathlib.Path(cp), kp

    # localhost/127.0.0.1 및 +N 패턴
    for base in ("127.0.0.1", "localhost"):
        c = pathlib.Path(f"{base}.pem"); k = pathlib.Path(f"{base}-key.pem")
        if c.exists() and k.exists(): return c, k
        for cp in glob.glob(f"{base}+*.pem"):
            kp = pathlib.Path(cp.replace(".pem", "-key.pem"))
            if pathlib.Path(cp).exists() and kp.exists():
                return pathlib.Path(cp), kp

    return None, None

# 유틸 함수들
def lm_to_px_dict(res_lm, w, h):
    d = {}
    if not res_lm: return d
    for p in PL:
        lm = res_lm.landmark[p.value]
        d[p.name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return d

def spinepose_infer_any(est, img_bgr, bboxes=None):
    """SpinePoseEstimator의 다양한 API를 시도해 (kpts_xy[(x,y)...], scores[...]) 반환."""
    if est is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _has(name): 
        return hasattr(est, name) and callable(getattr(est, name))

    candidates = []
    if callable(est):
        candidates.append(("__call__", True))
        candidates.append(("__call__", False))
    
    for name in ("predict", "run", "estimate", "detect", "predict_kpts", "forward"):
        if _has(name):
            candidates.append((name, True))
            candidates.append((name, False))

    last_err = None
    for name, try_bboxes in candidates:
        fn = est if name == "__call__" else getattr(est, name)
        try:
            if try_bboxes and bboxes is not None:
                out = fn(img_rgb, bboxes)
            else:
                out = fn(img_rgb)
        except TypeError as e:
            last_err = e
            continue
        except Exception as e:
            last_err = e
            continue

        # 출력 정규화
        kpts_xy, scores = None, None
        try:
            if isinstance(out, dict):
                kpts_xy = out.get("keypoints") or out.get("kpts_xy")
                scores  = out.get("scores")
            elif isinstance(out, (list, tuple)):
                if out and isinstance(out[0], np.ndarray):
                    kpts_xy = out[0]
                if len(out) > 1 and isinstance(out[1], np.ndarray):
                    scores = out[1]
            elif hasattr(out, "shape"):
                kpts_xy = out

            if kpts_xy is None:
                continue

            kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
            if kpts_xy.size == 0:
                continue

            if scores is None:
                scores = np.ones((len(kpts_xy),), dtype=np.float32)
            else:
                scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                if scores.shape[0] != kpts_xy.shape[0]:
                    scores = np.ones((len(kpts_xy),), dtype=np.float32)

            return kpts_xy, scores

        except Exception as e:
            last_err = e
            continue

    raise AttributeError(f"SpinePoseEstimator 호출 실패. 마지막 에러: {last_err}")


class SpineTracker:
    def __init__(self, history_size=5):
        self.history = deque(maxlen=history_size)
        self.prev_spine_map = {}
        
    def add_detection(self, spine_map):
        """새로운 탐지 결과를 히스토리에 추가"""
        self.history.append(spine_map.copy())
        
    def get_smoothed_spine_map(self, current_spine_map):
        """시간적 스무딩을 적용한 척추 맵 반환"""
        if len(self.history) < 2:
            return current_spine_map
            
        smoothed_map = {}
        
        for name in current_spine_map.keys():
            coords_history = []
            scores_history = []
            
            # 히스토리에서 해당 키포인트 수집
            for hist_map in self.history:
                if name in hist_map:
                    coords_history.append(hist_map[name]['coord'])
                    scores_history.append(hist_map[name]['score'])
            
            if len(coords_history) >= 2:
                # 가중 평균 (최근 프레임에 더 높은 가중치)
                weights = np.linspace(0.1, 1.0, len(coords_history))
                weights = weights / weights.sum()
                
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

# 전역 트래커 인스턴스
spine_tracker = SpineTracker(history_size=5)


# ===== 표시/파라미터 =====
WIN_W, WIN_H = 1280, 720
SHOW_ROI_BOX = False
SPINE_SCORE_TH = 0.15  # 점수 임계값 낮춤

# ============= 개선된 SpinePose 척추 분석 시스템 =============

# 전역 트래커 인스턴스
spine_tracker = SpineTracker(history_size=5)

def detect_outliers(points, threshold=2.0):
    """이상값 탐지 (Z-score 기반)"""
    if len(points) < 3:
        return [False] * len(points)
    
    coords = np.array([(p[1], p[2]) for p in points])  # x, y 좌표만
    mean = np.mean(coords, axis=0)
    std = np.std(coords, axis=0)
    
    outliers = []
    for i, (idx, x, y, score) in enumerate(points):
        z_score_x = abs(x - mean[0]) / (std[0] + 1e-6)
        z_score_y = abs(y - mean[1]) / (std[1] + 1e-6)
        is_outlier = max(z_score_x, z_score_y) > threshold
        outliers.append(is_outlier)
    
    return outliers


# ===== 1단계: 동적 키포인트 탐지 =====
def detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th=0.15):
    """개선된 동적 척추 키포인트 탐지"""
    
    if not sp_kpts or len(sp_kpts) < 3:
        return {}
    
    # 1단계: 기본 필터링 (점수 기반)
    valid_points = []
    for i, (x, y) in enumerate(sp_kpts):
        if i < len(sp_scores) and sp_scores[i] >= score_th:
            valid_points.append((i, x, y, sp_scores[i]))
    
    if len(valid_points) < 3:
        return {}
    
    # 2단계: 이상값 제거
    outliers = detect_outliers(valid_points, threshold=2.5)
    filtered_points = [p for p, is_outlier in zip(valid_points, outliers) if not is_outlier]
    
    if len(filtered_points) < 3:
        filtered_points = valid_points  # 이상값 제거로 너무 적어지면 원본 사용
    
    # 3단계: Y좌표 기준 정렬
    filtered_points.sort(key=lambda p: p[2])
    
    # 4단계: 개선된 중앙선 체크 (적응적 임계값)
    center_x = np.median([p[1] for p in filtered_points])
    x_spread = np.std([p[1] for p in filtered_points])
    adaptive_threshold = max(25, min(50, x_spread * 1.5))  # 25-50픽셀 범위
    
    spine_candidates = []
    for idx, x, y, score in filtered_points:
        if abs(x - center_x) <= adaptive_threshold:
            spine_candidates.append((idx, x, y, score))
    
    if len(spine_candidates) < 3:
        # 임계값을 완화하여 재시도
        for idx, x, y, score in filtered_points:
            if abs(x - center_x) <= 60:  # 더 관대한 기준
                spine_candidates.append((idx, x, y, score))
    
    if len(spine_candidates) < 3:
        return {}
    
    # 5단계: 개선된 연속성 체크 (적응적)
    spine_candidates.sort(key=lambda p: p[2])  # Y좌표 재정렬
    filtered_candidates = []
    
    # 평균 Y 간격 계산
    if len(spine_candidates) > 1:
        y_distances = []
        for i in range(len(spine_candidates) - 1):
            y_diff = abs(spine_candidates[i+1][2] - spine_candidates[i][2])
            y_distances.append(y_diff)
        avg_y_gap = np.median(y_distances) if y_distances else 100
        max_gap = max(150, avg_y_gap * 2.5)  # 적응적 최대 간격
    else:
        max_gap = 150
    
    prev_y = None
    for idx, x, y, score in spine_candidates:
        if prev_y is None:
            filtered_candidates.append((idx, x, y, score))
            prev_y = y
        else:
            y_diff = abs(y - prev_y)
            if y_diff <= max_gap:
                filtered_candidates.append((idx, x, y, score))
                prev_y = y
    
    spine_candidates = filtered_candidates
    
    if len(spine_candidates) < 3:
        return {}
    
    # 6단계: 점수 기반 가중치를 고려한 키포인트 할당
    spine_map = {}
    
    # 점수 기반 정렬 (높은 점수 우선)
    spine_candidates_scored = sorted(spine_candidates, key=lambda p: p[3], reverse=True)
    spine_candidates_y_sorted = sorted(spine_candidates, key=lambda p: p[2])  # Y좌표 순서
    
    if len(spine_candidates_y_sorted) >= 9:
        spine_names = ["C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "S1"]
        indices = np.linspace(0, len(spine_candidates_y_sorted)-1, len(spine_names), dtype=int)
        for i, name in enumerate(spine_names):
            idx = indices[i]
            candidate = spine_candidates_y_sorted[idx]
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    elif len(spine_candidates_y_sorted) >= 6:
        spine_names = ["C1", "C7", "T3", "T8", "L1", "L5"]
        indices = np.linspace(0, len(spine_candidates_y_sorted)-1, len(spine_names), dtype=int)
        for i, name in enumerate(spine_names):
            idx = indices[i]
            candidate = spine_candidates_y_sorted[idx]
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    else:
        # 적은 수면 기본 키포인트만 (높은 점수 우선)
        key_names = ["C7", "T8", "L5"]
        for i, name in enumerate(key_names[:len(spine_candidates_scored)]):
            candidate = spine_candidates_scored[i]
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    
    return spine_map

# 개선된 하드코딩 매핑 (경추 인덱스 조정)
HARDCODED_SPINE_INDICES = {
    "C1": [17, 16, 18],   # 여러 후보 인덱스 (메인, 보조1, 보조2)
    "C4": [36, 35, 37],   
    "C7": [0, 1, 2],      
    "T3": [35, 34, 36],   
    "T8": [30, 29, 31],   
    "L1": [27, 26, 28],   
    "L3": [28, 27, 29],   
    "L5": [12, 11, 13],   
    "S1": [26, 25, 24],   
}

def get_spine_keypoints_hardcoded(sp_kpts, sp_scores, score_th=0.12):
    """개선된 하드코딩 척추 키포인트 추출"""
    spine_map = {}
    missing_indices = []
    
    for name, idx_list in HARDCODED_SPINE_INDICES.items():
        best_idx = None
        best_score = 0
        
        # 여러 후보 중 가장 높은 점수 선택
        for idx in idx_list:
            if idx < len(sp_kpts) and idx < len(sp_scores):
                if sp_scores[idx] >= score_th and sp_scores[idx] > best_score:
                    best_idx = idx
                    best_score = sp_scores[idx]
        
        if best_idx is not None:
            spine_map[name] = {
                'index': best_idx,
                'coord': sp_kpts[best_idx],
                'score': best_score
            }
        else:
            available_scores = [f"idx{idx}:{sp_scores[idx]:.3f}" for idx in idx_list 
                              if idx < len(sp_scores)]
            missing_indices.append(f"{name}({','.join(available_scores)})")
    
    # 디버깅 출력 (경추 문제 진단)
    if len(spine_map) < 7:  # 7개 미만일 때
        cervical_found = sum(1 for name in ['C1', 'C4', 'C7'] if name in spine_map)
        print(f"[SPINE] Cervical points: {cervical_found}/3, Total: {len(spine_map)}/9")
        if missing_indices:
            print(f"[SPINE] Missing: {', '.join(missing_indices[:3])}...")  # 처음 3개만 출력
    
    return spine_map

def extract_spine_keypoints(sp_kpts, sp_scores, method='dynamic', score_th=0.15):
    """개선된 척추 키포인트 통합 추출기"""
    
    # 동적 탐지 시도
    if method == 'dynamic':
        spine_map = detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th)
        
        # 동적 탐지 실패 시 하드코딩 방법으로 보완
        if len(spine_map) < 4:
            hardcoded_map = get_spine_keypoints_hardcoded(sp_kpts, sp_scores, score_th * 0.8)
            
            # 두 방법 결합: 동적 방법 우선, 빠진 부분은 하드코딩으로 보완
            for name, data in hardcoded_map.items():
                if name not in spine_map:
                    spine_map[name] = data
    else:
        spine_map = get_spine_keypoints_hardcoded(sp_kpts, sp_scores, score_th)
    
    # 시간적 스무딩 적용
    if spine_map:
        spine_tracker.add_detection(spine_map)
        spine_map = spine_tracker.get_smoothed_spine_map(spine_map)
    
    return spine_map

# ===== 개선된 자세 분석 함수들 =====

def compute_spine_lumbar_lordosis(spine_map):
    """L1-L3-L5 기반 요추 전만곡 계산"""
    required = ['L1', 'L3', 'L5']
    if not all(name in spine_map for name in required):
        return None, None, None
    
    L1_xy = spine_map['L1']['coord']
    L3_xy = spine_map['L3']['coord'] 
    L5_xy = spine_map['L5']['coord']
    
    xs = np.array([L1_xy[0], L3_xy[0], L5_xy[0]], dtype=np.float64)
    ys = np.array([L1_xy[1], L3_xy[1], L5_xy[1]], dtype=np.float64)
    A = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    
    try:
        a, b, c = np.linalg.lstsq(A, ys, rcond=None)[0]
        
        def slope_at(x): return 2*a*x + b
        def angle_at(x): return abs(np.degrees(np.arctan(slope_at(x))))
        
        th_L1 = angle_at(L1_xy[0])
        th_L5 = angle_at(L5_xy[0])
        lordosis = abs(th_L5 - th_L1)
        
        return lordosis, th_L1, th_L5
    except:
        return None, None, None

def compute_spine_thoracic_kyphosis(spine_map):
    """C7-T8 기반 흉추 후만곡 계산"""
    if 'C7' not in spine_map or 'T8' not in spine_map:
        return None
    
    c7_coord = spine_map['C7']['coord']
    t8_coord = spine_map['T8']['coord']
    
    dx = t8_coord[0] - c7_coord[0] 
    dy = t8_coord[1] - c7_coord[1]
    
    angle = abs(np.degrees(np.arctan2(dx, dy)))
    return min(angle, 180 - angle)

def compute_spine_cva(spine_map):
    """개선된 CVA 계산 (여러 키포인트 활용)"""
    
    # 우선순위: C1-C7, C4-C7, 또는 사용 가능한 경추 조합
    cervical_pairs = [('C1', 'C7'), ('C4', 'C7'), ('C1', 'C4')]
    
    for top, bottom in cervical_pairs:
        if top in spine_map and bottom in spine_map:
            top_coord = spine_map[top]['coord']
            bottom_coord = spine_map[bottom]['coord']
            
            dx = top_coord[0] - bottom_coord[0]
            dy = top_coord[1] - bottom_coord[1]
            
            if abs(dy) > 10:  # 최소 Y 차이 확보
                angle = abs(np.degrees(np.arctan2(dx, dy)))
                return min(angle, 180 - angle)
    
    return None

def compute_spine_cva(spine_map):
    """개선된 CVA 계산 (여러 키포인트 활용)"""
    
    # 우선순위: C1-C7, C4-C7, 또는 사용 가능한 경추 조합
    cervical_pairs = [('C1', 'C7'), ('C4', 'C7'), ('C1', 'C4')]
    
    for top, bottom in cervical_pairs:
        if top in spine_map and bottom in spine_map:
            top_coord = spine_map[top]['coord']
            bottom_coord = spine_map[bottom]['coord']
            
            dx = top_coord[0] - bottom_coord[0]
            dy = top_coord[1] - bottom_coord[1]
            
            if abs(dy) > 10:  # 최소 Y 차이 확보
                angle = abs(np.degrees(np.arctan2(dx, dy)))
                return min(angle, 180 - angle)
    
    return None

def visualize_spine_analysis(img, sp_kpts, sp_scores, method='dynamic', show_indices=False):
    """개선된 척추 시각화"""
    
    spine_map = extract_spine_keypoints(sp_kpts, sp_scores, method)
    
    if not spine_map:
        cv2.putText(img, "No valid spine keypoints detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    
    # 개선된 자세 분석
    cva = compute_spine_cva(spine_map)
    lordosis, th_L1, th_L5 = compute_spine_lumbar_lordosis(spine_map)
    kyphosis = compute_spine_thoracic_kyphosis(spine_map)
    
    # 키포인트 시각화 (신뢰도별 색상)
    spine_coords = []
    spine_names = ["C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "S1"]
    
    for name in spine_names:
        if name in spine_map:
            coord = spine_map[name]['coord']
            score = spine_map[name]['score']
            x, y = int(coord[0]), int(coord[1])
            
            # 신뢰도 기반 색상 (높음:녹색, 중간:노랑, 낮음:빨강)
            if score > 0.7:
                color = (0, 255, 0)  # 녹색
            elif score > 0.4:
                color = (0, 255, 255)  # 노랑
            else:
                color = (0, 100, 255)  # 주황
                
            # 키포인트 그리기
            radius = int(4 + score * 4)  # 신뢰도에 따른 크기
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.circle(img, (x, y), radius + 2, (255, 255, 255), 2)
            
            # 경추 키포인트는 특별 표시
            if name in ['C1', 'C4', 'C7']:
                cv2.circle(img, (x, y), radius + 4, (255, 0, 255), 2)  # 보라색 외곽
            
            if show_indices:
                cv2.putText(img, f"{name}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            spine_coords.append((x, y))
    
    # 척추 연결선
    if len(spine_coords) >= 2:
        cv2.polylines(img, [np.array(spine_coords, np.int32)], 
                     False, (0, 255, 255), 3)
    
    # 결과 텍스트
    y_pos = 30
    if cva is not None:
        color = (0, 0, 255) if cva > 15 else (0, 255, 0)  # 15도 기준으로 조정
        cv2.putText(img, f"CVA: {cva:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y_pos += 35
        
    if lordosis is not None:
        cv2.putText(img, f"Lumbar: {lordosis:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        y_pos += 35
        
    if kyphosis is not None:
        cv2.putText(img, f"Thoracic: {kyphosis:.1f}deg", (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    # 안정성 정보 표시
    cervical_count = sum(1 for name in ['C1', 'C4', 'C7'] if name in spine_map)
    stability_text = f"Cervical: {cervical_count}/3, Total: {len(spine_map)}/9"
    cv2.putText(img, stability_text, (img.shape[1] - 300, img.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img

# ===== ROI 및 SpinePose 실행 함수들 =====
def make_side_roi_from_mp(lm_px, w, h, margin=0.12):
    """BlazePose 픽셀 좌표에서 측면 상체 ROI 박스 반환"""
    def get(name):
        v = lm_px.get(name)
        return v if (v and v[2] > 0.5) else None

    sh = [get("RIGHT_SHOULDER"), get("LEFT_SHOULDER")]
    sh = [p for p in sh if p]
    if not sh:
        return (0, 0, w, h)

    sx = sum(p[0] for p in sh)/len(sh); sy = sum(p[1] for p in sh)/len(sh)

    hips = [get("RIGHT_HIP"), get("LEFT_HIP")]
    hips = [p for p in hips if p]
    if hips:
        hx = sum(p[0] for p in hips)/len(hips); hy = sum(p[1] for p in hips)/len(hips)
        torso_h = abs(hy - sy)
    else:
        ear = get("RIGHT_EAR") or get("LEFT_EAR") or (sx, sy-80, 1.0)
        torso_h = max(120, abs(ear[1]-sy)*1.6)

    cx, cy = sx, sy + 0.25*torso_h
    H = torso_h * 2.3
    W = H * 0.85

    W *= (1+margin); H *= (1+margin)

    x1 = int(max(0, cx - W/2)); y1 = int(max(0, cy - H/2))
    x2 = int(min(w-1, cx + W/2)); y2 = int(min(h-1, cy + H/2))
    return (x1, y1, x2, y2)

def run_spinepose_on_crop(frame_bgr, roi=None, resize=(256, 256)):
    """ROI 크롭→리사이즈→SpinePose→원본 좌표로 역변환"""
    if roi is Ellipsis:
        roi = None

    h, w = frame_bgr.shape[:2]
    if roi is None:
        roi = (0, 0, w, h)

    x1, y1, x2, y2 = roi
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return [], []
    
    rw, rh = resize
    crop_rs = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)

    try:
        crop_w, crop_h = resize
        bbox = [[0, 0, crop_w, crop_h]]
        kpts_crop, scores = spinepose_infer_any(spine_est, crop_rs, bboxes=bbox)
        
        if kpts_crop is not None and len(kpts_crop) > 0:
            kpts_crop = np.asarray(kpts_crop, dtype=np.float32)
            kpts_crop = np.squeeze(kpts_crop)
            
            if kpts_crop.ndim == 1 and len(kpts_crop) == 2:
                kpts_crop = np.expand_dims(kpts_crop, axis=0)
            elif kpts_crop.ndim == 1 and len(kpts_crop) % 2 == 0:
                kpts_crop = kpts_crop.reshape(-1, 2)
            
            if kpts_crop.ndim != 2 or kpts_crop.shape[1] != 2:
                return [], []
            
            if scores is not None:
                scores = np.asarray(scores, dtype=np.float32).reshape(-1)
                if len(scores) != len(kpts_crop):
                    scores = np.ones(len(kpts_crop), dtype=np.float32) * 0.5
            else:
                scores = np.ones(len(kpts_crop), dtype=np.float32) * 0.5
            
            # 원본 좌표로 역변환
            sx = (x2 - x1) / float(rw)
            sy = (y2 - y1) / float(rh)
            kpts_full = [(int(p[0] * sx + x1), int(p[1] * sy + y1)) for p in kpts_crop]
            
            return kpts_full, scores
        else:
            return [], []
            
    except Exception as e:
        return [], []

def get_center_roi(w, h, ratio=0.7):
    """중앙 영역 ROI 생성"""
    center_w = int(w * ratio)
    center_h = int(h * ratio)
    x1 = (w - center_w) // 2
    y1 = (h - center_h) // 2
    return (x1, y1, x1 + center_w, y1 + center_h)

async def stream_video(track, img):
    """OpenCV 윈도우 표시 및 종료 처리"""
    disp = cv2.resize(img, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Preview", disp)
    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.destroyAllWindows()
        raise Exception("Quit")

# ========= HTML 페이지 =========
HTML_PAGE = f"""
<!doctype html><html><head><meta charset="utf-8"><title>SideCam</title></head>
<body style="font-family:sans-serif;background:#111;color:#eee;">
  <h3>Android Side Camera</h3>
  <video id="preview" autoplay playsinline muted style="width: 60vw; transform: scaleX(-1); background:#000;"></video><br/>
  <button id="start">Start Streaming</button>
  <pre id="log" style="white-space:pre-wrap;background:#222;color:#eee;padding:8px;margin-top:8px;max-width:60vw;"></pre>

  <script>
    const W={WIN_W}, H={WIN_H};
    const startBtn = document.getElementById('start');
    const preview  = document.getElementById('preview');
    const $log     = document.getElementById('log');
    function log(...a) {{ $log.textContent += a.join(' ') + '\\n'; console.log(...a); }}

    function waitForIceGatheringComplete(pc) {{
      return new Promise(resolve => {{
        if (!pc) return resolve();
        if (pc.iceGatheringState === 'complete') return resolve();
        const onchg = () => {{
          if (pc.iceGatheringState === 'complete') {{
            pc.removeEventListener('icegatheringstatechange', onchg);
            resolve();
          }}
        }};
        pc.addEventListener('icegatheringstatechange', onchg);
      }});
    }}

    async function start() {{
      startBtn.disabled = true;
      try {{
        log('isSecureContext =', window.isSecureContext);
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
          throw new Error('getUserMedia unsupported');
        }}
        log('getUserMedia...');
        const stream = await navigator.mediaDevices.getUserMedia({{
          video: {{
            facingMode: {{ ideal: "environment" }},
            width:  {{ ideal: W, max: W }},
            height: {{ ideal: H, max: H }},
            frameRate: {{ ideal: 24, max: 24 }}
          }},
          audio: false
        }});
        if (!stream) throw new Error('No stream returned');
        preview.srcObject = stream;

        const tracks = stream.getVideoTracks();
        log('video tracks =', tracks.length);
        const track = tracks[0];
        if (!track) throw new Error('No video track');

        try {{
          await track.applyConstraints({{
            width:  {{ ideal: W, max: W }},
            height: {{ ideal: H, max: H }},
            frameRate: {{ ideal: 24, max: 24 }}
          }});
        }} catch(e) {{ log('applyConstraints error:', e && (e.message || e)); }}
        try {{ track.contentHint = 'motion'; }} catch(e){{}}

        const pc = new RTCPeerConnection({{ iceServers: [{{ urls: ["stun:stun.l.google.com:19302"] }}] }});
        stream.getTracks().forEach(t => pc.addTrack(t, stream));

        pc.oniceconnectionstatechange = () => log('ICE state:', pc.iceConnectionState);
        pc.onconnectionstatechange     = () => log('PC state:',  pc.connectionState);

        pc.onnegotiationneeded = async () => {{
          try {{
            const s = pc.getSenders().find(s => s.track && s.track.kind==='video');
            if (s) {{
              const p = s.getParameters();
              p.encodings = [{{ maxBitrate: 1500000, maxFramerate: 24, scaleResolutionDownBy: 1.0 }}];
              await s.setParameters(p);
              log('Sender params set (maxBitrate=1.5Mbps)');
            }}
          }} catch(e) {{ log('onnegotiationneeded error:', e && (e.message || e)); }}
        }};

        log('createOffer...');
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        log('waiting ICE gathering complete...');
        await waitForIceGatheringComplete(pc);

        log('POST /offer');
        const r = await fetch('/offer', {{
          method:'POST',
          headers:{{'Content-Type':'application/json'}},
          body: JSON.stringify({{ sdp: pc.localDescription.sdp, type: pc.localDescription.type }})
        }});
        log('POST /offer status', r.status);
        if (!r.ok) throw new Error('offer POST failed: '+r.status);

        const answer = await r.json();
        await pc.setRemoteDescription(answer);
        log('setRemoteDescription done');
      }} catch (e) {{
        log('ERROR:', e && (e.stack || e.message || e));
        if (!window.isSecureContext) {{
          log('Hint: Android Chrome는 보통 HTTPS에서만 카메라 허용. HTTPS(예: https://<IP>:PORT)로 접속하거나 Firefox로 테스트하세요.');
        }}
        alert('Error: ' + (e && (e.message || e)));
      }} finally {{
        startBtn.disabled = false;
      }}
    }}

    startBtn.onclick = start;
  </script>
</body></html>
"""

# ========= aiohttp 서버 =========
pcs = set()

async def index(request):
    return web.Response(content_type="text/html", text=HTML_PAGE)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection(); pcs.add(pc)

    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", WIN_W, WIN_H)

    @pc.on("track")
    async def on_track(track):
        if track.kind != "video":
            return

        q = deque(maxlen=1)

        # SpinePose 결과만 저장
        last_results = {
            "sp_kpts": [],
            "sp_scores": [], 
            "roi": None,
        }

        async def reader():
            while True:
                try:
                    frame: VideoFrame = await track.recv()
                except Exception:
                    break
                q.append(frame)

        reader_task = asyncio.create_task(reader())
        frame_count = 0
        
        try:
            while True:
                if not q:
                    await asyncio.sleep(0.001)
                    continue

                frame = q.pop()
                img = frame.to_ndarray(format="bgr24")
                h, w = img.shape[:2]

                # 4프레임마다 SpinePose 추론
                if frame_count % 4 == 0:
                    try:
                        # MediaPipe로 ROI 구하기
                        res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                        if res.pose_landmarks:
                            lm_px = lm_to_px_dict(res.pose_landmarks, w, h)
                            roi = make_side_roi_from_mp(lm_px, w, h, margin=0.20)
                        else:
                            # MediaPipe 실패 시 중앙 ROI 사용
                            roi = get_center_roi(w, h, 0.7)
                            
                        last_results["roi"] = roi

                        # SpinePose 실행 - 직접 전체 이미지에서 테스트
                        full_bbox = [[0, 0, w, h]]
                        test_kpts, test_scores = spinepose_infer_any(spine_est, img, bboxes=full_bbox)
                        
                        if test_kpts is not None and len(test_kpts) > 0:
                            # 전체 이미지 결과 사용
                            last_results["sp_kpts"] = [(int(x), int(y)) for x, y in test_kpts]
                            last_results["sp_scores"] = test_scores if test_scores is not None else [0.5] * len(test_kpts)
                        else:
                            # ROI로 재시도
                            if roi and (roi[2] - roi[0] >= 40) and (roi[3] - roi[1] >= 40):
                                sp_kpts, sp_scores = run_spinepose_on_crop(img, roi, resize=(320, 320))
                                if sp_kpts and sp_scores is not None:
                                    last_results["sp_kpts"] = sp_kpts
                                    last_results["sp_scores"] = sp_scores
                                
                    except Exception:
                        pass

                # 척추 키포인트만 표시하도록 단순화
                if last_results["sp_kpts"] and len(last_results["sp_kpts"]) >= 3:
                    try:
                        # 척추 분석 및 시각화 (인덱스 번호 숨김)
                        img = visualize_spine_analysis(
                            img, 
                            last_results["sp_kpts"], 
                            last_results["sp_scores"], 
                            method='dynamic',
                            show_indices=False  # 인덱스 번호 숨김
                        )
                    except Exception:
                        # 에러 발생 시 기본 정보만 표시
                        cv2.putText(img, "Spine analysis error", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(img, f"SpinePose: {len(last_results['sp_kpts'])} points detected", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    # 키포인트가 부족할 때
                    kpt_count = len(last_results["sp_kpts"]) if last_results["sp_kpts"] else 0
                    cv2.putText(img, f"SpinePose: {kpt_count} points (need ≥3 for analysis)", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                # ROI 박스 표시 (옵션)
                if SHOW_ROI_BOX and last_results["roi"]:
                    x1, y1, x2, y2 = last_results["roi"]
                    if (x2 - x1) >= 40 and (y2 - y1) >= 40:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 128, 255), 2)

                await stream_video(track, img)
                frame_count += 1

        except Exception:
            logging.exception("Error processing video track")
        finally:
            reader_task.cancel()
            cv2.destroyAllWindows()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

async def on_shutdown(app):
    for pc in pcs:
        await pc.close()
    pcs.clear()
    try:
        pose.close()
    except Exception:
        pass

app = web.Application()
app.on_shutdown.append(on_shutdown)
app.router.add_get("/", index)
app.router.add_post("/offer", offer)

if __name__ == "__main__":
    LAN_IP = get_lan_ip()
    CERT_FILE, KEY_FILE = find_cert_pair_for_ip(LAN_IP)

    print("===== HTTPS Only SpinePose 척추 분석 시스템 =====")

    # 인증서가 없으면 에러
    if not (CERT_FILE and KEY_FILE):
        print("ERROR: SSL 인증서를 찾을 수 없습니다!")
        print(f"필요한 파일:")
        print(f"  - {LAN_IP}.pem (또는 localhost.pem)")
        print(f"  - {LAN_IP}-key.pem (또는 localhost-key.pem)")
        print("또는 환경변수로 설정:")
        print("  - OVERRIDE_CERT=인증서파일경로")
        print("  - OVERRIDE_KEY=키파일경로")
        sys.exit(1)

    # HTTPS로만 실행
    try:
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(certfile=str(CERT_FILE), keyfile=str(KEY_FILE))
        
        print("======== Running on HTTPS Only ========")
        print(f"   URL: https://{LAN_IP}:8443")
        print(f"   인증서: {CERT_FILE}")
        print(f"   키파일: {KEY_FILE}")
        print("=======================================", flush=True)
        
        web.run_app(app, host="0.0.0.0", port=8443, ssl_context=ssl_ctx)
        
    except Exception as e:
        print(f"HTTPS 서버 시작 실패: {e}")
        print("인증서 파일을 확인하고 다시 시도해주세요.")
        sys.exit(1)