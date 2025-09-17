def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SpinePose Analysis WebRTC Server (Intel i5 Optimized)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large", "xlarge"],
                       help="SpinePose model size (Intel i5 권장: medium) - 파라미터명: mode")
    parser.add_argument("--spine-only", action="store_true", 
                       help="Start with spine-only mode (toggle with 's' key)")
    args = parser.parse_args()

    global spine_only_mode, spine_est
    spine_only_mode = args.spine_only

    print(f"===== SpinePose Analysis System (Intel i5 4C/8T 최적화) =====")
    print(f"Model: {args.model_size} | CPU: Intel i5 4코어/8스레드")
    print(f"🚀 최적화: BlazePose Lite + 멀티스레딩 + 8프레임 간격")
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Controls:")
    print(f"  - Press 'q' to quit")
    print(f"  - Press 's' to toggle spine-only mode")
    print(f"Pipeline: BlazePose Lite → ROI → SpinePose → Spine Analysis")
    print(f"spine_only 모드: {'활성화' if spine_only_mode else '비활성화'}")
    
    # SpinePose 모델 초기화
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
    
    # OpenCV 윈도우 초기화
    cv2.namedWindow("SpinePose Analysis (Intel i5 Optimized)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SpinePose Analysis (Intel i5 Optimized)", WIN_W, WIN_H)

    # 서버에 프레임 처리 콜백 등록
    server.set_frame_callback(process_frame_callback)

    # 서버 실행
    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    finally:
        frame_queue.put(None)  # 워커 종료
        cv2.destroyAllWindows()
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

# 상위 디렉토리를 path에 추가 (server 폴더 접근용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# server.py import
from server import server

logging.basicConfig(level=logging.INFO)

# Intel i5 4C/8T 최적화 설정
CPU_CORES = 4
CPU_THREADS = 8
cv2.setNumThreads(CPU_THREADS)  # OpenCV 멀티스레딩 최적화

# AI 모델 초기화 - Intel i5 최적화
spine_est = SpinePoseEstimator(mode="medium", device="cpu")  # Intel i5에 최적
mp_pose = mp.solutions.pose
# BlazePose Lite + Intel i5 최적화
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # 0: Lite (Intel i5에 최적)
    smooth_landmarks=True,
    enable_segmentation=False,  # 세그멘테이션 비활성화로 성능 향상
    smooth_segmentation=False,
    min_detection_confidence=0.25,  # Intel i5 최적화
    min_tracking_confidence=0.25
)
PL = mp_pose.PoseLandmark

# 표시 파라미터
WIN_W, WIN_H = 720, 1440
SPINE_SCORE_TH = 0.1  # Intel i5에서 더 많은 키포인트 탐지

# 프레임 버퍼링 (Intel i5 멀티스레딩 활용)
frame_queue = queue.Queue(maxsize=3)
result_queue = queue.Queue(maxsize=3)

def lm_to_px_dict(res_lm, w, h):
    """MediaPipe 랜드마크를 픽셀 좌표로 변환"""
    d = {}
    if not res_lm: return d
    for p in PL:
        lm = res_lm.landmark[p.value]
        d[p.name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return d

def spinepose_infer_any(est, img_bgr, bboxes=None):
    """SpinePoseEstimator 추론 - Intel i5 최적화"""
    if est is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        # Intel i5에서는 bbox 사용 권장 (연산량 최적화)
        if bboxes is not None:
            out = est(img_rgb, bboxes)
        else:
            out = est(img_rgb)
    except Exception as e:
        print(f"SpinePose inference error: {e}")
        return [], []

    kpts_xy, scores = None, None
    try:
        if isinstance(out, dict):
            kpts_xy = out.get("keypoints") or out.get("kpts_xy")
            scores = out.get("scores")
        elif isinstance(out, (list, tuple)):
            if out and isinstance(out[0], np.ndarray):
                kpts_xy = out[0]
            if len(out) > 1 and isinstance(out[1], np.ndarray):
                scores = out[1]
        elif hasattr(out, "shape"):
            kpts_xy = out

        if kpts_xy is None:
            return [], []

        kpts_xy = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
        if kpts_xy.size == 0:
            return [], []

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
    """척추 키포인트 시간적 스무딩 - Intel i5 최적화"""
    def __init__(self, history_size=3):  # Intel i5에서는 메모리 절약
        self.history = deque(maxlen=history_size)
        
    def add_detection(self, spine_map):
        self.history.append(spine_map.copy())
        
    def get_smoothed_spine_map(self, current_spine_map):
        if len(self.history) < 2:
            return current_spine_map
            
        smoothed_map = {}
        for name in current_spine_map.keys():
            coords_history = []
            scores_history = []
            
            for hist_map in self.history:
                if name in hist_map:
                    coords_history.append(hist_map[name]['coord'])
                    scores_history.append(hist_map[name]['score'])
            
            if len(coords_history) >= 2:
                weights = np.linspace(0.2, 1.0, len(coords_history))
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

spine_tracker = SpineTracker(history_size=3)

def get_spine_keypoint_indices():
    """SpinePose의 실제 척추 키포인트 인덱스 반환"""
    # SpinePose는 33개 키포인트를 제공하며, 이 중 척추 관련 인덱스는:
    # 기본 COCO 17개 + 발 6개 + 척추 9개 = 32개 (0-31 인덱스)
    # 척추 키포인트들은 보통 뒤쪽 인덱스에 위치
    spine_indices = list(range(24, 33))  # 24-32번이 척추 키포인트로 추정
    return spine_indices

def filter_spine_keypoints(all_kpts, all_scores, spine_only=False):
    """spine_only 옵션에 따라 키포인트 필터링"""
    if not spine_only or not all_kpts:
        return all_kpts, all_scores
    
    # 모든 키포인트를 일단 반환 (실제 SpinePose는 척추 키포인트를 구분해서 제공)
    # 실제 구현에서는 SpinePose가 제공하는 척추 키포인트만 선택해야 함
    return all_kpts, all_scores

def detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th=0.1):
    """동적 척추 키포인트 탐지 - Intel i5 최적화"""
    if not sp_kpts or len(sp_kpts) < 3:
        return {}
    
    valid_points = []
    for i, (x, y) in enumerate(sp_kpts):
        if i < len(sp_scores) and sp_scores[i] >= score_th:
            valid_points.append((i, x, y, sp_scores[i]))
    
    if len(valid_points) < 3:
        return {}
    
    valid_points.sort(key=lambda p: p[2])  # Y좌표 정렬
    
    # 중앙선 필터링 (Intel i5에서 빠른 연산)
    center_x = np.median([p[1] for p in valid_points])
    spine_candidates = []
    tolerance = 45  # Intel i5 최적화
    
    for idx, x, y, score in valid_points:
        if abs(x - center_x) <= tolerance:
            spine_candidates.append((idx, x, y, score))
    
    if len(spine_candidates) < 3:
        return {}
    
    # 키포인트 할당
    spine_map = {}
    spine_candidates.sort(key=lambda p: p[2])
    
    if len(spine_candidates) >= 6:
        spine_names = ["C7", "T3", "T8", "T12", "L1", "L5"]
        indices = np.linspace(0, len(spine_candidates)-1, len(spine_names), dtype=int)
        for i, name in enumerate(spine_names):
            idx = indices[i]
            candidate = spine_candidates[idx]
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    else:
        key_names = ["C7", "T8", "L5"]
        spine_candidates_scored = sorted(spine_candidates, key=lambda p: p[3], reverse=True)
        for i, name in enumerate(key_names[:len(spine_candidates_scored)]):
            candidate = spine_candidates_scored[i]
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    
    return spine_map

def extract_spine_keypoints(sp_kpts, sp_scores, score_th=0.1):
    """척추 키포인트 추출 및 스무딩"""
    spine_map = detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th)
    
    if spine_map:
        spine_tracker.add_detection(spine_map)
        spine_map = spine_tracker.get_smoothed_spine_map(spine_map)
    
    return spine_map

def compute_spine_cva(spine_map):
    """CVA 계산"""
    cervical_pairs = [('C7', 'T8'), ('C7', 'T3')]
    
    for top, bottom in cervical_pairs:
        if top in spine_map and bottom in spine_map:
            top_coord = spine_map[top]['coord']
            bottom_coord = spine_map[bottom]['coord']
            
            dx = top_coord[0] - bottom_coord[0]
            dy = top_coord[1] - bottom_coord[1]
            
            if abs(dy) > 10:
                angle = abs(np.degrees(np.arctan2(dx, dy)))
                return min(angle, 180 - angle)
    
    return None

def make_side_roi_from_mp(lm_px, w, h, margin=0.15):
    """MediaPipe 결과로 ROI 생성 - Intel i5 최적화"""
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
        torso_h = 120

    cx, cy = sx, sy + 0.25*torso_h
    H = torso_h * 2.2
    W = H * 0.8

    W *= (1+margin); H *= (1+margin)

    x1 = int(max(0, cx - W/2)); y1 = int(max(0, cy - H/2))
    x2 = int(min(w-1, cx + W/2)); y2 = int(min(h-1, cy + H/2))
    return (x1, y1, x2, y2)

def calculate_forward_head_posture(spine_coords):
    """거북목 감지 - 목과 어깨 정렬 분석"""
    if len(spine_coords) < 3:
        return None
    
    try:
        # 상위 25% (목 부분)과 중간 부분 비교
        neck_idx = len(spine_coords) // 4
        shoulder_idx = len(spine_coords) // 2
        
        if neck_idx >= len(spine_coords) or shoulder_idx >= len(spine_coords):
            return None
        
        neck_point = spine_coords[neck_idx]
        shoulder_point = spine_coords[shoulder_idx]
        
        # 목이 어깨보다 얼마나 앞으로 나와있는지 측정
        forward_distance = neck_point[0] - shoulder_point[0]
        vertical_distance = abs(neck_point[1] - shoulder_point[1])
        
        if vertical_distance > 20:  # 충분한 수직 거리
            forward_angle = np.degrees(np.arctan(abs(forward_distance) / vertical_distance))
            return forward_angle
        
        return None
    except:
        return None

def calculate_spinal_curvature(spine_coords):
    """허리 굴곡 감지 - 척추 전체 곡률 분석"""
    if len(spine_coords) < 4:
        return None
    
    try:
        # 상위, 중간, 하위 지점으로 곡률 계산
        upper_idx = len(spine_coords) // 4
        middle_idx = len(spine_coords) // 2  
        lower_idx = 3 * len(spine_coords) // 4
        
        if lower_idx >= len(spine_coords):
            return None
        
        upper = spine_coords[upper_idx]
        middle = spine_coords[middle_idx]
        lower = spine_coords[lower_idx]
        
        # 세 점으로 곡률 계산
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
    """척추 분석 결과 시각화"""
    global spine_only_mode
    
    if spine_only is None:
        spine_only = spine_only_mode
    
    if not sp_kpts or len(sp_kpts) == 0:
        return img
    
    spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]
    
    if spine_only:
        # spine_only 모드: 척추만 강조하여 표시
        spine_coords = []
        
        for i in spine_indices:
            if i < len(sp_kpts) and i < len(sp_scores):
                kpt = sp_kpts[i]
                score = sp_scores[i]
                
                if score > 0.2:
                    x, y = int(kpt[0]), int(kpt[1])
                    
                    # 민트색 + 빨간색 강조
                    cv2.circle(img, (x, y), 6, (255, 255, 0), -1)  # 민트색
                    cv2.circle(img, (x, y), 8, (0, 0, 255), 2)     # 빨간 테두리
                    
                    spine_coords.append((x, y))
        
        # 노란색 연결선
        if len(spine_coords) >= 2:
            spine_coords.sort(key=lambda p: p[1])
            cv2.polylines(img, [np.array(spine_coords, np.int32)], 
                         False, (0, 255, 255), 3)  # 노란색 연결선
            
            # 거북목 분석
            forward_head = calculate_forward_head_posture(spine_coords)
            if forward_head is not None:
                color = (0, 0, 255) if forward_head > 20 else (0, 255, 0)
                cv2.putText(img, f"Forward Head: {forward_head:.1f}deg", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 허리 굴곡 분석  
            spinal_curve = calculate_spinal_curvature(spine_coords)
            if spinal_curve is not None:
                color = (0, 0, 255) if spinal_curve > 25 else (0, 255, 0)
                cv2.putText(img, f"Spinal Curve: {spinal_curve:.1f}deg", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(img, f"[SPINE-ONLY MODE]", 
                   (10, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    else:
        # 일반 모드: 모든 키포인트를 민트색으로만 표시
        for i, (kpt, score) in enumerate(zip(sp_kpts, sp_scores)):
            if score > 0.3:
                x, y = int(kpt[0]), int(kpt[1])
                cv2.circle(img, (x, y), 3, (255, 255, 0), -1)  # 민트색만
        
        cv2.putText(img, f"[ALL KEYPOINTS]", 
                   (10, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 공통 정보
    cv2.putText(img, f"Press 's' to toggle spine-only mode", (10, img.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return img

async def process_frame_callback(img):
    """서버에서 받은 프레임 처리 - Intel i5 최적화"""
    global frame_count, last_results, spine_only_mode
    
    h, w = img.shape[:2]
    
    # Intel i5에서는 8프레임마다 추론 (4C/8T 고려)
    if frame_count % 8 == 0:
        try:
            # 프레임을 추론 큐에 넣기 (논블로킹)
            if not frame_queue.full():
                frame_queue.put((img.copy(), w, h))
        except Exception as e:
            print(f"Frame queue error: {e}")
    
    # 추론 결과 가져오기 (논블로킹)
    try:
        while not result_queue.empty():
            last_results = result_queue.get_nowait()
    except queue.Empty:
        pass
    
    # 척추 분석 및 시각화 (전역 변수 사용)
    if last_results["sp_kpts"] and len(last_results["sp_kpts"]) >= 3:
        try:
            img = visualize_spine_analysis(img, last_results["sp_kpts"], last_results["sp_scores"])
        except Exception as e:
            cv2.putText(img, f"Analysis error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        kpt_count = len(last_results["sp_kpts"]) if last_results["sp_kpts"] else 0
        mode_text = "[SPINE-ONLY]" if spine_only_mode else "[ALL KEYPOINTS]"
        cv2.putText(img, f"SpinePose: {kpt_count} points (need ≥3) {mode_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # OpenCV 윈도우 표시
    disp = cv2.resize(img, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("SpinePose Analysis (Intel i5 Optimized)", disp)
    
    # 키 입력 처리
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        frame_queue.put(None)  # 워커 종료 신호
        print("프로그램 종료")
        exit(0)
    elif k == ord('s'):
        spine_only_mode = not spine_only_mode
        mode_text = "SPINE-ONLY" if spine_only_mode else "ALL KEYPOINTS"
        print(f"모드 변경: {mode_text}")

    frame_count += 1
    return img

def detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th=0.1):
    """동적 척추 키포인트 탐지"""
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
    spine_candidates = []
    tolerance = 45
    
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
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    else:
        key_names = ["C7", "T8", "L5"]
        spine_candidates_scored = sorted(spine_candidates, key=lambda p: p[3], reverse=True)
        for i, name in enumerate(key_names[:len(spine_candidates_scored)]):
            candidate = spine_candidates_scored[i]
            spine_map[name] = {
                'index': candidate[0],
                'coord': (candidate[1], candidate[2]),
                'score': candidate[3]
            }
    
    return spine_map

def extract_spine_keypoints(sp_kpts, sp_scores, score_th=0.1):
    """척추 키포인트 추출 및 스무딩"""
    spine_map = detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th)
    
    if spine_map:
        spine_tracker.add_detection(spine_map)
        spine_map = spine_tracker.get_smoothed_spine_map(spine_map)
    
    return spine_map

# 프레임 처리 변수
frame_count = 0
last_results = {"sp_kpts": [], "sp_scores": [], "roi": None}
spine_only_mode = True  # 기본적으로 spine_only 모드

def inference_worker():
    """별도 스레드에서 추론 수행 (Intel i5 멀티스레딩 활용)"""
    while True:
        try:
            frame_data = frame_queue.get(timeout=1.0)
            if frame_data is None:  # 종료 신호
                break
                
            img, w, h = frame_data
            
            # MediaPipe 처리
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm_px = lm_to_px_dict(res.pose_landmarks, w, h)
                roi = make_side_roi_from_mp(lm_px, w, h, margin=0.20)
            else:
                roi = (int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))

            # SpinePose 실행
            x1, y1, x2, y2 = roi
            bbox = [[x1, y1, x2, y2]]
            test_kpts, test_scores = spinepose_infer_any(spine_est, img, bboxes=bbox)
            
            result = {
                "sp_kpts": [(int(x), int(y)) for x, y in test_kpts] if test_kpts is not None else [],
                "sp_scores": test_scores if test_scores is not None else [],
                "roi": roi
            }
            
            result_queue.put(result)
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Inference worker error: {e}")

# 추론 워커 스레드 시작
inference_thread = threading.Thread(target=inference_worker, daemon=True)
inference_thread.start()

async def process_frame_callback(img):
    """서버에서 받은 프레임 처리 - Intel i5 최적화"""
    global frame_count, last_results, spine_only_mode
    
    h, w = img.shape[:2]
    
    # Intel i5에서는 8프레임마다 추론 (4C/8T 고려)
    if frame_count % 8 == 0:
        try:
            # 프레임을 추론 큐에 넣기 (논블로킹)
            if not frame_queue.full():
                frame_queue.put((img.copy(), w, h))
        except Exception as e:
            print(f"Frame queue error: {e}")
    
    # 추론 결과 가져오기 (논블로킹)
    try:
        while not result_queue.empty():
            last_results = result_queue.get_nowait()
    except queue.Empty:
        pass
    
    # 척추 분석 및 시각화
    if last_results["sp_kpts"] and len(last_results["sp_kpts"]) >= 3:
        try:
            img = visualize_spine_analysis(img, last_results["sp_kpts"], last_results["sp_scores"], spine_only_mode)
        except Exception as e:
            cv2.putText(img, f"Analysis error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        kpt_count = len(last_results["sp_kpts"]) if last_results["sp_kpts"] else 0
        mode_text = "[SPINE-ONLY]" if spine_only_mode else "[ALL KEYPOINTS]"
        cv2.putText(img, f"SpinePose: {kpt_count} points (need ≥3) {mode_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # OpenCV 윈도우 표시
    disp = cv2.resize(img, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("SpinePose Analysis (Intel i5 Optimized)", disp)
    
    # 키 입력 처리
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        cv2.destroyAllWindows()
        frame_queue.put(None)  # 워커 종료 신호
        print("프로그램 종료")
        exit(0)
    elif k == ord('s'):
        spine_only_mode = not spine_only_mode
        mode_text = "SPINE-ONLY" if spine_only_mode else "ALL KEYPOINTS"
        print(f"모드 변경: {mode_text}")

    frame_count += 1
    return img

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="SpinePose Analysis WebRTC Server (Intel i5 Optimized)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    parser.add_argument("--model-size", default="medium", choices=["small", "medium", "large", "xlarge"],
                       help="SpinePose model size (Intel i5 권장: medium) - 파라미터명: mode")
    parser.add_argument("--spine-only", action="store_true", 
                       help="Start with spine-only mode (toggle with 's' key)")
    args = parser.parse_args()

    global spine_only_mode, spine_est
    spine_only_mode = args.spine_only

    print(f"===== SpinePose Analysis System (Intel i5 4C/8T 최적화) =====")
    print(f"Model: {args.model_size} | CPU: Intel i5 4코어/8스레드")
    print(f"🚀 최적화: BlazePose Lite + 멀티스레딩 + 8프레임 간격")
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Controls:")
    print(f"  - Press 'q' to quit")
    print(f"  - Press 's' to toggle spine-only mode")
    print(f"Pipeline: BlazePose Lite → ROI → SpinePose → Spine Analysis")
    print(f"spine_only 모드: {'활성화' if spine_only_mode else '비활성화'}")
    
    # SpinePose 모델 재초기화 (올바른 파라미터 사용)
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
    
    # OpenCV 윈도우 초기화
    cv2.namedWindow("SpinePose Analysis (Intel i5 Optimized)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("SpinePose Analysis (Intel i5 Optimized)", WIN_W, WIN_H)

    # 서버에 프레임 처리 콜백 등록
    server.set_frame_callback(process_frame_callback)

    # 서버 실행
    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    finally:
        frame_queue.put(None)  # 워커 종료
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()