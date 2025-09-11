import cv2, argparse, logging, sys, os
import numpy as np
import mediapipe as mp
from collections import deque
from spinepose import SpinePoseEstimator

# 상위 디렉토리를 path에 추가 (server 폴더 접근용)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# server.py import
from server import server

logging.basicConfig(level=logging.INFO)

# AI 모델 초기화
spine_est = SpinePoseEstimator(device="cpu")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
PL = mp_pose.PoseLandmark

# 표시 파라미터
WIN_W, WIN_H = 1280, 720
SPINE_SCORE_TH = 0.15

def lm_to_px_dict(res_lm, w, h):
    """MediaPipe 랜드마크를 픽셀 좌표로 변환"""
    d = {}
    if not res_lm: return d
    for p in PL:
        lm = res_lm.landmark[p.value]
        d[p.name] = (int(lm.x * w), int(lm.y * h), lm.visibility)
    return d

def spinepose_infer_any(est, img_bgr, bboxes=None):
    """SpinePoseEstimator 추론"""
    if est is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    try:
        if bboxes is not None:
            out = est(img_rgb, bboxes)
        else:
            out = est(img_rgb)
    except Exception:
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
    except Exception:
        return [], []

class SpineTracker:
    """척추 키포인트 시간적 스무딩"""
    def __init__(self, history_size=5):
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

spine_tracker = SpineTracker(history_size=5)

def detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th=0.15):
    """동적 척추 키포인트 탐지"""
    if not sp_kpts or len(sp_kpts) < 3:
        return {}
    
    valid_points = []
    for i, (x, y) in enumerate(sp_kpts):
        if i < len(sp_scores) and sp_scores[i] >= score_th:
            valid_points.append((i, x, y, sp_scores[i]))
    
    if len(valid_points) < 3:
        return {}
    
    valid_points.sort(key=lambda p: p[2])  # Y좌표 정렬
    
    # 중앙선 필터링
    center_x = np.median([p[1] for p in valid_points])
    spine_candidates = []
    for idx, x, y, score in valid_points:
        if abs(x - center_x) <= 50:
            spine_candidates.append((idx, x, y, score))
    
    if len(spine_candidates) < 3:
        return {}
    
    # 키포인트 할당
    spine_map = {}
    spine_candidates.sort(key=lambda p: p[2])
    
    if len(spine_candidates) >= 6:
        spine_names = ["C1", "C7", "T3", "T8", "L1", "L5"]
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

def extract_spine_keypoints(sp_kpts, sp_scores, score_th=0.15):
    """척추 키포인트 추출 및 스무딩"""
    spine_map = detect_spine_keypoints_dynamically(sp_kpts, sp_scores, score_th)
    
    if spine_map:
        spine_tracker.add_detection(spine_map)
        spine_map = spine_tracker.get_smoothed_spine_map(spine_map)
    
    return spine_map

def compute_spine_cva(spine_map):
    """CVA 계산"""
    cervical_pairs = [('C1', 'C7'), ('C1', 'T8')]
    
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

def make_side_roi_from_mp(lm_px, w, h, margin=0.12):
    """MediaPipe 결과로 ROI 생성"""
    def get(name):
        v = lm_px.get(name)
        return v if (v and v[2] > 0.5) else None

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
    H = torso_h * 2.3
    W = H * 0.85

    W *= (1+margin); H *= (1+margin)

    x1 = int(max(0, cx - W/2)); y1 = int(max(0, cy - H/2))
    x2 = int(min(w-1, cx + W/2)); y2 = int(min(h-1, cy + H/2))
    return (x1, y1, x2, y2)

def visualize_spine_analysis(img, sp_kpts, sp_scores):
    """척추 분석 결과 시각화"""
    spine_map = extract_spine_keypoints(sp_kpts, sp_scores)
    
    if not spine_map:
        cv2.putText(img, "No spine keypoints detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return img
    
    # CVA 계산
    cva = compute_spine_cva(spine_map)
    
    # 키포인트 시각화
    spine_coords = []
    for name in ["C1", "C7", "T3", "T8", "L1", "L5"]:
        if name in spine_map:
            coord = spine_map[name]['coord']
            score = spine_map[name]['score']
            x, y = int(coord[0]), int(coord[1])
            
            # 신뢰도 기반 색상
            if score > 0.7:
                color = (0, 255, 0)  # 녹색
            elif score > 0.4:
                color = (0, 255, 255)  # 노랑
            else:
                color = (0, 100, 255)  # 주황
                
            radius = int(4 + score * 4)
            cv2.circle(img, (x, y), radius, color, -1)
            cv2.circle(img, (x, y), radius + 2, (255, 255, 255), 2)
            
            spine_coords.append((x, y))
    
    # 척추 연결선
    if len(spine_coords) >= 2:
        cv2.polylines(img, [np.array(spine_coords, np.int32)], 
                     False, (0, 255, 255), 3)
    
    # CVA 결과 표시
    if cva is not None:
        color = (0, 0, 255) if cva > 15 else (0, 255, 0)
        cv2.putText(img, f"CVA: {cva:.1f}deg", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # 상태 정보
    cv2.putText(img, f"Spine points: {len(spine_map)}", (10, img.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

# 프레임 처리 변수
frame_count = 0
last_results = {"sp_kpts": [], "sp_scores": [], "roi": None}

async def process_frame_callback(img):
    """서버에서 받은 프레임 처리"""
    global frame_count, last_results
    
    h, w = img.shape[:2]
    
    # 4프레임마다 처리 (성능 최적화)
    if frame_count % 4 == 0:
        try:
            # MediaPipe로 ROI 구하기
            res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                lm_px = lm_to_px_dict(res.pose_landmarks, w, h)
                roi = make_side_roi_from_mp(lm_px, w, h, margin=0.20)
            else:
                roi = (int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.85))
                
            last_results["roi"] = roi

            # SpinePose 실행
            full_bbox = [[0, 0, w, h]]
            test_kpts, test_scores = spinepose_infer_any(spine_est, img, bboxes=full_bbox)
            
            if test_kpts is not None and len(test_kpts) > 0:
                last_results["sp_kpts"] = [(int(x), int(y)) for x, y in test_kpts]
                last_results["sp_scores"] = test_scores if test_scores is not None else [0.5] * len(test_kpts)
                        
        except Exception as e:
            print(f"SpinePose error: {e}")
    
    # 척추 분석 및 시각화
    if last_results["sp_kpts"] and len(last_results["sp_kpts"]) >= 3:
        try:
            img = visualize_spine_analysis(img, last_results["sp_kpts"], last_results["sp_scores"])
        except Exception:
            cv2.putText(img, "Analysis error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        kpt_count = len(last_results["sp_kpts"]) if last_results["sp_kpts"] else 0
        cv2.putText(img, f"SpinePose: {kpt_count} points (need ≥3)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # OpenCV 윈도우 표시
    disp = cv2.resize(img, (WIN_W, WIN_H), interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Spine Analysis", disp)
    
    # 종료 처리
    k = cv2.waitKey(1)
    if k == ord('q'):
        cv2.destroyAllWindows()
        print("프로그램 종료")
        exit(0)

    frame_count += 1
    return img

def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="BlazePose + SpinePose Analysis WebRTC Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8080, help="Port number")
    args = parser.parse_args()

    print(f"===== BlazePose + SpinePose Analysis System =====")
    print(f"Starting server on http://{args.host}:{args.port}")
    print(f"Press 'q' in OpenCV window to quit")
    print(f"Pipeline: BlazePose → Upper Body Crop → SpinePose")
    
    # OpenCV 윈도우 초기화
    cv2.namedWindow("BlazePose + SpinePose Analysis", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("BlazePose + SpinePose Analysis", WIN_W, WIN_H)

    # 서버에 프레임 처리 콜백 등록
    server.set_frame_callback(process_frame_callback)

    # 서버 실행
    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n프로그램이 중단되었습니다.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()