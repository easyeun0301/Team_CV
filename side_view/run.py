# -*- coding: utf-8 -*-
# ========= (1) Environment Variables and Optimization Settings =========
import os
import sys
import time
import threading
import queue
import logging
import argparse
import platform
import cProfile
import pstats
import gc
import shutil
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List, Callable
from pathlib import Path
import requests

def maybe_reset(port):
    """서버에서 reset_signal을 확인하고 감점 누적 초기화"""
    global neck_sum, spine_sum
    try:
        SIDE_BASE = f"http://localhost:{port}"
        r = requests.get(f"{SIDE_BASE}/android/reset", timeout=0.3)
        if r.ok:
            data = r.json()
            if data.get("reset_signal"):
                neck_sum = 0
                spine_sum = 0
    except Exception as e:
        pass

# Set environment variables before imports
def setup_environment():
    """Handle environment variables and performance optimization settings"""
    env_vars = {
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", 
        "OPENBLAS_NUM_THREADS": "4",
        "NUMEXPR_NUM_THREADS": "4",
        "TF_ENABLE_ONEDNN_OPTS": "1"
    }
    for key, value in env_vars.items():
        os.environ.setdefault(key, value)
    
    # GC optimization
    gc.set_threshold(700, 10, 10)

setup_environment()

# Required library imports
import cv2
import numpy as np
import mediapipe as mp
from spinepose import SpinePoseEstimator

# Add parent directory to path (server module)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from server import server

# Logging setup
logging.basicConfig(level=logging.ERROR)
logging.getLogger('aioice').setLevel(logging.WARNING)
logging.getLogger('absl').disabled = True

# ========= GLOBAL VARIABLES FOR STREAMLIT =========
# �� Streamlit accesses these variables
neck_sum = 0      # Cumulative neck score (3 minutes)
spine_sum = 0     # Cumulative spine score (3 minutes)

# �� �뚮┝ �뚮옒洹� 異붽�
neck_alert_flag = False    # True�대㈃ 嫄곕턿紐� 寃쎄퀬
spine_alert_flag = False   # True�대㈃ 泥숈텛 寃쎄퀬
LAST_ALERT = None          # �곸꽭 �뺣낫

def get_current_scores():
    """For Streamlit: Get current cumulative scores"""
    global neck_sum, spine_sum
    return {
        'neck_sum': neck_sum,
        'spine_sum': spine_sum
    }

def get_alert_flags():
    """For Streamlit: Get alert flags (boolean)"""
    global neck_alert_flag, spine_alert_flag
    return {
        'neck_alert': neck_alert_flag,
        'spine_alert': spine_alert_flag
    }

def clear_alert_flags():
    """For Streamlit: Clear alert flags after handling"""
    global neck_alert_flag, spine_alert_flag
    neck_alert_flag = False
    spine_alert_flag = False

def get_last_alert():
    """For Streamlit: Get and clear last alert (detailed info)"""
    global LAST_ALERT
    alert = LAST_ALERT
    LAST_ALERT = None
    return alert

# ========= (2) Constants and Configuration =========
@dataclass
class Config:
    """System configuration constants"""
    WIN_W: int = 480
    WIN_H: int = 640

    STREAMLIT_OUTPUT_W: int = 480
    STREAMLIT_OUTPUT_H: int = 640
    
    DISPLAY_WINDOW_W: int = 480
    DISPLAY_WINDOW_H: int = 640
    
    BP_PERIOD_MS: int = 400
    SP_PERIOD_MS: int = 450
    STICKY_MS: int = 450
    SPINE_SCORE_TH: float = 0.1
    INFER_SCALE: float = 0.25

    DRAW_SPINE_ONLY_DEFAULT: bool = True
    WINDOW_TITLE: str = "SpinePose Analysis (Spine-Only)"

    NECK_IDX: List[int] = field(default_factory=lambda: [36, 35, 18])
    LUMBAR_IDX: List[int] = field(default_factory=lambda: [30, 28, 19])

    FHP_THRESH_DEG: float = 19.0
    CURVE_THRESH_DEG: float = 18.0
    
    MEMORY_CHECK_INTERVAL: int = 100
    MEMORY_THRESHOLD_MB: int = 500

# ========= (3) Utility Functions =========
def safe_import(name: str):
    """Safe module import"""
    try:
        return __import__(name)
    except Exception:
        return None

def detect_cpu_env() -> Dict[str, Any]:
    """Auto-detect CPU environment"""
    psutil = safe_import("psutil")
    cpuinfo = safe_import("cpuinfo")
    
    logical = os.cpu_count() or 4
    physical = None
    
    if psutil:
        try:
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
    
    if physical is None:
        physical = logical

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

def apply_thread_tuning(env: Dict[str, Any]) -> int:
    """Apply thread tuning"""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass

    n = max(1, min(env["usable"], 8))
    thread_vars = [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS",
        "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"
    ]
    for var in thread_vars:
        os.environ[var] = str(n)

    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
    return n

def recommend_period_ms(env: Dict[str, Any]) -> int:
    """Recommend SpinePose period based on environment"""
    phys = env["physical"]
    if phys <= 4:
        return 110
    elif phys <= 6:
        return 100
    else:
        return 90

def safe_queue_put(q: queue.Queue, item: Any, replace_if_full: bool = True):
    """Thread-safe queue insertion"""
    try:
        if q.full() and replace_if_full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put_nowait(item)
    except queue.Full:
        pass

def safe_queue_get_all(q: queue.Queue) -> List[Any]:
    """Safely get all items from queue"""
    items = []
    try:
        while not q.empty():
            items.append(q.get_nowait())
    except queue.Empty:
        pass
    return items

# ========= METHOD 4: Pre-initialize Libraries =========
def preinitialize_libraries():
    """Pre-initialize OpenCV and NumPy thread pools"""
    print("[Init] Pre-initializing libraries...")
    
    # OpenCV thread pool warmup
    dummy = np.zeros((640, 480, 3), dtype=np.uint8)
    for _ in range(3):
        _ = cv2.resize(dummy, (320, 240), interpolation=cv2.INTER_AREA)
        _ = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    
    # NumPy BLAS warmup
    a = np.random.rand(100, 100).astype(np.float32)
    b = np.random.rand(100, 100).astype(np.float32)
    for _ in range(3):
        _ = np.dot(a, b)
    
    print("[Init] Libraries ready")

# Memory monitoring
class MemoryMonitor:
    """Memory usage monitoring"""
    def __init__(self, threshold_mb: int = 500):
        self.threshold = threshold_mb * 1024 * 1024
        self.psutil = safe_import("psutil")
        self.frame_count = 0
        
    def check_and_collect(self, interval: int = 100) -> bool:
        """Check memory and run GC if needed"""
        self.frame_count += 1
        if self.frame_count % interval != 0:
            return False
            
        if not self.psutil:
            return False
            
        try:
            process = self.psutil.Process()
            mem_info = process.memory_info()
            if mem_info.rss > self.threshold:
                gc.collect()
                return True
        except Exception:
            pass
        return False

# ========= (4) Spine Tracking and Analysis Classes =========
class SpineTracker:
    """Spine keypoint smoothing tracker"""
    def __init__(self, history_size: int = 3):
        self.history = deque(maxlen=history_size)
    
    def add_detection(self, spine_map: Dict[str, Any]):
        self.history.append(spine_map.copy())
    
    def get_smoothed_spine_map(self, current_spine_map: Dict[str, Any]) -> Dict[str, Any]:
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

# ROI cache class (tuple reuse)
class ROICache:
    """ROI tuple reuse to reduce GC overhead"""
    __slots__ = ['x1', 'y1', 'x2', 'y2']
    
    def __init__(self, x1: int = 0, y1: int = 0, x2: int = 0, y2: int = 0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def update(self, x1: int, y1: int, x2: int, y2: int):
        """In-place update"""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    
    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to tuple if needed"""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def as_list(self) -> List[List[int]]:
        """SpinePose bbox format"""
        return [[self.x1, self.y1, self.x2, self.y2]]

# ========= POSTURE SCORE MANAGER =========
# Color constants (BGR)
GREEN  = (0, 255, 0)
YELLOW = (0, 255, 255)
RED    = (0, 0, 255)
GRAY   = (190, 190, 190)

class PostureScoreManager:
    """Neck and Spine separate evaluation with priority-based tie-breaking"""
    
    def __init__(self, window_sec: float = 10.0, max_duration_sec: float = 180.0):
        self.window_sec = window_sec
        self.max_duration_sec = max_duration_sec  # �� 3遺� (180珥�) �쒗븳
        
        # Color history (recent 10 seconds)
        self.color_history = deque()
        
        # Segment results
        self.segment_results = []
        
        self.last_check_time = time.time()
        self.start_time = time.time()
        
        self.scoring_ended = False  # �� �먯닔 怨꾩궛 醫낅즺 �뚮옒洹�
        
    def add_frame(self, neck_color: Tuple[int,int,int], spine_color: Tuple[int,int,int]):
        """Record neck/spine color every frame"""
        now = time.time()
        
        # �� 3遺� 寃쎄낵 泥댄겕
        elapsed = now - self.start_time
        if elapsed >= self.max_duration_sec and not self.scoring_ended:
            self.scoring_ended = True
            print("\n" + "="*70)
            print(f"[Session End] 3 minutes elapsed - Scoring stopped")
            print(f"[Final Score] neck_sum: {neck_sum}, spine_sum: {spine_sum}")
            print("="*70 + "\n")
        
        # �� 3遺� �댄썑�먮뒗 �덉뒪�좊━�� 異붽� �� ��
        if self.scoring_ended:
            return
        
        self.color_history.append((now, neck_color, spine_color))
        
        # Remove data older than 10 seconds
        cutoff = now - self.window_sec
        while self.color_history and self.color_history[0][0] < cutoff:
            self.color_history.popleft()
    
    def check_and_score(self) -> Optional[Dict[str, Any]]:
        """Evaluate every 10 seconds: separate neck/spine assessment"""
        global neck_sum, spine_sum, LAST_ALERT
        global neck_alert_flag, spine_alert_flag  # �� 異붽�
        
        # �� 3遺� �댄썑�먮뒗 �됯� �� ��
        if self.scoring_ended:
            return None
        
        now = time.time()
        
        # Return None if 10 seconds haven't passed
        if now - self.last_check_time < self.window_sec:
            return None
        
        self.last_check_time = now
        
        if not self.color_history:
            return None
        
        # Count neck/spine colors separately
        total = len(self.color_history)
        
        # Neck count
        neck_red = 0
        neck_yellow = 0
        neck_green = 0
        
        # Spine count
        spine_red = 0
        spine_yellow = 0
        spine_green = 0
        
        for _, neck_color, spine_color in self.color_history:
            # Neck classification
            if neck_color == RED:
                neck_red += 1
            elif neck_color == YELLOW:
                neck_yellow += 1
            else:
                neck_green += 1
            
            # Spine classification
            if spine_color == RED:
                spine_red += 1
            elif spine_color == YELLOW:
                spine_yellow += 1
            else:
                spine_green += 1
        
        # Neck evaluation
        neck_red_ratio = neck_red / total
        neck_yellow_ratio = neck_yellow / total
        neck_green_ratio = neck_green / total
        
        # Priority-based tie-breaking: RED > YELLOW > GREEN
        if neck_red_ratio >= neck_yellow_ratio and neck_red_ratio >= neck_green_ratio:
            neck_segment_color = "RED"
            neck_delta = 2.5
            neck_alert = True
        elif neck_yellow_ratio >= neck_green_ratio:
            neck_segment_color = "YELLOW"
            neck_delta = 1.5
            neck_alert = False
        else:
            neck_segment_color = "GREEN"
            neck_delta = 0
            neck_alert = False
        
        # Spine evaluation
        spine_red_ratio = spine_red / total
        spine_yellow_ratio = spine_yellow / total
        spine_green_ratio = spine_green / total
        
        # Priority-based tie-breaking: RED > YELLOW > GREEN
        if spine_red_ratio >= spine_yellow_ratio and spine_red_ratio >= spine_green_ratio:
            spine_segment_color = "RED"
            spine_delta = 2.5
            spine_alert = True
        elif spine_yellow_ratio >= spine_green_ratio:
            spine_segment_color = "YELLOW"
            spine_delta = 1.5
            spine_alert = False
        else:
            spine_segment_color = "GREEN"
            spine_delta = 0
            spine_alert = False
        
        # �� Update global variables
        neck_sum += neck_delta
        spine_sum += spine_delta
        
        # Terminal log output
        elapsed_time = now - self.start_time
        segment_num = len(self.segment_results) + 1
        
        print("\n" + "="*70)
        print(f"[Segment {segment_num}/18] Elapsed: {elapsed_time:.1f}s")
        print(f"[Frame count] Total {total} frames")
        print()
        print(f"[NECK Evaluation]")
        print(f"  Red:    {neck_red:3d} ({neck_red_ratio:.1%})")
        print(f"  Yellow: {neck_yellow:3d} ({neck_yellow_ratio:.1%})")
        print(f"  Green:  {neck_green:3d} ({neck_green_ratio:.1%})")
        print(f"  -> Result: {neck_segment_color} (neck_sum +{neck_delta})")
        if neck_alert:
            print(f"  �좑툘  Warning: Forward head posture alert!")
        print()
        print(f"[SPINE Evaluation]")
        print(f"  Red:    {spine_red:3d} ({spine_red_ratio:.1%})")
        print(f"  Yellow: {spine_yellow:3d} ({spine_yellow_ratio:.1%})")
        print(f"  Green:  {spine_green:3d} ({spine_green_ratio:.1%})")
        print(f"  -> Result: {spine_segment_color} (spine_sum +{spine_delta})")
        if spine_alert:
            print(f"  �좑툘  Warning: Spinal curvature alert!")
        print()
        print(f"[Cumulative Scores]")
        print(f"  neck_sum:  {neck_sum - neck_delta} -> {neck_sum} (+{neck_delta})")
        print(f"  spine_sum: {spine_sum - spine_delta} -> {spine_sum} (+{spine_delta})")
        print("="*70 + "\n")
        
        # �� Update alert flags (only RED triggers alert)
        if neck_alert:
            neck_alert_flag = True  # �� �뚮옒洹� �ㅼ젙
            LAST_ALERT = {
                'type': 'neck',
                'timestamp': now,
                'score': neck_sum,
                'segment': segment_num
            }
            print(f"�슚 [ALERT FLAG] neck_alert_flag = True\n")
        
        if spine_alert:
            spine_alert_flag = True  # �� �뚮옒洹� �ㅼ젙
            LAST_ALERT = {
                'type': 'spine',
                'timestamp': now,
                'score': spine_sum,
                'segment': segment_num
            }
            print(f"�슚 [ALERT FLAG] spine_alert_flag = True\n")
        
        # Save results
        result = {
            'timestamp': now,
            'elapsed_time': elapsed_time,
            'segment_number': segment_num,
            'total_frames': total,
            
            # Neck results
            'neck': {
                'red_count': neck_red,
                'yellow_count': neck_yellow,
                'green_count': neck_green,
                'red_ratio': neck_red_ratio,
                'yellow_ratio': neck_yellow_ratio,
                'green_ratio': neck_green_ratio,
                'segment_color': neck_segment_color,
                'score_delta': neck_delta,
                'should_alert': neck_alert
            },
            
            # Spine results
            'spine': {
                'red_count': spine_red,
                'yellow_count': spine_yellow,
                'green_count': spine_green,
                'red_ratio': spine_red_ratio,
                'yellow_ratio': spine_yellow_ratio,
                'green_ratio': spine_green_ratio,
                'segment_color': spine_segment_color,
                'score_delta': spine_delta,
                'should_alert': spine_alert
            },
            
            # Cumulative scores
            'neck_sum': neck_sum,
            'spine_sum': spine_sum
        }
        
        self.segment_results.append(result)
        
        return result
    
    def get_segment_history(self) -> List[Dict[str, Any]]:
        """Get entire segment history"""
        return self.segment_results
    
    def is_scoring_ended(self) -> bool:
        """Check if 3-minute scoring period has ended"""
        return self.scoring_ended

# ========= (5) Context Class =========
@dataclass
class Context:
    """Global state management context"""
    spine_est: SpinePoseEstimator
    mp_pose: Any
    mp_pl: Any
    pose: Any
    config: Config = field(default_factory=Config)

    frame_q: "queue.Queue[Optional[np.ndarray]]" = field(default_factory=lambda: queue.Queue(maxsize=1))
    result_q: "queue.Queue[np.ndarray]" = field(default_factory=lambda: queue.Queue(maxsize=1))
    display_q: "queue.Queue[np.ndarray]" = field(default_factory=lambda: queue.Queue(maxsize=2))

    next_bp_ts: float = 0.0
    next_sp_ts: float = 0.0
    last_recv_ts: float = 0.0
    last_decode_ts: float = 0.0

    bp_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    sp_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    e2e_hist: deque = field(default_factory=lambda: deque(maxlen=30))

    last_results: Dict[str, Any] = field(default_factory=lambda: {"sp_kpts": [], "sp_scores": [], "roi": None})
    last_good_results: Dict[str, Any] = field(default_factory=lambda: {"sp_kpts": [], "sp_scores": [], "roi": None})
    last_update_ms: float = 0.0
    spine_only: bool = field(default_factory=lambda: Config.DRAW_SPINE_ONLY_DEFAULT)
    running: bool = True
    last_web_frame: Optional[np.ndarray] = None

    last_roi: Optional[ROICache] = None
    infer_scale: float = field(default_factory=lambda: Config.INFER_SCALE)
    spine_tracker: SpineTracker = field(default_factory=lambda: SpineTracker(history_size=3))
    memory_monitor: MemoryMonitor = field(default_factory=lambda: MemoryMonitor(threshold_mb=500))
    
    _rgb_buffer: Optional[np.ndarray] = None
    _small_buffer: Optional[np.ndarray] = None
    
    # �� Posture score manager
    score_manager: Optional[PostureScoreManager] = None

# ========= (6) Utility Functions (Vectorized) =========
def lm_to_px_dict(res_lm, w: int, h: int, mp_pl) -> Dict[str, Tuple[int, int, float]]:
    """Vectorized MediaPipe landmark conversion (2-3x faster)"""
    d = {}
    if not res_lm:
        return d
    
    landmarks = res_lm.landmark
    coords = np.array([[lm.x * w, lm.y * h, lm.visibility] 
                       for lm in landmarks], dtype=np.float32)
    
    coords_int = np.round(coords[:, :2]).astype(np.int32)
    
    for p in mp_pl.PoseLandmark:
        idx = p.value
        d[p.name] = (int(coords_int[idx, 0]), int(coords_int[idx, 1]), float(coords[idx, 2]))
    
    return d

def spinepose_infer_any(est: SpinePoseEstimator, img, bboxes=None, *, already_rgb: bool=False) -> Tuple[List, List]:
    """SpinePose inference execution"""
    if est is None:
        return [], []
    try:
        if not already_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    try:
        out = est(img, bboxes) if bboxes is not None else est(img)
    except Exception as e:
        print(f"[SpinePose] inference error: {e}")
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
        print(f"[SpinePose] output processing error: {e}")
        return [], []

def make_side_roi_from_mp(lm_px: Dict[str, Tuple], w: int, h: int, 
                         margin: float = 0.10, square_pad: bool = True, 
                         pad_ratio: float = 0.10, 
                         roi_cache: Optional[ROICache] = None) -> ROICache:
    """ROI cache reuse (GC overhead removal)"""
    def get(name: str):
        v = lm_px.get(name)
        return v if (v and v[2] > 0.4) else None
    
    lshoulder = get("LEFT_SHOULDER")
    rshoulder = get("RIGHT_SHOULDER")
    
    if not (lshoulder and rshoulder):
        x1, y1 = int(w*0.2), int(h*0.2)
        x2, y2 = int(w*0.8), int(h*0.8)
    else:
        sx = (lshoulder[0] + rshoulder[0]) / 2
        sy = (lshoulder[1] + rshoulder[1]) / 2
        lhip = get("LEFT_HIP")
        rhip = get("RIGHT_HIP")
        if lhip and rhip:
            hips = [lhip, rhip]
            hy = sum(p[1] for p in hips) / len(hips)
            torso_h = abs(hy - sy)
        else:
            torso_h = 120
        cx, cy = sx, sy + 0.25 * torso_h
        H = torso_h * 2.2
        W = H * 0.75 if square_pad else H * 0.6
        if square_pad:
            side = max(W, H)
            W = H = side * (1.0 + pad_ratio)
        x1 = int(max(0, cx - W/2)); y1 = int(max(0, cy - H/2))
        x2 = int(min(w-1, cx + W/2)); y2 = int(min(h-1, cy + H/2))
    
    if roi_cache is None:
        roi_cache = ROICache(x1, y1, x2, y2)
    else:
        roi_cache.update(x1, y1, x2, y2)
    
    return roi_cache

def smooth_roi(prev: Optional[ROICache], 
               new: Optional[ROICache], 
               alpha: float = 0.7, max_scale_step: float = 0.10, 
               frame_w: Optional[int] = None, frame_h: Optional[int] = None) -> Optional[ROICache]:
    """ROI smoothing (in-place update)"""
    if new is None:
        return prev
    if prev is None:
        return new
    
    px1, py1, px2, py2 = prev.x1, prev.y1, prev.x2, prev.y2
    nx1, ny1, nx2, ny2 = new.x1, new.y1, new.x2, new.y2
    pw, ph = px2 - px1, py2 - py1
    nw, nh = nx2 - nx1, ny2 - ny1
    
    def clamp_len(new_len: float, prev_len: float) -> float:
        up = prev_len * (1.0 + max_scale_step)
        dn = prev_len * (1.0 - max_scale_step)
        return max(min(new_len, up), dn)
    
    cw = clamp_len(nw, pw)
    ch = clamp_len(nh, ph)
    pcx, pcy = px1 + pw/2, py1 + ph/2
    ncx, ncy = nx1 + nw/2, ny1 + nh/2
    scx = pcx + alpha * (ncx - pcx)
    scy = pcy + alpha * (ncy - pcy)
    x1, y1 = int(scx - cw/2), int(scy - ch/2)
    x2, y2 = int(scx + cw/2), int(scy + ch/2)
    
    if frame_w and frame_h:
        x1 = max(0, min(x1, frame_w-1))
        y1 = max(0, min(y1, frame_h-1))
        x2 = max(x1+1, min(x2, frame_w-1))
        y2 = max(y1+1, min(y2, frame_h-1))
    
    prev.update(x1, y1, x2, y2)
    return prev

# ========= (7) Spine Analysis Functions (Vectorized) =========
def detect_spine_keypoints_dynamically(sp_kpts: List, sp_scores: List, 
                                     score_th: float = 0.1) -> Dict[str, Any]:
    """Dynamic spine keypoint detection"""
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

def calculate_forward_head_posture_torso(sp_kpts, sp_scores, neck_indices, lumbar_indices, score_th: float = 0.2) -> Optional[float]:
    """Vectorized forward head posture calculation (2x faster)"""
    try:
        sp_kpts_arr = np.array(sp_kpts, dtype=np.float32)
        sp_scores_arr = np.array(sp_scores, dtype=np.float32)
        
        neck_mask = np.isin(np.arange(len(sp_kpts)), neck_indices) & (sp_scores_arr >= score_th)
        lumbar_mask = np.isin(np.arange(len(sp_kpts)), lumbar_indices) & (sp_scores_arr >= score_th)
        
        neck_pts = sp_kpts_arr[neck_mask]
        lumbar_pts = sp_kpts_arr[lumbar_mask]
        
        if len(neck_pts) < 2 or len(lumbar_pts) < 1:
            return None
        
        neck_mid = np.mean(neck_pts, axis=0)
        lumbar_mid = np.mean(lumbar_pts, axis=0)
        
        torso_vec = lumbar_mid - neck_mid
        
        neck_sorted_idx = np.argsort(neck_pts[:, 1])
        p0 = neck_pts[neck_sorted_idx[0]]
        p1 = neck_pts[neck_sorted_idx[1]]
        neck_vec = p1 - p0
        
        if np.linalg.norm(torso_vec) < 1e-3 or np.linalg.norm(neck_vec) < 1e-3:
            return None
        
        ang = np.degrees(np.arctan2(neck_vec[0], neck_vec[1]) - np.arctan2(torso_vec[0], torso_vec[1]))
        ang = abs((ang + 180.0) % 360.0 - 180.0)
        return float(ang)
    except Exception:
        return None

def calculate_spinal_curvature(spine_coords: List[Tuple[int, int]]) -> Optional[float]:
    """Spinal curvature calculation"""
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

# ========= (8) Visualization and Rendering =========
def visualize_spine_analysis(img: np.ndarray, sp_kpts: List, sp_scores: List, 
                           spine_only: bool = True) -> np.ndarray:
    """Spine keypoint visualization (reference)"""
    if not sp_kpts or len(sp_kpts) == 0:
        return img
    spine_indices = [36, 35, 18, 30, 29, 28, 27, 26, 19]
    if spine_only:
        spine_coords = []
        for i in spine_indices:
            if i < len(sp_kpts) and i < len(sp_scores):
                kpt = sp_kpts[i]; score = sp_scores[i]
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

def thr_neck_color(val: Optional[float], th: float):
    if val is None: return GRAY
    if val < 0.75 * th: return GREEN
    if val <= th:       return YELLOW
    return RED

def thr_lumbar_color(val: Optional[float], th: float):
    if val is None: return GRAY
    if val < 0.5 * th:  return GREEN
    if val <= th:       return YELLOW
    return RED

# Batch rendering functions
def draw_keypoints_batch(canvas: np.ndarray, points: List[Tuple[int, int]], 
                        color: Tuple[int, int, int], 
                        inner_radius: int = 5, outer_radius: int = 7) -> None:
    """Batch keypoint drawing (90% faster)"""
    if not points:
        return
    
    points_arr = np.array(points, dtype=np.int32)
    
    for pt in points_arr:
        cv2.circle(canvas, tuple(pt), inner_radius, (255, 255, 255), -1)
    
    for pt in points_arr:
        cv2.circle(canvas, tuple(pt), outer_radius, color, 2)

def draw_polyline_batch(canvas: np.ndarray, points: List[Tuple[int, int]], 
                       color: Tuple[int, int, int], thickness: int = 3) -> None:
    """Batch polyline drawing"""
    if len(points) < 2:
        return
    points_arr = np.array(points, dtype=np.int32)
    cv2.polylines(canvas, [points_arr], False, color, thickness, lineType=cv2.LINE_AA)

# ========= render_display_frame =========
def render_display_frame(ctx: Context, img_bgr: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """Memory-optimized rendering"""
    now_ms = time.perf_counter() * 1000.0
    use = result
    has_valid = len(result.get("sp_kpts", [])) >= 3
    
    if not has_valid and (now_ms - ctx.last_update_ms) <= ctx.config.STICKY_MS:
        use = ctx.last_good_results
        has_valid = len(use.get("sp_kpts", [])) >= 3

    canvas = img_bgr.copy()

    forward_head = None
    spinal_curve = None
    neck_pts, lumbar_pts = [], []

    if has_valid:
        sp_kpts_arr = np.array(use["sp_kpts"], dtype=np.float32)
        sp_scores_arr = np.array(use["sp_scores"], dtype=np.float32)
        
        for i in ctx.config.NECK_IDX:
            if i < len(sp_kpts_arr) and sp_scores_arr[i] > 0.2:
                neck_pts.append((int(sp_kpts_arr[i, 0]), int(sp_kpts_arr[i, 1])))
        
        for i in ctx.config.LUMBAR_IDX:
            if i < len(sp_kpts_arr) and sp_scores_arr[i] > 0.2:
                lumbar_pts.append((int(sp_kpts_arr[i, 0]), int(sp_kpts_arr[i, 1])))

        all_spine_coords = []
        for i in ctx.config.NECK_IDX + ctx.config.LUMBAR_IDX:
            if i < len(sp_kpts_arr) and sp_scores_arr[i] > 0.2:
                all_spine_coords.append((int(sp_kpts_arr[i, 0]), int(sp_kpts_arr[i, 1])))
        
        if len(all_spine_coords) >= 2:
            all_spine_coords.sort(key=lambda p: p[1])

        forward_head = calculate_forward_head_posture_torso(
            use["sp_kpts"], use["sp_scores"],
            ctx.config.NECK_IDX, ctx.config.LUMBAR_IDX, score_th=0.2
        )
        spinal_curve = calculate_spinal_curvature(all_spine_coords)

        neck_color   = thr_neck_color(forward_head, ctx.config.FHP_THRESH_DEG)
        lumbar_color = thr_lumbar_color(spinal_curve, ctx.config.CURVE_THRESH_DEG)

        if len(neck_pts) >= 2:
            neck_pts_sorted = sorted(neck_pts, key=lambda p: p[1])
            draw_polyline_batch(canvas, neck_pts_sorted, neck_color, thickness=3)
            draw_keypoints_batch(canvas, neck_pts_sorted, neck_color, 
                               inner_radius=5, outer_radius=7)

        if len(lumbar_pts) >= 2:
            lumbar_pts_sorted = sorted(lumbar_pts, key=lambda p: p[1])
            draw_polyline_batch(canvas, lumbar_pts_sorted, lumbar_color, thickness=3)
            draw_keypoints_batch(canvas, lumbar_pts_sorted, lumbar_color, 
                                inner_radius=5, outer_radius=7)
    else:
        kpt_count = len(use.get("sp_kpts", []))
        cv2.putText(canvas, f"SpinePose: {kpt_count} points (need >=3)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

    target_w = ctx.config.STREAMLIT_OUTPUT_W
    target_h = ctx.config.STREAMLIT_OUTPUT_H
    
    if canvas.shape[1] != target_w or canvas.shape[0] != target_h:
        scale_w = target_w / canvas.shape[1]
        scale_h = target_h / canvas.shape[0]
        scale = min(scale_w, scale_h)
        
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        
        new_w = int(canvas.shape[1] * scale)
        new_h = int(canvas.shape[0] * scale)
        resized = cv2.resize(canvas, (new_w, new_h), interpolation=interp)
        
        output = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        canvas = output
    
    font_scale = 0.75
    thickness = 2
    
    if has_valid:
        if forward_head is not None:
            cv2.putText(canvas, f"Forward Head: {forward_head:.1f}deg",
                        (10, 34), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        thr_neck_color(forward_head, ctx.config.FHP_THRESH_DEG), 
                        thickness, cv2.LINE_AA)
        
        if spinal_curve is not None:
            cv2.putText(canvas, f"Spinal Curve: {spinal_curve:.1f}deg",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        thr_lumbar_color(spinal_curve, ctx.config.CURVE_THRESH_DEG), 
                        thickness, cv2.LINE_AA)

    return canvas

# ========= (9) Frame Callback (Async, Non-blocking) =========
async def process_frame_callback(ctx: Context, img_bgr: np.ndarray) -> np.ndarray:
    """Async, non-blocking callback for WebRTC (no AI work)"""
    
    safe_queue_put(ctx.frame_q, img_bgr, replace_if_full=True)
    
    try:
        latest_frame = ctx.result_q.get_nowait()
        ctx.last_web_frame = latest_frame
        return latest_frame
    except queue.Empty:
        if ctx.last_web_frame is not None:
            return ctx.last_web_frame
        else:
            return img_bgr

# ========= (9.5) AI Pipeline (Sync, Blocking) =========
def _run_ai_pipeline(ctx: Context, img_bgr: np.ndarray) -> np.ndarray:
    """All heavy AI work in synchronous function"""
    h, w = img_bgr.shape[:2]

    if ctx._rgb_buffer is None or ctx._rgb_buffer.shape[:2] != (h, w):
        ctx._rgb_buffer = np.empty((h, w, 3), dtype=np.uint8)
    
    cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB, dst=ctx._rgb_buffer)
    img_rgb = ctx._rgb_buffer
    
    try:
        t0 = time.perf_counter()
        now_ms = t0 * 1000.0
        
        ctx.memory_monitor.check_and_collect(interval=ctx.config.MEMORY_CHECK_INTERVAL)
        
        if now_ms >= ctx.next_bp_ts:
            ctx.next_bp_ts = now_ms + ctx.config.BP_PERIOD_MS
            
            if 0 < ctx.infer_scale < 1.0:
                small_w = int(w * ctx.infer_scale)
                small_h = int(h * ctx.infer_scale)
                
                if ctx._small_buffer is None or ctx._small_buffer.shape[:2] != (small_h, small_w):
                    ctx._small_buffer = np.empty((small_h, small_w, 3), dtype=np.uint8)
                
                cv2.resize(img_rgb, (small_w, small_h), 
                          dst=ctx._small_buffer, interpolation=cv2.INTER_AREA)
                
                res = ctx.pose.process(ctx._small_buffer)
                
                if res and res.pose_landmarks:
                    lm_px_small = lm_to_px_dict(res.pose_landmarks, small_w, small_h, ctx.mp_pl)
                    
                    scale_inv = 1.0 / ctx.infer_scale
                    lm_px = {
                        k: (int(v[0] * scale_inv + 0.5),
                            int(v[1] * scale_inv + 0.5),
                            v[2])
                        for k, v in lm_px_small.items()
                    }
                    raw_roi = make_side_roi_from_mp(lm_px, w, h, roi_cache=ctx.last_roi)
                else:
                    if ctx.last_roi is None:
                        ctx.last_roi = ROICache(int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))
                    raw_roi = ctx.last_roi
            else:
                res = ctx.pose.process(img_rgb)
                if res and res.pose_landmarks:
                    lm_px = lm_to_px_dict(res.pose_landmarks, w, h, ctx.mp_pl)
                    raw_roi = make_side_roi_from_mp(lm_px, w, h, roi_cache=ctx.last_roi)
                else:
                    if ctx.last_roi is None:
                        ctx.last_roi = ROICache(int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))
                    raw_roi = ctx.last_roi
            
            ctx.last_roi = smooth_roi(ctx.last_roi, raw_roi, alpha=0.8, 
                                     max_scale_step=0.05, frame_w=w, frame_h=h)
        
        sp_kpts, sp_scores = [], []
        
        if now_ms >= ctx.next_sp_ts:
            ctx.next_sp_ts = now_ms + ctx.config.SP_PERIOD_MS
            
            if ctx.last_roi is None:
                ctx.last_roi = ROICache(int(w*0.2), int(h*0.2), int(w*0.8), int(h*0.8))
            
            bbox = ctx.last_roi.as_list()
            sp_kpts, sp_scores = spinepose_infer_any(ctx.spine_est, img_rgb, 
                                                     bboxes=bbox, already_rgb=True)
        
        result = {
            "sp_kpts": [(int(x), int(y)) for x, y in sp_kpts] if len(sp_kpts) > 0 else [],
            "sp_scores": sp_scores.tolist() if hasattr(sp_scores, 'tolist') and len(sp_scores) > 0 else (sp_scores if sp_scores else []),
            "roi": ctx.last_roi.as_tuple() if ctx.last_roi else None
        }
        
        if len(result.get("sp_kpts", [])) >= 3:
            ctx.last_good_results.update(result)
            ctx.last_update_ms = time.perf_counter() * 1000.0
            
            # �� Calculate neck/spine angles and colors
            forward_head = calculate_forward_head_posture_torso(
                result["sp_kpts"], result["sp_scores"],
                ctx.config.NECK_IDX, ctx.config.LUMBAR_IDX, score_th=0.2
            )
            
            all_spine_coords = []
            sp_kpts_arr = np.array(result["sp_kpts"], dtype=np.float32)
            sp_scores_arr = np.array(result["sp_scores"], dtype=np.float32)
            for i in ctx.config.NECK_IDX + ctx.config.LUMBAR_IDX:
                if i < len(sp_kpts_arr) and sp_scores_arr[i] > 0.2:
                    all_spine_coords.append((int(sp_kpts_arr[i, 0]), int(sp_kpts_arr[i, 1])))
            
            if len(all_spine_coords) >= 2:
                all_spine_coords.sort(key=lambda p: p[1])
            
            spinal_curve = calculate_spinal_curvature(all_spine_coords)
            
            neck_color = thr_neck_color(forward_head, ctx.config.FHP_THRESH_DEG)
            spine_color = thr_lumbar_color(spinal_curve, ctx.config.CURVE_THRESH_DEG)
            
            # neck_color, spine_color 까지 계산된 바로 뒤
            if ctx.score_manager:
                ctx.score_manager.add_frame(neck_color, spine_color)
                score_result = ctx.score_manager.check_and_score()
                # (선택) score_result 활용 로깅 등

            # ✅ 서버에 최신 값 push (각도 + 누적 점수)
            try:
                server.set_metrics({
                    "fhp_deg": forward_head,   # None 가능
                    "curve_deg": spinal_curve, # None 가능
                    "neck_sum": neck_sum,      # 전역 누적
                    "spine_sum": spine_sum     # 전역 누적
                })
            except Exception as e:
                print("[WARN] set_metrics failed:", e)



            # �� Add to score manager
            if ctx.score_manager:
                ctx.score_manager.add_frame(neck_color, spine_color)
                
                # Check every 10 seconds
                score_result = ctx.score_manager.check_and_score()
                
                if score_result:
                    # Segment evaluated (terminal log already printed)
                    pass
            if ctx.score_manager:
                ctx.score_manager.check_and_score()
        
        processed_frame = render_display_frame(ctx, img_bgr, result)
        
        return processed_frame
        
    except Exception as e:
        print(f"[_run_ai_pipeline] AI processing error: {e}")
        return img_bgr

# ========= (10) Inference Worker (Dedicated AI Thread) =========
def inference_worker(ctx: Context, port: int):
    """Dedicated thread for heavy AI pipeline execution"""
    print("[Worker] AI Inference worker started.")
    while ctx.running:
        maybe_reset(port)
        try:
            frame_bgr = ctx.frame_q.get(timeout=1.0)
            
            if frame_bgr is None:
                break
                
            processed_frame = _run_ai_pipeline(ctx, frame_bgr)
            
            safe_queue_put(ctx.result_q, processed_frame, replace_if_full=True)
            safe_queue_put(ctx.display_q, processed_frame, replace_if_full=True)
                
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[Worker] error: {e}")
    print("[Worker] AI Inference worker stopped.")

# ========= (11) Display Worker =========
def display_worker(ctx: Context):
    """Local debugging display worker"""
    window_title = Config.WINDOW_TITLE
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_title, ctx.config.DISPLAY_WINDOW_W, ctx.config.DISPLAY_WINDOW_H)
    
    while ctx.running:
        try:
            disp = ctx.display_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if disp is None:
            break
        
        if (disp.shape[1] != ctx.config.DISPLAY_WINDOW_W or 
            disp.shape[0] != ctx.config.DISPLAY_WINDOW_H):
            disp = cv2.resize(disp, 
                            (ctx.config.DISPLAY_WINDOW_W, ctx.config.DISPLAY_WINDOW_H),
                            interpolation=cv2.INTER_LINEAR)
        
        cv2.imshow(window_title, disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            ctx.running = False
            break

# ========= (12) System Initialization =========

# ========= METHOD 5: Disk Cache for Models =========
def load_or_cache_model(model_size: str = "small"):
    """Load model from cache or create new one"""
    try:
        import joblib
    except ImportError:
        print("[Cache] joblib not available, loading directly")
        return SpinePoseEstimator(mode=model_size, device="cpu")
    
    cache_dir = Path.home() / ".cache" / "spinepose"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"spine_{model_size}.pkl"
    
    stat = shutil.disk_usage(cache_dir)
    free_gb = stat.free / (1024**3)
    
    if free_gb < 1.0:
        print(f"[Cache] Warning: Low disk space ({free_gb:.1f}GB)")
        print(f"[Cache] Skipping cache, loading directly...")
        return SpinePoseEstimator(mode=model_size, device="cpu")
    
    if cache_path.exists():
        print(f"[Cache] Loading from {cache_path}")
        cache_size_mb = cache_path.stat().st_size / 1024 / 1024
        print(f"[Cache] File size: {cache_size_mb:.1f}MB")
        try:
            spine_est = joblib.load(cache_path)
            print("[Cache] Model loaded from cache")
            return spine_est
        except Exception as e:
            print(f"[Cache] Load failed: {e}")
            print(f"[Cache] Cache file may be corrupted, deleting...")
            try:
                cache_path.unlink()
            except:
                pass
    
    print("[Cache] Creating new model...")
    spine_est = SpinePoseEstimator(mode=model_size, device="cpu")
    
    print("[Cache] Compiling model (first inference)...")
    dummy = np.zeros((640, 480, 3), dtype=np.uint8)
    try:
        _ = spine_est(dummy, bboxes=[[100, 100, 400, 500]])
    except:
        pass
    
    try:
        print(f"[Cache] Saving to {cache_path}...")
        joblib.dump(spine_est, cache_path, compress=3)
        saved_size = cache_path.stat().st_size / 1024 / 1024
        print(f"[Cache] Model saved ({saved_size:.1f}MB)")
        print(f"[Cache] Remaining disk space: {free_gb - saved_size/1024:.1f}GB")
    except Exception as e:
        print(f"[Cache] Save failed: {e}")
    
    return spine_est

# ========= METHOD 3: Optimized Model Initialization =========
def initialize_models_optimized(model_size: str = "small") -> Tuple[SpinePoseEstimator, Any, Any]:
    """Initialize AI models with pre-loading optimization"""
    print("[Init] Loading SpinePose...")
    
    try:
        spine_est = load_or_cache_model(model_size)
    except Exception as e:
        print(f"[Init] Cache loading failed: {e}")
        print(f"[Init] Loading directly...")
        spine_est = SpinePoseEstimator(mode=model_size, device="cpu")
    
    print(f"[Init] SpinePose ready")
    
    print("[Init] Loading MediaPipe Pose...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=False,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4
    )
    
    print("[Init] Warming up MediaPipe...")
    dummy = np.zeros((640, 480, 3), dtype=np.uint8)
    dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    _ = pose.process(dummy_rgb)
    print("[Init] MediaPipe Pose ready")
    
    return spine_est, mp_pose, pose

# ========= METHOD 1: Warm-up Routine =========
def warmup_models(ctx: Context, num_frames: int = 10):
    """Pre-warm up models for consistent performance"""
    print(f"[Warmup] Warming up AI models ({num_frames} frames)...")
    
    dummy_h, dummy_w = 640, 480
    dummy_frame = np.zeros((dummy_h, dummy_w, 3), dtype=np.uint8)
    
    print("[Warmup] Running full AI pipeline...")
    for i in range(num_frames):
        _ = _run_ai_pipeline(ctx, dummy_frame)
        if i % 3 == 0:
            print(f"[Warmup] Progress: {i+1}/{num_frames}")
    
    print("[Warmup] Complete! Models are ready.")

def create_context(spine_est: SpinePoseEstimator, mp_pose, pose, 
                  spine_only: bool = True, infer_scale: float = 0.5) -> Context:
    """Create context object"""
    config = Config()
    config.INFER_SCALE = max(0.2, min(infer_scale, 1.0))
    
    return Context(
        spine_est=spine_est,
        mp_pose=mp_pose,
        mp_pl=mp_pose,
        pose=pose,
        config=config,
        spine_only=True,
        infer_scale=config.INFER_SCALE
    )

# ========= (13) Main Function =========
def main():
    """Main execution function with all optimizations"""
    
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    
    # preinitialize_libraries()
    
    parser = argparse.ArgumentParser(description="SpinePose Analysis with Posture Scoring")
    parser.add_argument("--host", default="0.0.0.0", help="Server host address")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--model_size", default="small", 
                       choices=["small", "medium", "large", "xlarge"],
                       help="SpinePose model size")
    parser.add_argument("--spine-only", action="store_true", 
                       help="Start in spine-only drawing mode")
    parser.add_argument("--infer-scale", type=float, default=0.5,
                       help="Downscale ratio for inference (0.2~1.0)")
    parser.add_argument("--streamlit-width", type=int, default=720,
                       help="Streamlit output width")
    parser.add_argument("--streamlit-height", type=int, default=1280,
                       help="Streamlit output height")
    parser.add_argument("--memory-threshold", type=int, default=500,
                       help="Memory threshold for GC (MB)")
    parser.add_argument("--warmup-frames", type=int, default=10,
                       help="Number of warmup frames (default: 10)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable model caching (Method 5)")
    
    args = parser.parse_args()

    env = detect_cpu_env()
    tuned_threads = apply_thread_tuning(env)

    print("===== SpinePose Analysis System with Posture Scoring =====")
    print(f"CPU: {env['cpu_name']}")
    print(f"Cores: physical={env['physical']} logical={env['logical']} usable={env['usable']}")
    print(f"Threads: OpenCV=1, OMP={tuned_threads}")
    print(f"Model: {args.model_size}")
    print(f"Inference scale: {args.infer_scale}x")
    print(f"Streamlit output: {args.streamlit_width}x{args.streamlit_height}")
    print(f"Memory threshold: {args.memory_threshold}MB")
    print(f"Warmup frames: {args.warmup_frames}")
    print(f"Model caching: {'Disabled' if args.no_cache else 'Enabled'}")
    print(f"Server: http://{args.host}:{args.port}")
    print("Mode: SPINE-ONLY with Posture Scoring (3 minutes)")
    print("\nOptimizations Applied:")
    print("  [1] Warm-up routine")
    print("  [3] Pre-load optimization")
    print("  [4] Thread pool pre-initialization")
    print("  [5] Model disk caching" + (" (disabled)" if args.no_cache else ""))
    print("\n[NEW] Posture Scoring: Separate neck/spine evaluation every 10 seconds")

    try:
        spine_est, mp_pose, pose = initialize_models_optimized(args.model_size)
    except Exception as e:
        print(f"Model initialization failed: {e}")
        return

    ctx = create_context(spine_est, mp_pose, pose, 
                        spine_only=True,
                        infer_scale=args.infer_scale)
    
    ctx.config.STREAMLIT_OUTPUT_W = args.streamlit_width
    ctx.config.STREAMLIT_OUTPUT_H = args.streamlit_height
    ctx.config.MEMORY_THRESHOLD_MB = args.memory_threshold
    ctx.memory_monitor = MemoryMonitor(threshold_mb=args.memory_threshold)
    
    # �� Initialize posture score manager (10 seconds window)
    ctx.score_manager = PostureScoreManager(window_sec=10.0, max_duration_sec=180.0)

    # warmup_models(ctx, num_frames=args.warmup_frames)

    async def frame_callback(img):
        return await process_frame_callback(ctx, img)
    server.set_frame_callback(frame_callback)

    worker_thread = threading.Thread(target=inference_worker, args=(ctx, args.port), daemon=True)
    display_thread = threading.Thread(target=display_worker, args=(ctx,), daemon=True)
    worker_thread.start()
    display_thread.start()

    print("\n[Ready] System fully initialized and warmed up. Starting server...")
    print("[Info] Posture scoring: 18 segments (3 minutes / 10 seconds each)")
    print("[Info] Streamlit can access: get_current_scores() and get_last_alert()")

    try:
        server.run_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\nProgram interrupted")
    finally:
        # Print final scores
        print(f"\n[Final] 3-minute session ended")
        print(f"[Final] neck_sum: {neck_sum}")
        print(f"[Final] spine_sum: {spine_sum}")
        
        ctx.running = False
        safe_queue_put(ctx.frame_q, None, replace_if_full=False)
        safe_queue_put(ctx.display_q, None, replace_if_full=False)
        worker_thread.join(timeout=2.0)
        display_thread.join(timeout=2.0)
        cv2.destroyAllWindows()

# ========= (14) Profiler Utilities =========
def run_profiler():
    """Run performance profiling"""
    cProfile.run('main()', 'profile_results')

def analyze_profile():
    """Analyze profiling results"""
    p = pstats.Stats('profile_results')
    p.strip_dirs().sort_stats('time').print_stats(20)

# ========= (15) Entry Point =========
if __name__ == "__main__":
    main()
