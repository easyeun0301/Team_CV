import streamlit as st
import cv2
import time
import threading
import requests
from typing import Optional
import numpy as np
import subprocess
import os
import atexit
from collections import deque
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
from datetime import datetime, timedelta

# 명령행 인자 파싱 함수 추가
def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="Dual Pose Analysis Streamlit App")
    parser.add_argument("--port", type=int, default=8081, 
                       help="Side view 서버 포트 (기본값: 8081)")
    
    # Streamlit이 실행될 때 추가되는 인자들 무시
    args, unknown = parser.parse_known_args()
    return args

# 상위 디렉토리를 sys.path에 추가해 로컬 모듈 임포트 가능하도록
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from front_view.front_view_utils import FrontViewAnalyzer # 정면

# 전역 스트림 매니저 (종료시 통계 출력용)
_global_stream_manager = None

def print_stats_on_exit():
    """프로그램 종료시 통계 출력"""
    global _global_stream_manager
    if _global_stream_manager and (_global_stream_manager.front_process_times or _global_stream_manager.side_process_times):
        print("\n" + "="*60, flush=True)
        print("모델 처리 시간 통계", flush=True)
        print("="*60, flush=True)
        
        # Front View 통계
        if _global_stream_manager.front_process_times:
            front_avg = sum(_global_stream_manager.front_process_times) / len(_global_stream_manager.front_process_times)
            front_min = min(_global_stream_manager.front_process_times)
            front_max = max(_global_stream_manager.front_process_times)
            front_success_rate = (len(_global_stream_manager.front_process_times) / _global_stream_manager.front_total_frames) * 100 if _global_stream_manager.front_total_frames > 0 else 0
            
            print(f"Front View (FaceMesh + Pose):", flush=True)
            print(f"   평균 처리 시간: {front_avg:.1f}ms", flush=True)
            print(f"   최소 처리 시간: {front_min:.1f}ms", flush=True)
            print(f"   최대 처리 시간: {front_max:.1f}ms", flush=True)
            print(f"   처리 성공률: {front_success_rate:.1f}% ({len(_global_stream_manager.front_process_times)}/{_global_stream_manager.front_total_frames})", flush=True)
        else:
            print("Front View: 처리된 프레임 없음", flush=True)
        
        print("", flush=True)
        
        # Side View 통계
        if _global_stream_manager.side_process_times:
            side_avg = sum(_global_stream_manager.side_process_times) / len(_global_stream_manager.side_process_times)
            side_min = min(_global_stream_manager.side_process_times)
            side_max = max(_global_stream_manager.side_process_times)
            side_success_rate = (len(_global_stream_manager.side_process_times) / _global_stream_manager.side_total_frames) * 100 if _global_stream_manager.side_total_frames > 0 else 0
            
            print(f"Side View (HTTP + SpinePose):", flush=True)
            print(f"   평균 처리 시간: {side_avg:.1f}ms", flush=True)
            print(f"   최소 처리 시간: {side_min:.1f}ms", flush=True)
            print(f"   최대 처리 시간: {side_max:.1f}ms", flush=True)
            print(f"   연결 성공률: {side_success_rate:.1f}% ({len(_global_stream_manager.side_process_times)}/{_global_stream_manager.side_total_frames})", flush=True)
        else:
            print("Side View: 연결된 프레임 없음", flush=True)
        
        print("="*60, flush=True)

# 종료 시 통계 출력 등록
atexit.register(print_stats_on_exit)

class OptimizedDualStreamManager:
    """최적화된 Front View(가로)와 Side View(세로) 관리 클래스"""
    
    def __init__(self, port=8081):
        # Front view 관련 (웹캠 - 가로)
        self.front_analyzer = FrontViewAnalyzer()
        self.front_cap = None
        self.front_running = False
        self.front_frame_buffer = deque(maxlen=1)
        self.front_lock = threading.Lock()
        self.front_thread = None
        self.front_fps = 0
        self.front_fps_counter = 0
        self.front_fps_start = time.time()
        
        # 모델 처리 시간 측정
        self.front_process_times = []
        self.front_total_frames = 0
        
        # Side view 관련 (HTTP 서버 - 세로)
        self.side_port = port
        self.side_running = False
        self.side_frame_buffer = deque(maxlen=1)
        self.side_lock = threading.Lock()
        self.side_thread = None
        self.side_server_process = None
        self.side_server_url = f"http://localhost:{port}/android/frame"
        self.side_status_url = f"http://localhost:{port}/android/status"
        self.side_fps = 0
        self.side_fps_counter = 0
        self.side_fps_start = time.time()
        
        # Side view 처리 시간 측정
        self.side_process_times = []
        self.side_total_frames = 0
        
        # 미리 할당된 결합 버퍼
        self.combined_buffer = np.zeros((480, 1280, 3), dtype=np.uint8)

        # ⭐ 자세 플래그 추가
        self.bad_posture_flag = False      # 고개 기울기
        self.shoulder_bad_flag = False     # 어깨 비대칭
        self.bad_posture_lock = threading.Lock()

    def start_front_view(self):
        """웹캠 기반 Front View 시작 (최적화)"""
        if self.front_running:
            return "Front View가 이미 실행 중입니다."
            
        self.front_cap = cv2.VideoCapture(0)
        if not self.front_cap.isOpened():
            return "웹캠을 열 수 없습니다!"
            
        # 최적화된 웹캠 설정
        self.front_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.front_cap.set(cv2.CAP_PROP_FPS, 60)
            
        self.front_running = True
        self.front_thread = threading.Thread(target=self._optimized_front_worker, daemon=True)
        self.front_thread.start()
        
        return "Front View 시작됨"
    
    def _optimized_front_worker(self):
        """최적화된 Front view 처리 워커 (논블로킹)"""
        while self.front_running and self.front_cap and self.front_cap.isOpened():
            self.front_cap.grab()
            ret, frame = self.front_cap.read()
            if not ret:
                continue
                
            # FPS 계산
            self.front_fps_counter += 1
            if self.front_fps_counter % 30 == 0:
                elapsed = time.time() - self.front_fps_start
                self.front_fps = 30 / elapsed if elapsed > 0 else 0
                self.front_fps_start = time.time()
            
            # 모델 처리 시간 측정 시작
            process_start = time.time()
            self.front_total_frames += 1
            
            # 논블로킹 AI 처리
            try:
                processed_frame, bad_flag = self.front_analyzer.analyze_frame(frame)
                
                # ⭐ 스레드 안전하게 플래그 업데이트
                with self.bad_posture_lock:
                    self.bad_posture_flag = bad_flag
                    self.shoulder_bad_flag = bool(self.front_analyzer.shoulder_bad_flag)
                
                process_time = (time.time() - process_start) * 1000
                self.front_process_times.append(process_time)
            except:
                processed_frame = frame
            
            # 단일 버퍼 업데이트
            with self.front_lock:
                self.front_frame_buffer.clear()
                self.front_frame_buffer.append(processed_frame)
    
    def start_side_view(self):
        """Side view HTTP 서버 시작 (최적화)"""
        if self.side_running:
            return "Side View가 이미 실행 중입니다."
            
        current_dir = os.path.dirname(os.path.abspath(__file__))
        side_view_path = os.path.join(current_dir, '..', 'side_view', 'run.py')
        
        if not os.path.exists(side_view_path):
            return f"side_view/run.py 파일을 찾을 수 없습니다: {side_view_path}"
        
        try:
            self.side_server_process = subprocess.Popen([
                sys.executable, side_view_path,
                '--host', '0.0.0.0',
                '--port', str(self.side_port)
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            for i in range(10):
                time.sleep(0.5)
                try:
                    response = requests.get(self.side_status_url, timeout=1)
                    if response.status_code == 200:
                        break
                except:
                    continue
            else:
                if self.side_server_process:
                    stdout, stderr = self.side_server_process.communicate(timeout=3)
                    print(f"서버 stdout: {stdout.decode()}")
                    print(f"서버 stderr: {stderr.decode()}")
                    self.side_server_process.terminate()
                    self.side_server_process = None
                return f"서버 시작 후 응답이 없습니다. 포트 {self.side_port}이 사용중인지 확인하세요."
            
            self.side_running = True
            self.side_thread = threading.Thread(target=self._optimized_side_worker, daemon=True)
            self.side_thread.start()
            
            return f"Side View 서버가 포트 {self.side_port}에서 성공적으로 시작되었습니다!"
        
        except Exception as e:
            return f"Side View 서버 시작 실패: {str(e)}"
    
    def _optimized_side_worker(self):
        """최적화된 Side view HTTP 클라이언트 워커"""
        consecutive_errors = 0
        
        while self.side_running:
            try:
                request_start = time.time()
                self.side_total_frames += 1
                
                response = requests.get(self.side_server_url, timeout=0.1)

                if response.status_code == 200:
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        process_time = (time.time() - request_start) * 1000
                        self.side_process_times.append(process_time)
                        
                        self.side_fps_counter += 1
                        if self.side_fps_counter % 30 == 0:
                            elapsed = time.time() - self.side_fps_start
                            self.side_fps = 30 / elapsed if elapsed > 0 else 0
                            self.side_fps_start = time.time()
                        
                        frame_resized = cv2.resize(frame, (320, 480))
                        
                        with self.side_lock:
                            self.side_frame_buffer.clear()
                            self.side_frame_buffer.append(frame_resized)
                        
                        consecutive_errors = 0
                
            except requests.exceptions.RequestException:
                consecutive_errors += 1
                if consecutive_errors >= 10:
                    self._create_side_error_frame()
                    time.sleep(1.0)
    
    def _create_side_error_frame(self):
        """Side view 연결 실패시 에러 프레임 생성"""
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Side View", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(error_frame, "Server Required", (120, 340), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        with self.side_lock:
            self.side_frame_buffer.clear()
            self.side_frame_buffer.append(error_frame)
    
    def get_front_frame(self) -> Optional[np.ndarray]:
        """Front view 최신 프레임 가져오기 (논블로킹)"""
        with self.front_lock:
            return self.front_frame_buffer[0] if self.front_frame_buffer else None
    
    def get_side_frame(self) -> Optional[np.ndarray]:
        """Side view 최신 프레임 가져오기 (논블로킹)"""
        with self.side_lock:
            return self.side_frame_buffer[0] if self.side_frame_buffer else None
    
    def stop(self):
        """모든 스트리밍 중지"""
        # Front view 정지
        self.front_running = False
        if self.front_cap:
            self.front_cap.release()
        if self.front_thread:
            self.front_thread.join(timeout=2.0)
            
        # Side view 정지
        self.side_running = False
        if self.side_thread:
            self.side_thread.join(timeout=2.0)
            
        # Side view 서버 프로세스 종료
        if self.side_server_process:
            try:
                self.side_server_process.terminate()
                self.side_server_process.wait(timeout=3)
            except:
                self.side_server_process.kill()
            self.side_server_process = None

@st.cache_resource
def get_optimized_stream_manager(_port):
    """최적화된 스트림 매니저 싱글톤"""
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = OptimizedDualStreamManager(port=_port)
    return _global_stream_manager

def plot_posture_graph(history):
    import matplotlib.pyplot as plt
    import numpy as np
    
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    except:
        plt.rcParams['axes.unicode_minus'] = False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    posture_configs = [
        {
            'name': '고개 기울기',
            'timestamps_key': 'head_timestamps',
            'scores_key': 'head_scores',
            'color': 'b',
            'marker': 'o',
            'edge_color': 'darkblue'
        },
        {
            'name': '어깨수평',
            'timestamps_key': 'shoulder_timestamps',
            'scores_key': 'shoulder_scores',
            'color': 'g',
            'marker': 's',
            'edge_color': 'darkgreen'
        },
        {
            'name': '거북목',
            'timestamps_key': 'neck_timestamps',
            'scores_key': 'neck_scores',
            'color': 'y',
            'marker': '^',
            'edge_color': 'gold'
        },
        {
            'name': '굽은 허리',
            'timestamps_key': 'spine_timestamps',
            'scores_key': 'spine_scores',
            'color': 'orange',
            'marker': 'v',
            'edge_color': 'darkorange'
        },
    ]
    
    for config in posture_configs:
        timestamps = np.array(history.get(config['timestamps_key'], [0]))
        scores = np.array(history.get(config['scores_key'], [50]))

        # 배열 크기 확인 및 보정 추가
        if len(timestamps) != len(scores):
            min_len = min(len(timestamps), len(scores))
            if min_len > 0:
                timestamps = timestamps[:min_len]
                scores = scores[:min_len]
            else:
                timestamps = np.array([0])
                scores = np.array([50])

        if len(timestamps) > 0 and len(scores) > 0:
            # 정렬 및 보간
            order = np.argsort(timestamps)
            timestamps = timestamps[order]
            scores = scores[order]
            interp_times = np.linspace(0, 180, 300)
            interp_scores = np.interp(interp_times, timestamps, scores)

            # 선 그리기
            ax.plot(
                interp_times,
                interp_scores,
                linestyle='-',
                color=config['color'],
                linewidth=2.5,
                label=config['name'],
                alpha=0.9
            )
            
            # 30초 간격 위치에 마커만 찍기
            marker_times = [0, 30, 60, 90, 120, 150, 180]
            marker_scores = np.interp(marker_times, timestamps, scores)
            
            ax.plot(
                marker_times, marker_scores,
                linestyle='None',             
                marker=config['marker'],      
                color=config['color'],        
                markersize=10,
                markeredgecolor=config['edge_color'],
                markeredgewidth=2
            )
    
    ax.set_xlabel('시간 (초)', fontsize=13, fontweight='bold')
    ax.set_ylabel('점수', fontsize=13, fontweight='bold')
    ax.set_title('[자세 요약 그래프]', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True, fancybox=True)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    ax.set_ylim([0, 60])
    ax.set_xlim([0, 180])
    ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def main():
    args = parse_args()
    port = args.port

    st.set_page_config(
        page_title="Optimized Dual Pose Analysis",
        layout="wide"
    )

    # ===== 세션 상태 초기화 =====
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'analysis_active' not in st.session_state:
        st.session_state.analysis_active = False
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None
    if 'analysis_duration' not in st.session_state:
        st.session_state.analysis_duration = 180  # 3분 = 180초
    if 'show_report' not in st.session_state: 
        st.session_state.show_report = False
    if 'score_history' not in st.session_state: # 1104 수정
        st.session_state.score_history = {
            'head_timestamps': [0],      
            'head_scores': [50],
            'shoulder_timestamps': [0],  
            'shoulder_scores': [50],
            'neck_timestamps': [0],      
            'neck_scores': [50],
            'spine_timestamps': [0],     
            'spine_scores': [50],
            'start_time': time.time()
        }
    
    st.title("바르게 살자 !")
    st.markdown("안녕하세요! 2025 D&X:W Conference Tech_CV팀 부스에 오신 걸 환영합니다 😊")
    st.markdown("<br><br>", unsafe_allow_html=True)

    stream_manager = get_optimized_stream_manager(port)
    
    # ===== 컨트롤 패널 =====
    col1, col2 = st.columns(2)
    message_placeholder = st.empty()
    
    with col1:
        if not st.session_state.streaming:
            if st.button("듀얼 스트리밍 시작", type="primary", use_container_width=True, key="start_everything"):
                front_result = stream_manager.start_front_view()
                side_result = stream_manager.start_side_view()
                
                if "시작됨" in front_result or "성공적으로" in side_result:
                    # ⭐ 고개 관련 초기화
                    st.session_state.prev_bad_flag = False
                    st.session_state.last_bad_alert_ts = 0.0
                    st.session_state.last_penalty_ts = 0.0
                    st.session_state.score = 50
                    st.session_state.last_score_update_ts = 0.0
                    
                    # ⭐ 어깨 관련 초기화
                    st.session_state.prev_shoulder_bad_flag = False
                    st.session_state.last_shoulder_alert_ts = 0.0
                    st.session_state.last_shoulder_penalty_ts = 0.0
                    st.session_state.shoulder_score = 50
                    st.session_state.last_shoulder_score_update_ts = 0.0

                    # 측면 점수 초기화
                    st.session_state.neck_score = 50
                    st.session_state.spine_score = 50
                    st.session_state.prev_neck_sum = 0
                    st.session_state.prev_spine_sum = 0

                    st.session_state.streaming = True
                    message_placeholder.success("듀얼 스트리밍이 성공적으로 시작되었습니다!")
                    st.rerun()
                else:
                    message_placeholder.error(f"시작 실패 - Front: {front_result}, Side: {side_result}")
        else:
            if st.button("스트리밍 정지", use_container_width=True, key="stop_everything"):
                stream_manager.stop()
                st.session_state.streaming = False
                st.session_state.analysis_active = False
                st.session_state.analysis_start_time = None
                st.session_state.show_report = False
                message_placeholder.warning("스트리밍 정지됨")
                st.rerun()
    
    with col2:
        st.write(f"상태: {'실행 중' if st.session_state.streaming else '정지'}")
    
    # ===== 알림 상수 =====
    BAD_ALERT_COOL_S = 3.0       # 알림 쿨다운 (초)
    PENALTY_INTERVAL_S = 10.0    # 지속 감점 간격 (초)
    SCORE_UPDATE_THROTTLE = 0.5  # UI 업데이트 쓰로틀 (초)
    
    # ===== 스트리밍 표시 =====
    if st.session_state.streaming:
        # ===== 리포트 화면 =====
        if st.session_state.show_report:
            st.markdown("## 🍀자세 분석 리포트")
            st.success("3분간의 분석이 완료되었습니다! 분석 결과를 확인해보세요!")

            # 2열 구성: 왼쪽(그래프), 오른쪽(점수 요약)
            col_left, col_right = st.columns([2, 1])

            with col_left:
                # 왼쪽에는 자세 변화 그래프
                plot_posture_graph(st.session_state.score_history)

            with col_right:
                st.markdown("### 📊 요약 점수")

                # 정면 총점
                front_total = (
                    st.session_state.get("score", 50)
                    + st.session_state.get("shoulder_score", 50)
                )
                st.metric("## 정면 총점", f"{front_total} / 100")

                # 측면 총점
                side_total = (
                    st.session_state.get("neck_score", 50)
                    + st.session_state.get("spine_score", 50)
                )
                st.metric("## 측면 총점", f"{side_total} / 100")

                # 전체 총점 (정면 + 측면)
                overall_total = front_total + side_total
                st.metric("## 전체 총점", f"{overall_total} / 200")
                
                # 🎮 테트리스 점수 입력
                st.markdown("---")
                st.markdown("### 🎮 최종 점수 계산")

                tetris_score = st.number_input(
                    "테트리스 점수를 입력하세요!",
                    min_value=0,
                    max_value=999999,
                    step=1000,
                    key="tetris_score_input"
                )

                # 최종 점수 계산: (정면 총점 × 측면 총점 × 테트리스 점수) × 0.01
                final_score = round(((front_total + side_total) * 0.01) * tetris_score)

                st.metric("🏁 최종 점수", f"{final_score:,}점")
                st.caption("계산식: ((정면 총점 + 측면 총점) × 테트리스 점수) × 0.01")

            # 점수 표시
            st.markdown("---")
            st.markdown("## 📊 자세 점수")

            ## 정면
            st.markdown("### 정면")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(label="얼굴 기울기 점수", value=f"{st.session_state.get('score', 50)} / 50")
                if st.session_state.get('score', 50) >= 40:
                    st.success("✅ 훌륭한 자세입니다!")
                elif st.session_state.get('score', 50) >= 30:
                    st.warning("⚠️ 자세 개선이 필요합니다.")
                else:
                    st.error("❌ 자세 교정이 시급합니다!")
            
            with col2:
                st.metric(label="어깨 균형 점수", value=f"{st.session_state.get('shoulder_score', 50)} / 50")
                if st.session_state.get('shoulder_score', 50) >= 40:
                    st.success("✅ 어깨 균형이 좋습니다!")
                elif st.session_state.get('shoulder_score', 50) >= 30:
                    st.warning("⚠️ 어깨 균형 개선이 필요합니다.")
                else:
                    st.error("❌ 어깨 균형 교정이 필요합니다!")
            
            with col3:
                # 🚩 정면 총 점수 계산
                total_front_score = st.session_state.get('score', 50) + st.session_state.get('shoulder_score', 50)
                st.metric(label="정면 총 점수", value=f"{total_front_score} / 100")

            ## 측면
            st.markdown("### 측면")
            col4, col5, col6 = st.columns(3)

            with col4:
                neck_score = st.session_state.get('neck_score', 0)
                st.metric(label="거북목 점수", value=f"{neck_score} / 50")
                if neck_score >= 40:
                    st.success("✅ 당신은 거북이가 아닙니다!")
                elif neck_score >= 30:
                    st.warning("⚠️ 거북목이 조금 있습니다. 곧 거북이가 될지도..? 🐢")
                else:
                    st.error("❌ 거북이님 안녕하세요! 🐢🐢")

            with col5:
                spine_score = st.session_state.get('spine_score', 0)
                st.metric(label="굽은 허리 점수", value=f"{spine_score} / 50")
                if spine_score >= 40:
                    st.success("✅ 허리 곡선이 정상적입니다!")
                elif spine_score >= 30:
                    st.warning("⚠️ 허리가 다소 굽었습니다.")
                else:
                    st.error("❌ 허리 교정이 필요합니다!")

            with col6:
                total_side_score = neck_score + spine_score
                st.metric(label="측면 총 점수", value=f"{total_side_score} / 100")

            # 최고점/최저점 자세 리포트
            st.markdown("---")
            st.markdown("## 🏆 자세 비교 리포트")

            # 각 자세별 점수 가져오기
            scores_dict = {
                "고개 기울기": st.session_state.get("score", 0),
                "어깨 수평": st.session_state.get("shoulder_score", 0),
                "거북목": st.session_state.get("neck_score", 0),
                "굽은 허리": st.session_state.get("spine_score", 0)
            }

            # 최고 / 최저 자세 판별 (동점 허용)
            max_score = max(scores_dict.values())
            min_score = min(scores_dict.values())

            best_postures = [k for k, v in scores_dict.items() if v == max_score]
            worst_postures = [k for k, v in scores_dict.items() if v == min_score]

            # posture_configs (그래프용 설정)
            posture_configs = {
                "고개 기울기": ("head_timestamps", "head_scores", "b", "o", "darkblue"),
                "어깨 수평": ("shoulder_timestamps", "shoulder_scores", "g", "s", "darkgreen"),
                "거북목": ("neck_timestamps", "neck_scores", "y", "^", "gold"),
                "굽은 허리": ("spine_timestamps", "spine_scores", "orange", "v", "darkorange"),
            }

            # 코멘트 템플릿
            best_comments = {
                "고개 기울기": "와우~! 고개 수평의 신이군요! 대단해요!",
                "어깨 수평": "와우~! 어깨 수평의 신이군요! 대단해요!",
                "거북목": "🐢 거북이가 아니군요! 훌륭합니다!",
                "굽은 허리": "척추 수술 2000만원은 영원히 아낄 수 있겠군요! 훌륭합니다!"
            }

            worst_comments = {
                "고개 기울기": "고민이 많으셨나요? 고개가 자주 기울어졌어요!",
                "어깨 수평": "테트리스가 너무 신이나 어깨를 자주 들썩이셨군요!",
                "거북목": "엉금엉금... 지금 거의 거북이에요!!!🐢",
                "굽은 허리": "척추 수술비 2000만원.... 있으세요...?"
            }

            def plot_single_posture_graph(name, history):
                """단일 자세 그래프 그리기"""
                key_t, key_s, color, marker, edge = posture_configs[name]
                timestamps = np.array(history.get(key_t, [0]))
                scores = np.array(history.get(key_s, [50]))
                min_len = min(len(timestamps), len(scores))
                timestamps, scores = timestamps[:min_len], scores[:min_len]
                order = np.argsort(timestamps)
                timestamps, scores = timestamps[order], scores[order]
                interp_times = np.linspace(0, 180, 300)
                interp_scores = np.interp(interp_times, timestamps, scores)

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.plot(interp_times, interp_scores, color=color, linewidth=3, label=name)
                ax.scatter(
                    [0, 30, 60, 90, 120, 150, 180],
                    np.interp([0, 30, 60, 90, 120, 150, 180], timestamps, scores),
                    color=color,
                    marker=marker,
                    edgecolor=edge,
                    s=60
                )
                ax.set_ylim(0, 60)
                ax.set_xlim(0, 180)
                ax.set_title(f"[{name} 점수 변화]", fontsize=13, fontweight="bold")
                ax.grid(True, alpha=0.3, linestyle="--")
                st.pyplot(fig)
                plt.close()

            st.markdown("### 🤩 가장 바른 자세")
            if len(best_postures) > 1:
                st.info(f"두 가지 이상의 자세가 동일한 최고 점수({max_score:.1f}점)를 기록했습니다!")

            for posture in best_postures:
                col_best_left, col_best_right = st.columns([1.5, 1])
                with col_best_left:
                    plot_single_posture_graph(posture, st.session_state.score_history)
                with col_best_right:
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; height:100%; min-height:180px;">
                            <div>
                                <div style="font-size:18px; font-weight:bold; color:#0f5132; margin-bottom:8px;">
                                    ✅ {posture}가(이) 평균적으로 가장 안정적이었습니다! ({max_score:.1f}점)
                                </div>
                                <div style="font-size:18px; color:#155724;">
                                    {best_comments[posture]}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            st.markdown("### 😫 개선이 필요한 자세")
            if len(worst_postures) > 1:
                st.warning(f"두 가지 이상의 자세가 동일한 최저 점수({min_score:.1f}점)를 기록했습니다!")

            for posture in worst_postures:
                col_worst_left, col_worst_right = st.columns([1.5, 1])
                with col_worst_left:
                    plot_single_posture_graph(posture, st.session_state.score_history)
                with col_worst_right:
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; height:100%; min-height:180px;">
                            <div>
                                <div style="font-size:18px; font-weight:bold; color:#842029; margin-bottom:8px;">
                                    ⚠️ {posture}가(이) 평균적으로 가장 낮은 점수를 기록했습니다. ({min_score:.1f}점)
                                </div>
                                <div style="font-size:18px; color:#5a1a1a;">
                                    {worst_comments[posture]}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # 눈깜빡임 리포트
            st.markdown("---")
            st.markdown("### 👀 눈 깜빡임 분석")
            
            col1, col2, col3 = st.columns(3)
            
            blink_total = (
                stream_manager.front_analyzer.total_blink_count
                + stream_manager.front_analyzer.blink_count
            )
            duration_minutes = st.session_state.analysis_duration / 60
            blinks_per_minute = blink_total / duration_minutes if duration_minutes > 0 else 0
            
            with col1:
                st.metric(label="총 깜빡임 수", value=f"{blink_total}회")
            
            with col2:
                st.metric(label="분당 깜빡임", value=f"{blinks_per_minute:.1f}회/분")
            
            with col3:
                st.markdown("##### 📊 분석 결과")
                if blinks_per_minute >= 15:
                    st.info("눈 깜빡임이 정상 범위입니다 :) \n\n1분에 15~20회/분 깜빡여야 눈의 피로를 줄일 수 있습니다!")
                else:
                    st.error("눈 깜빡임이 적습니다 :( \n\n1분에 15~20회/분 깜빡여야 눈의 피로를 줄일 수 있습니다!")
            
            # 새로운 분석 시작 버튼
            if st.button("🔄 새로운 분석 시작", type="primary", use_container_width=True):
                stream_manager.stop()
                st.session_state.streaming = False
                st.session_state.analysis_active = False
                st.session_state.analysis_start_time = None
                st.session_state.show_report = False
                st.rerun()
            return
        
        st.markdown("### MISSION : 바른 자세로 테트리스하기!")
        st.markdown("다양한 옵션 버튼들을 통해 설정값을 조정해보세요 :) \n\n본인에게 맞는 바른 자세를 파악하신 후 '분석 시작' 버튼을 클릭하면 자세 분석이 시작됩니다!")
       
        # 분석 시작 버튼
        button_area = st.empty()

        if not st.session_state.analysis_active:
            with button_area.container():
                if st.button("⏰분석 시작 (3분)", type="primary", use_container_width=True):
                    st.session_state.analysis_active = True
                    st.session_state.analysis_start_time = time.time()
                    
                    # ⭐ 점수/상태 초기화 (고개 + 어깨)
                    st.session_state.score = 50
                    st.session_state.shoulder_score = 50
                    
                    # 측면 점수/상태 초기화
                    st.session_state.neck_score = 50
                    st.session_state.spine_score = 50
                    
                    # 서버의 현재 누적값을 기준점으로 설정
                    try:
                        SIDE_BASE = f"http://localhost:{stream_manager.side_port}"
                        r = requests.get(f"{SIDE_BASE}/android/metrics", timeout=0.5)
                        if r.ok:
                            m = r.json()
                            st.session_state.prev_neck_sum = m.get("neck_sum", 0)
                            st.session_state.prev_spine_sum = m.get("spine_sum", 0)
                        else:
                            st.session_state.prev_neck_sum = 0
                            st.session_state.prev_spine_sum = 0
                    except:
                        st.session_state.prev_neck_sum = 0
                        st.session_state.prev_spine_sum = 0

                    # 그래프 히스토리 초기화 - 1104 수정
                    st.session_state.score_history = {
                        'head_timestamps': [0],      
                        'head_scores': [50],
                        'shoulder_timestamps': [0],  
                        'shoulder_scores': [50],
                        'neck_timestamps': [0],
                        'neck_scores': [50],
                        'spine_timestamps': [0],
                        'spine_scores': [50],
                        'start_time': time.time()
                    }

                    st.session_state.prev_bad_flag = False
                    st.session_state.prev_shoulder_bad_flag = False

                    st.session_state.last_bad_alert_ts = 0.0
                    st.session_state.last_penalty_ts = 0.0
                    st.session_state.last_shoulder_alert_ts = 0.0
                    st.session_state.last_shoulder_penalty_ts = 0.0

                    st.session_state.last_score_update_ts = 0.0
                    st.session_state.last_shoulder_score_update_ts = 0.0

                    # 눈 깜빡임 리셋
                    stream_manager.front_analyzer.blink_count = 0
                    stream_manager.front_analyzer.total_blink_count = 0   # ⭐ 추가
                    stream_manager.front_analyzer.win_start = time.time()

                    button_area.empty()
                    st.rerun()

        # 한 행: Front / Side / 옵션+점수
        col_front, col_side, col_option = st.columns([1, 1, 1])
        
        front_placeholder = col_front.empty()
        side_placeholder = col_side.empty()

        # 정면 영상 아래 점수 박스
        front_score_box = col_front.container()

        # metric 크기 조정용 CSS 추가
        st.markdown("""
            <style>
            [data-testid="stMetricValue"] {
                font-size: 22px !important;  /* 기본값은 약 36px */
            }
            [data-testid="stMetricLabel"] {
                font-size: 14px !important;  /* 라벨 글자 */
            }
            </style>
        """, unsafe_allow_html=True)

        score_title_ph = front_score_box.markdown("#### 📊 현재 점수")

        # 2열로 나눔 — 왼쪽: 정면 / 오른쪽: 측면
        col_front_scores, col_side_scores = front_score_box.columns(2)

        # 왼쪽 (정면)
        with col_front_scores:
            st.markdown("🎯 정면")
            head_score_ph = st.empty()
            shoulder_score_ph = st.empty()
            # 기존 점수 렌더 유지
            head_score_ph.metric("얼굴 기울기", f"{st.session_state.get('score', 50)}/50")
            shoulder_score_ph.metric("어깨 균형", f"{st.session_state.get('shoulder_score', 50)}/50")

        # 오른쪽 (측면)
        with col_side_scores:
            st.markdown("🎯 측면")
            side_score_ph = st.empty()
            side_score2_ph = st.empty()
            # ✅ 초기 렌더링에서 기본 점수 표시
            st.session_state.setdefault("neck_score", 50)
            st.session_state.setdefault("spine_score", 50)
            side_score_ph.metric("거북목", f"{st.session_state.neck_score} / 50")
            side_score2_ph.metric("굽은 허리", f"{st.session_state.spine_score} / 50")

        front_img = None
        side_img = None

        # ===== 옵션 및 점수 표시 =====
        with col_option:
            st.markdown("### 💻 정면캠 옵션")

            # 옵션 설정
            with st.container():
                st.markdown("#### 옵션 설정")

                colA, colB = st.columns(2)

                if colA.button("판독 감도 조절", key="thr_btn_once"):
                    stream_manager.front_analyzer.cycle_threshold_profile(+1)

                if colA.button("Key Points 표시", key="pts_btn_once"):
                    stream_manager.front_analyzer.SHOW_POINTS = not stream_manager.front_analyzer.SHOW_POINTS
                
                if colA.button("눈 깜빡임 보정", key="ear_btn_once"):
                    if len(stream_manager.front_analyzer.ear_window) >= 10:
                        arr = np.array(stream_manager.front_analyzer.ear_window, dtype=np.float32)
                        med = float(np.median(arr))
                        p10 = float(np.percentile(arr, 10))
                        stream_manager.front_analyzer.T_LOW = max(0.08, min(med * 0.75, p10 + 0.02))
                        stream_manager.front_analyzer.T_HIGH = max(stream_manager.front_analyzer.T_LOW + 0.02, med * 0.92)
                        stream_manager.front_analyzer.calibrated = True
                    else:
                        st.toast("⚠️ EAR 데이터가 부족합니다")

                if colB.button("명암 대비 조정", key="clahe_btn_once"):
                    stream_manager.front_analyzer.use_clahe = not stream_manager.front_analyzer.use_clahe

                if colB.button("투명도 ↑", key="alpha_up_once"):
                    stream_manager.front_analyzer.ALPHA = min(1.0, stream_manager.front_analyzer.ALPHA + 0.1)
                if colB.button("투명도 ↓", key="alpha_dn_once"):
                    stream_manager.front_analyzer.ALPHA = max(0.1, stream_manager.front_analyzer.ALPHA - 0.1)

            # ⭐ 점수 표시
            #st.markdown("---")
            #st.markdown("### 📊 현재 점수")
            #score_placeholder = st.empty()
            #score_placeholder2 = st.empty()
            
            #score_placeholder.metric("얼굴 기울기", f"{st.session_state.get('score', 35)}/35")
            #score_placeholder2.metric("어깨 균형", f"{st.session_state.get('shoulder_score', 35)}/35")

            # 정면 점수 초기 렌더
            #score_title_ph.markdown("### 📊 현재 점수")
            #head_score_ph.metric("얼굴 기울기", f"{st.session_state.get('score', 35)}/35")
            #shoulder_score_ph.metric("어깨 균형", f"{st.session_state.get('shoulder_score', 35)}/35")

            # 상태 요약
            st.markdown("---")
            st.markdown("#### 설정 상태")

            analyzer = stream_manager.front_analyzer
            col_status1, col_status2 = st.columns(2)

            with col_status1:
                st.metric("판독 감도", analyzer.THR_PROFILES[analyzer.thr_profile_idx][1])
                st.metric("Key Points", "ON" if analyzer.SHOW_POINTS else "OFF")

            with col_status2:
                st.metric("명암 조정", "ON" if analyzer.use_clahe else "OFF")
                st.metric("투명도", f"{analyzer.ALPHA:.1f}")

        # ===== 스트리밍 루프 =====
        start_time = time.time()

        while st.session_state.streaming and (time.time() - start_time) < 6000:
            # 3분 경과 시 리포트 화면 전환
            if st.session_state.analysis_active and st.session_state.analysis_start_time:
                elapsed = time.time() - st.session_state.analysis_start_time
                
                if elapsed >= st.session_state.analysis_duration:
                    # 리포트 전환 직전 마지막 측면 점수 저장
                    elapsed = time.time() - st.session_state.score_history['start_time']
                    st.session_state.score_history['neck_timestamps'].append(elapsed)
                    st.session_state.score_history['neck_scores'].append(st.session_state.get('neck_score', 0))
                    st.session_state.score_history['spine_timestamps'].append(elapsed)
                    st.session_state.score_history['spine_scores'].append(st.session_state.get('spine_score', 0))

                    st.session_state.analysis_active = False
                    st.session_state.show_report = True
                    stream_manager.stop()
                    st.rerun()

            # 프레임 표시
            front_frame = stream_manager.get_front_frame()
            side_frame = stream_manager.get_side_frame()
            
            if front_frame is not None:
                front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)
                if front_img is None:
                    front_img = front_placeholder.image(front_rgb, channels="RGB", width=640)
                else:
                    front_img.image(front_rgb, channels="RGB", width=640)
            else:
                front_placeholder.text("Front AI Loading...")

            if side_frame is not None:
                side_rgb = cv2.cvtColor(side_frame, cv2.COLOR_BGR2RGB)
                if side_img is None:
                    side_img = side_placeholder.image(side_rgb, channels="RGB", width=480)
                else:
                    side_img.image(side_rgb, channels="RGB", width=480)
            else:
                side_placeholder.text("Side AI Loading...")
            
            # ===== 자세 분석 및 알림 =====
            with stream_manager.bad_posture_lock:
                cur_bad = bool(stream_manager.bad_posture_flag)
                cur_shoulder_bad = bool(stream_manager.shoulder_bad_flag)
            
            now = time.time()
            prev = bool(st.session_state.get('prev_bad_flag', False))
            prev_sh = bool(st.session_state.get('prev_shoulder_bad_flag', False))
            
            # ===== (1) 고개 기울기 알림 (False→True 전이) =====
            if (not prev) and cur_bad and (now - st.session_state.get('last_bad_alert_ts', 0.0) >= BAD_ALERT_COOL_S):
                # 점수 차감
                st.session_state.score = max(0, st.session_state.get('score', 50) - 1)

                # 히스토리 업데이트 - 1104
                elapsed = now - st.session_state.score_history['start_time']
                st.session_state.score_history['head_timestamps'].append(elapsed)
                st.session_state.score_history['head_scores'].append(st.session_state.score)

                st.session_state.last_bad_alert_ts = now
                st.session_state.last_penalty_ts = now
                
                # 토스트 알림
                st.toast("⚠️ 5초 이상 고개 기울기 감지! 바르게 앉으세요.")
                
                # ⭐ 간소화된 TTS (조건문 안에서만 실행)
                st.components.v1.html("""
                    <script>
                    (function(){
                        const s = (window.top && window.top.speechSynthesis) || window.speechSynthesis;
                        const U = (window.top && window.top.SpeechSynthesisUtterance) || SpeechSynthesisUtterance;
                        if (s && U) {
                            const u = new U("5초 이상 고개가 기울어졌습니다.");
                            u.lang = "ko-KR";
                            u.rate = 1.2;
                            u.pitch = 1.5;
                            s.cancel();
                            s.speak(u);
                        }
                    })();
                    </script>
                """, height=0)
            
            # ===== (2) 고개 기울기 지속 감점 (10초마다) =====
            if cur_bad and prev:
                if now - st.session_state.get('last_penalty_ts', now) >= PENALTY_INTERVAL_S:
                    st.session_state.score = max(0, st.session_state.get('score', 50) - 2)

                    # 히스토리 업데이트 추가 - 1104
                    elapsed = now - st.session_state.score_history['start_time']
                    st.session_state.score_history['head_timestamps'].append(elapsed)
                    st.session_state.score_history['head_scores'].append(st.session_state.score)

                    st.session_state.last_penalty_ts = now
                    st.toast(f"⏱ 지속 불량 자세: -2점 (현재 {st.session_state.score}점)")
            
            # ===== (3) 어깨 비대칭 알림 (False→True 전이) =====
            if (not prev_sh) and cur_shoulder_bad and (now - st.session_state.get('last_shoulder_alert_ts', 0.0) >= BAD_ALERT_COOL_S):
                st.session_state.shoulder_score = max(0, st.session_state.get('shoulder_score', 50) - 2)
                
                # 히스토리 업데이트 추가 - 1106
                elapsed = now - st.session_state.score_history['start_time']
                st.session_state.score_history['shoulder_timestamps'].append(elapsed)
                st.session_state.score_history['shoulder_scores'].append(st.session_state.shoulder_score)

                st.session_state.last_shoulder_alert_ts = now
                st.session_state.last_shoulder_penalty_ts = now
                st.toast(f"⚠️ 어깨 비대칭 10초 지속: -2점 (현재 {st.session_state.shoulder_score}점)")
                
                # TTS
                st.components.v1.html("""
                    <script>
                    (function(){
                        const s = (window.top && window.top.speechSynthesis) || window.speechSynthesis;
                        const U = (window.top && window.top.SpeechSynthesisUtterance) || SpeechSynthesisUtterance;
                        if (s && U) {
                            const u = new U("10초 이상 어깨가 기울어졌습니다.");
                            u.lang = "ko-KR";
                            u.rate = 1.2;
                            u.pitch = 1.5;
                            s.cancel();
                            s.speak(u);
                        }
                    })();
                    </script>
                """, height=0)
            
            # ===== (4) 어깨 비대칭 지속 감점 (10초마다) =====
            if cur_shoulder_bad and prev_sh:
                if now - st.session_state.get('last_shoulder_penalty_ts', now) >= PENALTY_INTERVAL_S:
                    st.session_state.shoulder_score = max(0, st.session_state.get('shoulder_score', 50) - 2)
                    
                    # 히스토리 업데이트 추가 - 1104
                    elapsed = now - st.session_state.score_history['start_time']
                    st.session_state.score_history['shoulder_timestamps'].append(elapsed)
                    st.session_state.score_history['shoulder_scores'].append(st.session_state.shoulder_score)

                    st.session_state.last_shoulder_penalty_ts = now
                    st.toast(f"⏱ 어깨 비대칭 지속: -2점 (현재 {st.session_state.shoulder_score}점)")
            
            # ===== (5) 점수 UI 업데이트 (쓰로틀링) =====
            if now - st.session_state.get('last_score_update_ts', 0.0) >= SCORE_UPDATE_THROTTLE:
                head_score_ph.metric("얼굴 기울기", f"{st.session_state.score}/50")
                shoulder_score_ph.metric("어깨 균형", f"{st.session_state.shoulder_score}/50")
                st.session_state.last_score_update_ts = now
                        
            # 이전 상태 갱신
            st.session_state.prev_bad_flag = cur_bad
            st.session_state.prev_shoulder_bad_flag = cur_shoulder_bad

            # 측면 점수 가져오기 (10Hz 이하 주기)
            SIDE_BASE = f"http://localhost:{stream_manager.side_port}"

            # 분석 시작 직후 1초 동안 metrics 업데이트 잠시 무시
            if st.session_state.analysis_active and (time.time() - st.session_state.analysis_start_time) < 1.0:
                time.sleep(0.001)
                continue

            if time.time() - st.session_state.get("last_side_metrics_ts", 0.0) >= 0.5:
                try:
                    r = requests.get(f"{SIDE_BASE}/android/metrics", timeout=0.4)
                    if r.ok:
                        m = r.json()
                        
                        neck_sum = m.get("neck_sum", 0)
                        spine_sum = m.get("spine_sum", 0)

                        # 분석 시작 기준점 대비 증가분만 감점
                        delta_neck = max(0, neck_sum - st.session_state.get("prev_neck_sum", 0))
                        delta_spine = max(0, spine_sum - st.session_state.get("prev_spine_sum", 0))

                        if st.session_state.analysis_active:
                            st.session_state.neck_score = max(0, 50 - delta_neck)
                            st.session_state.spine_score = max(0, 50 - delta_spine)

                            # 그래프 히스토리 업데이트
                            elapsed = now - st.session_state.score_history['start_time']
                            st.session_state.score_history['neck_timestamps'].append(elapsed)
                            st.session_state.score_history['neck_scores'].append(st.session_state.neck_score)
                            st.session_state.score_history['spine_timestamps'].append(elapsed)
                            st.session_state.score_history['spine_scores'].append(st.session_state.spine_score)

                        # 분석 중이든 아니든 metric은 항상 표시
                        side_score_ph.metric("거북목", f"{st.session_state.neck_score} / 50")
                        side_score2_ph.metric("굽은 허리", f"{st.session_state.spine_score} / 50")

                        st.session_state["last_side_metrics_ts"] = time.time()
                except Exception as e:
                    print("[WARN] side metrics fetch failed:", e)

            # CPU 양보
            time.sleep(0.001)
        
    else:
        st.info("모든 준비가 완료되었다면, '듀얼 스트리밍 시작' 버튼을 클릭해주세요!")

if __name__ == "__main__":
    main()
