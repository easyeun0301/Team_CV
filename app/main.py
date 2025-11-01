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
                
                if "시작됨" in front_result and "성공적으로" in side_result:
                    # ⭐ 고개 관련 초기화
                    st.session_state.prev_bad_flag = False
                    st.session_state.last_bad_alert_ts = 0.0
                    st.session_state.last_penalty_ts = 0.0
                    st.session_state.score = 35
                    st.session_state.last_score_update_ts = 0.0
                    
                    # ⭐ 어깨 관련 초기화
                    st.session_state.prev_shoulder_bad_flag = False
                    st.session_state.last_shoulder_alert_ts = 0.0
                    st.session_state.last_shoulder_penalty_ts = 0.0
                    st.session_state.shoulder_score = 35
                    st.session_state.last_shoulder_score_update_ts = 0.0

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
            
            # 점수 표시
            st.markdown("---")
            st.markdown("### 📊 자세 점수")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(label="얼굴 기울기 점수", value=f"{st.session_state.get('score', 35)} / 35")
                if st.session_state.get('score', 35) >= 30:
                    st.success("✅ 훌륭한 자세입니다!")
                elif st.session_state.get('score', 35) >= 20:
                    st.warning("⚠️ 자세 개선이 필요합니다.")
                else:
                    st.error("❌ 자세 교정이 시급합니다!")
            
            with col2:
                st.metric(label="어깨 균형 점수", value=f"{st.session_state.get('shoulder_score', 35)} / 35")
                if st.session_state.get('shoulder_score', 35) >= 30:
                    st.success("✅ 어깨 균형이 좋습니다!")
                elif st.session_state.get('shoulder_score', 35) >= 20:
                    st.warning("⚠️ 어깨 균형 개선이 필요합니다.")
                else:
                    st.error("❌ 어깨 균형 교정이 필요합니다!")
            
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
        
        st.markdown("### Front_view + Side_view")
        st.markdown("다양한 옵션 버튼들을 통해 설정값을 조정해보세요 :) \n\n본인에게 맞는 바른 자세를 파악하신 후 '분석 시작' 버튼을 클릭하면 자세 분석이 시작됩니다!")
       
        # 분석 시작 버튼
        button_area = st.empty()

        if not st.session_state.analysis_active:
            with button_area.container():
                if st.button("⏰분석 시작 (3분)", type="primary", use_container_width=True):
                    st.session_state.analysis_active = True
                    st.session_state.analysis_start_time = time.time()
                    
                    # ⭐ 점수/상태 초기화 (고개 + 어깨)
                    st.session_state.score = 35
                    st.session_state.shoulder_score = 35

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
        score_title_ph = front_score_box.markdown("### 📊 현재 점수")
        head_score_ph = front_score_box.empty()
        shoulder_score_ph = front_score_box.empty()

        front_img = None
        side_img = None

        # ===== 옵션 및 점수 표시 =====
        with col_option:
            st.markdown("### Front View 옵션")

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
            score_title_ph.markdown("### 📊 현재 점수")
            head_score_ph.metric("얼굴 기울기", f"{st.session_state.get('score', 35)}/35")
            shoulder_score_ph.metric("어깨 균형", f"{st.session_state.get('shoulder_score', 35)}/35")

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
                st.session_state.score = max(0, st.session_state.get('score', 35) - 1)
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
                    st.session_state.score = max(0, st.session_state.get('score', 35) - 1)
                    st.session_state.last_penalty_ts = now
                    st.toast(f"⏱ 지속 불량 자세: -1점 (현재 {st.session_state.score}점)")
            
            # ===== (3) 어깨 비대칭 알림 (False→True 전이) =====
            if (not prev_sh) and cur_shoulder_bad and (now - st.session_state.get('last_shoulder_alert_ts', 0.0) >= BAD_ALERT_COOL_S):
                st.session_state.shoulder_score = max(0, st.session_state.get('shoulder_score', 35) - 1)
                st.session_state.last_shoulder_alert_ts = now
                st.session_state.last_shoulder_penalty_ts = now
                st.toast(f"⚠️ 어깨 비대칭 10초 지속: -1점 (현재 {st.session_state.shoulder_score}점)")
                
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
                    st.session_state.shoulder_score = max(0, st.session_state.get('shoulder_score', 35) - 1)
                    st.session_state.last_shoulder_penalty_ts = now
                    st.toast(f"⏱ 어깨 비대칭 지속: -1점 (현재 {st.session_state.shoulder_score}점)")
            
            # ===== (5) 점수 UI 업데이트 (쓰로틀링) =====
            if now - st.session_state.get('last_score_update_ts', 0.0) >= SCORE_UPDATE_THROTTLE:
                head_score_ph.metric("얼굴 기울기", f"{st.session_state.score}/35")
                shoulder_score_ph.metric("어깨 균형", f"{st.session_state.shoulder_score}/35")
                st.session_state.last_score_update_ts = now
                        
            # 이전 상태 갱신
            st.session_state.prev_bad_flag = cur_bad
            st.session_state.prev_shoulder_bad_flag = cur_shoulder_bad
            
            # CPU 양보
            time.sleep(0.001)
        
    else:
        st.info("모든 준비가 완료되었다면, '듀얼 스트리밍 시작' 버튼을 클릭해주세요!")

if __name__ == "__main__":
    main()
