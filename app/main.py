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
        ## 웹캠 캡처 핸들/상태, 최신 프레임 1장만 유지하는 단일 데크 버퍼, FPS 측정용 카운터, 처리 시간 기록 리스트
        self.front_analyzer = FrontViewAnalyzer()
        self.front_cap = None
        self.front_running = False
        self.front_frame_buffer = deque(maxlen=1)  # 단일 버퍼
        self.front_lock = threading.Lock()
        self.front_thread = None
        self.front_fps = 0
        self.front_fps_counter = 0
        self.front_fps_start = time.time()
        
        # 모델 처리 시간 측정
        self.front_process_times = []
        self.front_total_frames = 0
        
        # Side view 관련 (HTTP 서버 - 세로)
        ## 서버 포트/상태, 최신 프레임 단일 버퍼, 워커 스레드, 서버 서브프로세스 핸들, HTTP 엔드포인트(URL), FPS/처리 시간 기록
        self.side_port = port  # 포트 저장
        self.side_running = False
        self.side_frame_buffer = deque(maxlen=1)  # 단일 버퍼
        self.side_lock = threading.Lock()
        self.side_thread = None
        self.side_server_process = None
        self.side_server_url = f"http://localhost:{port}/android/frame"      # 동적 포트
        self.side_status_url = f"http://localhost:{port}/android/status"     # 동적 포트
        self.side_fps = 0
        self.side_fps_counter = 0
        self.side_fps_start = time.time()
        
        # Side view 처리 시간 측정
        self.side_process_times = []
        self.side_total_frames = 0
        
        # 미리 할당된 결합 버퍼
        ## 초기엔 세로 480px, 가로 1280px, 3채널임 -> 필요시 새 크기로 재할당 가능
        self.combined_buffer = np.zeros((480, 1280, 3), dtype=np.uint8)

        # front_view 생성자 함수 변수 추가 - 1031
        self.bad_posture_flag = False  # 불량 자세 플래그
        self.bad_posture_lock = threading.Lock()  # 스레드 안전성

    def start_front_view(self):
        """웹캠 기반 Front View 시작 (최적화)"""
        if self.front_running:
            return "Front View가 이미 실행 중입니다."
            
        self.front_cap = cv2.VideoCapture(0)
        if not self.front_cap.isOpened():
            return "웹캠을 열 수 없습니다!"
            
        # 최적화된 웹캠 설정
        self.front_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최소 버퍼
        self.front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  ## 가로 640px
        self.front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) ## 세로 480px
        self.front_cap.set(cv2.CAP_PROP_FPS, 60)  # 높은 FPS 설정
            
        self.front_running = True
        self.front_thread = threading.Thread(target=self._optimized_front_worker, daemon=True)
        self.front_thread.start()
        
        return "Front View 시작됨"
    
    def _optimized_front_worker(self):
        ## 프레임 캡처 → FPS 갱신 → FrontViewAnalyzer로 분석/오버레이 → 최신 프레임 1장만 버퍼에 유지
        ## 실패시 원본 프레임으로 대체하여 끊김 최소화
        """최적화된 Front view 처리 워커 (논블로킹)"""
        while self.front_running and self.front_cap and self.front_cap.isOpened():
            self.front_cap.grab()   # 강제로 이전 프레임 버림, 1006 수정
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
            
            # 논블로킹 AI 처리 (실패시 원본 사용)
            try:
                processed_frame, bad_flag = self.front_analyzer.analyze_frame(frame)
                with self.bad_posture_lock:
                    self.bad_posture_flag = bad_flag
                # 성공한 경우만 처리 시간 기록
                process_time = (time.time() - process_start) * 1000  # ms 변환
                self.front_process_times.append(process_time)
            except:
                processed_frame = frame  # 처리 실패시 원본 즉시 사용
                # 실패한 경우는 처리 시간에 포함하지 않음
            
            # 단일 버퍼 업데이트
            with self.front_lock:                                # 동시에 접근할 수 없도록 lock을 걸어두기
                self.front_frame_buffer.clear()                  # 기존 프레임 제거
                self.front_frame_buffer.append(processed_frame)  # 새로 처리된 최신 프레임 추가
            
            # 지연 제거 - sleep 없음으로 최대 성능
    
    def start_side_view(self):
        """Side view HTTP 서버 시작 (최적화)"""
        if self.side_running:
            return "Side View가 이미 실행 중입니다."
            
        # side_view/run.py 경로 확인
        current_dir = os.path.dirname(os.path.abspath(__file__))
        side_view_path = os.path.join(current_dir, '..', 'side_view', 'run.py')
        
        if not os.path.exists(side_view_path):
            return f"side_view/run.py 파일을 찾을 수 없습니다: {side_view_path}"
        
        try:
            # 서버 프로세스 시작 (동적 포트 전달)
            self.side_server_process = subprocess.Popen([
                # ' python run.py --host 0.0.0.0 --port 8081 ' 같은 명령이 내부에서 실행되는 것과 같음
                sys.executable, side_view_path,
                '--host', '0.0.0.0',
                '--port', str(self.side_port)  # 동적 포트 전달
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # 서버 시작 대기 및 확인
            # 1초씩 쉬면서 최대 10번동안 /android/status 엔드포인트에 요청을 보냄
            for i in range(10):
                time.sleep(0.5) ## 1006 수정
                try:
                    response = requests.get(self.side_status_url, timeout=1)
                    if response.status_code == 200: # 서버가 정상적으로 켜져 있으면 돌려줌
                        break                       # 루프 종료
                except:
                    continue
            # for 루프가 다 돌때까지 서버가 응답이 없을 경우
            # Popen.communicate로 서버 로그를 3초간 수집하여 출력
            # 이후 프로세스를 terminate()로 종료하고 None으로 초기화 -> 에러 메시지 반환
            else:
                if self.side_server_process:
                    stdout, stderr = self.side_server_process.communicate(timeout=3)
                    print(f"서버 stdout: {stdout.decode()}")
                    print(f"서버 stderr: {stderr.decode()}")
                    self.side_server_process.terminate()
                    self.side_server_process = None
                return f"서버 시작 후 응답이 없습니다. 포트 {self.side_port}이 사용중인지 확인하세요."
            
            # 정상적으로 서버가 켜진 이후, 클라이언트 스레드를 백그라운드로 돌림
            # /android/frame으로 계속 GET 요청을 보내서 프레임(JPEG)을 가져오고, 이를 디코딩해 streamlit에 넘겨줌
            # demon=True : streamlit이 종료되면 자동으로 스레드도 함께 종료
            self.side_running = True
            self.side_thread = threading.Thread(target=self._optimized_side_worker, daemon=True)
            self.side_thread.start()
            
            return f"Side View 서버가 포트 {self.side_port}에서 성공적으로 시작되었습니다!"

        # 어떤 단계에서든 예외가 생기면(파일 접근, 포트 충돌, 네트워크 예외 등) 여기로 와서 실패 메세지 반환        
        except Exception as e:
            return f"Side View 서버 시작 실패: {str(e)}"
    
    def _optimized_side_worker(self):
        """최적화된 Side view HTTP 클라이언트 워커"""
        consecutive_errors = 0  # 실패 횟수 카운트
        
        while self.side_running:
            try:
                # HTTP 요청 시간 측정 시작
                request_start = time.time()
                self.side_total_frames += 1 # 총 요청 시도 횟수 카운트 (성공/실패 포함)
                
                # 타임아웃
                response = requests.get(self.side_server_url, timeout=0.1) # 0.1초 안에 서버로부터 최신 JPEG 프레임을 가져올게, 1006 수정

                if response.status_code == 200:
                    # JPEG 바이트를 OpenCV 이미지로 변환
                    img_array = np.frombuffer(response.content, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # 성공한 경우만 처리 시간 기록
                        process_time = (time.time() - request_start) * 1000  # ms 변환
                        self.side_process_times.append(process_time)
                        
                        # FPS 계산
                        self.side_fps_counter += 1
                        if self.side_fps_counter % 30 == 0:
                            elapsed = time.time() - self.side_fps_start
                            self.side_fps = 30 / elapsed if elapsed > 0 else 0
                            self.side_fps_start = time.time()
                        
                        # 320(가로) x 480(세로)
                        frame_resized = cv2.resize(frame, (320, 480))
                        
                        # 단일 버퍼 업데이트
                        with self.side_lock:
                            self.side_frame_buffer.clear() # 이전 프레임 지우기
                            self.side_frame_buffer.append(frame_resized) # 최신 프레임 한 장만
                        
                        consecutive_errors = 0 # 연속 실패 카운트 0으로 초기화
                
            except requests.exceptions.RequestException:
                consecutive_errors += 1     # 실패할 때마다 +1
                if consecutive_errors >= 10:
                    # 연결 실패시 에러 프레임 생성
                    self._create_side_error_frame()
                    time.sleep(1.0)  # 연결 실패시만 대기 (서버가 잠깐 꺼져 있는 동안 계속 폭주하지 않도록)
    
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

# streamlit 캐시 메커니즘을 이용해 OptimizedDualStreamManager 객체를 한 번만 생성 후 재사용
# streamlit 앱은 버튼 클릭시마다 스크립트를 're-run'하기 때문에 객체를 캐시하지 않으면 start_side_view()가 계속 새 프로세스를 띄워버려 서버 중복 실행 우려가 있음
@st.cache_resource
def get_optimized_stream_manager(_port):
    """최적화된 스트림 매니저 싱글톤"""
    global _global_stream_manager
    if _global_stream_manager is None:
        _global_stream_manager = OptimizedDualStreamManager(port=_port)
    return _global_stream_manager

def main():
    # 명령행 인자 파싱
    args = parse_args()
    port = args.port

    st.set_page_config(
        page_title="Optimized Dual Pose Analysis",  # 브라우저 탭 제목
        layout="wide"                               # 페이지 레이아웃 (가로 폭 전체를 사용하는 UI 모드)
    )

    # 세션 상태 초기화 (분석 관련 추가) - 1021
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False
    if 'analysis_active' not in st.session_state:
        st.session_state.analysis_active = False  # 분석 진행 여부
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None  # 분석 시작 시간
    if 'analysis_duration' not in st.session_state:
        st.session_state.analysis_duration = 10  # 실행 확인을 위해 10초로 테스트
    if 'show_report' not in st.session_state: 
        st.session_state.show_report = False  # 리포트 화면 표시 여부
    
    st.title("바르게 살자 !")                                                             # 페이지 상단 타이틀
    st.markdown("안녕하세요! 2025 D&X:W Conference Tech_CV팀 부스에 오신 걸 환영합니다 😊")  # 짧은 설명 문구
    st.markdown("<br><br>", unsafe_allow_html=True)  # 2줄 공백

    # 위에서 정의한 캐시된 OptimizedDualStreamManager 객체를 가져옴
    # -> 이 객체가 실제로 카메라 켜고, 서버 띄우고, 프레임 버퍼 관리
    stream_manager = get_optimized_stream_manager(port)
    
    # ────────────────────────────────
    # 컨트롤 패널
    # ────────────────────────────────
    col1, col2 = st.columns(2) # col1은 버튼, col2는 상태 표시 텍스트
    message_placeholder = st.empty() # 알림 메시지를 임시로 띄울 수 있는 공간
    
    # 시작/정지 버튼 제어
    with col1:
        if not st.session_state.streaming:
            if st.button("듀얼 스트리밍 시작", type="primary", use_container_width=True, key="start_everything"):
                # 모든 것을 한 번에 시작
                front_result = stream_manager.start_front_view() # 웹캠 스레드 실행
                side_result = stream_manager.start_side_view()   # run.py 서버 프로세스 + HTTP 클라이언트 스레드 실행
                
                if "시작됨" in front_result and "성공적으로" in side_result:  # 두 결과 문자열에 '성공적으로' / '시작됨'이 들어있으면 성공
                    st.session_state.prev_bad_flag = False
                    st.session_state.last_bad_alert_ts = 0.0

                    st.session_state.streaming = True # 상태 전환 후 st.rerun으로 페이지 새로고침 시켜 스트리밍 루프 표시 영역으로 이동
                    message_placeholder.success("듀얼 스트리밍이 성공적으로 시작되었습니다!")
                    st.rerun()
                else:
                    message_placeholder.error(f"시작 실패 - Front: {front_result}, Side: {side_result}")
        else:
            # '정지' 버튼 클릭 시 모든 스레드/프로세스 종료
            if st.button("스트리밍 정지", use_container_width=True, key="stop_everything"):
                stream_manager.stop()
                st.session_state.streaming = False
                message_placeholder.warning("스트리밍 정지됨")
                # 분석 상태 초기화를 위한 설정 - 1021
                st.session_state.analysis_active = False
                st.session_state.analysis_start_time = None
                st.session_state.show_report = False
                st.rerun()
    
    with col2:
        # 현재 세션의 상태를 간단한 텍스트로 보여줌
        st.write(f"상태: {'실행 중' if st.session_state.streaming else '정지'}")
    
    # 알림 엣지 트리거/쿨다운 상태
    if 'prev_bad_flag' not in st.session_state:
        st.session_state.prev_bad_flag = False
    if 'last_bad_alert_ts' not in st.session_state:
        st.session_state.last_bad_alert_ts = 0.0
    BAD_ALERT_COOL_S = 3.0  # 알림 연속 방지를 위한 쿨다운(초)
    
    # ────────────────────────────────
    # 스트리밍 표시
    # ────────────────────────────────
    if st.session_state.streaming:
        # 리포트 화면 표시 추가 (분석 완료 후) - 1021
        if st.session_state.show_report:
            st.markdown("## 🍀자세 분석 리포트")
            st.success("3분간의 분석이 완료되었습니다! 분석 결과를 확인해보세요!")
            
            ## 눈깜빡임 리포트 추가 - 1026
            st.markdown("---")
            st.markdown("### 👀 눈 깜빡임 분석")
            
            col1, col2, col3 = st.columns(3)
            
            blink_count = stream_manager.front_analyzer.blink_count
            duration_minutes = st.session_state.analysis_duration / 60
            blinks_per_minute = blink_count / duration_minutes if duration_minutes > 0 else 0
            
            with col1:
                st.metric(label="총 깜빡임 수", value=f"{blink_count}회")
            
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
        
        st.markdown("### Front_view + Side_view")  # 제목 표시
        st.markdown("다양한 옵션 버튼들을 통해 설정값을 조정해보세요 :) \n\n본인에게 맞는 바른 자세를 파악하신 후 '분석 시작' 버튼을 클릭하면 자세 분석이 시작됩니다!")
       
        # 분석 시작 버튼 추가 - 1021
        button_area = st.empty()

        if not st.session_state.analysis_active:
            with button_area.container():
                if st.button("⏰분석 시작 (3분)", type="primary", use_container_width=True):
                    st.session_state.analysis_active = True
                    st.session_state.analysis_start_time = time.time()
                    
                    ## 분석 시작하면 눈 깜빡임 리셋 - 1026
                    stream_manager.front_analyzer.blink_count = 0
                    stream_manager.front_analyzer.win_start = time.time()

                    button_area.empty()
                    st.rerun()

        # 한 행: Front / Side / 옵션+상태 (1009 수정)
        col_front, col_side, col_option = st.columns([1, 1, 1]) # 맨 오른쪽에 옵션+상태 배치
        
        # 한 번만 공간 확보 (빈 이미지 영역 생성), 1006 수정
        front_placeholder = col_front.empty()
        side_placeholder = col_side.empty()

        # 첫 프레임 때만 이미지 객체 생성, 1006 수정
        front_img = None
        side_img = None

        # ────────────────────────────────
        # Front View 옵션 제어 + 상태 요약 (한 열에 세로 정렬)
        # ────────────────────────────────
        with col_option:
            st.markdown("### Front View 옵션")

            # ----- 옵션 설정 -----
            with st.container():
                st.markdown("#### 옵션 설정")

                colA, colB = st.columns(2)

                # 임계값 설정
                if colA.button("판독 감도 조절(RELAX1 < RELAX2 < Strict)", key="thr_btn_once"):
                    stream_manager.front_analyzer.cycle_threshold_profile(+1)

                # keypoint 표시
                if colA.button("Key Points 표시", key="pts_btn_once"):
                    stream_manager.front_analyzer.SHOW_POINTS = not stream_manager.front_analyzer.SHOW_POINTS
                
                # EAR 보정
                if colA.button("눈 깜빡임 보정", key="ear_btn_once"):
                    if len(stream_manager.front_analyzer.ear_window) >= 10:
                        arr = np.array(stream_manager.front_analyzer.ear_window, dtype=np.float32)
                        med = float(np.median(arr))
                        p10 = float(np.percentile(arr, 10))
                        stream_manager.front_analyzer.T_LOW = max(0.08, min(med * 0.75, p10 + 0.02))
                        stream_manager.front_analyzer.T_HIGH = max(stream_manager.front_analyzer.T_LOW + 0.02, med * 0.92)
                        stream_manager.front_analyzer.calibrated = True
                    else:
                        st.toast("⚠️ EAR 데이터가 부족합니다 (눈 깜빡임 감지 후 다시 시도)")

                # CLAHE
                if colB.button("명암 대비 조정", key="clahe_btn_once"):
                    stream_manager.front_analyzer.use_clahe = not stream_manager.front_analyzer.use_clahe

                # 투명도 조절
                if colB.button("투명도 ↑ (+)", key="alpha_up_once"):
                    stream_manager.front_analyzer.ALPHA = min(1.0, stream_manager.front_analyzer.ALPHA + 0.1)
                if colB.button("투명도 ↓ (-)", key="alpha_dn_once"):
                    stream_manager.front_analyzer.ALPHA = max(0.1, stream_manager.front_analyzer.ALPHA - 0.1)

            # ----- 상태 요약 -----
            st.markdown("---")
            st.markdown("#### 현재 설정 상태")

            analyzer = stream_manager.front_analyzer
            col_status1, col_status2 = st.columns(2)

            with col_status1:
                st.metric("임계값", analyzer.THR_PROFILES[analyzer.thr_profile_idx][1])
                st.metric("Key Points 표시", "ON ⭕" if analyzer.SHOW_POINTS else "OFF ❌")
                st.metric("눈 깜빡임 보정", "보정 ⭕" if analyzer.calibrated else "자동")

            with col_status2:
                st.metric("명암 대비 조정", "ON ⭕" if analyzer.use_clahe else "OFF ❌")
                st.metric("투명도", f"{analyzer.ALPHA:.1f}")

        # 실시간 스트리밍 루프
        start_time = time.time()

        # 최대 100분(6000초)동안 루프를 돌며 stream_manager의 버퍼에서 최신 프레임을 읽어와 화면 갱신
        while st.session_state.streaming and (time.time() - start_time) < 6000:  ## 1006 수정
           # 3분 경과 시 리포트 화면으로 전환 - 1021
            if st.session_state.analysis_active and st.session_state.analysis_start_time:
                elapsed = time.time() - st.session_state.analysis_start_time
                
                if elapsed >= st.session_state.analysis_duration:
                    st.session_state.analysis_active = False
                    st.session_state.show_report = True  # 리포트 화면 표시
                    stream_manager.stop()  # 스트리밍 중지
                    st.rerun()

            # 프레임 가져오기 및 표시
            front_frame = stream_manager.get_front_frame()
            side_frame = stream_manager.get_side_frame()
            
            ## 1006 수정, 이미지 객체를 1회만 생성 후 재사용하도록 아래 코드들 싹 다 수정
            if front_frame is not None:
                front_rgb = cv2.cvtColor(front_frame, cv2.COLOR_BGR2RGB)

                # 첫 프레임일 때만 st.image() 생성
                if front_img is None:
                    front_img = front_placeholder.image(front_rgb, channels="RGB", width=640)
                else:
                    # 이후에는 기존 이미지 갱신만 수행 (DOM 재생성 없음)
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
            
             # 스트리밍 루프 내부 (프레임 렌더 뒤, sleep 전에 배치)
            with stream_manager.bad_posture_lock:
                cur_bad = bool(stream_manager.bad_posture_flag)
            now = time.time()

            # ⭐ 디버깅 로그
            if cur_bad:
                print(f"[DEBUG] Bad posture detected! prev={st.session_state.prev_bad_flag}, "
                    f"cooldown={(now - st.session_state.last_bad_alert_ts):.1f}s")

            if (not st.session_state.prev_bad_flag) and cur_bad and \
            (now - st.session_state.last_bad_alert_ts >= BAD_ALERT_COOL_S):
                print("[DEBUG] ✅ Alert triggered!")  # 이게 출력되는지 확인
                st.toast("⚠️ 5초 이상 고개 기울기 감지! 바르게 앉으세요.")


                # 브라우저 TTS
                st.components.v1.html("""
                <script>
                (async function(){
                try{
                    // 1) synth를 하나로 통일 (top 우선)
                    const synth = (window.top && window.top.speechSynthesis) || window.speechSynthesis;
                    const U = (window.top && window.top.SpeechSynthesisUtterance) || window.SpeechSynthesisUtterance;

                    // 2) voices 로드 대기 (일부 브라우저는 초기엔 빈 배열)
                    const waitVoices = () => new Promise(resolve => {
                    const tryGet = () => {
                        const v = synth.getVoices();
                        if (v && v.length) resolve(v);
                        else {
                        synth.onvoiceschanged = () => resolve(synth.getVoices());
                        // 안전망: 혹시 이벤트가 안 오면 타임아웃으로 한 번 더 시도
                        setTimeout(() => {
                            const v2 = synth.getVoices();
                            if (v2 && v2.length) resolve(v2);
                        }, 500);
                        }
                    };
                    tryGet();
                    });

                    const voices = await waitVoices();

                    // 3) Edge에서 흔한 한국어 보이스 후보들
                    const preferredNames = [
                    "Microsoft InJoon Online (Natural)",  // Microsoft InJoon Online (Natural) - Korean (Korea)
                    "Microsoft SunHi Online (Natural)",   // Microsoft SunHi Online (Natural) - Korean (Korea)
                    "Korean"   // 넓게 매칭 (환경별 이름 편차 대비)
                    ];

                    // 4) 우선순위: 이름 일부 매칭 → 언어 ko-* → 첫 번째
                    let v =
                    voices.find(x => preferredNames.some(p => x.name.includes(p))) ||
                    voices.find(x => x.lang && x.lang.toLowerCase().startsWith("ko")) ||
                    voices[0];

                    // 디버그: 어떤 보이스가 선택됐는지 확인
                    console.log("[TTS] voices:", voices.map(x => ({name:x.name, lang:x.lang})));
                    console.log("[TTS] picked:", v ? {name:v.name, lang:v.lang} : null);

                    // 5) 발화
                    const u = new U("5초 이상 고개가 틀어졌습니다. 바르게 앉으세요.");
                    u.lang = "ko-KR";   // 언어 힌트
                    if (v) u.voice = v;
                    u.rate = 1.3;       // 속도 (0.5~2.0)
                    u.pitch = 1.2;      // 톤 (0~2)
                    synth.cancel();
                    synth.speak(u);
                }catch(e){
                    console.error("[TTS] error:", e);
                }
                })();
                </script>
                """, height=0)


                st.session_state.last_bad_alert_ts = now

            # 이전 상태 갱신
            st.session_state.prev_bad_flag = cur_bad

            # 살~짝 sleep으로 CPU 양보
            time.sleep(0.001)    
        
    else:
        st.info("모든 준비가 완료되었다면, '듀얼 스트리밍 시작' 버튼을 클릭해주세요!")

if __name__ == "__main__":
    main()