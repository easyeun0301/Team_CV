import asyncio, json, logging, uuid, time
import numpy as np, cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRelay
from aiortc import RTCConfiguration, RTCIceServer
from av import VideoFrame

logger = logging.getLogger("webrtc")

# ─────────────────────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────────────────────
pcs = set()
relay = MediaRelay()

# AI 콜백 및 프레임 버스(최신 1장만 유지)
frame_callback = None
frame_bus: asyncio.Queue | None = None  # VideoFrame을 저장
_callback_gate = asyncio.Semaphore(1)   # 콜백 동시 1개만 실행

# 처리 결과 이미지 & 락
processed_frame = None                  # 최신 처리 결과 (ndarray, BGR)
processed_frame_lock = asyncio.Lock()
last_frame_time = 0.0                   # 마지막 수신 프레임 시각
last_processed_time = 0.0               # 마지막 처리 완료 시각

# JPEG 캐시 (결과/빈화면)
_empty_jpeg = None
_cached_jpeg = None
_cached_jpeg_ts = 0.0

# PC별 리더 태스크 (수신 전용)
pc_reader_tasks = {}  # pc_id -> asyncio.Task

# ─────────────────────────────────────────────────────────────
# 스마트폰 연결 페이지 (그대로 사용)
# ─────────────────────────────────────────────────────────────
HTML = """<!doctype html>
<meta charset="utf-8">
<title>카메라 연결</title>
<body style="font-family:system-ui;margin:24px;background:#f5f5f5">
  <div style="max-width:400px;margin:50px auto;background:white;padding:30px;border-radius:12px;box-shadow:0 4px 6px rgba(0,0,0,0.1)">
    <h2 style="text-align:center;color:#333;margin-bottom:30px">카메라 연결</h2>
    
    <button id="start" style="width:100%;padding:15px;font-size:18px;background:#28a745;color:white;border:none;border-radius:8px;cursor:pointer;margin-bottom:20px">
      카메라 시작
    </button>
    
    <video id="local" autoplay playsinline muted style="width:100%;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,0.2)"></video>
  </div>

  <script>
    async function start() {
      try {
        const pc = new RTCPeerConnection({
          iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]
        });
        const local = document.getElementById("local");
        const startBtn = document.getElementById("start");
        
        const constraints = {
          video: { 
            facingMode: "environment",
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
          },
          audio: false
        };
        
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        stream.getTracks().forEach(t => pc.addTrack(t, stream));
        local.srcObject = stream;
        
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);

        const resp = await fetch("/offer", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type })
        });
        
        const answer = await resp.json();
        await pc.setRemoteDescription(answer);
        
        startBtn.innerHTML = "카메라 연결됨";
        startBtn.style.background = "#007bff";
        
      } catch (error) {
        alert("카메라 연결 실패: " + error.message);
      }
    }
    
    document.getElementById("start").onclick = start;
  </script>
</body>
"""

# ─────────────────────────────────────────────────────────────
# API: 콜백 등록 & 결과 저장
# ─────────────────────────────────────────────────────────────
def set_frame_callback(callback):
    """외부에서 프레임 처리 콜백 등록: async def f(img_bgr: np.ndarray) -> Optional[np.ndarray]"""
    global frame_callback
    frame_callback = callback

async def store_processed_frame(frame):
    """처리된 프레임 저장 + JPEG 바이트 캐시 업데이트"""
    global processed_frame, last_processed_time, _cached_jpeg, _cached_jpeg_ts
    async with processed_frame_lock:
        if frame is None:
            return
        processed_frame = frame  # 콜백에서 새 버퍼를 반환한다고 가정
        last_processed_time = time.time()
        # 결과 JPEG 캐시(요청 때마다 재인코딩 방지)
        ok, buf = cv2.imencode('.jpg', processed_frame,
                               [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
        if ok:
            _cached_jpeg = buf.tobytes()
            _cached_jpeg_ts = last_processed_time

# ─────────────────────────────────────────────────────────────
# 프레임 소비 루프: VideoFrame -> (여기서만) ndarray 디코딩 1회 -> 콜백
# ─────────────────────────────────────────────────────────────
async def _consume_and_call():
    """frame_bus에서 VideoFrame을 꺼내 콜백을 단 1개만 실행(세마포어)"""
    global frame_bus, frame_callback
    assert frame_bus is not None

    while True:
        try:
            vf: VideoFrame = await frame_bus.get()

            # 콜백이 없으면 스킵
            if frame_callback is None:
                continue

            # 콜백 실행 중이면 스킵(최신 1장 유지 전략; frame_bus maxsize=1)
            if _callback_gate.locked():
                continue

            async with _callback_gate:
                # 여기서 단 1회만 디코딩
                img = vf.to_ndarray(format="bgr24")
                try:
                    out = await frame_callback(img)
                    await store_processed_frame(out if out is not None else img)
                except Exception as e:
                    logger.error(f"Frame callback error: {e}")
                    # 실패시 원본 송출
                    await store_processed_frame(img)

        except Exception as e:
            logger.error(f"Frame consumer error: {e}")

# ─────────────────────────────────────────────────────────────
# HTTP 핸들러
# ─────────────────────────────────────────────────────────────
async def index(request):
    return web.Response(content_type="text/html", text=HTML)

async def get_android_frame(request):
    """Streamlit 등에서 최신 처리 이미지를 가져감 (JPEG 캐시 사용)"""
    global processed_frame, last_processed_time, _empty_jpeg, _cached_jpeg

    # 최근 3초 이내 결과가 있으면 캐시된 JPEG 반환
    async with processed_frame_lock:
        fresh = processed_frame is not None and (time.time() - last_processed_time) < 3.0
        if fresh and _cached_jpeg is not None:
            return web.Response(
                body=_cached_jpeg,
                content_type='image/jpeg',
                headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                         'Pragma': 'no-cache', 'Expires': '0'}
            )

    # 연결 전: "Connecting..." 화면을 전역 1회 생성/재사용
    if _empty_jpeg is None:
        empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(empty_img, "Connecting...", (160, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode('.jpg', empty_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            # 전역에 캐시
            globals()['_empty_jpeg'] = buf.tobytes()

    return web.Response(body=_empty_jpeg, content_type='image/jpeg')

async def get_android_status(request):
    """연결 상태/지표"""
    global last_frame_time, processed_frame, last_processed_time, frame_callback
    current_time = time.time()
    frame_age = (current_time - last_frame_time) * 1000 if last_frame_time > 0 else 999
    processed_age = (current_time - last_processed_time) * 1000 if last_processed_time > 0 else 999

    status = {
        "connected": last_frame_time > 0 and frame_age < 5000,  # 5초 이내 프레임 수신
        "active_connections": len(pcs),
        "last_frame_age_ms": round(frame_age, 1),
        "has_processed_frame": processed_frame is not None,
        "processed_frame_age_ms": round(processed_age, 1),
        "callback_registered": frame_callback is not None
    }
    return web.Response(content_type="application/json", text=json.dumps(status))

# ─────────────────────────────────────────────────────────────
# WebRTC 오퍼/트랙 처리 (루프백 송신/녹화 제거, 리더 태스크로 대체)
# ─────────────────────────────────────────────────────────────
async def offer(request):
    """WebRTC 연결 핸드셋"""
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    config = RTCConfiguration(iceServers=[RTCIceServer(urls=["stun:stun.l.google.com:19302"])])
    pc = RTCPeerConnection(config)
    pcs.add(pc)
    pc_id = f"PeerConnection({uuid.uuid4()})"

    def log_info(msg, *args):
        logger.info(pc_id + " " + msg, *args)

    log_info("Created for %s", request.remote)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState in ("failed", "closed", "disconnected"):
            # 리더 태스크 정리
            t = pc_reader_tasks.pop(pc_id, None)
            if t:
                t.cancel()
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)

        if track.kind == "video":
            # 루프백 송신/녹화 제거
            # pc.addTrack(VideoCallbackTrack(relay.subscribe(track)))  # 제거됨
            # recorder.addTrack(relay.subscribe(track))                # 제거됨

            subscribed = relay.subscribe(track)

            async def reader():
                global frame_bus, last_frame_time
                try:
                    while True:
                        vf: VideoFrame = await subscribed.recv()
                        last_frame_time = time.time()
                        if frame_bus is not None:
                            # 최신 1장만 유지
                            if frame_bus.full():
                                try:
                                    frame_bus.get_nowait()
                                except Exception:
                                    pass
                            try:
                                frame_bus.put_nowait(vf)  # VideoFrame 그대로
                            except asyncio.QueueFull:
                                pass
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Video reader error: {e}")

            task = asyncio.create_task(reader())
            pc_reader_tasks[pc_id] = task

            @track.on("ended")
            async def on_ended():
                log_info("Track %s ended", track.kind)
                t = pc_reader_tasks.pop(pc_id, None)
                if t:
                    t.cancel()

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )

# ─────────────────────────────────────────────────────────────
# 앱 라이프사이클
# ─────────────────────────────────────────────────────────────
async def on_startup(app):
    """앱 시작 시 frame_bus 생성 및 소비자 태스크 시작"""
    global frame_bus
    frame_bus = asyncio.Queue(maxsize=1)  # 최신 1장만 유지 (VideoFrame)
    app['consumer_task'] = asyncio.create_task(_consume_and_call())

async def on_cleanup(app):
    """앱 종료 시 소비자/리더 태스크 취소"""
    # 소비자 태스크 취소
    task = app.get('consumer_task')
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # 리더 태스크 전부 취소
    for t in list(pc_reader_tasks.values()):
        t.cancel()
    pc_reader_tasks.clear()

async def on_shutdown(app):
    # 열린 pc 모두 닫기
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()

# ─────────────────────────────────────────────────────────────
# 앱/서버 생성 & 실행
# ─────────────────────────────────────────────────────────────
def create_app():
    app = web.Application()
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.on_shutdown.append(on_shutdown)

    app.router.add_get("/", index)
    app.router.add_get("/android/frame", get_android_frame)
    app.router.add_get("/android/status", get_android_status)
    app.router.add_post("/offer", offer)
    return app

def run_server(host="0.0.0.0", port=8080):
    logging.basicConfig(level=logging.WARNING)
    app = create_app()
    web.run_app(app, access_log=None, host=host, port=port, ssl_context=None)

# 사용 예시:
# if __name__ == "__main__":
#     # 예시 콜백 등록
#     async def demo_cb(img):
#         out = img.copy()
#         cv2.putText(out, "OK", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
#         return out
#     set_frame_callback(demo_cb)
#     run_server()
