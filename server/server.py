import asyncio, json, logging, uuid, time  # ★ 변경: time 추가
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from av import VideoFrame

logger = logging.getLogger("webrtc")
pcs = set()
relay = MediaRelay()

# ─────────────────────────────────────────────────────────────
# 최신 프레임만 유지하는 버스 + 콜백 + 타임스탬프  ★ 변경
# ─────────────────────────────────────────────────────────────
frame_callback = None
frame_bus: asyncio.Queue | None = None   # 최신 1장만
last_recv_ts = 0.0       # RTP 프레임 수신 직후(perf_counter)
last_decode_ts = 0.0     # VideoFrame→ndarray 디코드 직후(perf_counter)

def set_frame_callback(callback):
    """외부에서 프레임 처리 콜백을 설정"""
    global frame_callback
    frame_callback = callback

async def _consume_and_call():
    """frame_bus에서 최신 프레임만 꺼내 외부 콜백을 await로 호출"""
    global frame_bus, frame_callback
    assert frame_bus is not None
    while True:
        img = await frame_bus.get()
        if frame_callback is not None:
            try:
                # 콜백은 여기서 비동기로 대기 → recv()는 절대 기다리지 않음
                await frame_callback(img)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

# HTML 클라이언트
HTML = """<!doctype html>
<meta charset="utf-8">
<title>Spine Analysis</title>
<body style="font-family:system-ui;margin:24px;background:#111;color:#eee;">
  <h3>Spine Analysis Camera</h3>
  <button id="start">Start Camera</button>
  <video id="local" autoplay playsinline muted style="width:400px;border:1px solid #ccc;display:block;margin-top:12px"></video>
  <script>
    async function start() {
      const pc = new RTCPeerConnection({iceServers: [{urls: ["stun:stun.l.google.com:19302"]}]});
      const local = document.getElementById("local");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment", width: 640, height: 480 },
          audio: false
        });
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
      } catch(e) {
        alert('Camera error: ' + e.message);
      }
    }
    document.getElementById("start").onclick = start;
  </script>
</body>
"""

class VideoCallbackTrack(MediaStreamTrack):
    """비디오 프레임을 콜백으로 전달 (수신/디코드와 콜백을 분리)"""
    kind = "video"
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        global last_recv_ts, last_decode_ts, frame_bus
        frame = await self.track.recv()
        last_recv_ts = time.perf_counter()

        # ndarray로 디코드
        try:
            img = frame.to_ndarray(format="bgr24")
            last_decode_ts = time.perf_counter()
        except Exception as e:
            logger.error(f"to_ndarray error: {e}")
            return frame  # 디코드 실패 시 원본 프레임 반환

        # 최신 한 장만 유지(덮어쓰기)
        if frame_bus is not None:
            try:
                if frame_bus.full():
                    frame_bus.get_nowait()  # 오래된 것 버림
                frame_bus.put_nowait(img)
            except asyncio.QueueFull:
                pass
            except Exception as e:
                logger.error(f"frame_bus put error: {e}")

        # 서버는 굳이 처리된 프레임을 돌려줄 필요가 없으면 원본 반환
        # (만약 처리영상을 브라우저로 보내고 싶다면, 최신 결과를 별도 공유변수에 보관 후 여기서 교체해도 됨)
        return frame

async def index(request):
    return web.Response(content_type="text/html", text=HTML)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    pcs.add(pc)
    recorder = MediaBlackhole()

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        if pc.connectionState == "failed":
            await pc.close()
            pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoCallbackTrack(relay.subscribe(track)))
            recorder.addTrack(relay.subscribe(track))

    await pc.setRemoteDescription(offer)
    await recorder.start()
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs])
    pcs.clear()

async def on_startup(app):
    """앱 시작 시 frame_bus 생성 및 소비자 태스크 시작  ★ 변경"""
    global frame_bus
    frame_bus = asyncio.Queue(maxsize=1)
    app['consumer_task'] = asyncio.create_task(_consume_and_call())

async def on_cleanup(app):
    """앱 종료 시 소비자 태스크 취소  ★ 변경"""
    task = app.get('consumer_task')
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

def create_app():
    """웹 애플리케이션 생성"""
    app = web.Application()
    app.on_startup.append(on_startup)   # ★ 변경
    app.on_cleanup.append(on_cleanup)   # ★ 변경
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)
    app.router.add_post("/offer", offer)
    return app

def run_server(host="0.0.0.0", port=8080):
    """서버 실행"""
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    print(f"WebRTC Server starting on http://{host}:{port}")
    web.run_app(app, host=host, port=port, access_log=None)
