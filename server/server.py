import asyncio, json, logging, uuid
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaRelay
from av import VideoFrame

logger = logging.getLogger("webrtc")
pcs = set()
relay = MediaRelay()

# 전역 프레임 처리 콜백
frame_callback = None

def set_frame_callback(callback):
    """외부에서 프레임 처리 콜백을 설정"""
    global frame_callback
    frame_callback = callback

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
    """비디오 프레임을 콜백으로 전달"""
    kind = "video"
    
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        
        if frame_callback:
            try:
                img = frame.to_ndarray(format="bgr24")
                processed_img = await frame_callback(img)
                
                if processed_img is not None:
                    new_frame = VideoFrame.from_ndarray(processed_img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = frame.time_base
                    return new_frame
            except Exception as e:
                logger.error(f"Frame callback error: {e}")
        
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

def create_app():
    """웹 애플리케이션 생성"""
    app = web.Application()
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