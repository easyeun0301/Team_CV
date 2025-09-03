import argparse, asyncio, json, logging, os, ssl, uuid
import numpy as np, cv2
from aiohttp import web
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaBlackhole, MediaPlayer, MediaRecorder, MediaRelay
from av import VideoFrame

ROOT = os.path.dirname(__file__)
logger = logging.getLogger("pc")
pcs = set()
relay = MediaRelay()

# ──(내장 클라이언트)───────────────────────────────────────────────────────────
HTML = """<!doctype html>
<meta charset="utf-8">
<title>Team_CV WebRTC</title>
<body style="font-family:system-ui;margin:24px">
  <h3>Team_CV WebRTC Client (inline)</h3>
  <button id="start">Start</button>
  <video id="local" autoplay playsinline muted style="width:280px;border:1px solid #ccc;display:block;margin-top:12px"></video>
  <script>
    async function start() {
      const pc = new RTCPeerConnection();
      const local = document.getElementById("local");
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: true });
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
    }
    document.getElementById("start").onclick = start;
  </script>
</body>
"""

class VideoTransformTrack(MediaStreamTrack):
    """미러+레터박스로 '표시만' 축소하고, 전송은 원본 유지"""
    kind = "video"
    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")

        # 표시용만 축소/미러
        DISPLAY_W, DISPLAY_H = 360, 540
        h, w = img.shape[:2]
        scale = min(DISPLAY_W / w, DISPLAY_H / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=img.dtype)
        x, y = (DISPLAY_W - nw) // 2, (DISPLAY_H - nh) // 2
        canvas[y:y+nh, x:x+nw] = resized
        disp = cv2.flip(canvas, 1)
        cv2.imshow("Smartphone Camera (small, letterboxed, mirrored)", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 종료합니다."); cv2.destroyAllWindows(); os._exit(0)

        # 전송/후속처리는 원본으로
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts; new_frame.time_base = frame.time_base
        return new_frame

async def index(request):
    return web.Response(content_type="text/html", text=HTML)

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    pc_id = f"PeerConnection({uuid.uuid4()})"
    def log_info(msg, *args): logger.info(pc_id + " " + msg, *args)
    log_info("Created for %s", request.remote)

    # 선택 오디오 프롬프트(없으면 생략)
    audio_path = os.path.join(ROOT, "demo-instruct.wav")
    player = MediaPlayer(audio_path) if os.path.exists(audio_path) else None
    recorder = MediaRecorder(args.record_to) if args.record_to else MediaBlackhole()

    @pc.on("datachannel")
    def on_datachannel(channel):
        @channel.on("message")
        def on_message(message):
            if isinstance(message, str) and message.startswith("ping"):
                channel.send("pong" + message[4:])

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_info("Connection state is %s", pc.connectionState)
        if pc.connectionState == "failed":
            await pc.close(); pcs.discard(pc)

    @pc.on("track")
    def on_track(track):
        log_info("Track %s received", track.kind)
        if track.kind == "audio":
            if player and getattr(player, "audio", None):
                pc.addTrack(player.audio)
            recorder.addTrack(track)
        elif track.kind == "video":
            transform = params.get("video_transform", "none")
            pc.addTrack(VideoTransformTrack(relay.subscribe(track), transform))
            if args.record_to:
                recorder.addTrack(relay.subscribe(track))

        @track.on("ended")
        async def on_ended():
            log_info("Track %s ended", track.kind)
            await recorder.stop()

    await pc.setRemoteDescription(offer)
    await recorder.start()
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return web.Response(
        content_type="application/json",
        text=json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}),
    )

async def on_shutdown(app):
    await asyncio.gather(*[pc.close() for pc in pcs]); pcs.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC audio/video demo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--cert-file"); parser.add_argument("--key-file")
    parser.add_argument("--record-to")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    ssl_context = None
    if args.cert_file:
        ssl_context = ssl.SSLContext(); ssl_context.load_cert_chain(args.cert_file, args.key_file)

    app = web.Application()
    app.on_shutdown.append(on_shutdown)
    app.router.add_get("/", index)          # ← 내장 클라
    app.router.add_post("/offer", offer)    # ← 시그널링
    web.run_app(app, access_log=None, host=args.host, port=args.port, ssl_context=ssl_context)

# 명령프롬프트 입력 : python server.py --host 0.0.0.0 --port 8081
# 사이트 접속 : http://localhost:8081/