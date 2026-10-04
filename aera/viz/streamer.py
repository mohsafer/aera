"""Real-time web streaming: watch a live run from a browser — no display, no
X11, no extra dependencies (stdlib http.server + the headless renderer).

    python -m aera train --config ... --stream        # serve during training
    python -m aera watch --ckpt ... --stream 8001     # serve during playback

Then, through the usual SSH tunnel (`ssh -L 8001:localhost:8001 ...`), open
http://localhost:8001/ — the page shows an MJPEG video stream plus the live
event feed (anomalies, milestones, foods). Frames render at ~10 fps from the
same painter's-algorithm renderer used for GIFs; rendering never touches the
RNG, so streaming cannot change a run's outcome.
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from .camera import Camera
from .renderer3d import render_frame

_PAGE = """<!doctype html><html><head><title>AERA live</title>
<meta charset="utf-8"><style>
body{background:#12141a;color:#e8e6dc;font-family:monospace;margin:0;display:flex}
#v{image-rendering:pixelated;width:70vw;height:100vh;object-fit:contain}
#side{padding:12px;width:30vw;box-sizing:border-box}
h3{color:#8c8c87;font-weight:normal}
#feed{white-space:pre-wrap;font-size:13px;line-height:1.5}
</style></head><body>
<img id="v" src="/stream">
<div id="side"><h3>event feed (live)</h3><div id="feed">…</div></div>
<script>setInterval(()=>fetch('/events').then(r=>r.json()).then(
  e=>{document.getElementById('feed').textContent=e.feed.join('\\n')}),1000);
</script></body></html>"""


class FrameStreamer:
    """Publish frames/events from the training thread; serve them via HTTP."""

    def __init__(self, port: int = 8001, size=(426, 240), camera="chase"):
        self.port = port
        self.size = size
        self.cam_mode = camera
        self._cam = Camera(camera)
        self._frame: bytes | None = None
        self._events: list[str] = []
        self._lock = threading.Lock()
        self.quit_requested = False          # trainer hook compatibility
        self._server = None
        self._thread = None

    # ------------------------------------------------------- producer side
    def tick(self, env, min_interval: float = 0.1) -> bool:
        """Call once per env step (trainer hook). Renders at ~10 fps max."""
        now = time.monotonic()
        if not hasattr(self, "_last") or now - self._last >= min_interval:
            self._last = now
            try:
                arr = render_frame(env, self._cam, size=self.size,
                                   time_s=env.steps * env.config.sim.dt)
                self.publish(arr)
            except Exception:
                pass                          # never kill training over viz
        if env.last_events:
            with self._lock:
                self._events += [f"[ep {env.episode}] {e}" for e in env.last_events]
                self._events = self._events[-30:]
        return not self.quit_requested

    def publish(self, frame: np.ndarray) -> None:
        from PIL import Image
        import io
        buf = io.BytesIO()
        Image.fromarray(frame).save(buf, format="JPEG", quality=70)
        with self._lock:
            self._frame = buf.getvalue()

    # ----------------------------------------------------------- HTTP side
    def start(self) -> None:
        streamer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):       # silence request spam
                pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with streamer._lock:
                                jpg = streamer._frame
                            if jpg is not None:
                                self.wfile.write(
                                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                                self.wfile.write(jpg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.08)
                    except (ConnectionAbortedError, BrokenPipeError):
                        pass
                elif self.path == "/events":
                    with streamer._lock:
                        body = streamer._events
                    import json
                    data = json.dumps({"feed": body}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(_PAGE.encode())

        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True)
        self._thread.start()
        print(f"live view → http://localhost:{self.port}/ "
              f"(via ssh -L {self.port}:localhost:{self.port} user@host)")

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()


class StreamViewer:
    """Drop-in for the pygame Viewer inside Trainer (--stream): exposes
    .tick(env), .quit_requested and .chart.push so the trainer doesn't care
    which one is attached."""

    def __init__(self, port: int = 8001, size=(426, 240), camera: str = "chase"):
        from .hud import Chart
        self.streamer = FrameStreamer(port, size, camera)
        self.streamer.start()
        self.chart = Chart()
        self.quit_requested = False

    def tick(self, env) -> bool:
        return self.streamer.tick(env)

    def play(self, env, policy=None, seed: int = 0):
        """Standalone playback loop (the --stream equivalent of Viewer.play)."""
        obs, _ = env.reset(seed=seed)
        while not self.quit_requested:
            a = (policy.act(obs, deterministic=True)[0]
                 if policy is not None else env.action_space.sample())
            obs, _, term, trunc, _ = env.step(a)
            self.tick(env)
            if term or trunc:
                obs, _ = env.reset()
        self.streamer.stop()
