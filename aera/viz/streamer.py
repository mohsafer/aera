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
#v{image-rendering:pixelated;width:60vw;height:100vh;object-fit:contain}
#side{padding:12px;width:40vw;box-sizing:border-box;overflow-y:auto}
h3{color:#8c8c87;font-weight:normal;margin:10px 0 4px}
canvas{background:#181b22;border:1px solid #46484a;display:block;margin-bottom:6px}
#feed{white-space:pre-wrap;font-size:13px;line-height:1.5}
</style></head><body>
<img id="v" src="/stream">
<div id="side">
<h3>training curves (live)</h3>
<canvas id="c0" width="440" height="100"></canvas>
<canvas id="c1" width="440" height="100"></canvas>
<canvas id="c2" width="440" height="100"></canvas>
<canvas id="c3" width="440" height="100"></canvas>
<h3>event feed (live)</h3><div id="feed">…</div></div>
<script>
function line(cv, ys, color, label){
  const g = cv.getContext('2d');
  g.clearRect(0,0,cv.width,cv.height);
  g.fillStyle='#8c8c87'; g.font='12px monospace';
  if(!ys || ys.length<2){ g.fillText(label+' — waiting for data',8,18); return; }
  let lo=Math.min(...ys), hi=Math.max(...ys);
  if(hi-lo<1e-9){ hi=lo+1; }
  const W=cv.width-70, H=cv.height-28, X=8, Y=16;
  g.strokeStyle=color; g.lineWidth=1.6; g.beginPath();
  ys.forEach((v,i)=>{
    const x=X+i*(W/(ys.length-1)), y=Y+H-(v-lo)/(hi-lo)*H;
    i?g.lineTo(x,y):g.moveTo(x,y);
  });
  g.stroke();
  g.fillStyle='#e8e6dc';
  g.fillText(label, X, Y-3);
  g.fillStyle='#8c8c87';
  g.fillText(hi.toFixed(3), X+W+4, Y+8);
  g.fillText(lo.toFixed(3), X+W+4, Y+H);
  g.fillText('now '+ys[ys.length-1].toFixed(3), X, cv.height-4);
}
setInterval(()=>fetch('/curves').then(r=>r.json()).then(d=>{
  line(document.getElementById('c0'), d.episodes.map(p=>p[1]), '#78dc78', 'episode return');
  line(document.getElementById('c1'), d.updates.map(u=>u[1]), '#e8e6dc', 'policy loss (pi)');
  line(document.getElementById('c2'), d.updates.map(u=>u[2]), '#f0b446', 'value loss');
  line(document.getElementById('c3'), d.updates.map(u=>u[4]), '#e85a46', 'clip fraction');
}).catch(()=>{}),2000);
setInterval(()=>fetch('/events').then(r=>r.json()).then(
  e=>{document.getElementById('feed').textContent=e.feed.join('\\n')}),1000);
</script></body></html>"""


class FrameStreamer:
    """Publish frames/events from the training thread; serve them via HTTP."""

    def __init__(self, port: int = 8001, size=(640, 360), camera="chase"):
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
        # live training curves: [step, pi_loss, v_loss, entropy, clipfrac]
        self.updates: list[list[float]] = []
        self.episodes: list[list[float]] = []   # [episode, return]

    def push_update(self, step: int, stats: dict) -> None:
        """Trainer hook: one record per PPO update."""
        self.updates.append([float(step), float(stats.get("pi_loss", 0.0)),
                             float(stats.get("v_loss", 0.0)),
                             float(stats.get("entropy", 0.0)),
                             float(stats.get("clipfrac", 0.0))])
        self.updates = self.updates[-2000:]

    def push_episode(self, ep: dict) -> None:
        """Trainer hook: one record per finished episode."""
        self.episodes.append([float(ep.get("episode", 0)), float(ep.get("ret", 0.0))])
        self.episodes = self.episodes[-2000:]

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
                elif self.path == "/curves":
                    import json
                    with streamer._lock:
                        data = json.dumps({"updates": streamer.updates,
                                           "episodes": streamer.episodes}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
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

    def __init__(self, port: int = 8001, size=(640, 360), camera: str = "chase"):
        from .hud import Chart
        self.streamer = FrameStreamer(port, size, camera)
        self.streamer.start()
        self.chart = Chart()
        self.quit_requested = False

    def tick(self, env) -> bool:
        return self.streamer.tick(env)

    def push_update(self, step: int, stats: dict) -> None:
        self.streamer.push_update(step, stats)

    def push_episode(self, ep: dict) -> None:
        self.streamer.push_episode(ep)

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
