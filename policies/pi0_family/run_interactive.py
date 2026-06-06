# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live interactive demo for the Pi0 family (language steering in the browser).

Wraps a single RoboLab environment in a continuous step loop and exposes it over
a FastAPI + WebSocket server. A browser page streams the policy's camera view and
lets you type a new instruction at any time; the new prompt is fed to the policy
on the *next* step (no env rebuild), so pi0.5 re-steers live on the same scene.

This is a demo harness, not an evaluator: it does not write results/HDF5. It
reuses RoboLab's env + the same Pi0DroidJointposClient used by run.py.

Run (with the pi05 policy server already serving, e.g. on :8001):

    CUDA_VISIBLE_DEVICES=5 uv run python policies/pi0_family/run_interactive.py \
        --task BananaInBowlTask --remote-port 8001 --web-port 8090

Then open http://<host>:8090 in a browser.
"""

import argparse
import json
import logging
import sys
import threading
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

logger = logging.getLogger(__name__)

PI0_VARIANTS = ["pi0", "pi0_fast", "pi05", "paligemma", "paligemma_fast"]

parser = argparse.ArgumentParser(description="Live interactive Pi0-family language-steering demo.")
parser.add_argument("--policy", choices=PI0_VARIANTS, default="pi05",
                    help="Which Pi0-family variant to drive (default: pi05).")
parser.add_argument("--policy-backend", "--policy_backend",
                    choices=["pi05", "molmoact", "pi05_steer", "compare", "steer_vs_pi0"], default="pi05",
                    help="Inference backend: 'pi05' (OpenPI WebSocket), 'molmoact' (MolmoAct2 ZMQ), "
                         "'pi05_steer' (PyTorch pi0.5 ZMQ with VLS steering), or 'compare' "
                         "(toggle between two OpenPI servers: pi0.5 ON vs pi0 OFF, both unsteered).")
parser.add_argument("--model-host", type=str, default="localhost",
                    help="Host for the OpenPI model servers in 'compare' mode.")
parser.add_argument("--pi05-port", type=int, default=8000,
                    help="Port of the pi0.5 OpenPI server (toggle ON) in 'compare' mode.")
parser.add_argument("--pi0-port", type=int, default=8001,
                    help="Port of the pi0 OpenPI server (toggle OFF) in 'compare' mode.")
parser.add_argument("--molmo-host", "--molmo_host", type=str, default="localhost",
                    help="MolmoAct2 server host (used when --policy-backend molmoact).")
parser.add_argument("--molmo-port", "--molmo_port", type=int, default=5555,
                    help="MolmoAct2 server port (used when --policy-backend molmoact).")
parser.add_argument("--steer", action="store_true",
                    help="Enable VLS gradient steering (molmoact backend): Gemini guidance + "
                         "keypoint-gradient injection into the flow loop.")
parser.add_argument("--guide-scale", "--guide_scale", type=float, default=2.0,
                    help="Steering strength (molmoact --steer).")
parser.add_argument("--start-ratio", "--start_ratio", type=float, default=0.8,
                    help="Guidance active when flow time <= this (VLS default 0.8).")
parser.add_argument("--task", type=str, default="BananaInBowlTask",
                    help="Task/env loaded first in the interactive demo.")
parser.add_argument("--tasks", type=str, default=None,
                    help=("Comma-separated tasks selectable from the in-viewer scene dropdown. "
                          "Defaults to a curated set good for language steering. "
                          "--task is always included."))
parser.add_argument("--remote-host", "--remote_host", type=str, default="localhost",
                    help="Policy server host (default: localhost).")
parser.add_argument("--remote-port", "--remote_port", type=int, default=8000,
                    help="Policy server port (default: 8000).")
parser.add_argument("--remote-uri", "--remote_uri", type=str, default=None,
                    help="Full WebSocket URI for the policy server (overrides host/port).")
parser.add_argument("--open-loop-horizon", "--open_loop_horizon", type=int, default=None,
                    help="Actions to execute from each predicted chunk before re-querying.")
parser.add_argument("--web-host", "--web_host", type=str, default="0.0.0.0",
                    help="Bind host for the demo web server (default: 0.0.0.0).")
parser.add_argument("--web-port", "--web_port", type=int, default=8090,
                    help="Bind port for the demo web server (default: 8090).")
parser.add_argument("--frame-scale", "--frame_scale", type=float, default=1.0,
                    help="Resize factor for streamed camera frames (default: 1.0).")
parser.add_argument("--jpeg-quality", "--jpeg_quality", type=int, default=80,
                    help="JPEG quality for streamed frames, 1-100 (default: 80).")
parser.add_argument("--cam-width", type=int, default=320,
                    help="Demo-only policy camera render width (policy resizes to 224 anyway; "
                         "lower = much faster per-step RTX render. 1280 = benchmark default).")
parser.add_argument("--cam-height", type=int, default=180,
                    help="Demo-only policy camera render height (720 = benchmark default).")
parser.add_argument("--display-width", type=int, default=960,
                    help="Render res (width) of the third-person camera shown in the web UI. "
                         "Higher = crisper stream; speed is render-overhead-bound so impact is small.")
parser.add_argument("--display-height", type=int, default=540,
                    help="Render res (height) of the third-person display camera.")
parser.add_argument("--decimation", type=int, default=0,
                    help="Demo-only override of control decimation (physics substeps per action; "
                         "0 = keep task default of 8). Lower = faster/smoother sim stepping.")

# AppLauncher needs the flags known at parse time; heavy imports are deferred.
args_cli, _ = parser.parse_known_args()
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if not hasattr(args_cli, "headless") or args_cli.headless is None:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- isaaclab-dependent imports (only safe after AppLauncher) ---------------
import numpy as np  # noqa: E402
import torch  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402

try:
    from policies.molmoact import guidance_manual as _guidance_manual  # noqa: E402
except Exception:
    _guidance_manual = None

# Quiet, demo-friendly defaults.
robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False


class DemoState:
    """Thread-safe bridge between the sim loop (writer) and the web server (reader)."""

    def __init__(self, instruction: str, presets: list[str], task: str, menu: list[str]):
        self._lock = threading.Lock()
        self._instruction = instruction
        self._jpeg: bytes | None = None
        self._running = False          # start PAUSED; runs one episode per Steer press
        self._reset_requested = False  # True => sim loop ends current episode early
        self._load_task: str | None = None  # set => sim loop should rebuild this task
        self._steer_enabled = True     # VLS steering on/off (toggle from UI)
        self._status = {
            "instruction": instruction,
            "task": task,
            "menu": list(menu),
            "presets": list(presets),
            "step": 0,
            "max_steps": 0,
            "episode": 0,
            "success": False,
            "terminated": False,
            "paused": True,
            "loading": False,
            "steer": True,
            "mode": "",
            "guidance": "",
        }
        self._code_instr = None  # instruction the cached guidance code was built for
        self.presets = presets

    # -- run/pause + reset (web -> sim) --
    def set_running(self, running: bool) -> None:
        with self._lock:
            self._running = bool(running)
            self._status["paused"] = not self._running

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def take_reset(self) -> bool:
        with self._lock:
            if self._reset_requested:
                self._reset_requested = False
                return True
            return False

    # -- scene switching (web -> sim) --
    def request_load(self, task: str) -> None:
        with self._lock:
            self._load_task = str(task)

    def take_load(self) -> str | None:
        with self._lock:
            t, self._load_task = self._load_task, None
            return t

    def get_task(self) -> str:
        with self._lock:
            return self._status["task"]

    def set_task(self, task: str, presets: list[str], instruction: str | None = None) -> None:
        with self._lock:
            self._status["task"] = task
            self._status["presets"] = list(presets)
            self.presets = presets
            if instruction is not None:
                self._instruction = instruction
                self._status["instruction"] = instruction

    def set_loading(self, loading: bool) -> None:
        with self._lock:
            self._status["loading"] = bool(loading)

    # -- instruction (web -> sim) --
    def set_instruction(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            self._instruction = text
            self._status["instruction"] = text

    def get_instruction(self) -> str:
        with self._lock:
            return self._instruction

    # -- steering on/off (web -> sim) --
    def set_steer(self, enabled: bool) -> None:
        with self._lock:
            self._steer_enabled = bool(enabled)
            self._status["steer"] = bool(enabled)

    def steer_enabled(self) -> bool:
        with self._lock:
            return self._steer_enabled

    # -- VLS process viz (sim -> web): mode + generated reward code --
    def set_mode(self, mode: str) -> None:
        with self._lock:
            self._status["mode"] = mode

    def set_guidance(self, code: str, instr: str) -> None:
        with self._lock:
            self._status["guidance"] = code
            self._code_instr = instr

    def code_instr(self) -> str | None:
        with self._lock:
            return self._code_instr

    # -- frame + status (sim -> web) --
    def publish(self, jpeg: bytes | None, status_update: dict) -> None:
        with self._lock:
            if jpeg is not None:
                self._jpeg = jpeg
            self._status.update(status_update)
            self._status["instruction"] = self._instruction

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._status)


def _cam_from_obs(obs, *needles):
    """Read a named camera image (H,W,3 uint8) from obs without concatenating."""
    try:
        grp = obs["image_obs"]
        for key in grp:
            if any(n in key.lower() for n in needles):
                return grp[key][0].detach().cpu().numpy().astype(np.uint8)
    except Exception:
        return None
    return None


def render_third_person(obs, kps, quality: int) -> bytes | None:
    """Third-person (over-shoulder) camera as JPEG with the steering keypoints drawn,
    plus a wrist-camera picture-in-picture inset (no keypoints) in the top-right."""
    rgb = _cam_from_obs(obs, "shoulder", "exterior")
    if rgb is None:
        try:
            rgb = next(iter(obs["image_obs"].values()))[0].detach().cpu().numpy().astype(np.uint8)
        except Exception:
            return None
    if rgb is None or rgb.ndim != 3:
        return None
    # Upscale only if the camera is small; a high-res display camera streams ~native.
    up = max(1, round(960 / max(rgb.shape[1], 1)))
    img = cv2.resize(rgb, (rgb.shape[1] * up, rgb.shape[0] * up), interpolation=cv2.INTER_LINEAR) if up > 1 else np.ascontiguousarray(rgb)
    img = np.ascontiguousarray(img)
    for k in (kps or []):
        nm = str(k.get("name", ""))
        if nm.lower() == "table":
            continue
        uv = k.get("uv")
        if not uv:
            continue
        u, v = int(round(uv[0] * up)), int(round(uv[1] * up))
        if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            continue
        is_wall = nm.lower() == "wall"
        dot = (255, 140, 0) if is_wall else (0, 230, 90)     # wall = orange, objects = green
        cv2.circle(img, (u, v), 6, dot, -1)
        cv2.circle(img, (u, v), 6, (255, 255, 255), 1)
        cv2.putText(img, nm, (u + 9, v - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, nm, (u + 9, v - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, dot, 1, cv2.LINE_AA)

    # --- wrist camera picture-in-picture (top-right, no keypoints) ---
    wrist = _cam_from_obs(obs, "wrist")
    if wrist is not None and wrist.ndim == 3:
        iw = int(img.shape[1] * 0.26)
        ih = int(round(iw * wrist.shape[0] / max(wrist.shape[1], 1)))
        wr = cv2.resize(wrist, (iw, ih))
        x0, y0 = img.shape[1] - iw - 14, 14
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + iw + 2, y0 + ih + 2), (255, 255, 255), 2)
        img[y0:y0 + ih, x0:x0 + iw] = wr
        cv2.putText(img, "wrist cam", (x0 + 4, y0 + ih + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, "wrist cam", (x0 + 4, y0 + ih + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return encode_frame(img, quality)


def _overlay_kps(client, kps_full, steering):
    """[{name,uv}] to draw: scene keypoints (excl table) + the static wall label."""
    draw = []
    for k in (kps_full or []):
        if str(k.get("name", "")).lower() == "table" or not k.get("uv"):
            continue
        draw.append({"name": k["name"], "uv": k["uv"]})
    if steering and _guidance_manual is not None and hasattr(client, "project_world"):
        try:
            w = _guidance_manual._WALL
            uv = client.project_world([w["cx"], w["cy"], w["top"]])
            if uv:
                draw.append({"name": "wall", "uv": uv})
        except Exception:
            pass
    return draw


def encode_frame(rgb: np.ndarray, quality: int) -> bytes | None:
    """RGB uint8 HxWxC -> JPEG bytes (BGR conversion so browser colors are correct)."""
    if rgb is None or rgb.size == 0:
        return None
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else None


HTML_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>VLS Demo</title>
<style>
  :root{
    color-scheme:dark;
    --bg:#080a0e; --panel:#11151c; --panel2:#171c25; --border:#222936;
    --muted:#868f9f; --text:#eef1f6; --accent:#4f8cff; --accent2:#8b5cff;
    --green:#46d99a; --red:#ff6b6b;
  }
  *{box-sizing:border-box}
  body{margin:0; color:var(--text); font-family:'Inter',-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;
       -webkit-font-smoothing:antialiased;
       background:radial-gradient(1100px 560px at 50% -12%, #18203250 0%, transparent 60%), var(--bg);}
  .wrap{max-width:1060px; margin:0 auto; padding:30px 22px 48px;}
  header{display:flex; align-items:center; justify-content:space-between; gap:16px; margin-bottom:22px;}
  .brand{display:flex; align-items:center; gap:14px;}
  .logo{width:44px; height:44px; border-radius:13px; display:flex; align-items:center; justify-content:center;
        font-weight:800; font-size:15px; letter-spacing:.5px; color:#fff;
        background:linear-gradient(135deg,var(--accent),var(--accent2));
        box-shadow:0 8px 24px rgba(79,140,255,.4), inset 0 1px 0 rgba(255,255,255,.25);}
  .brand h1{font-size:19px; font-weight:700; margin:0; letter-spacing:-.3px; line-height:1.3;}
  .brand .sub{font-size:12.5px; color:var(--muted); margin-top:3px; letter-spacing:.2px;}
  .live{display:flex; align-items:center; gap:8px; font-size:11.5px; font-weight:600; letter-spacing:1px;
        color:var(--muted); border:1px solid var(--border); background:var(--panel);
        padding:8px 13px; border-radius:999px;}
  .live .ldot{width:8px; height:8px; border-radius:50%; background:var(--green); animation:pulse 2s infinite;}
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(70,217,154,.5)}70%{box-shadow:0 0 0 8px rgba(70,217,154,0)}100%{box-shadow:0 0 0 0 rgba(70,217,154,0)}}
  .camwrap{position:relative; border-radius:18px; overflow:hidden; border:1px solid var(--border);
        background:#000; box-shadow:0 22px 60px rgba(0,0,0,.55);}
  #cam{width:100%; background:#000; display:block; aspect-ratio:16/6; object-fit:contain;}
  #overlay{position:absolute; inset:0; display:none; align-items:center; justify-content:center;
        background:rgba(8,10,14,.74); backdrop-filter:blur(4px); font-size:15px; color:#cfd3d8; letter-spacing:.3px;}
  #overlay.show{display:flex;}
  .camcap{position:absolute; left:14px; bottom:13px; display:flex; align-items:center; gap:10px; max-width:84%;
        background:rgba(10,12,16,.55); backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,.09);
        padding:9px 14px; border-radius:12px; font-size:13.5px;}
  .camcap .tag{font-size:10px; font-weight:700; letter-spacing:1px; color:var(--accent);
        border:1px solid rgba(79,140,255,.4); border-radius:6px; padding:2px 6px;}
  .camcap b{color:#fff; font-weight:600;}
  .bar{display:flex; gap:10px; margin-top:18px;}
  #prompt{flex:1; padding:14px 16px; font-size:15px; border-radius:13px; border:1px solid var(--border);
        background:var(--panel); color:#fff; outline:none; transition:border .15s, box-shadow .15s;}
  #prompt:focus{border-color:var(--accent); box-shadow:0 0 0 3px rgba(79,140,255,.18);}
  #prompt::placeholder{color:#5b6373;}
  button{font-family:inherit; font-size:14.5px; font-weight:600; border:0; border-radius:13px; cursor:pointer;
        padding:14px 22px; color:#fff; transition:transform .06s, filter .15s, background .15s, border-color .15s;}
  button:active{transform:translateY(1px);}
  #startbtn{background:linear-gradient(135deg,var(--accent),#3a6fe0); box-shadow:0 8px 22px rgba(79,140,255,.34);}
  #startbtn:hover{filter:brightness(1.08);}
  .ctrl{background:var(--panel2); border:1px solid var(--border);}
  .ctrl:hover{border-color:#3a4252;}
  #stopbtn{color:#ff8a8a;}
  #steertgl{min-width:120px; text-align:center;}
  .row{display:flex; align-items:center; justify-content:space-between; gap:14px; margin-top:16px;}
  .hint{font-size:13px; color:var(--muted);}
  .hint b{color:#cfd6e2; font-weight:600;}
  .status{font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums; white-space:nowrap;}
  .status b{color:var(--text); font-weight:700;}
  .panel{margin-top:16px; border:1px solid var(--border); border-radius:14px; overflow:hidden; background:var(--panel);}
  .panel-h{display:flex; align-items:center; gap:9px; padding:10px 14px; border-bottom:1px solid var(--border);
           font-size:12.5px; color:var(--muted); background:var(--panel2);}
  .panel-h .pdot{width:8px;height:8px;border-radius:50%;background:var(--accent2);}
  .panel-h b{color:var(--text); font-weight:600;}
  .code{margin:0; padding:14px 16px; font-family:'SF Mono',ui-monospace,Menlo,Consolas,monospace;
        font-size:12.5px; line-height:1.5; color:#cfe3ff; white-space:pre-wrap; overflow-x:auto;
        max-height:340px; overflow-y:auto;}
</style></head>
<body><div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">VLS</div>
      <div>
        <h1>VLS: Steering Pretrained Robot Policies<br>via Vision–Language Models</h1>
      </div>
    </div>
    <div class="live"><span class="ldot"></span> LIVE</div>
  </header>
  <div class="camwrap">
    <img id="cam" alt="camera stream"/>
    <div id="overlay">Loading scene…</div>
    <div class="camcap"><span class="tag">PROMPT</span> <b id="curinstr">—</b></div>
  </div>
  <div class="bar">
    <input id="prompt" placeholder="Type an instruction…  e.g. put the blue cube in the box" autocomplete="off"/>
    <button id="startbtn">Start</button>
    <button id="stopbtn" class="ctrl">Stop</button>
    <button id="steertgl" class="ctrl">ON</button>
  </div>
  <div class="row">
    <div class="hint">Type a prompt → <b>Start</b> · Toggle <b>ON</b> = VLS + pi0.5 · <b>OFF</b> = pi0.5</div>
    <div class="status">step <b id="step">0</b> / <b id="maxsteps">0</b></div>
  </div>
  <div class="panel">
    <div class="panel-h"><span class="pdot"></span> Generated VLS reward · <b id="mode">—</b></div>
    <pre id="code" class="code"># Press Start with steering ON to see the VLS reward generated for your prompt.</pre>
  </div>
</div>
<script>
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = "blob";
  const cam = document.getElementById("cam");
  let lastUrl = null;
  let curTask = null;
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") {
      const s = JSON.parse(ev.data);
      if (s.type === "presets") { return; }  // superseded by status payload
      document.getElementById("overlay").className = s.loading ? "show" : "";
      document.getElementById("curinstr").textContent = s.instruction || "—";
      document.getElementById("step").textContent = s.step;
      document.getElementById("maxsteps").textContent = s.max_steps;
      if (s.mode !== undefined) document.getElementById("mode").textContent = s.mode;
      const code = document.getElementById("code");
      if (s.guidance !== undefined && s.guidance !== code._last) {
        code.textContent = s.guidance || "# (no reward — steering off)";
        code._last = s.guidance;
      }
      paused = !!s.paused;
      steerOn = (s.steer !== false);
      const stbtn = document.getElementById("steertgl");
      stbtn.textContent = steerOn ? "ON" : "OFF";
      stbtn.style.background = steerOn ? "#2563eb" : "#52585f";
    } else {
      const url = URL.createObjectURL(ev.data);
      cam.src = url;
      if (lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl = url;
    }
  };
  let paused = false, steerOn = true;
  function command(cmd){ if (ws.readyState === 1) ws.send(JSON.stringify({command: cmd})); }
  const input = document.getElementById("prompt");
  // Fresh run with the CURRENT prompt: set instruction, restart the episode, run.
  function freshStart(){
    if (ws.readyState !== 1) return;
    if (input.value) ws.send(JSON.stringify({instruction: input.value}));
    ws.send(JSON.stringify({command: "reset"}));
    ws.send(JSON.stringify({command: "start"}));
  }
  // Single Start button = restart-from-scratch: every press runs a fresh episode
  // with whatever prompt is in the box.
  document.getElementById("startbtn").onclick = () => { freshStart(); };
  document.getElementById("stopbtn").onclick = () => { command("stop"); };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") freshStart(); });
  document.getElementById("steertgl").onclick = () => { command(steerOn ? "steer_off" : "steer_on"); };
</script>
</body></html>
"""


def build_app(state: DemoState, task_name: str):
    import asyncio

    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    app = FastAPI()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML_PAGE)

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        # Send presets + task once on connect.
        await websocket.send_text(json.dumps({"type": "presets", "presets": state.presets, "task": task_name}))

        async def receiver():
            try:
                while True:
                    msg = await websocket.receive_text()
                    try:
                        data = json.loads(msg)
                    except json.JSONDecodeError:
                        continue
                    if "instruction" in data:
                        state.set_instruction(str(data["instruction"]))
                    cmd = data.get("command")
                    if cmd == "stop":
                        state.set_running(False)
                    elif cmd == "start":
                        state.set_running(True)
                    elif cmd == "reset":
                        state.request_reset()
                    elif cmd == "steer_on":
                        state.set_steer(True)
                    elif cmd == "steer_off":
                        state.set_steer(False)
            except WebSocketDisconnect:
                pass

        recv = asyncio.create_task(receiver())
        try:
            while True:
                jpg = state.get_jpeg()
                if jpg is not None:
                    await websocket.send_bytes(jpg)
                await websocket.send_text(json.dumps({"type": "status", **state.get_status()}))
                await asyncio.sleep(1 / 15)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            recv.cancel()

    return app


def start_web_server(state: DemoState, task_name: str, host: str, port: int) -> None:
    import uvicorn

    app = build_app(state, task_name)
    # loop="asyncio" is REQUIRED: the default ("auto") installs uvloop as the global
    # event-loop policy, which then makes Isaac's env.close() crash on the main thread
    # ("no current event loop in thread 'MainThread'") when it schedules USD-cleanup
    # callbacks. Plain asyncio in the server thread leaves the main thread's loop intact.
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", loop="asyncio")
    server = uvicorn.Server(config)
    # Run uvicorn in a daemon thread; the sim loop owns the main thread.
    threading.Thread(target=server.run, daemon=True).start()


class ModelToggleClient:
    """Routes inference to one of two policy servers based on the UI toggle:
    steering-ON selects pi0.5, OFF selects pi0 (both unsteered). Lets the demo
    A/B two whole models live without a restart."""

    def __init__(self, on_client, off_client):
        self.on, self.off = on_client, off_client
        self._state = None

    def bind_state(self, state):
        self._state = state

    def _cur(self):
        return self.on if (self._state is None or self._state.steer_enabled()) else self.off

    def infer(self, *a, **k):
        return self._cur().infer(*a, **k)

    def reset(self):
        for c in (self.on, self.off):
            if hasattr(c, "reset"):
                try:
                    c.reset()
                except Exception:
                    pass

    def attach_env(self, env):
        for c in (self.on, self.off):
            if hasattr(c, "attach_env"):
                c.attach_env(env)

    # Keypoint/overlay viz comes from the steered (ON) client.
    def _scene_keypoints(self, env_id: int = 0):
        return self.on._scene_keypoints(env_id) if hasattr(self.on, "_scene_keypoints") else []

    def project_world(self, xyz, env_id: int = 0):
        return self.on.project_world(xyz, env_id) if hasattr(self.on, "project_world") else None


def make_client():
    if args_cli.policy_backend == "steer_vs_pi0":
        # ON = VLS-steered pi0.5 (PyTorch ZMQ steer server); OFF = pi0 (OpenPI server).
        from policies.molmoact.client import MolmoActDroidClient
        from policies.pi0_family.client import Pi0DroidJointposClient
        on_kwargs = {"open_loop_horizon": args_cli.open_loop_horizon} if args_cli.open_loop_horizon else {}
        on = MolmoActDroidClient(steer=True, guide_scale=args_cli.guide_scale,
                                 start_ratio=args_cli.start_ratio,
                                 remote_host=args_cli.molmo_host, remote_port=args_cli.molmo_port, **on_kwargs)
        off = Pi0DroidJointposClient(remote_host=args_cli.model_host, remote_port=args_cli.pi0_port,
                                     policy_variant="pi0")
        return ModelToggleClient(on, off)

    if args_cli.policy_backend == "compare":
        from policies.pi0_family.client import Pi0DroidJointposClient
        on = Pi0DroidJointposClient(remote_host=args_cli.model_host, remote_port=args_cli.pi05_port,
                                    open_loop_horizon=args_cli.open_loop_horizon, policy_variant="pi05")
        off = Pi0DroidJointposClient(remote_host=args_cli.model_host, remote_port=args_cli.pi0_port,
                                     open_loop_horizon=args_cli.open_loop_horizon, policy_variant="pi0")
        return ModelToggleClient(on, off)

    if args_cli.policy_backend in ("molmoact", "pi05_steer"):
        # Both speak the same ZMQ+keypoints protocol; only the server differs.
        from policies.molmoact.client import MolmoActDroidClient
        kwargs = dict(
            remote_host=args_cli.molmo_host,
            remote_port=args_cli.molmo_port,
            open_loop_horizon=args_cli.open_loop_horizon,
        )
        return MolmoActDroidClient(
            steer=args_cli.steer, guide_scale=args_cli.guide_scale,
            start_ratio=args_cli.start_ratio,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

    kwargs = dict(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        remote_uri=args_cli.remote_uri,
        open_loop_horizon=args_cli.open_loop_horizon,
        policy_variant=args_cli.policy,
    )
    return Pi0DroidJointposClient(**{k: v for k, v in kwargs.items() if v is not None})


DEFAULT_MENU = [
    "BananaInBowlTask",
    "GrabAFruitTask",
    "GrabABagelTask",
    "FruitsMovingOrangeOrLimeTask",
    "MoveBananaToBagelPlateTask",
]


def presets_from_cfg(env_cfg) -> tuple[list[str], str]:
    """Return (preset instruction strings, default instruction) from a task's variants."""
    variants = getattr(env_cfg, "_instruction_variants", env_cfg.instruction)
    if isinstance(variants, dict):
        presets = [str(v) for v in variants.values()]
        default_instr = str(variants.get("default", next(iter(variants.values()))))
    else:
        presets = [str(variants)]
        default_instr = str(variants)
    return presets, str(env_cfg.instruction or default_instr)


def _shrink_scene_cameras(env_cfg, width: int, height: int) -> None:
    """Demo-only: render the policy cameras at low res (the policy resizes to 224
    regardless), cutting per-step RTX render cost by ~(1280*720)/(w*h) and making the
    stream much smoother. Mutates the parsed env_cfg's scene sensors in-process only —
    repo files and other runs are untouched. Keep 16:9 (e.g. 320x180) for framing."""
    if width <= 0 or height <= 0:
        return
    scene = getattr(env_cfg, "scene", None)
    if scene is None:
        return
    dw, dh = int(args_cli.display_width), int(args_cli.display_height)
    n = 0
    for name, v in list(vars(scene).items()):
        if hasattr(v, "width") and hasattr(v, "height") and hasattr(v, "prim_path"):
            try:
                # The third-person (over-shoulder) camera is the one streamed to the
                # web UI -> render it at high res for a crisp image. All other cameras
                # (wrist, etc.) stay small for speed (the policy resizes to 224 anyway).
                if "shoulder" in name.lower() or "exterior" in name.lower() or "head" in name.lower():
                    v.width, v.height = dw, dh
                else:
                    v.width, v.height = int(width), int(height)
                n += 1
            except Exception:
                pass
    print(f"[demo] cameras: display(over-shoulder)->{dw}x{dh}, others->{width}x{height} ({n} sensors)", flush=True)


# Wrap parse_env_cfg so every env built in this process renders cameras at the
# demo resolution. create_env() (str path) calls runtime.parse_env_cfg, then
# gym.make builds the cameras from the cfg we just shrank.
def _install_camera_shrink():
    import robolab.core.environments.runtime as _rt
    if getattr(_rt, "_demo_cam_wrapped", False):
        return
    _orig = _rt.parse_env_cfg

    def _wrapped(*a, **k):
        cfg = _orig(*a, **k)
        try:
            _shrink_scene_cameras(cfg, args_cli.cam_width, args_cli.cam_height)
            if args_cli.decimation and args_cli.decimation > 0:
                cfg.decimation = int(args_cli.decimation)
                if getattr(cfg, "sim", None) is not None and hasattr(cfg.sim, "render_interval"):
                    cfg.sim.render_interval = int(args_cli.decimation)
                print(f"[demo] decimation -> {args_cli.decimation} (faster sim stepping)", flush=True)
        except Exception:
            logger.exception("env_cfg tweak failed; keeping defaults")
        return cfg

    _rt.parse_env_cfg = _wrapped
    _rt._demo_cam_wrapped = True


def make_env(task: str):
    _install_camera_shrink()
    return create_env(task, device=args_cli.device, num_envs=1, instruction_type="default", policy=args_cli.policy)


def run_one_env(env, env_cfg, client, state: DemoState):
    """Run episodes on one env until a scene switch is requested or the app exits.

    Returns the task name to switch to (str), or None if the app is shutting down.
    Reads the live instruction each step; honors stop/resume and reset.
    """
    import omni.kit.app
    import omni.timeline

    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    action_dim = getattr(getattr(env, "action_manager", None), "total_action_dim", None) \
        or env.action_space.shape[-1]
    max_steps = env.max_episode_length
    cur_task = state.get_task()
    episode = 0

    def pending_switch():
        t = state.take_load()
        return t if (t and t != cur_task) else None

    while simulation_app.is_running():
        sw = pending_switch()
        if sw:
            return sw
        episode += 1
        obs, _ = env.reset()
        obs, _ = env.reset()
        try:
            client.reset()
        except Exception:
            pass

        # Hold at the fresh episode until the user presses Steer (no auto-run).
        # Honor scene switches / app shutdown while holding. Keep rendering + streaming
        # the (static) scene so the camera is visible while paused.
        _hold_i = 0
        while not state.is_running():
            if not simulation_app.is_running():
                return None
            sw = pending_switch()
            if sw:
                return sw
            state.take_reset()  # discard stale resets while idle
            kit_app.update()
            if _hold_i % 4 == 0:
                try:
                    kps_p = None
                    if state.steer_enabled() and hasattr(client, "_scene_keypoints"):
                        try:
                            kps_p = client._scene_keypoints(env_id=0)
                        except Exception:
                            kps_p = None
                    jpeg = render_third_person(obs, _overlay_kps(client, kps_p, state.steer_enabled()),
                                               args_cli.jpeg_quality)
                    state.publish(jpeg, {"step": 0, "episode": episode, "success": False})
                except Exception:
                    pass
            _hold_i += 1

        for step in range(max_steps):
            if not simulation_app.is_running():
                return None
            while not timeline.is_playing():
                kit_app.update()

            sw = pending_switch()
            if sw:
                return sw
            if state.take_reset():
                break

            # Paused from the UI: hold the robot, keep the app + stream alive.
            interrupt = None  # "reset" | "switch:<task>"
            while not state.is_running():
                if not simulation_app.is_running():
                    return None
                sw = pending_switch()
                if sw:
                    interrupt = f"switch:{sw}"
                    break
                if state.take_reset():
                    interrupt = "reset"
                    break
                kit_app.update()
            if interrupt == "reset":
                break
            if isinstance(interrupt, str) and interrupt.startswith("switch:"):
                return interrupt.split(":", 1)[1]
            if not state.is_running():
                continue

            instruction = state.get_instruction()
            # Live steering on/off: flip the client's per-request steer flag so the
            # user can compare base pi0.5 (off) vs VLS-steered (on) without a restart.
            if hasattr(client, "_steer"):
                client._steer = state.steer_enabled()
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            for env_id in env.active_env_ids:
                try:
                    ret = client.infer(obs, instruction, env_id=env_id)
                    actions[env_id] = torch.tensor(ret["action"], device=env.device)
                except Exception:
                    # Transient policy/transport failure: hold this step (zero action)
                    # and keep the demo alive rather than crashing the whole loop.
                    logger.exception("policy infer failed; holding this step")

            obs, _, _, _, _ = env.step(actions)

            # --- VLS process viz: when steering is ON, fetch the privileged keypoints
            # the policy is conditioned on (for overlay) and the generated reward code.
            steering = state.steer_enabled()
            kps_overlay = None
            if steering and hasattr(client, "_scene_keypoints"):
                try:
                    kps_overlay = client._scene_keypoints(env_id=0)
                except Exception:
                    kps_overlay = None
            state.set_mode("VLS steering · pi0.5" if steering else "pi0.5 (base, no steering)")
            if steering and kps_overlay and _guidance_manual is not None and instruction != state.code_instr():
                try:
                    names = [k["name"] for k in kps_overlay]
                    xyz = np.asarray([k["xyz"] for k in kps_overlay], dtype=np.float32)
                    state.set_guidance(_guidance_manual.describe(instruction, names, xyz), instruction)
                except Exception:
                    pass
            elif not steering and state.code_instr() != "__base__":
                state.set_guidance("# OFF — running pi0.5 base policy (no VLS steering / reward).", "__base__")

            # Stream the third-person camera, with keypoints + wall overlaid when steering.
            try:
                jpeg = render_third_person(obs, _overlay_kps(client, kps_overlay, steering),
                                           args_cli.jpeg_quality)
            except Exception:
                jpeg = None

            results = env.get_env_results()
            success = bool(results[0].get("success", False)) if results else False
            state.publish(jpeg, {
                "step": step + 1,
                "max_steps": int(max_steps),
                "episode": episode,
                "success": success,
                "terminated": bool(env.all_terminated),
            })

            if env.all_terminated:
                break

        # Very brief hold on the final frame so the outcome is visible, then reset.
        # Short so an explicit Start (restart) feels snappy.
        for _ in range(6):
            if not simulation_app.is_running():
                return None
            kit_app.update()
        env.reset_eval_state()
        # NOTE: do NOT auto-pause here — once the user presses Steer, keep running
        # (auto-resetting) for a smooth demo until they press Stop. The start-paused
        # hold at the top of the loop handles "off until first Steer".

    return None


def run_demo(client, state: DemoState, env, env_cfg) -> None:
    """Orchestrate: run one env; on a scene-switch request, rebuild and continue."""
    try:
        while simulation_app.is_running():
            switch_to = run_one_env(env, env_cfg, client, state)
            if switch_to is None:
                break
            # Rebuild the env for the newly requested scene.
            print(f"\033[96m[RoboLab] Switching scene -> {switch_to}\033[0m")
            state.set_loading(True)
            try:
                env.close()
            except Exception:
                logger.exception("env.close() failed during scene switch")
            env, env_cfg = make_env(switch_to)
            if hasattr(client, "attach_env"):
                client.attach_env(env)
            presets, default_instr = presets_from_cfg(env_cfg)
            state.set_task(switch_to, presets, instruction=default_instr)
            state.set_loading(False)
    finally:
        try:
            env.close()
        except Exception:
            logger.exception("env.close() failed")


def main() -> None:
    # Guarantee the main thread has an asyncio loop. Isaac's stage-close callbacks call
    # asyncio.get_event_loop()/ensure_future() during env.close(); without a loop set on
    # this thread (and with uvicorn running its own loop in another thread) those raise.
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Build the selectable scene menu (--task is always first/included).
    if args_cli.tasks:
        menu = [t.strip() for t in args_cli.tasks.split(",") if t.strip()]
    else:
        menu = list(DEFAULT_MENU)
    if args_cli.task not in menu:
        menu.insert(0, args_cli.task)

    # Register every selectable task up front so create_env() can build any of them.
    auto_register_droid_envs(task=menu)

    env, env_cfg = make_env(args_cli.task)
    presets, default_instr = presets_from_cfg(env_cfg)
    state = DemoState(
        instruction=str(env_cfg.instruction or default_instr),
        presets=presets,
        task=args_cli.task,
        menu=menu,
    )
    client = make_client()
    if hasattr(client, "bind_state"):
        client.bind_state(state)  # 'compare' toggle reads the steer flag to pick pi0.5/pi0
    if hasattr(client, "attach_env"):
        client.attach_env(env)

    start_web_server(state, args_cli.task, args_cli.web_host, args_cli.web_port)
    backend = "molmoact2-droid" if args_cli.policy_backend == "molmoact" else args_cli.policy
    print(f"\033[96m[RoboLab] Interactive demo on http://{args_cli.web_host}:{args_cli.web_port} "
          f"(task: {args_cli.task}, backend: {backend}, scenes: {len(menu)})\033[0m")

    run_demo(client, state, env, env_cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Interactive demo terminated with error: {e}\033[0m")
        traceback.print_exc()
    finally:
        simulation_app.close()
        sys.exit(0)
