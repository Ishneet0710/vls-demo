# VLS Demo — Language Steering of Pretrained Robot Policies (RoboLab)

Interactive browser demo of **VLS (Vision-Language Steering)** applied to **pi0.5** on
NVIDIA RoboLab (Isaac Lab / Isaac Sim). A FastAPI + WebSocket page streams the
robot's third-person camera with the privileged keypoints overlaid, shows the
**generated reward function** for each prompt, and lets you type instructions that
re-steer the policy live. A toggle switches between **VLS-steered pi0.5** and a
**base model**.

This repo holds the **custom files we added/modified** on top of RoboLab + OpenPI.
It is an overlay, not a standalone runnable package — see setup below.

## What's here
```
policies/molmoact/
  pi05_steer_server.py   # pi0.5 (PyTorch) + VLS gradient steering ZMQ server
  guidance_manual.py     # hand-written guidance: parses prompt -> per-stage reward
                         #   (reach / gated-place / stack / hold / wall-avoid), describe()
  guidance_gemini.py     # Gemini-3.x fallback that emits guidance in the same style
  client.py              # RoboLab client: privileged keypoints + camera projection
  molmoact_steer.py      # differentiable Franka FK + steering core
  server.py              # MolmoAct2 server (alt policy backend)
  _fk_base_transform.npy / _fk_calib_*  # FK calibration to the sim EE
policies/pi0_family/
  run_interactive.py     # the FastAPI+WS demo (UI, overlay, reward panel, toggle)
robolab/tasks/benchmark/
  colored_blocks_demo.py # the demo task (colored cubes + box + bowl + fruit + wall)
assets/scenes/
  colored_blocks.usda    # the scene (cubes, containers, fruit, static wall obstacle)
VLS_SESSION_SUMMARY.md   # full design notes + results
```

## Reproducible setup FROM SCRATCH (clean machine)

This repo is an **overlay** — it holds only our custom files. To run, you also need
RoboLab, OpenPI, Isaac Sim/Lab, and the pi0.5 checkpoint. Everything below is pinned
to the exact versions this was built and tested against.

### Versions (what this was built on)
| Component | Pin |
|---|---|
| RoboLab | `github.com/NVLabs/RoboLab` @ `7d45d74904eade3b578a8eb1f2f9f89bc3d40326` (branch main) |
| OpenPI (fork) | `github.com/xuningy/openpi.git` @ `aa6420561529593114160d05e5ad155792b272f3` |
| Vision-Language-Steering | `github.com/Vision-Language-Steering/code.git` @ `b6e87edc217a2a7682caadba285677fc79eb6f81` (optional; only the Gemini guidance template) |
| Python | 3.11.13 |
| torch (RoboLab venv) | 2.7.0+cu128 · (OpenPI venv) 2.11.0+cu128 — **CUDA 12.8** |
| Isaac Sim / Isaac Lab | 5.0 / 2.2.0 (installed automatically by RoboLab's `uv sync`) |

### Hardware / system prereqs
- NVIDIA GPU with **CUDA 12.8** drivers. (Built on Blackwell sm_120 → needs cu128 torch; the
  pins above already use cu128.) ~24 GB free VRAM is comfortable for one steered demo.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) and system `ffmpeg`.
- ~30 GB disk (Isaac Sim + checkpoints).

### Step 1 — RoboLab (this installs Isaac Sim 5.0 + Isaac Lab 2.2.0 automatically)
```bash
cd ~ && git clone https://github.com/NVLabs/RoboLab && cd RoboLab
git checkout 7d45d74904eade3b578a8eb1f2f9f89bc3d40326
uv sync                 # creates .venv with Isaac Sim/Lab; accepts EULA headlessly
export ROBOLAB=$(pwd)
```

### Step 2 — OpenPI (policy serving + JAX→PyTorch conversion)
```bash
cd ~ && git clone https://github.com/xuningy/openpi.git && cd openpi
git checkout aa6420561529593114160d05e5ad155792b272f3
uv sync                 # creates openpi/.venv
# Blackwell only: ensure cu128 torch in this venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch
export OPENPI=$(pwd)
```

### Step 3 — pi0.5 PyTorch checkpoint (for the steering server)
The steered server needs a **PyTorch** copy of `pi05_droid_jointpos` at
`~/.cache/openpi/pi05_droid_jointpos_pytorch` (`config.json` + `model.safetensors`
+ `assets/droid/norm_stats.json`). You don't back this up — you **reconvert** it:
```bash
cd $OPENPI
# (3a) download the JAX checkpoint locally: start the server once and Ctrl-C after it
#      prints "listening". Downloads to ~/.cache/openpi/openpi-assets-simeval/...
.venv/bin/python scripts/serve_policy.py --port 8999 policy:checkpoint \
  --policy.config=pi05_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos      # Ctrl-C once "listening"

# (3b) convert JAX -> PyTorch (tyro CLI). checkpoint_dir = the .../params subdir; the
#      script auto-copies the sibling assets/ (norm stats); pi05 is detected from the path.
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --config_name pi05_droid_jointpos \
  --checkpoint_dir ~/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos/params \
  --output_path   ~/.cache/openpi/pi05_droid_jointpos_pytorch

# (3c) safety net: ensure norm stats are in place (no-op if convert already copied them)
cp -rn ~/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos/assets \
       ~/.cache/openpi/pi05_droid_jointpos_pytorch/ 2>/dev/null || true
# verify -> should show config.json, model.safetensors, assets/droid/norm_stats.json
find ~/.cache/openpi/pi05_droid_jointpos_pytorch -maxdepth 3 -type f
```
The pi0 / pi0.5 **JAX** checkpoints used by `serve_policy.py` (incl. the OFF=pi0 side)
auto-download from `gs://openpi-assets-simeval/...` on first launch — no manual step.

### Step 4 — overlay these custom files into RoboLab
```bash
cd <this-repo>
cp -r policies/molmoact                       $ROBOLAB/policies/
cp    policies/pi0_family/run_interactive.py  $ROBOLAB/policies/pi0_family/
cp    robolab/tasks/benchmark/colored_blocks_demo.py $ROBOLAB/robolab/tasks/benchmark/
cp    assets/scenes/colored_blocks.usda       $ROBOLAB/assets/scenes/
cp    assets/scenes/_images/colored_blocks.png $ROBOLAB/assets/scenes/_images/
```

### Step 5 — (optional) Gemini guidance fallback
Only needed if a prompt can't be parsed by the hand-written `guidance_manual.py`
(the demo scene never needs it). Set `GOOGLE_API_KEY` in the env and optionally
clone the VLS template repo (pin above) to `VLS_TEMPLATE_DIR`.

Then jump to **Launch** below.

## Launch (the runbook)
GPU layout we used: **5** = steering server, **6** = demo, **7** = pi0 server.

1) **VLS steering server** (pi0.5 PyTorch, OpenPI venv):
```bash
cd $ROBOLAB
GOOGLE_API_KEY=$GOOGLE_API_KEY CUDA_VISIBLE_DEVICES=5 TORCHDYNAMO_DISABLE=1 \
  $OPENPI/.venv/bin/python policies/molmoact/pi05_steer_server.py \
  --port 5556 --steer --num-steps 5
```

2) **pi0 server** (for the OFF side of the toggle; OpenPI venv):
```bash
cd $OPENPI
XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/serve_policy.py \
  --port 8001 policy:checkpoint --policy.config=pi0_droid_jointpos \
  --policy.dir=gs://openpi-assets-simeval/pi0_droid_jointpos
```

3) **Demo** (RoboLab venv) — `steer_vs_pi0`: ON = steered pi0.5, OFF = pi0:
```bash
cd $ROBOLAB
OMNI_KIT_ACCEPT_EULA=YES CUDA_VISIBLE_DEVICES=6 .venv/bin/python \
  policies/pi0_family/run_interactive.py \
  --policy-backend steer_vs_pi0 --molmo-port 5556 --pi0-port 8001 \
  --task ColoredBlocksDemoTask --tasks ColoredBlocksDemoTask \
  --decimation 4 --web-port 8090
```
Open `http://<host>:8090`.

### Other backends `run_interactive.py` supports
- `--policy-backend pi05_steer --steer` : single pi0.5; toggle = steered vs base pi0.5.
- `--policy-backend compare` : ON = pi0.5 (`--pi05-port`), OFF = pi0 (`--pi0-port`), both unsteered (JAX).

### Useful demo flags
`--decimation 4` (faster sim stepping), `--display-width/--display-height` (third-person
render res, default 960x540), `--cam-width/--cam-height` (other cams, default 320x180).

## Public link (optional, for remote presenting)
ngrok with a reserved (free) domain + a restart watchdog:
```bash
ngrok config add-authtoken <YOUR_TOKEN>
# watchdog: relaunch ngrok if it ever drops (same URL)
while true; do ngrok http 8090 --domain=<your>.ngrok-free.dev --log=stdout >> /tmp/ngrok.log 2>&1; sleep 3; done
```
(Free tier shows a one-time "Visit Site" interstitial in the browser.)

## Demo prompts
- `put the blue cube in the box` / `put the red block in the bowl`
- `put the blue cube on top of the red block` (stacking)
- `pick up the red cube and hold it there` (grasp + hold)
- `put the blue cube in the box avoiding the wall` (obstacle avoidance)
- chain with "then": `put red on top of blue, then yellow on top of red`

## Notes
- VLS steers the **end-effector trajectory** via `autograd.grad(reward)` injected into the
  flow-matching Euler step; it does **not** control the gripper (the policy does) — see
  `VLS_SESSION_SUMMARY.md` for the honest results and design rationale.
- The wall is a static collider; its position is hardcoded as `_WALL` in `guidance_manual.py`
  (and used for both the avoidance reward and the on-screen label).
- License: RoboLab / OpenPI / VLS are third-party projects under their own licenses; this
  overlay only contains our additions.
