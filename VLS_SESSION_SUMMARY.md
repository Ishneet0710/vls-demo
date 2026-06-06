# RoboLab + VLS Steering — Session Summary

Date: 2026-06-02. Machine: research-8xpro6000 (8× RTX PRO 6000 Blackwell, sm_120).

## TL;DR
Set up RoboLab, got pi0.5 (OpenPI) running, built an **interactive browser demo**
(language steering of a robot policy on RoboLab scenes), then integrated two
policies and added **VLS gradient steering** to both:
- **MolmoAct2-DROID** (PyTorch) — drop-in policy swap + gradient steering.
- **pi0.5 droid** (converted JAX→PyTorch) — gradient steering, the policy that was
  "good and fast".

Both verified end-to-end in the browser: type a prompt → Gemini writes a
differentiable guidance fn → gradient injected into the policy's flow-matching
loop → robot steers toward the chosen object. Real task successes observed.

---

## Environment / infra facts
- **GPUs 0–3**: another team's MiniMax-M2 vLLM server (port 8000). DO NOT TOUCH.
- **GPUs 4–7**: free for our use.
- **Disk**: freed 214 GB at session start by deleting a stale Codex WAL backup
  (`/home/ubuntu/snap/codex/58/logs_2.sqlite-wal.codex-repair-*.bak`).
- **Venvs**:
  - RoboLab: `/root/ishneet/RoboLab/.venv` (python 3.11, isaaclab/isaacsim, cu128 torch).
  - OpenPI (pi0.5 JAX server + pi0.5 PyTorch): `/root/ishneet/openpi/.venv`
    (torch upgraded to **2.11.0+cu128** for Blackwell sm_120 support).
  - MolmoAct2: `/root/ishneet/molmoact-venv` (transformers + torch 2.12+cu130).
- **EULA**: `export OMNI_KIT_ACCEPT_EULA=Y` before RoboLab runs (one-time cached).
- **Gemini key**: `GOOGLE_API_KEY` in `/root/ishneet/.env` (also `HF_TOKEN`).
- Permissions: broad allowlist in `.claude/settings.local.json` (gitignored).

---

## What was built (all in the working tree, NOT committed)

### Interactive browser demo
`policies/pi0_family/run_interactive.py` — FastAPI+WebSocket browser UI:
- Live camera stream, prompt box (live language steering), **scene dropdown**
  (rebuilds env in-browser), Stop/Resume, Reset.
- Flags: `--policy-backend {pi05,molmoact,pi05_steer}`, `--steer`,
  `--guide-scale`, `--start-ratio`, `--task`, `--web-port`, `--molmo-port`.

### MolmoAct2 integration + steering (`policies/molmoact/`)
- `server.py` — MolmoAct2-DROID ZMQ server (plain + `--steer`).
- `client.py` — RoboLab `InferenceClient` (`MolmoActDroidClient`): extracts
  cameras + state + **privileged scene keypoints** (object world positions),
  ZMQ wire. Reused by the pi0.5-steer backend too.
- `molmoact_steer.py` — VLS gradient steering: monkeypatches MolmoAct2's flow
  loop, injects `autograd.grad(reward)` via differentiable decode→FK→reward.
  Contains `DifferentiableFrankaFK` (reused everywhere).
- `guidance_gemini.py` — loads the REAL VLS template
  (`/root/ishneet/vision-language-steering/vlm_query/guidance_template.txt`),
  calls **gemini-2.5-pro**, compiles `stageN_guidance(keypoints, action_seq)`.

### pi0.5 droid PyTorch + steering
- Converted JAX→PyTorch via OpenPI's `examples/convert_jax_model_to_pytorch.py`
  → `/root/.cache/openpi/pi05_droid_jointpos_pytorch` (6.8 GB).
- `policies/molmoact/pi05_steer_server.py` — loads the PyTorch pi0.5 via OpenPI's
  `create_trained_policy`, monkeypatches `sample_actions` flow loop for gradient
  steering. pi0.5 outputs joint DELTAS → decode = quantile-unnorm + current joints
  → absolute → FK. Same ZMQ protocol, so the same client drives it.

### Shared / calibration
- `_fk_base_transform.npy` — Franka FK base→world transform, Kabsch-calibrated
  to the sim's real EE (residual **2.5 mm**). `_fk_calib_collect.py` (RoboLab) +
  `_fk_calib_fit.py` (Kabsch).
- Scratch/tests: `/root/ishneet/_vls_scratch/` (perf + parity scripts).

---

## Design choices
- **Privileged sim info** instead of DINO/SAM/depth: keypoints = exact object
  world positions (`obj.data.root_pos_w`); EE trajectory = differentiable FK of
  predicted joints (mm-accurate). Only Gemini is non-privileged (intended — it's
  the VL decision-maker; it picks keypoint INDEX by name legend, reward uses exact xyz).
- **Gradient steering mechanism**: gradient taken only w.r.t. the action sample
  through decode→FK→reward (NOT through the policy net), so it's cheap and only
  needs the flow loop's per-step seam (`x_t = x_t + dt*v_t`). Monkeypatch that.

---

## Verification results
- MolmoAct2 steering: guidance-off == normal (Δ=0), guidance-on shifts actions; live demo works across scenes + prompts.
- pi0.5 PyTorch **parity vs JAX** (same noise): max |Δ| **0.0086 rad** — same policy.
- pi0.5 **perf**: JAX **58 ms** | PyTorch eager **193 ms** | steered ~**293 ms**
  (+~14 s once per new prompt for Gemini). ~3–5× slower than JAX but demo-smooth.
- pi0.5 steering: steered vs unsteered max |Δ| 0.87; **real task SUCCESS** observed
  on BananaInBowl; scene switch + new-scene guidance regen confirmed.

---

## How to run

### pi0.5 (plain, fast JAX) — original
```bash
# server (GPU 4)
cd /root/ishneet/openpi && CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 \
  uv run scripts/serve_policy.py --port 8001 policy:checkpoint \
  --policy.config=pi05_droid_jointpos --policy.dir=gs://openpi-assets-simeval/pi05_droid_jointpos
# demo (GPU 5)
cd /root/ishneet/RoboLab && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=Y && \
CUDA_VISIBLE_DEVICES=5 python policies/pi0_family/run_interactive.py --remote-port 8001 \
  --task BananaInBowlTask --web-port 8090
```

### MolmoAct2 (plain or steered)
```bash
# server (GPU 6); add --steer for VLS steering
GOOGLE_API_KEY=$(sed -nE 's/^GOOGLE_API_KEY=//p' /root/ishneet/.env) \
CUDA_VISIBLE_DEVICES=6 /root/ishneet/molmoact-venv/bin/python policies/molmoact/server.py --port 5555 --dtype bfloat16 --steer
# demo (GPU 5)
cd /root/ishneet/RoboLab && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=Y && \
CUDA_VISIBLE_DEVICES=5 python policies/pi0_family/run_interactive.py --policy-backend molmoact --steer --molmo-port 5555 --task BananaInBowlTask --web-port 8090
```

### pi0.5 droid PyTorch + gradient steering (VLS)
```bash
# server (GPU 5)
GOOGLE_API_KEY=$(sed -nE 's/^GOOGLE_API_KEY=//p' /root/ishneet/.env) \
CUDA_VISIBLE_DEVICES=5 TORCHDYNAMO_DISABLE=1 /root/ishneet/openpi/.venv/bin/python \
  policies/molmoact/pi05_steer_server.py --port 5556 --steer
# demo (GPU 6)
cd /root/ishneet/RoboLab && source .venv/bin/activate && export OMNI_KIT_ACCEPT_EULA=Y && \
CUDA_VISIBLE_DEVICES=6 python policies/pi0_family/run_interactive.py --policy-backend pi05_steer --steer --molmo-port 5556 --task BananaInBowlTask --web-port 8090
```
Then open `http://localhost:8090` (or `http://10.200.108.106:8090`).
Scenes menu: BananaInBowlTask, GrabAFruitTask, GrabABagelTask,
FruitsMovingOrangeOrLimeTask, MoveBananaToBagelPlateTask.

---

## Caveats / next steps
- PyTorch pi0.5 ~3–5× slower than JAX (eager; torch.compile incompatible with the
  gradient monkeypatch). For JAX speed + gradient, the alternative is injecting
  `jax.grad` into OpenPI's JAX flow loop (not done).
- `--guide-scale` (default 2.0) needs tuning per scene — too high destabilizes.
- Steering mechanism + a real success verified, but no measured success-RATE yet.
  For that, run the proper evaluator (`run.py --num-runs N`) steered vs unsteered.
- Multi-stage guidance is generated (grasp→place) but only stage 1 is auto-used;
  stage advancement is a future improvement.
- Re-init OpenPI submodules / `transformers_replace` is already done; converted
  checkpoint + FK transform persist on disk.
```
```

---
## Addendum (later session): FULL VLS parity + RoboLab tuning — all verified
- Adaptive sigmoid scaling (k=25, x0=0.75, init_reward = first chunk's final reward) ✅
- RBF diversity + FKD resampling (FKD imported verbatim from VLS repo; --num-particles, default 1 like VLS config) — implemented; particle mode not live-run
- Multi-stage advancement ✅ verified live: "ADVANCE to stage 2/2 ('banana' lifted 0.042m)" (privileged lift detector) and norm-reward Schmitt trigger
- Stage reset on episode AND live task change ✅ (bug found+fixed)
- Prompt: real 3D coords (+optional sizes via ROBOLAB_VLS_SIZES=1), full RoboLab task_specific_patterns, stage_targets + recommended_guide_scale output (rec_scale verified varying 5.0/6.0/8.0 per task) ✅
- TCP-offset FK (--tcp-offset 0.15, fingertip targeting) ✅
- Keypoint overlay projection FIXED (look-direction frame from camera config; markers verified ON objects in overlay image) ✅
- Stability note: avoid two demos on one GPU (zombie after crash held :8090); USD bbox sizes opt-in (suspected physx-view invalidation)
- FINAL RESULT: full-VLS steered pi0.5 on BananaInBowl = **17/18 SUCCESS (94%)**,
  all stage advances via the privileged lift trigger; fast completions (~110-130 steps).
  Key tuning: grasp stage advances on LIFT ONLY (reward-trigger advanced too early and
  broke grasps); guide_scale capped at 2.5 (Gemini rec 6.0 was too strong).

---
## Hand-written guidance (guidance_manual.py) — the demo win
- Server prefers HAND-WRITTEN guidance (instant, deterministic, no Gemini), parsing the
  instruction string + privileged object names/coords; Gemini is the fallback.
  Resolves left/right bins from world Y; cube-or-banana -> nearest; etc.
- KEY FIX: gentle PLACE stage — bias EE x-y over the container, light z weight, so pi0.5
  controls its own descent + gripper release (aggressive full-3D place pull was fighting
  the policy and dropping success to 0).
- A/B on mustard bins (commanded-bin success, RoboLab predicate):
    manual gentle-steer:  Left 1/3, Right 2/4  (~43% into the COMMANDED bin)
    aggressive auto-steer: 0/8
    unsteered:            ~25% (and cannot be told which bin)
- Takeaway: steering's value = enforcing the language-specified CHOICE (object/destination)
  while letting the competent policy execute; works when guidance is gentle on the place.
- Run: same pi05_steer_server (manual on by default; --no-manual forces Gemini).
