# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""pi0.5-DROID (PyTorch) inference server with FULL VLS gradient steering.

Faithful port of Vision-Language-Steering (core/pi05_steer.py) onto the
JAX->PyTorch-converted pi05_droid_jointpos checkpoint, with the perception
stack replaced by RoboLab privileged info:

- gradient guidance (late denoising): grad of the Gemini-authored reward w.r.t.
  the action sample through decode -> differentiable FK -> reward;
- ADAPTIVE SIGMOID scaling (sigmoid_k/x0; init_reward = final reward of the
  stage's first chunk, normalized_reward in [0, 1.2]) — exactly as VLS;
- RBF DIVERSITY between particles (early denoising, num_particles > 1);
- FEYNMAN-KAC particle resampling (FKD imported directly from the VLS repo);
- best-particle selection at the end;
- MULTI-STAGE execution with privileged stage advancement: Gemini outputs
  `stage_targets`; the server advances grasp->place when the target object has
  LIFTED (sim ground truth) or the stage's normalized reward crosses a Schmitt
  threshold (VLS's reward trigger, resolved privileged instead of via a VLM
  stage recognizer); per-episode reset via a client episode counter.
- TCP offset on the FK so guidance pulls the FINGERTIPS (not the gripper mount).

Runs in the OpenPI venv. ZMQ/msgpack protocol (same client as molmoact):

    GOOGLE_API_KEY=... CUDA_VISIBLE_DEVICES=5 TORCHDYNAMO_DISABLE=1 \
    /root/ishneet/openpi/.venv/bin/python policies/molmoact/pi05_steer_server.py \
        --port 5556 --steer [--num-particles 4 --use-fkd]
"""

import argparse
import json
import os
import sys
import time
import traceback

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")  # eager (compile breaks the grad path)

import msgpack
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
VLS_REPO = os.environ.get("VLS_REPO", "/root/ishneet/vision-language-steering")
sys.path.insert(0, VLS_REPO)

from openpi.training import config as _config  # noqa: E402
from openpi.policies import policy_config  # noqa: E402

from policies.molmoact.molmoact_steer import DifferentiableFrankaFK  # noqa: E402
from policies.molmoact import guidance_manual  # noqa: E402


def pack_arr(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.tobytes()}


def unpack_arr(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"]).copy()


# ---------------------------------------------------------------------------
# Full-VLS guided sampler
# ---------------------------------------------------------------------------
def build_guided_sample_actions(model, fk, q01_7, q99_7, default_num_steps=5):
    """Drop-in `sample_actions(device, observation, noise, num_steps)` implementing
    the complete VLS algorithm, configured via `model._vls_steer` (None = unsteered)."""
    import jax  # used only for pytree mapping over the Observation (torch tensors)

    def decode_ee(xg: torch.Tensor, current_joints: torch.Tensor) -> torch.Tensor:
        # normalized delta -> quantile unnorm -> + current joints -> FK -> TCP xyz
        dn = xg[:, :, :7].clamp(-1.0, 1.0)
        delta = (dn + 1.0) / 2.0 * (q99_7 - q01_7) + q01_7
        joints = current_joints.view(1, 1, 7) + delta
        return fk(joints)  # (B, T, 3) world frame

    def per_particle_rewards(x: torch.Tensor, cfg: dict) -> torch.Tensor:
        with torch.no_grad():
            ee = decode_ee(x.detach().float(), cfg["current_joints"])
            fn, kp = cfg["guidance_fn"], cfg["keypoints"]
            return torch.tensor([float(fn(kp, ee[b:b + 1])) for b in range(x.shape[0])],
                                device=x.device, dtype=torch.float32)

    def guidance_grad_and_reward(x_t: torch.Tensor, cfg: dict):
        with torch.enable_grad():
            xg = x_t.detach().float().requires_grad_(True)
            ee = decode_ee(xg, cfg["current_joints"])
            reward = cfg["guidance_fn"](cfg["keypoints"], ee)
            g = torch.autograd.grad(reward, xg)[0]
        return g / (torch.norm(g) + 1e-8), float(reward)

    def diversity_grad(x_t: torch.Tensor, cfg: dict):
        # VLS: pairwise 1/dist repulsion potential on flattened EE trajectories.
        B = x_t.shape[0]
        if B < 2:
            return None
        with torch.enable_grad():
            xg = x_t.detach().float().requires_grad_(True)
            traj = decode_ee(xg, cfg["current_joints"]).reshape(B, -1)
            diff = traj.unsqueeze(1) - traj.unsqueeze(0)
            dist = torch.sqrt((diff ** 2).sum(dim=2) + 1e-6)
            mask = ~torch.eye(B, dtype=torch.bool, device=x_t.device)
            potential = (1.0 / (dist + 1e-6) * mask.float()).sum()
            g = torch.autograd.grad(potential, xg)[0]
        n = torch.norm(g)
        return g / (n + 1e-8) if float(n) > 1e-8 else g

    @torch.no_grad()
    def guided_sample_actions(device, observation, noise=None, num_steps=None):
        if num_steps is None:
            num_steps = default_num_steps   # fewer flow steps = lower latency (smoother demo)
        cfg = getattr(model, "_vls_steer", None)
        P = int(cfg.get("num_particles", 1)) if cfg else 1
        if P > 1:
            observation = jax.tree.map(
                lambda x: x.repeat(P, *([1] * (x.ndim - 1))) if torch.is_tensor(x) else x,
                observation)

        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        bsize = observation.state.shape[0]
        noise = model.sample_noise((bsize, model.config.action_horizon, model.config.action_dim), device)

        images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d, position_ids=prefix_position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
        )

        start_ratio = float(cfg["start_ratio"]) if cfg else 0.0
        start_idx = int((1.0 - start_ratio) * num_steps)  # step index where late phase begins

        fkd = None
        if cfg is not None and P > 1 and cfg.get("use_fkd") and cfg.get("guidance_fn") is not None:
            from core.fkd_class import FKD  # from the VLS repo, verbatim
            fkd = FKD(
                potential_type=cfg.get("fkd_potential", "max"),
                lmbda=float(cfg.get("fkd_lmbda", 10.0)),
                num_particles=P,
                adaptive_resampling=True,
                resample_frequency=1,
                resampling_t_start=start_idx,
                resampling_t_end=num_steps - 1,
                timesteps=list(range(num_steps)),
                reward_fn=lambda x0: per_particle_rewards(x0, cfg),
                device=device,
            )

        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time_t = torch.tensor(1.0, dtype=torch.float32, device=device)
        step_idx = 0
        last_reward = None
        ss = cfg.get("stage_state") if cfg else None  # per-stage mutable dict

        while time_t >= -dt / 2:
            v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time_t.expand(bsize))
            in_late = cfg is not None and float(time_t) <= start_ratio
            if cfg is not None and not in_late and cfg.get("use_diversity", True) and P > 1:
                dg = diversity_grad(x_t, cfg)
                if dg is not None:
                    v_t = v_t + float(cfg.get("diversity_scale", 20.0)) * dg
            if in_late and cfg.get("guidance_fn") is not None:
                g, reward = guidance_grad_and_reward(x_t, cfg)
                last_reward = reward
                init_r = ss.get("init_reward") if ss else None
                if init_r is not None and init_r < -1e-6:
                    nr = max(0.0, min(1.2, 1.0 - reward / init_r))
                else:
                    nr = 0.0
                if ss is not None:
                    ss["last_norm_reward"] = nr
                strength = 1.0 / (1.0 + np.exp(float(cfg.get("sigmoid_k", 25.0)) * (nr - float(cfg.get("sigmoid_x0", 0.75)))))
                v_t = v_t - (float(cfg["guide_scale"]) * strength) * g  # dt<0 => ascend
            x_t = x_t + dt * v_t
            if fkd is not None and in_late:
                x_t, _ = fkd.resample(sampling_idx=step_idx, latents=x_t, x0_preds=x_t)
            time_t = time_t + dt
            step_idx += 1

        # VLS: baseline = final reward of the stage's FIRST chunk
        if ss is not None and ss.get("init_reward") is None and last_reward is not None:
            ss["init_reward"] = last_reward

        # best-particle selection
        if P > 1:
            if cfg is not None and cfg.get("guidance_fn") is not None:
                rs = per_particle_rewards(x_t, cfg)
                x_t = x_t[int(torch.argmax(rs)):int(torch.argmax(rs)) + 1]
            else:
                x_t = x_t[:1]
        return x_t

    return guided_sample_actions


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="pi0.5-DROID PyTorch full-VLS steering server.")
    parser.add_argument("--config-name", default="pi05_droid_jointpos")
    parser.add_argument("--checkpoint", default="/root/.cache/openpi/pi05_droid_jointpos_pytorch")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--steer", action="store_true")
    parser.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    parser.add_argument("--no-centroid-place", action="store_true",
                        help="disable object-centroid-aware place target shift (A/B baseline)")
    parser.add_argument("--num-steps", type=int, default=5,
                        help="flow-matching denoising steps per inference (lower = faster/smoother, "
                             "slightly lower quality; 10 = original)")
    parser.add_argument("--no-manual", action="store_true",
                        help="Disable hand-written guidance; always use Gemini.")
    parser.add_argument("--place-release", action="store_true",
                        help="Stop steering once EE is over the container (needs tuning with --release-xy).")
    parser.add_argument("--release-xy", type=float, default=0.15,
                        help="x-y distance (m) to the container at which to release steering.")
    # VLS algorithm knobs (defaults follow the VLS repo config)
    parser.add_argument("--num-particles", type=int, default=1)
    parser.add_argument("--use-fkd", action="store_true", help="FK resampling (needs particles>1).")
    parser.add_argument("--fkd-potential", default="max")
    parser.add_argument("--fkd-lmbda", type=float, default=10.0)
    parser.add_argument("--diversity-scale", type=float, default=20.0)
    parser.add_argument("--no-diversity", action="store_true")
    parser.add_argument("--sigmoid-k", type=float, default=25.0)
    parser.add_argument("--sigmoid-x0", type=float, default=0.75)
    # RoboLab-specific
    parser.add_argument("--tcp-offset", type=float, default=0.15,
                        help="Tool offset (m) along the flange z so guidance targets the fingertips.")
    parser.add_argument("--advance-lift", type=float, default=0.03,
                        help="Stage advance when the stage-target object lifts this many meters.")
    parser.add_argument("--advance-reward", type=float, default=0.9,
                        help="Stage advance when the stage's normalized reward crosses this (Schmitt upper).")
    parser.add_argument("--overlay-path", default="/root/ishneet/_vls_scratch/overlay_last.png")
    args = parser.parse_args()

    print(f"[pi05] loading PyTorch policy from {args.checkpoint} ...", flush=True)
    tc = _config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(tc, args.checkpoint, pytorch_device="cuda")
    model = policy._model
    device = policy._pytorch_device

    ns = json.load(open(os.path.join(args.checkpoint, "assets/droid/norm_stats.json")))
    ns = ns.get("norm_stats", ns)
    q01 = torch.tensor(ns["actions"]["q01"][:7], dtype=torch.float32, device=device)
    q99 = torch.tensor(ns["actions"]["q99"][:7], dtype=torch.float32, device=device)

    guidance = None
    guidance_cache: dict = {}      # task -> (stages, stage_targets)
    fk = None
    if args.steer:
        from policies.molmoact.guidance_gemini import GeminiGuidance
        bt_path = os.path.join(os.path.dirname(__file__), "_fk_base_transform.npy")
        base_T = np.load(bt_path) if os.path.exists(bt_path) else None
        fk = DifferentiableFrankaFK(base_transform=base_T, device=device,
                                    tool_offset=(0.0, 0.0, args.tcp_offset))
        policy._sample_actions = build_guided_sample_actions(model, fk, q01, q99,
                                                             default_num_steps=args.num_steps)
        model._vls_steer = None
        guidance = GeminiGuidance(model=args.gemini_model)
        os.makedirs(os.path.dirname(args.overlay_path), exist_ok=True)
        print(f"[pi05] FULL VLS steering enabled (base_T={'loaded' if base_T is not None else 'identity'}, "
              f"tcp={args.tcp_offset}m, particles={args.num_particles}, fkd={args.use_fkd}, "
              f"sigmoid k={args.sigmoid_k}/x0={args.sigmoid_x0}, gemini={args.gemini_model})", flush=True)

    # per-episode stage machine
    ep_state = {"episode": None, "stage": 0, "kp_z0": {}, "stage_state": {}}

    import zmq
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[pi05] server listening on tcp://0.0.0.0:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        if req.get("ping"):
            sock.send(msgpack.packb({"ok": True, "kind": "pi05-droid-pytorch-vls", "steer": bool(args.steer)},
                                    use_bin_type=True))
            continue
        try:
            images = [unpack_arr(im) for im in req["images"]]
            state = np.asarray(req["state"], dtype=np.float32)
            task = str(req["task"])
            obs = {
                "observation/exterior_image_1_left": images[0],
                "observation/wrist_image_left": images[-1],
                "observation/joint_position": state[:7],
                "observation/gripper_position": state[7:8],
                "prompt": task,
            }
            extra = {}
            do_steer = bool(req.get("steer")) and args.steer and req.get("keypoints")
            if do_steer:
                kps = req["keypoints"]
                kp_world = torch.tensor([k["xyz"] for k in kps], dtype=torch.float32, device=device)

                # ---- guidance (cached per task): hand-written first, Gemini fallback ----
                if task not in guidance_cache:
                    kp_names = [str(k["name"]) for k in kps]
                    kp_xyz = np.asarray([k["xyz"] for k in kps], dtype=np.float32)
                    man = None
                    if not args.no_manual:
                        try:
                            man = guidance_manual.build(task, kp_names, kp_xyz)
                        except Exception:
                            traceback.print_exc(); man = None
                    if man is not None:
                        stages, stage_targets, rec_scale, stage_kinds = man
                        print(f"[pi05] MANUAL guidance for task {task!r}: {len(stages)} stage(s), "
                              f"seq={[f'{stage_kinds[i]}:{kp_names[t]}' for i, t in enumerate(stage_targets)]}, "
                              f"scale={rec_scale}", flush=True)
                    else:
                        named = [(str(k["name"]), tuple(k.get("uv", (0, 0))), tuple(k["xyz"]),
                                  tuple(k["size"]) if k.get("size") else None) for k in kps]
                        stages, stage_targets, _raw = guidance.generate(
                            images[0], task, named, save_overlay_path=args.overlay_path)
                        rec_scale = getattr(guidance, "last_recommended_scale", None)
                        # Gemini path: first stage grasps, the rest place.
                        stage_kinds = (["grasp"] + ["place"] * (len(stages) - 1)) if stages else []
                        print(f"[pi05] GEMINI guidance for task: {task!r} "
                              f"(stage_targets={stage_targets}, rec_scale={rec_scale})", flush=True)
                    guidance_cache[task] = (stages, stage_targets, rec_scale, stage_kinds)
                    extra["num_stages"] = len(stages)
                stages, stage_targets, rec_scale, stage_kinds = guidance_cache[task]
                # HOLD / pick-up: pi0.5 closes the gripper reliably for a concrete
                # grasp instruction but hovers for "hold it there". Feed it a clean
                # grasp prompt while the steering lifts + holds the object in the air.
                if "hold" in stage_kinds and stage_targets:
                    gname = str(kps[stage_targets[0]]["name"]).replace("_", " ")
                    # Stage-aware prompt: until grasped, feed "put X in the box" (the
                    # phrasing pi0.5 reliably closes the gripper for); once grasped
                    # (hold stage), switch to "pick up X" so it keeps holding instead
                    # of carrying on to place it.
                    cur_stage = ep_state.get("stage", 0) if ep_state.get("task") == task else 0
                    obs["prompt"] = (f"put the {gname} in the box" if cur_stage == 0
                                     else f"pick up the {gname}")
                    extra["policy_prompt"] = obs["prompt"]

                # ---- episode / stage machine (reset on NEW EPISODE or LIVE TASK CHANGE) ----
                ep = req.get("episode", 0)
                if ep != ep_state["episode"] or task != ep_state.get("task"):
                    ep_state.update(episode=ep, task=task, stage=0,
                                    kp_z0={str(k["name"]): float(k["xyz"][2]) for k in kps},
                                    stage_state={i: {"init_reward": None, "last_norm_reward": 0.0}
                                                 for i in range(max(len(stages), 1))})
                    print(f"[pi05] episode {ep}: stage machine reset ({len(stages)} stages, task={task!r})", flush=True)

                # Current EE (x-y) for held-object / placement detection.
                ee_xy = None
                if fk is not None:
                    cj0 = torch.tensor(state[:7], dtype=torch.float32, device=device)
                    ee_xy = fk(cj0.view(1, 1, 7))[0, 0, :2].detach().cpu().numpy()

                if stages and ep_state["stage"] < len(stages) - 1:
                    adv = None
                    s = ep_state["stage"]
                    kind = stage_kinds[s] if s < len(stage_kinds) else ("grasp" if s % 2 == 0 else "place")
                    if kind == "grasp":
                        # Advance when the object the gripper is actually holding (nearest the
                        # EE in x-y, not the nominal target) has LIFTED. Robust to "X or Y"
                        # (grabbed the other one) AND to sequences (an already-placed block
                        # stays elevated but is far from the EE, so it won't false-trigger).
                        held, hd = None, 1e9
                        if ee_xy is not None:
                            for i in range(len(kps)):
                                if str(kps[i]["name"]).lower() == "table":
                                    continue
                                d = float(np.linalg.norm(np.asarray(kps[i]["xyz"][:2]) - ee_xy))
                                if d < hd:
                                    hd, held = d, i
                        if held is not None and hd < 0.12:
                            z0 = ep_state["kp_z0"].get(str(kps[held]["name"]))
                            if z0 is not None and float(kps[held]["xyz"][2]) - z0 > args.advance_lift:
                                adv = f"held '{kps[held]['name']}' lifted {float(kps[held]['xyz'][2]) - z0:.3f}m"
                    else:  # place: advance only once the carried object reaches the target
                        g = stage_targets[s - 1] if s >= 1 else None
                        p = stage_targets[s]
                        if g is not None and 0 <= g < len(kps) and 0 <= p < len(kps):
                            dxy = float(np.linalg.norm(np.asarray(kps[g]["xyz"][:2]) - np.asarray(kps[p]["xyz"][:2])))
                            dz = float(kps[g]["xyz"][2]) - float(kps[p]["xyz"][2])
                            # over the target in x-y AND descended (not still held high) =>
                            # placed/released; require 2 consecutive hits to avoid flicker.
                            if dxy < 0.07 and dz < 0.07:
                                ss = ep_state["stage_state"][s]
                                ss["place_hits"] = ss.get("place_hits", 0) + 1
                                if ss["place_hits"] >= 2:
                                    adv = f"placed '{kps[g]['name']}' at '{kps[p]['name']}' (dxy={dxy:.3f})"
                            else:
                                ep_state["stage_state"][s]["place_hits"] = 0
                    if adv:
                        ep_state["stage"] += 1
                        print(f"[pi05] episode {ep}: ADVANCE to stage {ep_state['stage'] + 1}"
                              f"/{len(stages)} ({adv})", flush=True)

                s = min(ep_state["stage"], max(len(stages) - 1, 0))
                # Per-stage scale: rec_scale may be a list (hand-written: grasp strong,
                # place gentle) or a scalar (Gemini). Grasp gets more pull than place.
                if isinstance(rec_scale, (list, tuple)):
                    gscale = float(rec_scale[s]) if s < len(rec_scale) else float(rec_scale[-1])
                else:
                    gscale = float(rec_scale) if rec_scale else float(req.get("guide_scale", 2.0))
                gscale = float(np.clip(gscale, 0.3, 2.5))

                active_fn = stages[s] if stages else None
                # PLACE-THEN-RELEASE (opt-in via --place-release): once the gripper is
                # over the container in x-y, stop steering so pi0.5 does its own descent +
                # release. NOTE: at the default place scale the EE often doesn't get within
                # the threshold, so this rarely fires; enable + tune --release-xy together.
                if (getattr(args, "place_release", False) and active_fn is not None and s > 0
                        and fk is not None and s < len(stage_targets)):
                    pidx = stage_targets[s]
                    if 0 <= pidx < kp_world.shape[0]:
                        cj = torch.tensor(state[:7], dtype=torch.float32, device=device)
                        ee_now = fk(cj.view(1, 1, 7))[0, 0]
                        xy = float(torch.norm(ee_now[:2] - kp_world[pidx][:2]))
                        if xy < float(args.release_xy):
                            active_fn = None
                            extra["release"] = True

                # OBJECT-CENTROID-AWARE PLACE: a grasped object's centroid is offset
                # from the TCP (e.g. a banana grabbed at one end), so centering the
                # GRIPPER over the container leaves the object hanging over the rim and
                # it drops OUTSIDE on release. Shift the place target by the measured
                # gripper->object x-y offset so the OBJECT (not the TCP) lands centered.
                # Shape-agnostic: ~0 for a compact cube, large for a banana. Applies to
                # both manual and Gemini guidance since it only transforms keypoints.
                kind_s = stage_kinds[s] if s < len(stage_kinds) else ("place" if s % 2 else "grasp")
                kp_steer = kp_world
                if (not getattr(args, "no_centroid_place", False) and active_fn is not None
                        and kind_s.startswith("place") and fk is not None and len(stage_targets) > s):
                    pidx = stage_targets[s]          # place target (container or support block)
                    if 0 <= pidx < kp_world.shape[0]:
                        cj = torch.tensor(state[:7], dtype=torch.float32, device=device)
                        ee_now = fk(cj.view(1, 1, 7))[0, 0]
                        # The object actually in the gripper = the keypoint nearest the
                        # TCP in x-y (excl. the place target + table). Robust to "X or Y"
                        # where the grabbed object differs from the nominal grasp target.
                        held, held_d = None, 1e9
                        for i in range(kp_world.shape[0]):
                            if i == pidx or str(kps[i]["name"]).lower() == "table":
                                continue
                            d = float(torch.norm(kp_world[i][:2] - ee_now[:2]))
                            if d < held_d:
                                held_d, held = d, i
                        if held is not None and held_d < 0.12:   # something really is held
                            offset_xy = torch.clamp(kp_world[held][:2] - ee_now[:2], -0.15, 0.15)
                            kp_steer = kp_world.clone()
                            kp_steer[pidx, :2] = kp_world[pidx][:2] - offset_xy
                            extra["centroid_obj"] = str(kps[held]["name"])
                            extra["centroid_shift"] = [round(float(offset_xy[0]), 3),
                                                       round(float(offset_xy[1]), 3)]
                            # STACK z-correction: the held block sits BELOW the fingertip,
                            # so aiming the TCP at the support top drops the block beside it
                            # (lands "horizontal"). Aim so the block's CENTER ends one block-
                            # height ABOVE the support center -> it rests squarely on top.
                            # (Z_OFF here must match _place_above's stacking z_off = 0.07.)
                            if kind_s == "place_on":
                                offset_z = float(kp_world[held][2] - ee_now[2])
                                STACK_H, Z_OFF = 0.055, 0.07
                                kp_steer[pidx, 2] = kp_world[pidx][2] + STACK_H - offset_z - Z_OFF
                                extra["stack_target_z"] = round(float(kp_steer[pidx, 2] + Z_OFF), 3)

                model._vls_steer = {
                    "keypoints": kp_steer,
                    "guidance_fn": active_fn,
                    "guide_scale": gscale,
                    "start_ratio": float(req.get("start_ratio", 0.8)),
                    "current_joints": torch.tensor(state[:7], dtype=torch.float32, device=device),
                    "stage_state": ep_state["stage_state"].get(s),
                    "num_particles": args.num_particles,
                    "use_diversity": not args.no_diversity,
                    "diversity_scale": args.diversity_scale,
                    "use_fkd": args.use_fkd,
                    "fkd_potential": args.fkd_potential,
                    "fkd_lmbda": args.fkd_lmbda,
                    "sigmoid_k": args.sigmoid_k,
                    "sigmoid_x0": args.sigmoid_x0,
                }
                extra["stage"] = s + 1
            elif args.steer:
                model._vls_steer = None

            t0 = time.perf_counter()
            out = policy.infer(obs)
            if args.steer:
                model._vls_steer = None
            actions = np.asarray(out["actions"], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[None, :]
            dt_ms = (time.perf_counter() - t0) * 1000.0
            sock.send(msgpack.packb({"ok": True, "actions": pack_arr(actions), "infer_ms": dt_ms,
                                     "steered": bool(do_steer), **extra}, use_bin_type=True))
        except Exception as e:
            traceback.print_exc()
            if args.steer:
                model._vls_steer = None
            sock.send(msgpack.packb({"ok": False, "error": f"{type(e).__name__}: {e}"}, use_bin_type=True))


if __name__ == "__main__":
    main()
