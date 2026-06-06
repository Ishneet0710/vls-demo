# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gradient-based VLS-style steering for MolmoAct2-DROID.

MolmoAct2 produces actions by a flow-matching (Euler) loop:
    trajectory = trajectory + dt * velocity        (modeling_molmoact2._run_action_flow_loop)
where `velocity = action_expert.forward_with_context(...)`.

VLS (Vision-Language-Steering) steers such a loop by injecting, on the late
denoising steps, the gradient of a VLM-authored spatial reward w.r.t. the action
sample, nudging the sampled trajectory toward task-relevant 3D keypoints.

Key realization that makes this cheap: the guidance gradient is taken w.r.t. the
*action sample* only (a one-step leaf), NOT through the 5B action expert. So the
sole differentiable path is:

    normalized_action  --(affine unnormalize)-->  joint angles  --(diff. FK)-->  EE xyz  -->  reward

We therefore (a) reuse all of `predict_action`'s preprocessing untouched, (b)
monkeypatch the (non-`@torch.no_grad`) `_run_action_flow_loop` to inject the
guidance term, computing the velocity as a plain value and the guidance gradient
under a local `torch.enable_grad()` (valid even inside `predict_action`'s
`@torch.no_grad`).

This module runs in the MolmoAct2 (server) venv where the model lives.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Differentiable Franka Panda forward kinematics (modified-DH / Craig)
# ---------------------------------------------------------------------------
# NOTE: These are the standard Franka Emika Panda modified-DH parameters. The
# resulting EE position is in the robot BASE frame. For guidance against
# world-frame keypoints you must set `base_transform` (4x4) to the robot's
# base->world pose, AND validate FK against RoboLab's reported EE pose before
# trusting the steering (see `validate_against` / the demo-side check).
class DifferentiableFrankaFK(torch.nn.Module):
    # (a_{i-1}, d_i, alpha_{i-1}) for joints 1..7 plus flange, modified DH (meters / rad)
    _DH = [
        (0.0,      0.333, 0.0),
        (0.0,      0.0,   -math.pi / 2),
        (0.0,      0.316,  math.pi / 2),
        (0.0825,   0.0,    math.pi / 2),
        (-0.0825,  0.384, -math.pi / 2),
        (0.0,      0.0,    math.pi / 2),
        (0.088,    0.0,    math.pi / 2),
        (0.0,      0.107,  0.0),  # flange (theta = 0)
    ]

    def __init__(self, base_transform: Optional[np.ndarray] = None, device="cuda", dtype=torch.float32,
                 tool_offset: Optional[tuple] = None):
        """tool_offset: (x,y,z) translation in the FLANGE frame appended after the flange
        transform — e.g. (0,0,0.15) targets the gripper fingertips (TCP) instead of the
        mount, which is what grasp guidance should pull toward."""
        super().__init__()
        a = torch.tensor([p[0] for p in self._DH], dtype=dtype)
        d = torch.tensor([p[1] for p in self._DH], dtype=dtype)
        alpha = torch.tensor([p[2] for p in self._DH], dtype=dtype)
        self.register_buffer("a", a)
        self.register_buffer("d", d)
        self.register_buffer("alpha", alpha)
        bt = np.eye(4) if base_transform is None else np.asarray(base_transform)
        self.register_buffer("base", torch.tensor(bt, dtype=dtype))
        to = (0.0, 0.0, 0.0) if tool_offset is None else tuple(tool_offset)
        self.register_buffer("tool", torch.tensor(to, dtype=dtype))
        self.to(device)

    def _link_T(self, theta_i: torch.Tensor, i: int) -> torch.Tensor:
        # theta_i: (B,) ; returns (B,4,4) modified-DH transform for link i.
        B = theta_i.shape[0]
        a, d, al = self.a[i], self.d[i], self.alpha[i]
        ct, st = torch.cos(theta_i), torch.sin(theta_i)
        ca, sa = torch.cos(al), torch.sin(al)
        T = torch.zeros(B, 4, 4, dtype=theta_i.dtype, device=theta_i.device)
        T[:, 0, 0] = ct
        T[:, 0, 1] = -st
        T[:, 0, 3] = a
        T[:, 1, 0] = st * ca
        T[:, 1, 1] = ct * ca
        T[:, 1, 2] = -sa
        T[:, 1, 3] = -d * sa
        T[:, 2, 0] = st * sa
        T[:, 2, 1] = ct * sa
        T[:, 2, 2] = ca
        T[:, 2, 3] = d * ca
        T[:, 3, 3] = 1.0
        return T

    def forward(self, joints: torch.Tensor) -> torch.Tensor:
        """joints: (..., 7) joint angles (rad) -> EE position (..., 3) in world frame."""
        lead = joints.shape[:-1]
        q = joints.reshape(-1, 7).to(self.a.dtype)
        B = q.shape[0]
        T = self.base.unsqueeze(0).expand(B, 4, 4).clone()
        for i in range(7):
            T = torch.bmm(T, self._link_T(q[:, i], i))
        T = torch.bmm(T, self._link_T(torch.zeros(B, dtype=q.dtype, device=q.device), 7))  # flange
        # tool/TCP offset expressed in the flange frame
        pos = T[:, :3, 3] + torch.einsum("bij,j->bi", T[:, :3, :3], self.tool.to(q.dtype))
        return pos.reshape(*lead, 3)


# ---------------------------------------------------------------------------
# Steering wrapper
# ---------------------------------------------------------------------------
class MolmoActSteer:
    """Wraps a loaded MolmoAct2 model, injecting VLS-style guidance into its flow loop."""

    def __init__(self, model, norm_tag: str = "franka_droid",
                 base_transform: Optional[np.ndarray] = None):
        self.model = model            # MolmoAct2ForConditionalGeneration (predict_action, _get_robot_stats)
        self.inner = model.model      # MolmoAct2Model (flow loop, action expert, masks, config)
        self.norm_tag = norm_tag
        self.device = next(model.parameters()).device
        self._active: Optional[dict] = None  # current guidance config or None

        # Differentiable joint-dim unnormalizer from the checkpoint's norm stats.
        # All of min_max / q01_q99 / q10_q90 unnormalize as: (x+1)*(hi-lo)/2 + lo.
        stats = model._get_robot_stats()
        norm = stats.action_normalizers.get(norm_tag)
        if norm is None:
            raise RuntimeError(f"No action normalizer for norm_tag '{norm_tag}'")
        if norm.mode == "min_max":
            lo, hi = norm.min_val, norm.max_val
        elif norm.mode in {"q01_q99", "q10_q90"}:
            lo, hi = norm.q_low, norm.q_high
        else:
            raise RuntimeError(f"Unsupported affine normalizer mode '{norm.mode}' for '{norm_tag}'")
        self._alo = torch.tensor(np.asarray(lo), dtype=torch.float32, device=self.device)
        self._ahi = torch.tensor(np.asarray(hi), dtype=torch.float32, device=self.device)
        self._norm_mode = norm.mode
        self.fk = DifferentiableFrankaFK(base_transform=base_transform, device=self.device)

        self._install_hook()

    # -- differentiable decode: normalized action sample -> real joint angles --
    def _decode_joints(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, action_dim) normalized in [-1, 1]; affine-unnormalize first 7 dims.
        xj = torch.clamp(x[..., :7].to(torch.float32), -1.0, 1.0)
        lo, hi = self._alo[:7], self._ahi[:7]
        return (xj + 1.0) * (hi - lo) / 2.0 + lo

    def _guidance_grad(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        g = self._active
        if g is None or g.get("guidance_fn") is None or g.get("keypoints") is None:
            return None
        with torch.enable_grad():
            xg = x.detach().to(torch.float32).requires_grad_(True)
            joints = self._decode_joints(xg)            # (B,T,7) real joint angles
            ee = self.fk(joints)                        # (B,T,3) world-frame EE
            reward = g["guidance_fn"](g["keypoints"], ee)  # scalar (higher = better)
            grad = torch.autograd.grad(reward, xg, retain_graph=False)[0]
        grad = grad.to(x.dtype)
        n = torch.norm(grad) + 1e-8
        return grad / n

    def _install_hook(self):
        inner = self.inner
        steer = self
        orig_loop = inner._run_action_flow_loop

        def guided_loop(inputs, steps):
            if steer._active is None:
                return orig_loop(inputs, steps)
            ae = inner._require_action_expert()
            dt = 1.0 / steps
            x = inputs.trajectory
            apad = inputs.action_dim_is_pad
            mask_enabled = inner.config.mask_action_dim_padding
            g = steer._active
            start_idx = int((1.0 - float(g["start_ratio"])) * steps)
            sign = float(g.get("guide_sign", 1.0))  # +1: dt>0 gradient ascent on reward
            for idx in range(steps):
                velocity = ae.forward_with_context(
                    x, inputs.modulations[idx].conditioning,
                    context=inputs.context, modulation=inputs.modulations[idx],
                )
                velocity = inner._mask_action_dim_tensor(velocity, action_dim_is_pad=apad, enabled=mask_enabled)
                if idx >= start_idx:
                    gr = steer._guidance_grad(x)
                    if gr is not None:
                        scale = float(g["guide_scale"])
                        velocity = velocity + sign * scale * gr  # nudge toward higher reward
                x = x + dt * velocity
                x = inner._mask_action_dim_tensor(x, action_dim_is_pad=apad, enabled=mask_enabled)
            return x

        inner._run_action_flow_loop = guided_loop
        self._orig_loop = orig_loop

    # -- public API --
    def set_guidance(self, keypoints, guidance_fn: Callable, *, guide_scale: float = 1.0,
                     start_ratio: float = 0.5, guide_sign: float = 1.0) -> None:
        """Arm guidance for subsequent predict_action calls. keypoints: (N,3) array/tensor."""
        kp = keypoints
        if kp is not None and not torch.is_tensor(kp):
            kp = torch.tensor(np.asarray(kp), dtype=torch.float32, device=self.device)
        self._active = dict(keypoints=kp, guidance_fn=guidance_fn, guide_scale=guide_scale,
                            start_ratio=start_ratio, guide_sign=guide_sign)

    def clear_guidance(self) -> None:
        self._active = None


# ---------------------------------------------------------------------------
# Standalone self-test (run in the molmoact venv):
#   CUDA_VISIBLE_DEVICES=7 /root/ishneet/molmoact-venv/bin/python \
#       policies/molmoact/molmoact_steer.py
# Verifies: (1) guidance OFF reproduces normal predict_action output (same seed),
#           (2) guidance ON shifts the actions, (3) FK is differentiable.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    from transformers import AutoModelForImageTextToText, AutoProcessor

    MODEL = "allenai/MolmoAct2-DROID"
    print("loading model ...", flush=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda").eval()

    steer = MolmoActSteer(model)
    print(f"norm mode: {steer._norm_mode} | joint lo: {steer._alo[:7].tolist()} | hi: {steer._ahi[:7].tolist()}", flush=True)

    # FK differentiability sanity
    q = torch.zeros(1, 1, 7, device="cuda", requires_grad=True)
    ee = steer.fk(q)
    (ee.sum()).backward()
    print("FK ok, ee(home)=", ee.detach().cpu().numpy().reshape(-1), "grad finite:", bool(torch.isfinite(q.grad).all()), flush=True)

    imgs = [np.zeros((224, 224, 3), np.uint8) for _ in range(3)]
    state = np.zeros(8, np.float32)
    task = "pick up the banana"

    def run(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        out = model.predict_action(processor=proc, images=imgs, task=task, state=state,
                                   norm_tag="franka_droid", inference_action_mode="continuous",
                                   num_steps=10, normalize_language=True,
                                   enable_depth_reasoning=False, enable_cuda_graph=False)
        a = out.actions
        return (a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a))

    steer.clear_guidance()
    a_off1 = run(123)
    a_off2 = run(123)
    print("OFF reproducible (same seed):", np.allclose(a_off1, a_off2, atol=1e-4),
          "| max|Δ|=", float(np.max(np.abs(a_off1 - a_off2))), flush=True)

    # Guidance ON: pull EE toward a fixed target point in front of the robot.
    target = torch.tensor([0.5, 0.0, 0.2], device="cuda")
    def guidance_fn(keypoints, ee_traj):  # ee_traj: (B,T,3); reward higher = closer to target
        return -((ee_traj - target) ** 2).sum(-1).mean()
    steer.set_guidance(keypoints=target[None, :], guidance_fn=guidance_fn,
                       guide_scale=2.0, start_ratio=1.0)  # guide on all steps
    a_on = run(123)
    print("ON vs OFF max|Δ|=", float(np.max(np.abs(a_on - a_off1))),
          "(should be > 0 => guidance influences actions)", flush=True)
    print("SELFTEST DONE", flush=True)
