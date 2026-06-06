# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemini-backed VLS guidance-function generator (runs in the molmoact venv).

Given the scene image (with keypoints overlaid), the language instruction, and a
legend mapping keypoint index -> object name, ask Gemini to author differentiable
`stageN_guidance(keypoints, action_sequence)` functions (the VLS contract:
keypoints (K,3), action_sequence (B,T,3) -> scalar reward, higher = better).

We compile the returned functions with `torch`/`np` injected (the prompt forbids
`import torch`). Returns the ordered list of stage guidance callables.
"""

from __future__ import annotations

import os
import re
import textwrap
from typing import Callable, Optional

import cv2
import numpy as np

# Use the ACTUAL VLS guidance template from the Vision-Language-Steering repo, so
# the prompt matches the published method (not a paraphrase). Override with
# VLS_TEMPLATE_DIR if the repo lives elsewhere.
VLS_TEMPLATE_DIR = os.environ.get(
    "VLS_TEMPLATE_DIR", "/root/ishneet/vision-language-steering/vlm_query"
)

# RoboLab-specific notes substituted into the template's {task_specific_patterns}
# slot (the base template is env-agnostic; VLS fills this with per-benchmark notes).
_ROBOLAB_NOTES = textwrap.dedent("""
    ## RoboLab (Franka tabletop) environment notes
    - A Franka Panda arm with a Robotiq gripper manipulates objects on a tabletop.
    - Coordinates are WORLD frame, METERS, Z is UP. The table surface is roughly at
      z ~= 0; objects rest on it. The action_sequence is the trajectory of the
      GRIPPER FINGERTIPS (TCP) in this world frame.
    - Keypoints are the exact CENTER positions of the scene objects (simulator
      ground truth). Their index -> object-name legend AND their current 3D
      coordinates are given below — use the coordinates to reason about geometry
      (left/right = y [+y is the robot's LEFT], forward = x, up = z, meters).
    - The policy opens/closes the gripper automatically — NEVER write gripper stages,
      and never steer z down hard: the policy already does the descent + release well.
      Your job is only to bias WHICH object and WHICH container, and to keep the EE
      CENTERED over the container so the release lands in the middle (not the rim).

    ## REQUIRED reward style — emit EXACTLY these two primitives (proven best here)
    Use time-weighting `w = torch.linspace(0.3, 1.0, action_sequence.shape[1], device=action_sequence.device)`
    in every stage (later timesteps matter more).

    (A) REACH a target object (grasp stage, or a single "pick up X" task) — full 3D pull:
        def stageN_guidance(keypoints, action_sequence):
            tgt = keypoints[IDX]
            ee = action_sequence
            d2 = ((ee - tgt) ** 2).sum(-1)
            w = torch.linspace(0.3, 1.0, ee.shape[1], device=ee.device)
            return -(d2 * w).mean()

    (B) PLACE above a container (place stage of "put X in/on Y") — GATED DESCENT.
        Pull x-y to the container center ALWAYS, but only pull z down once centered,
        so the gripper opens over the MIDDLE, never the edge. Use Z_OFF = 0.10 for a
        tall bin/crate/basket, else 0.05 for a bowl/plate/tray:
        def stageN_guidance(keypoints, action_sequence):
            tgt = keypoints[IDX]
            ee = action_sequence
            xy2 = ((ee[..., :2] - tgt[:2]) ** 2).sum(-1)
            z2 = (ee[..., 2] - (tgt[2] + Z_OFF)) ** 2
            gate = torch.exp(-xy2 / (0.06 ** 2))
            w = torch.linspace(0.3, 1.0, ee.shape[1], device=ee.device)
            return -((xy2 + 0.25 * gate * z2) * w).mean()

    ## Staging convention
      * pick / grasp ("pick up the X"): 1 stage — primitive (A) on X.
      * place-into-container ("put X in/on the Y"): 2 stages — stage 1 = (A) on X,
        stage 2 = (B) on Y.
      * relational / spatial ("the left X", "X closest to Y"): use the 3D coords to
        choose the correct keypoint index (left = larger y, right = smaller y).
      * conjunction ("X or Y"): pick ONE index — the nearer/easier candidate.

    ## REQUIRED trailing lines (same python block)
        stage_targets = [<keypoint index whose LIFT advances each stage>]
        recommended_guide_scale = 2.0
      e.g. for "put the orange in the bowl": stage_targets = [<orange idx>, <bowl idx>].
      Keep recommended_guide_scale = 2.0 (uniform, gentle): it preserves the policy's
      own grasp/descent skill. Do NOT exceed ~2.5 — aggressive pulls fight the policy
      and DROP success. Higher is only warranted if the choice is very ambiguous.
""").strip()

_FALLBACK_PROMPT = textwrap.dedent("""
You write differentiable guidance functions (PyTorch) that steer a robot
flow-matching policy to MAKE A CHOICE consistent with the instruction.

Inputs to each function:
- keypoints: torch.Tensor (K, 3), world-frame keypoint positions.
- action_sequence: torch.Tensor (B, T, 3), proposed end-effector xyz trajectory
  (world frame); the first item per batch is the current EE position.
Return: a scalar reward (higher = better). Use NEGATIVE squared distance to pull
the EE toward a target: `-(d**2).mean()`. Always return a SCALAR (`.mean()`).

Rules:
- DO NOT write "import torch" (torch and np are already available).
- No "if" statements, no obstacle/distractor penalties, no helper functions —
  each stageN_guidance must be fully self-contained.
- Only spatial-motion stages (gripper open/close is automatic). Typically:
  stage 1 = move EE to the object to grasp; stage 2 = move to the place target.
- Select a single keypoint as: `idx = torch.tensor([i], dtype=torch.long, device=keypoints.device); p = keypoints[idx][0]` (shape (3,)).
- The keypoint legend (index -> object) is given below; pick indices from it.

Keypoint legend (index: object):
{legend}

Instruction: "{instruction}"

Output exactly one python code block:
```python
num_stages = <int>
def stage1_guidance(keypoints, action_sequence):
    ...
    return reward.mean()
# ... stage2_guidance etc if needed
```
""").strip()


def _load_vls_template() -> Optional[str]:
    """Load the real VLS guidance_template.txt and fill {task_specific_patterns}."""
    base_path = os.path.join(VLS_TEMPLATE_DIR, "guidance_template.txt")
    if not os.path.exists(base_path):
        return None
    with open(base_path) as f:
        base = f.read()
    return base.replace("{task_specific_patterns}", _ROBOLAB_NOTES)


def _fill_template(template: str, *, instruction: str,
                   keypoints_named: list) -> str:
    """keypoints_named: list of (name, (u,v), (x,y,z)) in keypoint-index order."""
    legend = "\n".join(f"{i}: {kp[0]}" for i, kp in enumerate(keypoints_named))

    def _coord_line(i, kp):
        if len(kp) > 2 and kp[2] is not None:
            line = f"{i}: {kp[0]} at [{kp[2][0]:.3f}, {kp[2][1]:.3f}, {kp[2][2]:.3f}]"
            if len(kp) > 3 and kp[3] is not None:
                sx, sy, sz = kp[3]
                line += f" size [{sx:.2f}, {sy:.2f}, {sz:.2f}]"
            return line
        return f"{i}: {kp[0]} (position unknown)"

    coords = "\n".join(_coord_line(i, kp) for i, kp in enumerate(keypoints_named))
    prompt = template
    for ph, val in {
        "{instruction}": str(instruction),
        "{key_points_objects_map}": legend,
        "{init_keypoint_positions}": coords,
        "{num_keypoints}": str(len(keypoints_named)),
    }.items():
        prompt = prompt.replace(ph, val)
    return prompt


def _draw_overlay(image_rgb: np.ndarray, kps_uv: list[tuple[float, float]]) -> np.ndarray:
    img = np.ascontiguousarray(image_rgb[..., ::-1].copy())  # to BGR for cv2 draw
    for i, (u, v) in enumerate(kps_uv):
        u, v = int(round(u)), int(round(v))
        cv2.circle(img, (u, v), 6, (0, 0, 255), -1)
        cv2.circle(img, (u, v), 6, (255, 255, 255), 1)
        cv2.putText(img, str(i), (u + 7, v - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return img[..., ::-1]  # back to RGB


def _compile_stages_and_targets(code: str) -> tuple[list[Callable], list[int]]:
    ns: dict = {}
    import numpy as _np
    import torch as _torch
    ns["torch"] = _torch
    ns["np"] = _np
    # strip any stray 'import torch' lines the model may emit despite instructions
    code = re.sub(r"^\s*import\s+torch.*$", "", code, flags=re.MULTILINE)
    exec(code, ns)  # noqa: S102 - trusted: our own prompt, sandboxed namespace
    n = int(ns.get("num_stages", 0) or 0)
    stages = []
    i = 1
    while True:
        fn = ns.get(f"stage{i}_guidance")
        if fn is None:
            break
        stages.append(fn)
        i += 1
    if n and len(stages) >= 1:
        stages = stages[:max(n, 1)] if len(stages) >= n else stages
    raw_targets = ns.get("stage_targets") or []
    try:
        stage_targets = [int(t) for t in list(raw_targets)]
    except Exception:
        stage_targets = []
    return stages, stage_targets


def _compile_stages(code: str) -> list[Callable]:  # backwards-compat
    return _compile_stages_and_targets(code)[0]


class GeminiGuidance:
    def __init__(self, model: str = "gemini-3.1-pro-preview", api_key: Optional[str] = None):
        from google import genai
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        self.client = genai.Client(api_key=key)
        self.model = model

    def generate(self, image_rgb: np.ndarray, instruction: str,
                 keypoints_named: list,
                 save_overlay_path: Optional[str] = None) -> tuple[list[Callable], list[int], str]:
        """keypoints_named: list of (object_name, (u,v) pixel, (x,y,z) world) in
        keypoint index order. Returns (stage_fns, stage_targets, raw_text)."""
        from google.genai import types

        template = _load_vls_template()
        if template is not None:
            prompt = _fill_template(template, instruction=instruction, keypoints_named=keypoints_named)
            self.last_template_source = "vls"
        else:
            legend = "\n".join(f"{i}: {kp[0]}" for i, kp in enumerate(keypoints_named))
            prompt = _FALLBACK_PROMPT.format(legend=legend, instruction=instruction)
            self.last_template_source = "fallback"
        overlay = _draw_overlay(image_rgb, [kp[1] for kp in keypoints_named])
        ok, buf = cv2.imencode(".png", overlay[..., ::-1])  # encode as BGR png
        if save_overlay_path:
            try:
                with open(save_overlay_path, "wb") as f:
                    f.write(buf.tobytes())
            except Exception:
                pass
        img_part = types.Part.from_bytes(data=buf.tobytes(), mime_type="image/png")

        resp = self.client.models.generate_content(model=self.model, contents=[prompt, img_part])
        text = resp.text or ""
        m = re.search(r"```python\s*(.*?)```", text, flags=re.DOTALL)
        code = m.group(1) if m else text
        stages, stage_targets = _compile_stages_and_targets(code)
        # VLS-style per-task recommended guide scale (Gemini judges from scene geometry)
        rec = re.search(r"recommended_guide_scale\s*=\s*([0-9.]+)", code)
        self.last_recommended_scale = float(rec.group(1)) if rec else None
        return stages, stage_targets, text


if __name__ == "__main__":
    # Standalone test: synthetic scene + 2 keypoints + instruction -> compilable guidance.
    import torch
    img = np.full((224, 224, 3), 60, np.uint8)
    kps_named = [("banana", (60.0, 120.0), (0.55, 0.12, 0.04)),
                 ("bowl", (170.0, 130.0), (0.52, -0.18, 0.05))]
    g = GeminiGuidance()
    stages, stage_targets, raw = g.generate(img, "put the banana in the bowl", kps_named)
    print("num stage fns:", len(stages), "| stage_targets:", stage_targets)
    print("---- raw (head) ----\n", raw[:600])
    # exercise stage1 with dummy tensors
    kp = torch.tensor([[0.1, 0.2, 0.0], [0.4, 0.1, 0.0]], dtype=torch.float32)
    traj = torch.zeros(2, 8, 3, requires_grad=True)
    r = stages[0](kp, traj)
    r.backward()
    print("stage1 reward:", float(r), "| grad finite:", bool(torch.isfinite(traj.grad).all()))
    print("GUIDANCE_GEMINI OK")
