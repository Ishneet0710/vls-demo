# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hand-authored VLS guidance for the RoboLab demo tasks.

Why hand-written instead of Gemini for the demo:
- instant (no ~14 s VLM call per prompt) and deterministic (no LLM variance);
- smarter use of privileged info: resolve 'left/right' bins from world coords,
  and — crucially — make the PLACE stage GENTLE: bias the end-effector in x-y over
  the correct container but only lightly in z, so pi0.5 keeps control of the
  descent + gripper release it already does well (an aggressive full-3D pull on
  the place stage was fighting the policy and dropping success to 0).

Each builder returns (stage_fns, stage_targets) where stage_fns follow the VLS
contract `fn(keypoints (K,3), action_sequence (B,T,3)) -> scalar reward` (higher
better), and stage_targets[i] = the keypoint index whose LIFT advances stage i.

`build(task, kp_names, kp_world)` returns (stage_fns, stage_targets, rec_scale)
for a known task, or None so the server can fall back to Gemini.
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np
import torch


# ---- reward primitives (closures over resolved keypoint indices) ----------
def _reach(idx: int):
    """Pull the TCP to object `idx` (full 3D), weighting later timesteps more."""
    def fn(keypoints, action_sequence):
        tgt = keypoints[idx]
        ee = action_sequence                                   # (B, T, 3)
        d2 = ((ee - tgt) ** 2).sum(-1)                         # (B, T)
        T = ee.shape[1]
        w = torch.linspace(0.3, 1.0, T, device=ee.device)
        return -(d2 * w).mean()
    return fn


def _place_above(idx: int, z_off: float = 0.08, z_weight: float = 0.25,
                 center_sigma: float = 0.06):
    """Bias the TCP to be ABOVE the CENTER of container `idx` before it descends.

    Failure mode this fixes: pi0.5 grasps fine but opens the gripper at the EDGE of
    the container (object catches the rim / drops outside). We can't steer the
    gripper directly (the policy decides when to release), so the only lever is
    WHERE the EE is when it releases. A naive place reward pulls x-y and z together,
    so the arm starts descending while still off to the side and releases at the rim.

    GATED DESCENT: always pull x-y toward the container center, but only pull z DOWN
    once x-y is already centered (gate = exp(-xy2/sigma^2), ~1 centered, ~0 off to
    the side). The EE therefore floats over the center first, then settles — so
    whenever the policy opens the gripper, the object is over the middle. Smooth and
    differentiable (no branches), so it composes with the VLS gradient.
    """
    def fn(keypoints, action_sequence):
        tgt = keypoints[idx]
        ee = action_sequence                                   # (B, T, 3)
        xy2 = ((ee[..., :2] - tgt[:2]) ** 2).sum(-1)           # (B, T)
        z_target = tgt[2] + z_off
        z2 = (ee[..., 2] - z_target) ** 2
        gate = torch.exp(-xy2 / (center_sigma ** 2))           # only descend when centered
        T = ee.shape[1]
        w = torch.linspace(0.3, 1.0, T, device=ee.device)
        return -((xy2 + z_weight * gate * z2) * w).mean()
    return fn


# Known STATIC wall obstacle in colored_blocks.usda ('wall': center (0.60,-0.16),
# x-half 0.09, y-half 0.01, top z 0.10). Hardcoded because it's a static collider
# (no rigid body / keypoint). Update if the wall is moved in the scene.
_WALL = {"cx": 0.60, "cy": -0.16, "top": 0.10, "sx": 0.11, "sy": 0.05}


def _avoid_wall(over_z: float = 0.10, weight: float = 4.0):
    """Obstacle avoidance: while the TCP is horizontally OVER the wall, force it
    HIGH (above wall_top + over_z) so the carried object clears the wall. Smooth &
    differentiable: a Gaussian 'over the wall' gate times a 'too low' penalty."""
    wx, wy, wt, sx, sy = (_WALL["cx"], _WALL["cy"], _WALL["top"], _WALL["sx"], _WALL["sy"])

    def fn(keypoints, action_sequence):
        ee = action_sequence
        over = torch.exp(-((ee[..., 0] - wx) / sx) ** 2 - ((ee[..., 1] - wy) / sy) ** 2)
        too_low = torch.clamp((wt + over_z) - ee[..., 2], min=0.0)
        T = ee.shape[1]
        w = torch.linspace(0.3, 1.0, T, device=ee.device)
        return -weight * (over * too_low ** 2 * w).mean()
    return fn


def _combine(*fns):
    """Sum several reward functions into one stage reward."""
    def fn(keypoints, action_sequence):
        return sum(f(keypoints, action_sequence) for f in fns)
    return fn


def _hold_up(idx: int, hold_z: float = 0.22):
    """After grasping object `idx`, LIFT it and HOLD it in the air: pull the TCP
    up to `hold_z` (strong) while staying over the object's x-y (gentle). Used for
    'pick up X and hold it' — there is no place target, so we just keep it raised."""
    def fn(keypoints, action_sequence):
        tgt = keypoints[idx]
        ee = action_sequence
        xy2 = ((ee[..., :2] - tgt[:2]) ** 2).sum(-1)
        z2 = (ee[..., 2] - hold_z) ** 2
        T = ee.shape[1]
        w = torch.linspace(0.3, 1.0, T, device=ee.device)
        return -((0.3 * xy2 + z2) * w).mean()
    return fn


# ---- name resolution ------------------------------------------------------
def _find(names: list[str], *cands: str) -> Optional[int]:
    low = [n.lower() for n in names]
    for c in cands:                       # exact first
        if c in low:
            return low.index(c)
    for c in cands:                       # then substring
        for i, n in enumerate(low):
            if c in n:
                return i
    return None


def _pick_nearest(names, kp_world, options: list[str], to_name: str) -> Optional[int]:
    """Among `options` (object names), choose the one whose world pos is nearest to
    `to_name`'s — used for 'or'/ambiguous picks (closest = easiest)."""
    ref = _find(names, to_name)
    cand_idxs = [i for o in options if (i := _find(names, o)) is not None]
    if ref is None or not cand_idxs:
        return cand_idxs[0] if cand_idxs else None
    rp = kp_world[ref]
    return min(cand_idxs, key=lambda i: float(np.linalg.norm(kp_world[i] - rp)))


def _bin_by_side(names, kp_world, side: str) -> Optional[int]:
    """Resolve left/right bin from world Y (privileged), cross-checked with names.
    RoboLab world frame: +Y is to the robot's LEFT."""
    idxs = [i for i, n in enumerate(names) if "bin" in n.lower()]
    if not idxs:
        return None
    if len(idxs) == 1:
        return idxs[0]
    # left = max Y, right = min Y
    by_y = sorted(idxs, key=lambda i: float(kp_world[i][1]))
    return by_y[-1] if side == "left" else by_y[0]


# ---- generic INSTRUCTION parser (works for any typed prompt) ---------------
_CONTAINER_TOKENS = {"bowl", "plate", "bin", "crate", "basket", "tray", "box"}
# NOTE: colors are intentionally NOT generic — in scenes like colored_blocks the
# color IS the discriminating feature ("blue cube" vs "red block"). "grey" stays
# generic so it doesn't fight bin-side resolution (grey_bin_left/right).
_GENERIC_TOKENS = {"grey", "gray", "large", "small", "left", "right", "the", "a",
                   "wood", "wooden", "serving",
                   # common nouns shared across objects -> NOT discriminating. In
                   # colored_blocks every object is a "block"/"cube", so matching on
                   # those would make every block "mentioned"; the COLOR discriminates.
                   "block", "blocks", "cube", "cubes"}
# phrases that mean "stack the grasped object ON TOP OF another (graspable) object"
_STACK_PHRASES = (" on top of ", " on top ", " ontop ", " onto ", " stack ", " stacked ")
# phrases that mean "grasp X and hold it up" (no destination container)
_HOLD_PHRASES = (" hold ", " pick up ", " pickup ", " pick it up ", " lift ", " grab ", " raise ")


def _kp_tokens(name: str) -> list[str]:
    toks = [t for t in name.lower().replace("-", "_").split("_") if len(t) > 2]
    return [t for t in toks if not t.isdigit()]


def _z_off_for(name: str) -> float:
    n = name.lower()
    if "bin" in n or "crate" in n or "basket" in n or "box" in n:
        return 0.10            # taller container -> release higher
    return 0.05                # bowl / plate / tray


# Sequence connectors: "put X in box AND THEN stack Y on Z" -> two ordered ops.
_SEQ_SPLIT = re.compile(r"\s*(?:,?\s*and\s+then|,?\s*then|;|\.\s+|after\s+that|next,?|followed\s+by)\s+", re.I)


def _split_clauses(text: str) -> list[str]:
    return [c.strip() for c in _SEQ_SPLIT.split(text) if c and c.strip()]


def _parse_clause(clause, container_idxs, graspable_idxs, kp_names, kp_world):
    """Parse ONE clause -> (grasp_idx, place_idx_or_None, is_stack, z_off) or None."""
    instr = " " + clause.lower() + " "

    def mentioned(i):
        return any(f" {t} " in instr for t in _kp_tokens(kp_names[i]) if t not in _GENERIC_TOKENS)

    mentioned_grasp = [i for i in graspable_idxs if mentioned(i)]
    if not mentioned_grasp:
        return None

    # STACKING: "X on top of Y" -> grasp X, place ABOVE block Y
    if any(p in instr for p in _STACK_PHRASES) and len(mentioned_grasp) >= 2:
        def _pos(i):
            ps = [instr.find(f" {t} ") for t in _kp_tokens(kp_names[i]) if t not in _GENERIC_TOKENS]
            ps = [p for p in ps if p >= 0]
            return min(ps) if ps else 10**9
        pivot = max(instr.find(" on top"), instr.find(" onto "), instr.find(" stack"))
        after = sorted((i for i in mentioned_grasp if _pos(i) > pivot), key=_pos)
        before = sorted((i for i in mentioned_grasp if _pos(i) <= pivot), key=_pos)
        support = after[0] if after else mentioned_grasp[-1]
        grasp = before[0] if before else mentioned_grasp[0]
        if support != grasp:
            return (grasp, support, True, 0.07)

    # PICK-AND-PLACE into a container (resolve side for bins)
    side = "left" if " left " in instr else ("right" if " right " in instr else None)
    mentioned_containers = [i for i in container_idxs if mentioned(i)]
    bin_idxs = [i for i in container_idxs if "bin" in kp_names[i].lower()]
    place_idx = None
    if side and bin_idxs:
        place_idx = _bin_by_side(kp_names, kp_world, side)
    elif mentioned_containers:
        place_idx = mentioned_containers[0]
    elif len(container_idxs) == 1 and (" in " in instr or " into " in instr or " inside " in instr):
        place_idx = container_idxs[0]

    if len(mentioned_grasp) > 1 and place_idx is not None:
        ref = kp_world[place_idx]
        grasp = min(mentioned_grasp, key=lambda i: float(np.linalg.norm(kp_world[i] - ref)))
    else:
        grasp = mentioned_grasp[0]
    if place_idx is not None and place_idx != grasp:
        return (grasp, place_idx, False, _z_off_for(kp_names[place_idx]))
    return (grasp, None, False, 0.05)


def build(task: str, kp_names: list[str], kp_world: np.ndarray):
    """`task` is the INSTRUCTION string. Returns (stage_fns, stage_targets, scales,
    stage_kinds) or None (unknown -> server falls back to Gemini).

    Handles SEQUENCES: "put red on top of blue, then yellow on top of red" splits
    into ordered ops, each chained as grasp -> place stages. stage_kinds tells the
    server how to advance ("grasp" = wait for that object to be lifted; "place" =
    wait for it to reach the target) so multi-step plans don't skip ahead."""
    kp_world = np.asarray(kp_world, dtype=np.float32)
    container_idxs, graspable_idxs = [], []
    for i, nm in enumerate(kp_names):
        if nm.lower() == "table":
            continue
        toks = _kp_tokens(nm)
        (container_idxs if (set(toks) & _CONTAINER_TOKENS) else graspable_idxs).append(i)

    # ---- HOLD / PICK-UP (no destination): grasp X and hold it up ----
    low = " " + task.lower() + " "

    def _ment(i):
        return any(f" {t} " in low for t in _kp_tokens(kp_names[i]) if t not in _GENERIC_TOKENS)

    has_place = (any(f" {c} " in low for c in ("in", "into", "inside")) or " on top" in low
                 or " onto " in low or any(_ment(i) for i in container_idxs))
    if any(h in low for h in _HOLD_PHRASES) and not has_place:
        g = next((i for i in graspable_idxs if _ment(i)), None)
        if g is not None:
            return ([_reach(g), _hold_up(g)], [g, g], [2.0, 2.0], ["grasp", "hold"])

    clauses = _split_clauses(task.lower()) or [task.lower()]
    ops = []
    for cl in clauses:
        op = _parse_clause(cl, container_idxs, graspable_idxs, kp_names, kp_world)
        if op is not None:
            ops.append(op)
    if not ops:
        return None  # couldn't identify any target -> let Gemini try

    # Obstacle avoidance: "...avoiding the wall" / "around the wall" / "over the wall"
    # adds a repulsion that lifts the TCP over the static wall while still reaching.
    avoid = any(k in low for k in (" avoid", " around", " wall", "over the"))

    stage_fns, stage_targets, scales, kinds = [], [], [], []
    for (g, p, is_stack, z_off) in ops:
        reach_fn = _combine(_reach(g), _avoid_wall()) if avoid else _reach(g)
        stage_fns.append(reach_fn); stage_targets.append(g); scales.append(2.0); kinds.append("grasp")
        if p is not None:
            if is_stack:
                # stronger z + tight x-y so the block is set down ON TOP of the support
                place_fn = _place_above(p, z_off=z_off, z_weight=0.5, center_sigma=0.035)
                kinds.append("place_on")
            else:
                place_fn = _place_above(p, z_off=z_off)
                kinds.append("place_in")
            if avoid:
                place_fn = _combine(place_fn, _avoid_wall())
            stage_fns.append(place_fn); stage_targets.append(p); scales.append(2.0)
    return (stage_fns, stage_targets, scales, kinds)


def describe(task: str, kp_names: list[str], kp_world: np.ndarray) -> str:
    """Human-readable VLS reward for the current prompt (for the demo UI). Mirrors
    build(): one block per stage with the actual targets, plus the wall-avoidance
    term when requested. Returns '' if there's no manual guidance."""
    try:
        out = build(task, kp_names, kp_world)
    except Exception:
        return ""
    if out is None:
        return "# no manual guidance for this prompt (Gemini fallback)"
    _, targets, scales, kinds = out
    low = " " + task.lower() + " "
    avoid = any(k in low for k in (" avoid", " around", " wall", "over the"))
    L = [f'# prompt: "{task}"',
         "# VLS injects this into pi0.5's flow loop:  x_t += dt * (v_t + scale * grad_x reward)",
         "# ee = end-effector trajectory, kp[...] = privileged object keypoint, w_t weights later steps", ""]
    for i, (kind, t) in enumerate(zip(kinds, targets)):
        nm = kp_names[t] if 0 <= t < len(kp_names) else str(t)
        n = i + 1
        if kind == "grasp":
            L += [f"stage {n} — REACH '{nm}':",
                  f"    reward = -mean_t  w_t * || ee_t - kp['{nm}'] ||^2"]
        elif kind == "hold":
            L += [f"stage {n} — HOLD '{nm}' up:",
                  f"    reward = -mean_t  w_t * ( 0.3*|| ee_xy - kp['{nm}']_xy ||^2 + (ee_z - 0.22)^2 )"]
        elif kind == "place_on":
            L += [f"stage {n} — STACK on '{nm}':",
                  f"    reward = -mean_t  w_t * ( || ee_xy - kp['{nm}']_xy ||^2",
                  f"                    + 0.5 * gate(ee) * (ee_z - (kp['{nm}']_z + block_h))^2 )   # land on top"]
        else:  # place_in
            L += [f"stage {n} — PLACE into '{nm}':",
                  f"    reward = -mean_t  w_t * ( || ee_xy - kp['{nm}']_xy ||^2",
                  f"                    + 0.25 * gate(ee) * (ee_z - (kp['{nm}']_z + z_off))^2 )   # gated descent"]
        if avoid:
            L += ["    reward += -4.0 * over_wall(ee) * relu(wall_top + 0.10 - ee_z)^2   # lift over the wall"]
        L += [""]
    L += [f"guide_scale = {scales[0] if scales else 2.0}    # advance grasp->next when the object lifts"]
    return "\n".join(L)
