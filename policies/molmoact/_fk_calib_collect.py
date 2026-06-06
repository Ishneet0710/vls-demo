# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect (arm_joint_pos, EE world pos) pairs from a RoboLab rollout for FK calibration.

Runs in the RoboLab venv. Dumps JSON consumed by `_fk_calib_fit.py` (molmoact venv),
which Kabsch-fits the base->world rigid transform for DifferentiableFrankaFK.
"""

import argparse
import json

import cv2  # noqa: F401 - before isaaclab
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="BananaInBowlTask")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--out", default="policies/molmoact/_fk_calib_pairs.json")
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.enable_cameras = True
args.headless = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402

auto_register_droid_envs(task=[args.task])
env, env_cfg = create_env(args.task, device=args.device, num_envs=1, instruction_type="default", policy="molmoact")

robot = env.scene["robot"]
body_names = list(robot.data.body_names)
print("BODY_NAMES:", body_names, flush=True)

# Pick an EE-ish body: prefer hand/tcp/gripper/link8, else last body.
def pick_ee_body(names):
    for kw in ["panda_hand", "tcp", "gripper", "hand", "base_link", "link8", "link7"]:
        for n in names:
            if kw in n.lower():
                return n
    return names[-1]

ee_body = pick_ee_body(body_names)
ee_idx = body_names.index(ee_body)
print("EE_BODY:", ee_body, "idx", ee_idx, flush=True)

action_dim = getattr(getattr(env, "action_manager", None), "total_action_dim", None) or env.action_space.shape[-1]

import omni.kit.app
import omni.timeline
timeline = omni.timeline.get_timeline_interface()
kit_app = omni.kit.app.get_app()

obs, _ = env.reset()
obs, _ = env.reset()

pairs = []
base_pos = robot.data.root_pos_w[0].detach().cpu().numpy().tolist()
base_quat = robot.data.root_quat_w[0].detach().cpu().numpy().tolist()
env_origin = env.scene.env_origins[0].detach().cpu().numpy().tolist()

torch.manual_seed(0)
for step in range(args.steps):
    while not timeline.is_playing():
        kit_app.update()
    # small random joint targets around current to explore configs
    q_now = robot.data.joint_pos[0, :7].detach().cpu().numpy()
    target = q_now + np.random.uniform(-0.3, 0.3, size=7).astype(np.float32)
    act = torch.zeros(env.num_envs, action_dim, device=env.device)
    act[0, :7] = torch.tensor(target, device=env.device)
    obs, _, _, _, _ = env.step(act)

    q = robot.data.joint_pos[0, :7].detach().cpu().numpy()
    ee_w = robot.data.body_pos_w[0, ee_idx, :].detach().cpu().numpy()
    pairs.append({"q": q.tolist(), "ee_world": ee_w.tolist()})

out = {
    "ee_body": ee_body,
    "base_pos_w": base_pos,
    "base_quat_w": base_quat,
    "env_origin": env_origin,
    "pairs": pairs,
}
with open(args.out, "w") as f:
    json.dump(out, f)
print(f"WROTE {len(pairs)} pairs -> {args.out}", flush=True)
app.close()
