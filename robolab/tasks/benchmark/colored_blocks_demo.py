# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_in_container
from robolab.core.task.task import Task


@configclass
class ColoredBlocksDemoTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # Demo task: any colored block ending up inside the box counts as success.
    # (The stacking prompt is exercised live via steering; this predicate covers
    #  the "put <color> block in the box" prompts.)
    success = DoneTerm(
        func=object_in_container,
        params={
            "object": ["blue_block", "red_block", "green_block", "mug", "lemon"],
            "container": "box",
            "logical": "any",
            "require_gripper_detached": True,
        },
    )


@dataclass
class ColoredBlocksDemoTask(Task):
    """Interactive demo scene: 4 colored cubes + an open box, for language-steering.

    Supports prompts like:
      - "put the blue cube in the box"
      - "put the red block in the box"
      - "put the blue cube on top of the red block"  (stacking)
    """
    contact_object_list = ["blue_block", "red_block", "green_block", "box", "bowl", "mug", "lemon", "table"]
    scene = import_scene("colored_blocks.usda", contact_object_list)
    terminations = ColoredBlocksDemoTerminations
    instruction = {
        "default": "Put the blue cube in the box",
        "red_in_box": "Put the red block in the box",
        "stack": "Put the blue cube on top of the red block",
    }
    episode_length_s: int = 120
    attributes = ["color", "pick_and_place", "stacking"]
