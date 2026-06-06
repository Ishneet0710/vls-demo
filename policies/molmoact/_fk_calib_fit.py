# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kabsch-fit the base->world rigid transform for DifferentiableFrankaFK.

Runs in the molmoact venv. Loads (q, ee_world) pairs collected by
`_fk_calib_collect.py`, runs base-frame FK on q, fits the rigid transform T
(rotation+translation, no scale) minimizing ||T·FK_base(q) - ee_world||, reports
the residual, and saves T (4x4) to `_fk_base_transform.npy`.
"""

import json
import os

import numpy as np
import torch

from policies.molmoact.molmoact_steer import DifferentiableFrankaFK

HERE = os.path.dirname(__file__)
PAIRS = os.path.join(HERE, "_fk_calib_pairs.json")
OUT = os.path.join(HERE, "_fk_base_transform.npy")


def kabsch(P: np.ndarray, Q: np.ndarray):
    """Find R (3x3, det=+1), t (3,) minimizing ||R P + t - Q|| for paired rows."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = Qc - R @ Pc
    return R, t


def main():
    with open(PAIRS) as f:
        data = json.load(f)
    pairs = data["pairs"]
    q = np.array([p["q"] for p in pairs], dtype=np.float64)        # (N,7)
    ee = np.array([p["ee_world"] for p in pairs], dtype=np.float64)  # (N,3)
    print(f"loaded {len(q)} pairs | ee_body={data.get('ee_body')}")

    fk = DifferentiableFrankaFK(device="cpu", dtype=torch.float64)
    with torch.no_grad():
        base_pos = fk(torch.tensor(q)).numpy()  # (N,3) base-frame FK

    R, t = kabsch(base_pos, ee)
    pred = (base_pos @ R.T) + t
    resid = np.linalg.norm(pred - ee, axis=1)
    print(f"FK fit residual: mean={resid.mean()*1000:.1f}mm  max={resid.max()*1000:.1f}mm")

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    np.save(OUT, T)
    print(f"saved base->world transform -> {OUT}")
    print("translation (m):", np.round(t, 3).tolist())


if __name__ == "__main__":
    main()
