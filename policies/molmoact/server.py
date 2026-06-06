# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MolmoAct2-DROID inference server (plain + VLS gradient steering).

Loads ``allenai/MolmoAct2-DROID`` once and serves action predictions over a ZMQ
REQ/REP socket (msgpack). Two modes per request:

- plain: {images, state, task}  -> MolmoAct2.predict_action (unsteered)
- steered: additionally {steer: true, keypoints:[{name,xyz,uv}], guide_scale,
  start_ratio, stage} -> VLS-style gradient steering (see molmoact_steer.py).
  On a new task string it asks Gemini (guidance_gemini.py) to author the
  differentiable guidance functions, then injects their reward gradient into the
  flow loop each step. Keypoints (object world positions) come from the client.

Run in the molmoact venv with the Gemini key in env:

    GOOGLE_API_KEY=... CUDA_VISIBLE_DEVICES=6 /root/ishneet/molmoact-venv/bin/python \
        policies/molmoact/server.py --port 5555 --dtype bfloat16 --steer
"""

import argparse
import os
import sys
import time
import traceback

# Allow `from policies.molmoact...` imports when run as a script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import msgpack
import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def pack_arr(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.tobytes()}


def unpack_arr(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"]).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="MolmoAct2-DROID ZMQ inference server.")
    parser.add_argument("--model", default="allenai/MolmoAct2-DROID")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-solver steps.")
    parser.add_argument("--norm-tag", default="franka_droid")
    parser.add_argument("--steer", action="store_true", help="Enable VLS gradient steering support.")
    parser.add_argument("--gemini-model", default="gemini-2.5-pro")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"[MolmoAct] loading {args.model} ({args.dtype}) ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=dtype,
    ).to("cuda").eval()
    print("[MolmoAct] model loaded.", flush=True)

    steer = None
    guidance = None
    guidance_cache: dict = {}      # task -> list[stage_fn]
    if args.steer:
        from policies.molmoact.molmoact_steer import MolmoActSteer
        from policies.molmoact.guidance_gemini import GeminiGuidance
        bt_path = os.path.join(os.path.dirname(__file__), "_fk_base_transform.npy")
        base_T = np.load(bt_path) if os.path.exists(bt_path) else None
        steer = MolmoActSteer(model, norm_tag=args.norm_tag, base_transform=base_T)
        guidance = GeminiGuidance(model=args.gemini_model)
        print(f"[MolmoAct] steering enabled (base_transform={'loaded' if base_T is not None else 'identity'}, "
              f"gemini={args.gemini_model})", flush=True)

    ctx = __import__("zmq").Context()
    sock = ctx.socket(__import__("zmq").REP)
    sock.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"[MolmoAct] server listening on tcp://0.0.0.0:{args.port}", flush=True)

    def run_plain(images, state, task):
        with torch.no_grad():
            out = model.predict_action(
                processor=processor, images=images, task=task, state=state,
                norm_tag=args.norm_tag, inference_action_mode="continuous",
                num_steps=args.num_steps, normalize_language=True,
                enable_depth_reasoning=False, enable_cuda_graph=False,
            )
        return out.actions

    while True:
        raw = sock.recv()
        req = msgpack.unpackb(raw, raw=False)

        if req.get("ping"):
            sock.send(msgpack.packb({"ok": True, "kind": "molmoact2-droid", "steer": bool(args.steer)},
                                    use_bin_type=True))
            continue

        try:
            images = [unpack_arr(im) for im in req["images"]]
            state = np.asarray(req["state"], dtype=np.float32)
            task = str(req["task"])
            extra = {}

            do_steer = bool(req.get("steer")) and steer is not None and req.get("keypoints")
            if do_steer:
                kps = req["keypoints"]  # list of {name, xyz, uv}
                kp_world = torch.tensor([k["xyz"] for k in kps], dtype=torch.float32, device=steer.device)
                # (Re)generate guidance functions on a new task.
                if task not in guidance_cache:
                    named = [(str(k["name"]), tuple(k.get("uv", (0, 0))), tuple(k["xyz"])) for k in kps]
                    stages, _stage_targets, rawtext = guidance.generate(images[0], task, named)
                    guidance_cache[task] = stages
                    extra["num_stages"] = len(stages)
                    extra["guidance_text"] = rawtext[:1200]
                    print(f"[MolmoAct] generated {len(stages)} guidance stage(s) for task: {task!r}", flush=True)
                stages = guidance_cache[task]
                stage_idx = min(int(req.get("stage", 0)), max(len(stages) - 1, 0))
                if stages:
                    steer.set_guidance(
                        keypoints=kp_world, guidance_fn=stages[stage_idx],
                        guide_scale=float(req.get("guide_scale", 2.0)),
                        start_ratio=float(req.get("start_ratio", 0.6)),
                    )
                else:
                    steer.clear_guidance()
            elif steer is not None:
                steer.clear_guidance()

            t0 = time.perf_counter()
            acts = run_plain(images, state, task)
            if isinstance(acts, torch.Tensor):
                acts = acts.detach().to(torch.float32).cpu().numpy()
            actions = np.asarray(acts, dtype=np.float32)
            if actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            if actions.ndim == 1:
                actions = actions[None, :]
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if steer is not None:
                steer.clear_guidance()
            sock.send(msgpack.packb({"ok": True, "actions": pack_arr(actions), "infer_ms": dt_ms,
                                     "steered": bool(do_steer), **extra}, use_bin_type=True))
        except Exception as e:
            traceback.print_exc()
            if steer is not None:
                steer.clear_guidance()
            sock.send(msgpack.packb({"ok": False, "error": f"{type(e).__name__}: {e}"}, use_bin_type=True))


if __name__ == "__main__":
    main()
