# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RoboLab inference client for the MolmoAct2-DROID policy.

Observation extraction mirrors ``Pi0DroidJointposClient`` (same RoboLab camera
and proprio keys); the wire transport is a ZMQ REQ/REP socket (msgpack) to the
MolmoAct2 server in ``policies/molmoact/server.py``.

MolmoAct2-DROID takes 3 RGB views ``[exterior_1, exterior_2, wrist]`` + an 8-dim
state ``[7 joint positions, gripper]`` + a task string, and returns absolute
joint-pose actions (8-dim), which map directly onto RoboLab's
``DroidJointPositionActionCfg``.
"""

import logging

import msgpack
import numpy as np
import zmq

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


def _pack_arr(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "data": a.tobytes()}


def _unpack_arr(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=d["dtype"]).reshape(d["shape"]).copy()


def _quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = (x * x + y * y + z * z + w * w) ** 0.5 or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=float)


class MolmoActDroidClient(InferenceClient):
    """Drives MolmoAct2-DROID over ZMQ. One connection, per-env chunk state."""

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 5555,
        open_loop_horizon: int = 8,
        timeout_ms: int = 60000,
        binarize_gripper: bool = True,
        steer: bool = False,
        guide_scale: float = 2.0,
        start_ratio: float = 0.6,
    ) -> None:
        super().__init__()
        self.open_loop_horizon = int(open_loop_horizon)
        self._host = remote_host
        self._port = remote_port
        self._timeout_ms = timeout_ms
        self._binarize_gripper = binarize_gripper
        self._steer = bool(steer)
        self._guide_scale = float(guide_scale)
        self._start_ratio = float(start_ratio)
        self._env = None              # set via attach_env for keypoint extraction
        self._cam_name = "over_shoulder_left_camera"
        self._episode = 0             # incremented on reset(); lets the server reset stage state
        self._ctx = zmq.Context.instance()
        self._connect()
        print(f"[MolmoActDroidClient] Connecting to tcp://{remote_host}:{remote_port} ...")
        meta = self._query_server({"ping": True})
        print(f"[MolmoActDroidClient] Connected. server: {meta}")

    def _connect(self) -> None:
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self._timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self._host}:{self._port}")

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        exterior = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id].clone().detach().cpu().numpy()
        wrist = raw_obs["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        proprio = raw_obs["proprio_obs"]
        joint = proprio["arm_joint_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)
        gripper = proprio["gripper_pos"][env_id].clone().detach().cpu().numpy().astype(np.float32)
        state = np.concatenate([joint.reshape(-1), gripper.reshape(-1)]).astype(np.float32)  # (8,)

        return {"exterior": exterior.astype(np.uint8), "wrist": wrist.astype(np.uint8), "state": state}

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        ext = extracted_obs["exterior"]
        wrist = extracted_obs["wrist"]
        # Franka DROID layout is [exterior_1, exterior_2, wrist]; RoboLab gives one
        # exterior view, so duplicate it (matches the model card's 2-camera fallback).
        req = {
            "images": [_pack_arr(ext), _pack_arr(ext), _pack_arr(wrist)],
            "state": [float(x) for x in extracted_obs["state"].tolist()],
            "task": instruction,
        }
        if self._steer and self._env is not None:
            kps = self._scene_keypoints(env_id=0)
            if kps:
                req.update(steer=True, keypoints=kps, episode=self._episode,
                           guide_scale=self._guide_scale, start_ratio=self._start_ratio)
        return req

    def reset(self, *, env_id=None) -> None:
        self._episode += 1  # new episode -> server resets its stage state
        super().reset(env_id=env_id)

    # ---- steering: privileged scene keypoints --------------------------

    def attach_env(self, env) -> None:
        """Give the client the RoboLab env so it can read object world positions."""
        self._env = env
        self._kp_logged = False           # re-log keypoints for the new scene
        self._obj_sizes = self._object_sizes()  # privileged AABB extents, once per scene

    def _object_sizes(self) -> dict:
        """Best-effort per-object AABB extents (m) from the USD stage (privileged).

        OPT-IN via ROBOLAB_VLS_SIZES=1: BBoxCache reads on a live PhysX stage are a
        suspected (unconfirmed) trigger of simulation-view invalidation crashes.
        """
        import os as _os
        if _os.environ.get("ROBOLAB_VLS_SIZES", "0") != "1":
            return {}
        sizes = {}
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
            stage = omni.usd.get_context().get_stage()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            for name, obj in self._enumerate_objects().items():
                try:
                    path = str(getattr(obj.cfg, "prim_path", ""))
                    path = path.replace("env_.*", "env_0").replace(".*", "0")
                    prim = stage.GetPrimAtPath(path)
                    if prim and prim.IsValid():
                        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                        mn, mx = r.GetMin(), r.GetMax()
                        ext = [float(mx[i] - mn[i]) for i in range(3)]
                        if all(0.0 < e < 2.0 for e in ext):
                            sizes[name] = ext
                except Exception:
                    continue
        except Exception:
            pass
        return sizes

    def _enumerate_objects(self):
        """Return {name: rigid_object} for manipulable scene objects, robustly."""
        env = self._env
        cands = {}
        for attr in ("rigid_objects", "_rigid_objects"):
            try:
                d = getattr(env.scene, attr, None)
                if d:
                    cands = dict(d)
                    break
            except Exception:
                pass
        # Drop the robot articulation if present.
        return {n: o for n, o in cands.items() if n not in ("robot",)}

    def _scene_keypoints(self, env_id: int = 0):
        """World-frame positions of manipulable scene objects + best-effort image uv.

        The xyz (used in the reward gradient) is exact sim ground truth; uv is only
        a visual aid for the VLM, which selects keypoints by the name legend.
        """
        objs = self._enumerate_objects()
        K, R_wc, cam_t = self._camera_params(env_id)
        out = []
        for name, obj in objs.items():
            try:
                xyz = obj.data.root_pos_w[env_id].detach().cpu().numpy().astype(float)
            except Exception:
                continue
            uv = self._project(xyz, K, R_wc, cam_t)
            entry = {"name": str(name), "xyz": [float(v) for v in xyz],
                     "uv": [float(uv[0]), float(uv[1])]}
            size = getattr(self, "_obj_sizes", {}).get(name)
            if size:
                entry["size"] = size
            out.append(entry)
        if not getattr(self, "_kp_logged", False):
            self._kp_logged = True
            print(f"[MolmoActDroidClient] keypoints: {[o['name'] for o in out]} "
                  f"(steer={self._steer}, cam_K={'ok' if K is not None else 'none'})", flush=True)
        return out

    def _camera_params(self, env_id: int):
        try:
            cam = self._env.scene[self._cam_name]
            K = cam.data.intrinsic_matrices[env_id].detach().cpu().numpy()
            cam_t = cam.data.pos_w[env_id].detach().cpu().numpy()
            # Build the optical frame from the camera preset's DOCUMENTED look
            # direction (robolab/variations/camera.py) instead of decoding sensor
            # quaternion conventions: z_opt = look dir, x_opt = right, y_opt = down.
            # Valid for the default over-shoulder preset (all demo scenes).
            f = np.array([0.63, -0.48, -0.62]); f = f / np.linalg.norm(f)
            up = np.array([0.0, 0.0, 1.0])
            r = np.cross(f, up); r = r / np.linalg.norm(r)
            d = np.cross(f, r)  # right-handed optical: x(right) x y(down) = z(fwd)
            R_wc = np.stack([r, d, f], axis=1)  # columns = optical axes in world
            return K, R_wc, cam_t
        except Exception:
            return None, None, None

    def project_world(self, xyz, env_id: int = 0):
        """Project a world-frame point to over-shoulder pixel (u,v), or None.
        Used to label fixed scene elements (e.g. the static wall) that aren't keypoints."""
        K, R_wc, cam_t = self._camera_params(env_id)
        if K is None:
            return None
        u, v = self._project(xyz, K, R_wc, cam_t)
        return [float(u), float(v)]

    @staticmethod
    def _project(xyz, K, R_wc, cam_t):
        if K is None:
            return (0.0, 0.0)
        Xc = R_wc.T @ (np.asarray(xyz) - cam_t)  # world -> camera (ROS optical)
        z = float(Xc[2]) if abs(Xc[2]) > 1e-6 else 1e-6
        u = float(K[0, 0] * Xc[0] / z + K[0, 2])
        v = float(K[1, 1] * Xc[1] / z + K[1, 2])
        return (u, v)

    def _query_server(self, request: dict, max_retries: int = 3) -> dict:
        """Send + recv with reconnect-and-resend on transport hiccups.

        A REQ socket that times out (e.g. the TCP link went stale during a long
        pause) is stuck in recv state, so we must close + reopen and resend the
        request rather than propagate — otherwise a single hiccup kills the demo.
        """
        payload = msgpack.packb(request, use_bin_type=True)
        last_err = None
        for attempt in range(max_retries):
            try:
                self._sock.send(payload)
                resp = msgpack.unpackb(self._sock.recv(), raw=False)
                if not resp.get("ok", True):
                    raise RuntimeError(f"MolmoAct server error: {resp.get('error')}")
                return resp
            except zmq.ZMQError as e:
                last_err = e
                logger.warning(
                    "[MolmoActDroidClient] ZMQ error (%s), reconnecting (attempt %d/%d)...",
                    e, attempt + 1, max_retries,
                )
                try:
                    self._sock.close(0)
                except Exception:
                    pass
                self._connect()
        raise RuntimeError(f"MolmoAct server unreachable after {max_retries} attempts: {last_err}")

    def _unpack_response(self, response: dict) -> np.ndarray:
        return _unpack_arr(response["actions"]).astype(np.float32)  # (horizon, 8)

    # ---- optional hooks -----------------------------------------------

    def _needs_refresh(self, env_id: int) -> bool:
        if env_id not in self._chunks:
            return True
        horizon = min(self.open_loop_horizon, self._chunks[env_id].shape[0])
        return self._counters[env_id] >= horizon

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        if self._binarize_gripper and chunk.shape[-1] >= 1:
            chunk = chunk.copy()
            chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        ext, wr = extracted_obs["exterior"], extracted_obs["wrist"]
        if ext.shape[0] != wr.shape[0]:  # cameras may render at different heights
            import cv2
            new_w = max(1, int(round(wr.shape[1] * ext.shape[0] / wr.shape[0])))
            wr = cv2.resize(wr, (new_w, ext.shape[0]))
        return np.concatenate([ext, wr], axis=1)

    def close(self) -> None:
        try:
            self._sock.close(0)
        except Exception:
            pass
