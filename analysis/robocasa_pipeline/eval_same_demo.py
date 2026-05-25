#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any
import uuid

import h5py
import imageio.v2 as imageio
import numpy as np

from rldx.policy.server_client import PolicyClient
from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
from robocasa.utils.env_utils import convert_action
from robocasa.wrappers.gym_wrapper import RoboCasaGymEnv


FULL_STATE_KEYS = [
    ("state.base_position", 3),
    ("state.base_rotation", 4),
    ("state.end_effector_position_absolute", 3),
    ("state.end_effector_position_relative", 3),
    ("state.end_effector_rotation_absolute", 4),
    ("state.end_effector_rotation_relative", 4),
    ("state.gripper_qpos", 2),
    ("state.gripper_qvel", 2),
    ("state.joint_position", 7),
    ("state.joint_position_cos", 7),
    ("state.joint_position_sin", 7),
    ("state.joint_velocity", 7),
]


def load_demo(demo_hdf5: Path, demo_key: str | None) -> dict[str, Any]:
    with h5py.File(demo_hdf5, "r") as f:
        data = f["data"]
        if demo_key is None:
            demo_key = sorted(data.keys())[0]
        grp = data[demo_key]
        model_file = grp.attrs["model_file"]
        ep_meta = grp.attrs.get("ep_meta", None)
        actions = grp["actions"][:]
        states = grp["states"][:]
        env_args = json.loads(data.attrs["env_args"])
    if isinstance(model_file, bytes):
        model_file = model_file.decode("utf-8")
    if isinstance(ep_meta, bytes):
        ep_meta = ep_meta.decode("utf-8")
    return {
        "demo_key": demo_key,
        "model_file": model_file,
        "ep_meta": ep_meta,
        "actions": actions,
        "states": states,
        "env_args": env_args,
    }


class SameDemoRoboCasaEnv(RoboCasaGymEnv):
    def __init__(self, env_name: str, initial_state: dict[str, Any], **kwargs):
        self._initial_state = initial_state
        super().__init__(env_name=env_name, **kwargs)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.rng = np.random.default_rng(int(seed) % (2**32))
        self.env.reset()
        reset_to(self.env, self._initial_state)
        raw_obs = (
            self.env.viewer._get_observations(force_update=True)
            if self.env.viewer_get_obs
            else self.env._get_observations(force_update=True)
        )
        obs = self._remap_observation(self.get_observation(raw_obs))
        return obs, {"success": False}

    def step(self, action_dict):
        obs, reward, terminated, truncated, info = super().step(action_dict)
        return self._remap_observation(obs), reward, terminated, truncated, info

    def _remap_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        remapped = {
            "video.left_view": obs.pop("video.robot0_agentview_left"),
            "video.right_view": obs.pop("video.robot0_agentview_right"),
            "video.wrist_view": obs.pop("video.robot0_eye_in_hand"),
            **obs,
        }
        for key, dim in FULL_STATE_KEYS:
            if key not in remapped:
                remapped[key] = np.zeros(dim, dtype=np.float32)
        return remapped


def build_initial_state(demo: dict[str, Any]) -> dict[str, Any]:
    state = {"states": demo["states"][0], "model": demo["model_file"]}
    if demo["ep_meta"] is not None:
        state["ep_meta"] = demo["ep_meta"]
    return state


def stack_history(
    history: deque[dict[str, Any]],
    video_delta_indices: np.ndarray,
    state_delta_indices: np.ndarray,
) -> dict[str, Any]:
    latest = len(history) - 1
    stacked: dict[str, Any] = {}
    sample = history[-1]
    for key in sample:
        if key.startswith("video."):
            deltas = video_delta_indices
            stacked[key] = np.stack([history[latest + int(delta)][key] for delta in deltas], axis=0)
        elif key.startswith("state."):
            deltas = state_delta_indices
            stacked[key] = np.stack([history[latest + int(delta)][key] for delta in deltas], axis=0)
        elif key.startswith("annotation."):
            stacked[key] = sample[key]
    return stacked


def batch_observation(obs: dict[str, Any]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            batched[key] = value[None, ...]
        else:
            batched[key] = (str(value),)
    return batched


def write_video(path: Path, frames: list[np.ndarray], fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)


def run_ground_truth_rollout(
    env: SameDemoRoboCasaEnv,
    demo: dict[str, Any],
    video_path: Path,
) -> dict[str, Any]:
    obs, info = env.reset(seed=0)
    frames = [env.render()]
    success = bool(info.get("success", False))
    steps = 0
    for action_row in demo["actions"]:
        action = convert_action(action_row.astype(np.float32))
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        steps += 1
        success = success or bool(info.get("success", False)) or reward > 0
        if terminated or truncated or success:
            break
    write_video(video_path, frames)
    return {"success": success, "steps": steps, "video_path": str(video_path)}


def run_policy_rollout(
    env: SameDemoRoboCasaEnv,
    policy: PolicyClient,
    video_delta_indices: np.ndarray,
    state_delta_indices: np.ndarray,
    max_episode_steps: int,
    n_action_steps: int,
    video_path: Path,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=0)
    max_history = int(max(np.max(-video_delta_indices), np.max(-state_delta_indices))) + 1
    history: deque[dict[str, Any]] = deque([obs] * max_history, maxlen=max_history)
    frames = [env.render()]
    session_id = f"same_demo_{uuid.uuid4().hex[:8]}"
    is_first_step = True
    success = False
    primitive_steps = 0

    while primitive_steps < max_episode_steps and not success:
        stacked_obs = stack_history(history, video_delta_indices, state_delta_indices)
        batched_obs = batch_observation(stacked_obs)
        options = {"reset_memory": [is_first_step], "session_ids": [session_id]}
        action_chunk, _ = policy.get_action(batched_obs, options=options)
        is_first_step = False

        for step_idx in range(n_action_steps):
            action = {
                key: value[0, step_idx].astype(np.float32)
                for key, value in action_chunk.items()
            }
            obs, reward, terminated, truncated, info = env.step(action)
            history.append(obs)
            frames.append(env.render())
            primitive_steps += 1
            success = success or bool(info.get("success", False)) or reward > 0
            if terminated or truncated or success or primitive_steps >= max_episode_steps:
                break

    write_video(video_path, frames)
    return {
        "success": success,
        "steps": primitive_steps,
        "video_path": str(video_path),
        "session_id": session_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RLDX checkpoint on the same RoboCasa demo instance used for training.")
    parser.add_argument("--demo-hdf5", type=Path, required=True, help="Path to MimicGen or RoboCasa demo.hdf5")
    parser.add_argument("--policy-host", type=str, default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--demo-key", type=str, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--n-action-steps", type=int, default=16)
    args = parser.parse_args()

    demo = load_demo(args.demo_hdf5, args.demo_key)
    env_name = demo["env_args"]["env_name"]
    initial_state = build_initial_state(demo)

    env = SameDemoRoboCasaEnv(
        env_name=env_name,
        initial_state=initial_state,
        enable_render=True,
        split=None,
    )
    policy = PolicyClient(host=args.policy_host, port=args.policy_port)
    modality_config = policy.get_modality_config()
    video_delta_indices = np.array(modality_config["video"].delta_indices, dtype=np.int64)
    state_delta_indices = np.array(modality_config["state"].delta_indices, dtype=np.int64)
    policy.reset()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_video = args.output_dir / "same_demo_ground_truth.mp4"
    policy_video = args.output_dir / "same_demo_policy.mp4"

    gt_result = run_ground_truth_rollout(env, demo, gt_video)
    policy.reset()
    policy_result = run_policy_rollout(
        env=env,
        policy=policy,
        video_delta_indices=video_delta_indices,
        state_delta_indices=state_delta_indices,
        max_episode_steps=min(args.max_episode_steps, int(demo["actions"].shape[0])),
        n_action_steps=args.n_action_steps,
        video_path=policy_video,
    )

    summary = {
        "demo_key": demo["demo_key"],
        "env_name": env_name,
        "ground_truth": gt_result,
        "policy": policy_result,
    }
    summary_path = args.output_dir / "same_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
