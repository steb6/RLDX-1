from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio
import numpy as np
import pandas as pd

from common import (
    ACTION_DIM,
    ACTION_KEY_MAP,
    DEFAULT_CAMERAS,
    STATE_DIM,
    STATE_KEY_MAP,
    build_native_modality_meta,
    compute_vector_stats,
    load_json_maybe_bytes,
    make_video_feature,
    task_spec,
    upload_folder_to_hub,
    write_json,
    write_jsonl,
)


def _resolve_task_name(source_hdf5: Path, explicit_task: str | None) -> tuple[str, str]:
    if explicit_task is not None:
        spec = task_spec(explicit_task)
        return explicit_task, spec["env_name"]

    with h5py.File(source_hdf5, "r") as f:
        env_args = load_json_maybe_bytes(f["data"].attrs["env_args"])
    env_name = env_args["env_name"]
    spec = task_spec(env_name)
    return spec["mg_name"], spec["env_name"]


def _encode_episode_video(video_path: Path, frames: np.ndarray, fps: int) -> tuple[int, int]:
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        str(video_path),
        fps=fps,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
    ) as writer:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)
    height, width = frames.shape[1:3]
    return int(height), int(width)


def convert(source_hdf5: Path, output_root: Path, *, fps: int, task_name: str | None) -> None:
    mg_task_name, env_task_name = _resolve_task_name(source_hdf5, task_name)
    output_root.mkdir(parents=True, exist_ok=True)
    data_dir = output_root / "data" / "chunk-000"
    meta_dir = output_root / "meta"
    videos_root = output_root / "videos" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = []
    state_batches: list[np.ndarray] = []
    action_batches: list[np.ndarray] = []
    video_shape: tuple[int, int] | None = None

    with h5py.File(source_hdf5, "r") as f:
        demos = sorted([key for key in f["data"].keys() if key != "mask"], key=lambda key: int(key.split("_")[-1]))

        for episode_index, demo_key in enumerate(demos):
            grp = f["data"][demo_key]
            obs = grp["obs"]
            actions = grp["actions"][:].astype(np.float32)
            num_steps = actions.shape[0]

            state_parts = []
            for hdf5_key, _state_key, dim in STATE_KEY_MAP:
                arr = obs[hdf5_key][:].astype(np.float32)
                if arr.shape[1] != dim:
                    raise ValueError(f"{demo_key}:{hdf5_key} expected dim {dim}, got {arr.shape[1]}")
                state_parts.append(arr)
            states = np.concatenate(state_parts, axis=1)
            if states.shape[1] != STATE_DIM:
                raise ValueError(f"{demo_key}: expected state dim {STATE_DIM}, got {states.shape[1]}")
            if actions.shape[1] != ACTION_DIM:
                raise ValueError(f"{demo_key}: expected action dim {ACTION_DIM}, got {actions.shape[1]}")

            frame_dict: dict[str, np.ndarray] = {}
            for raw_camera, lerobot_camera in DEFAULT_CAMERAS:
                obs_key = f"{raw_camera}_image"
                if obs_key not in obs:
                    raise KeyError(
                        f"{demo_key} is missing {obs_key}. "
                        "Run MimicGen with obs.collect_obs=true and camera_names configured."
                    )
                frames = obs[obs_key][:]
                frame_dict[lerobot_camera] = frames
                current_shape = (int(frames.shape[1]), int(frames.shape[2]))
                video_shape = current_shape if video_shape is None else video_shape
                if video_shape != current_shape:
                    raise ValueError(f"Inconsistent video shapes: {video_shape} vs {current_shape}")

            df = pd.DataFrame(
                {
                    "observation.state": list(states),
                    "action": list(actions),
                    "next.reward": np.zeros(num_steps, dtype=np.float32),
                    "next.done": np.array([False] * (num_steps - 1) + [True]),
                    "timestamp": np.arange(num_steps, dtype=np.float32) / fps,
                    "frame_index": np.arange(num_steps, dtype=np.int64),
                    "episode_index": np.full(num_steps, episode_index, dtype=np.int64),
                    "index": np.arange(num_steps, dtype=np.int64),
                    "task_index": np.zeros(num_steps, dtype=np.int64),
                    "annotation.human.task_description": np.zeros(num_steps, dtype=np.int64),
                    "annotation.human.task_name": np.zeros(num_steps, dtype=np.int64),
                }
            )
            df.to_parquet(data_dir / f"episode_{episode_index:06d}.parquet", index=False)

            for lerobot_camera, frames in frame_dict.items():
                video_path = (
                    videos_root
                    / f"observation.images.{lerobot_camera}"
                    / f"episode_{episode_index:06d}.mp4"
                )
                _encode_episode_video(video_path, frames, fps)

            episode_rows.append(
                {
                    "episode_index": episode_index,
                    "tasks": [env_task_name],
                    "length": num_steps,
                }
            )
            state_batches.append(states)
            action_batches.append(actions)

    if video_shape is None:
        raise RuntimeError(f"No episodes were converted from {source_hdf5}")

    total_frames = int(sum(item["length"] for item in episode_rows))
    total_videos = len(episode_rows) * len(DEFAULT_CAMERAS)
    all_states = np.concatenate(state_batches, axis=0)
    all_actions = np.concatenate(action_batches, axis=0)

    write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": env_task_name}])
    write_json(
        meta_dir / "stats.json",
        {
            "observation.state": compute_vector_stats(all_states),
            "action": compute_vector_stats(all_actions),
        },
    )
    write_json(meta_dir / "relative_stats.json", {})
    write_json(meta_dir / "modality.json", build_native_modality_meta())

    height, width = video_shape
    features = {
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM]},
        "action": {"dtype": "float32", "shape": [ACTION_DIM]},
    }
    for _, lerobot_camera in DEFAULT_CAMERAS:
        features[f"observation.images.{lerobot_camera}"] = make_video_feature(height, width, fps)

    write_json(
        meta_dir / "info.json",
        {
            "codebase_version": "v2.1",
            "robot_type": "PandaOmron",
            "total_episodes": len(episode_rows),
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_videos,
            "total_chunks": 1,
            "chunks_size": len(episode_rows),
            "fps": fps,
            "splits": {"train": f"0:{len(episode_rows)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        },
    )

    print(f"Created {output_root}")
    print(f"  task: {mg_task_name} / {env_task_name}")
    print(f"  episodes: {len(episode_rows)}")
    print(f"  frames:   {total_frames}")
    print(f"  videos:   {total_videos} ({width}x{height})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MimicGen demo.hdf5 to a RoboCasa-native LeRobot dataset with direct videos."
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Input MimicGen demo.hdf5 or merged all_successes.hdf5")
    parser.add_argument("--output", type=Path, required=True, help="Output LeRobot dataset root")
    parser.add_argument("--task", type=str, default=None, help="Optional canonical task name override")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--hf-repo-id", type=str, default=None)
    args = parser.parse_args()

    convert(args.hdf5.resolve(), args.output.resolve(), fps=args.fps, task_name=args.task)
    if args.hf_repo_id:
        upload_folder_to_hub(
            args.output.resolve(),
            args.hf_repo_id,
            commit_message=f"Upload MimicGen LeRobot dataset for {args.hdf5.stem}",
        )
        print(f"Pushed to hf.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
