from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICGEN_ROOT = Path(
    os.environ.get("RLDX_MIMICGEN_ROOT", str(Path.home() / "mimicgen"))
).expanduser()
ROBOCASA_ROOT = Path(
    os.environ.get(
        "RLDX_ROBOCASA_ROOT",
        str(REPO_ROOT / "external_dependencies" / "robocasa"),
    )
).expanduser()
ROBOMIMIC_ROOT = Path(
    os.environ.get("RLDX_ROBOMIMIC_ROOT", str(MIMICGEN_ROOT / "external" / "robomimic"))
).expanduser()
DEFAULT_MIMICGEN_PYTHON = Path(
    os.environ.get(
        "RLDX_MIMICGEN_PYTHON",
        str(Path.home() / "miniconda3" / "envs" / "mimicgen310" / "bin" / "python"),
    )
).expanduser()
DEFAULT_RLDX_PYTHON = Path(
    os.environ.get(
        "RLDX_RLDX_PYTHON",
        str(Path.home() / "miniconda3" / "envs" / "rldx" / "bin" / "python"),
    )
).expanduser()
DEFAULT_LEROBOT_PYTHON = Path(
    os.environ.get(
        "RLDX_LEROBOT_PYTHON",
        str(Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "python"),
    )
).expanduser()
DEFAULT_DATASETS_ROOT = Path.home() / "datasets"
DEFAULT_CAMERAS = [
    ("robot0_agentview_left", "left_view"),
    ("robot0_agentview_right", "right_view"),
    ("robot0_eye_in_hand", "wrist_view"),
]
DEFAULT_CAMERA_NAMES = [raw_key for raw_key, _ in DEFAULT_CAMERAS]

# RoboCasa-native state keys expected by the sim wrapper / modality config.
STATE_KEY_MAP: list[tuple[str, str, int]] = [
    ("robot0_base_pos", "base_position", 3),
    ("robot0_base_quat", "base_rotation", 4),
    ("robot0_eef_pos", "end_effector_position_absolute", 3),
    ("robot0_base_to_eef_pos", "end_effector_position_relative", 3),
    ("robot0_eef_quat", "end_effector_rotation_absolute", 4),
    ("robot0_base_to_eef_quat", "end_effector_rotation_relative", 4),
    ("robot0_gripper_qpos", "gripper_qpos", 2),
    ("robot0_gripper_qvel", "gripper_qvel", 2),
    ("robot0_joint_pos", "joint_position", 7),
    ("robot0_joint_pos_cos", "joint_position_cos", 7),
    ("robot0_joint_pos_sin", "joint_position_sin", 7),
    ("robot0_joint_vel", "joint_velocity", 7),
]
ACTION_KEY_MAP: list[tuple[str, int]] = [
    ("end_effector_position", 3),
    ("end_effector_rotation", 3),
    ("gripper_close", 1),
    ("base_motion", 4),
    ("control_mode", 1),
]
STATE_DIM = sum(dim for _, _, dim in STATE_KEY_MAP)
ACTION_DIM = sum(dim for _, dim in ACTION_KEY_MAP)

TASK_SPECS = {
    "TurnSinkSpout": {
        "env_name": "TurnSinkSpout",
        "mg_name": "TurnSinkSpout",
        "env_interface": "MG_TurnSinkSpout",
    },
    "TurnOnStove": {
        "env_name": "TurnOnStove",
        "mg_name": "TurnOnStove",
        "env_interface": "MG_TurnOnStove",
    },
    "TurnOnSinkFaucet": {
        "env_name": "TurnOnSinkFaucet",
        "mg_name": "TurnOnSinkFaucet",
        "env_interface": "MG_TurnOnSinkFaucet",
    },
    "TurnOnMicrowave": {
        "env_name": "TurnOnMicrowave",
        "mg_name": "TurnOnMicrowave",
        "env_interface": "MG_TurnOnMicrowave",
    },
    "TurnOffStove": {
        "env_name": "TurnOffStove",
        "mg_name": "TurnOffStove",
        "env_interface": "MG_TurnOffStove",
    },
    "TurnOffSinkFaucet": {
        "env_name": "TurnOffSinkFaucet",
        "mg_name": "TurnOffSinkFaucet",
        "env_interface": "MG_TurnOffSinkFaucet",
    },
    "TurnOffMicrowave": {
        "env_name": "TurnOffMicrowave",
        "mg_name": "TurnOffMicrowave",
        "env_interface": "MG_TurnOffMicrowave",
    },
    "PnPStoveToCounter": {
        "env_name": "PickPlaceStoveToCounter",
        "mg_name": "PnPStoveToCounter",
        "env_interface": "MG_PnPStoveToCounter",
    },
    "PnPSinkToCounter": {
        "env_name": "PickPlaceSinkToCounter",
        "mg_name": "PnPSinkToCounter",
        "env_interface": "MG_PnPSinkToCounter",
    },
    "PnPMicrowaveToCounter": {
        "env_name": "PickPlaceMicrowaveToCounter",
        "mg_name": "PnPMicrowaveToCounter",
        "env_interface": "MG_PnPMicrowaveToCounter",
    },
    "PnPCounterToStove": {
        "env_name": "PickPlaceCounterToStove",
        "mg_name": "PnPCounterToStove",
        "env_interface": "MG_PnPCounterToStove",
    },
    "PnPCounterToSink": {
        "env_name": "PickPlaceCounterToSink",
        "mg_name": "PnPCounterToSink",
        "env_interface": "MG_PnPCounterToSink",
    },
    "PnPCounterToMicrowave": {
        "env_name": "PickPlaceCounterToMicrowave",
        "mg_name": "PnPCounterToMicrowave",
        "env_interface": "MG_PnPCounterToMicrowave",
    },
    "PnPCounterToCab": {
        "env_name": "PickPlaceCounterToCabinet",
        "mg_name": "PnPCounterToCab",
        "env_interface": "MG_PnPCounterToCab",
    },
    "PnPCabToCounter": {
        "env_name": "PickPlaceCabinetToCounter",
        "mg_name": "PnPCabToCounter",
        "env_interface": "MG_PnPCabToCounter",
    },
    "OpenSingleDoor": {
        "env_name": "OpenSingleDoor",
        "mg_name": "OpenSingleDoor",
        "env_interface": "MG_OpenSingleDoor",
    },
    "OpenDrawer": {
        "env_name": "OpenDrawer",
        "mg_name": "OpenDrawer",
        "env_interface": "MG_OpenDrawer",
    },
    "OpenDoubleDoor": {
        "env_name": "OpenDoubleDoor",
        "mg_name": "OpenDoubleDoor",
        "env_interface": "MG_OpenDoubleDoor",
    },
    "CoffeeSetupMug": {
        "env_name": "CoffeeSetupMug",
        "mg_name": "CoffeeSetupMug",
        "env_interface": "MG_CoffeeSetupMug",
    },
    "CoffeeServeMug": {
        "env_name": "CoffeeServeMug",
        "mg_name": "CoffeeServeMug",
        "env_interface": "MG_CoffeeServeMug",
    },
    "CoffeePressButton": {
        "env_name": "CoffeePressButton",
        "mg_name": "CoffeePressButton",
        "env_interface": "MG_CoffeePressButton",
    },
    "CloseSingleDoor": {
        "env_name": "CloseSingleDoor",
        "mg_name": "CloseSingleDoor",
        "env_interface": "MG_CloseSingleDoor",
    },
    "CloseDrawer": {
        "env_name": "CloseDrawer",
        "mg_name": "CloseDrawer",
        "env_interface": "MG_CloseDrawer",
    },
    "CloseDoubleDoor": {
        "env_name": "CloseDoubleDoor",
        "mg_name": "CloseDoubleDoor",
        "env_interface": "MG_CloseDoubleDoor",
    },
}
TASK_ALIASES = {name.lower(): name for name in TASK_SPECS}
TASK_ALIASES.update({spec["env_name"].lower(): name for name, spec in TASK_SPECS.items()})


def ensure_runtime_library_paths() -> None:
    runtime_paths = [
        Path.home() / ".mujoco" / "mujoco210" / "bin",
        Path("/usr/lib/nvidia"),
        Path("/usr/lib/x86_64-linux-gnu/nvidia/current"),
    ]
    current = [entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":") if entry]
    for path in runtime_paths:
        if path.exists() and str(path) not in current:
            current.append(str(path))
    if current:
        os.environ["LD_LIBRARY_PATH"] = ":".join(current)


def build_subprocess_env(python_bin: Path) -> dict[str, str]:
    ensure_runtime_library_paths()
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(MIMICGEN_ROOT),
            str(ROBOCASA_ROOT),
            str(ROBOMIMIC_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    ).rstrip(":")

    python_path = python_bin.resolve()
    conda_prefix = python_path.parent.parent
    if (conda_prefix / "include").exists():
        env["CPATH"] = f"{conda_prefix / 'include'}:{env.get('CPATH', '')}".rstrip(":")
    if (conda_prefix / "lib").exists():
        lib_prefix = str(conda_prefix / "lib")
        env["LIBRARY_PATH"] = f"{lib_prefix}:{env.get('LIBRARY_PATH', '')}".rstrip(":")
        env["LD_LIBRARY_PATH"] = f"{lib_prefix}:{env.get('LD_LIBRARY_PATH', '')}".rstrip(":")
    env["PATH"] = f"{python_path.parent}:{env.get('PATH', '')}"
    return env


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None, env=env)


def read_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_json_maybe_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value)
    return value


def canonical_task_name(name: str) -> str:
    task = TASK_ALIASES.get(name.lower())
    if task is None:
        valid = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"Unknown task '{name}'. Valid tasks: {valid}")
    return task


def task_spec(name: str) -> dict[str, str]:
    return TASK_SPECS[canonical_task_name(name)]


def default_repo_name(task_name: str, suffix: str) -> str:
    task_slug = canonical_task_name(task_name).replace("PnP", "pnp-")
    task_slug = (
        task_slug.replace("To", "-to-")
        .replace("Counter", "counter")
        .replace("Stove", "stove")
        .replace("Sink", "sink")
        .replace("Microwave", "microwave")
        .replace("Cab", "cab")
    )
    task_slug = task_slug.lower()
    return f"robocasa-{task_slug}-{suffix}"


def compute_vector_stats(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def build_native_modality_meta() -> dict[str, Any]:
    state_meta: dict[str, dict[str, int]] = {}
    action_meta: dict[str, dict[str, int]] = {}
    offset = 0
    for _, key, dim in STATE_KEY_MAP:
        state_meta[key] = {"start": offset, "end": offset + dim}
        offset += dim
    offset = 0
    for key, dim in ACTION_KEY_MAP:
        action_meta[key] = {"start": offset, "end": offset + dim}
        offset += dim
    return {
        "state": state_meta,
        "action": action_meta,
        "video": {
            out_key: {"original_key": f"observation.images.{out_key}"}
            for _, out_key in DEFAULT_CAMERAS
        },
        "annotation": {
            "human.task_description": {},
            "human.task_name": {},
        },
    }


def make_video_feature(height: int, width: int, fps: int) -> dict[str, Any]:
    return {
        "dtype": "video",
        "shape": [height, width, 3],
        "names": ["height", "width", "channel"],
        "video_info": {
            "video.fps": float(fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        },
    }


def upload_folder_to_hub(local_dir: Path, repo_id: str, *, commit_message: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(local_dir),
        commit_message=commit_message,
    )


def copy_episode_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def list_episode_parquets(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "data").rglob("episode_*.parquet"))


def rebuild_lerobot_subset(
    input_root: Path,
    output_root: Path,
    *,
    max_episodes: int | None = None,
    copy_extras: bool = True,
    copy_videos: bool = True,
) -> list[int]:
    parquet_files = list_episode_parquets(input_root)
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {input_root / 'data'}")

    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]

    output_root.mkdir(parents=True, exist_ok=True)
    dst_data = output_root / "data" / "chunk-000"
    dst_data.mkdir(parents=True, exist_ok=True)

    src_episodes = read_jsonl(input_root / "meta" / "episodes.jsonl")
    episode_meta_map = {int(item["episode_index"]): item for item in src_episodes}
    selected_old_indices: list[int] = []
    total_frames = 0
    state_arrays: list[np.ndarray] = []
    action_arrays: list[np.ndarray] = []

    for new_idx, parquet_path in enumerate(parquet_files):
        old_idx = int(parquet_path.stem.split("_")[-1])
        selected_old_indices.append(old_idx)
        df = pd.read_parquet(parquet_path)
        total_frames += len(df)

        df = df.copy()
        if "episode_index" in df.columns:
            df["episode_index"] = np.int64(new_idx)
        if "index" in df.columns:
            df["index"] = np.arange(len(df), dtype=np.int64)
        if "frame_index" in df.columns:
            df["frame_index"] = np.arange(len(df), dtype=np.int64)
        df.to_parquet(dst_data / f"episode_{new_idx:06d}.parquet", index=False)

        state_arrays.append(np.stack(df["observation.state"].map(np.asarray).values).astype(np.float32))
        action_arrays.append(np.stack(df["action"].map(np.asarray).values).astype(np.float32))

        if copy_extras:
            src_extras = input_root / "extras" / f"episode_{old_idx:06d}"
            if src_extras.exists():
                copy_episode_tree(src_extras, output_root / "extras" / f"episode_{new_idx:06d}")

    if copy_videos and (input_root / "videos").exists():
        for src_video in (input_root / "videos").rglob("episode_*.mp4"):
            old_idx = int(src_video.stem.split("_")[-1])
            if old_idx not in selected_old_indices:
                continue
            new_idx = selected_old_indices.index(old_idx)
            rel = src_video.relative_to(input_root / "videos")
            dst_dir = output_root / "videos" / rel.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_video, dst_dir / f"episode_{new_idx:06d}.mp4")

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    new_episodes = []
    for new_idx, old_idx in enumerate(selected_old_indices):
        original = episode_meta_map.get(old_idx, {})
        entry = dict(original)
        entry["episode_index"] = new_idx
        if "length" not in entry:
            entry["length"] = int(len(pd.read_parquet(dst_data / f"episode_{new_idx:06d}.parquet")))
        new_episodes.append(entry)
    write_jsonl(meta_dir / "episodes.jsonl", new_episodes)

    if (input_root / "meta" / "tasks.jsonl").exists():
        shutil.copy2(input_root / "meta" / "tasks.jsonl", meta_dir / "tasks.jsonl")

    for name in ["modality.json", "relative_stats.json", "embodiment.json"]:
        src = input_root / "meta" / name
        if src.exists():
            shutil.copy2(src, meta_dir / name)

    stats = {
        "observation.state": compute_vector_stats(np.concatenate(state_arrays, axis=0)),
        "action": compute_vector_stats(np.concatenate(action_arrays, axis=0)),
    }
    write_json(meta_dir / "stats.json", stats)

    info = read_json(input_root / "meta" / "info.json")
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    info["total_chunks"] = 1
    info["chunks_size"] = max(1, len(new_episodes))
    info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    if (output_root / "videos").exists():
        num_videos = len(list((output_root / "videos").rglob("episode_*.mp4")))
        info["total_videos"] = num_videos
        info["video_path"] = (
            "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
        )
    write_json(meta_dir / "info.json", info)
    return selected_old_indices


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def infer_task_from_lerobot_root(lerobot_root: Path) -> str:
    task_name = lerobot_root.parent.parent.name
    return canonical_task_name(task_name)


def build_env_args(template_hdf5: Path | None, env_name: str) -> str:
    if template_hdf5 is None:
        env_args = {
            "type": 1,
            "env_name": env_name,
            "env_version": "1.0.1",
            "robosuite_version": "1.5.2",
            "mujoco_version": "3.3.1",
            "env_kwargs": {
                "env_name": env_name,
                "robots": "PandaOmron",
            },
        }
        return json.dumps(env_args)

    with h5py.File(template_hdf5, "r") as f:
        env_args = load_json_maybe_bytes(f["data"].attrs["env_args"])
    env_args["env_name"] = env_name
    env_args.setdefault("env_kwargs", {})["env_name"] = env_name
    return json.dumps(env_args)


def convert_lerobot_to_source_hdf5(
    lerobot_root: Path,
    output_hdf5: Path,
    *,
    env_name: str,
    template_hdf5: Path | None = None,
    max_episodes: int | None = None,
) -> None:
    parquet_files = list_episode_parquets(lerobot_root)
    if max_episodes is not None:
        parquet_files = parquet_files[:max_episodes]
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files found under {lerobot_root / 'data'}")

    env_args_str = build_env_args(template_hdf5, env_name)
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with h5py.File(output_hdf5, "w") as f:
        data_group = f.create_group("data")
        data_group.attrs["env"] = env_name
        data_group.attrs["env_args"] = env_args_str
        for episode_index, parquet_path in enumerate(parquet_files):
            stem = parquet_path.stem
            extras_dir = lerobot_root / "extras" / stem
            if not extras_dir.exists():
                raise FileNotFoundError(f"Missing extras directory for {stem}: {extras_dir}")

            df = pd.read_parquet(parquet_path)
            actions = np.stack(df["action"].to_numpy()).astype(np.float32)
            states = np.load(extras_dir / "states.npz")["states"]
            with open(extras_dir / "ep_meta.json") as fp:
                ep_meta = json.dumps(json.load(fp))
            with gzip.open(extras_dir / "model.xml.gz", "rt") as fp:
                model_xml = fp.read()

            demo_group = data_group.create_group(f"demo_{episode_index}")
            demo_group.create_dataset("actions", data=actions)
            demo_group.create_dataset("states", data=states)
            demo_group.attrs["ep_meta"] = ep_meta
            demo_group.attrs["model_file"] = model_xml
            demo_group.attrs["num_samples"] = int(len(actions))
            total += int(len(actions))
        data_group.attrs["total"] = total


def build_mimicgen_config(
    *,
    mg_name: str,
    source_hdf5: Path,
    output_root: Path,
    num_trials: int,
    max_num_failures: int,
    env_name: str,
    env_interface: str,
    camera_names: list[str],
    camera_height: int,
    camera_width: int,
    seed: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(MIMICGEN_ROOT))
    from mimicgen.configs import config_factory

    base_cfg = config_factory(mg_name, config_type="robosuite")
    subtask_names = list(base_cfg.task.task_spec.keys())
    return {
        "name": mg_name,
        "type": "robosuite",
        "experiment": {
            "name": "demo",
            "seed": seed,
            "render_video": False,
            "num_demo_to_render": 0,
            "num_fail_demo_to_render": 0,
            "max_num_failures": max_num_failures,
            "source": {
                "dataset_path": str(source_hdf5),
                "filter_key": None,
                "start": 0,
                "n": None,
            },
            "generation": {
                "path": str(output_root),
                "guarantee": True,
                "keep_failed": True,
                "num_trials": num_trials,
                "select_src_per_subtask": False,
                "transform_first_robot_pose": True,
            },
            "task": {
                "name": env_name,
                "robot": "PandaOmron",
                "gripper": None,
                "interface": env_interface,
                "interface_type": "robosuite",
            },
        },
        "task": {"task_spec": {subtask_name: {} for subtask_name in subtask_names}},
        "obs": {
            "collect_obs": True,
            "camera_names": camera_names,
            "camera_height": camera_height,
            "camera_width": camera_width,
        },
    }


def patch_generation_env_args(
    source_hdf5: Path,
    output_hdf5: Path,
    *,
    env_name: str,
    layout_id: int | None,
    style_id: int | None,
) -> None:
    shutil.copy2(source_hdf5, output_hdf5)
    with h5py.File(output_hdf5, "r+") as f:
        env_args = load_json_maybe_bytes(f["data"].attrs["env_args"])
        env_args["env_name"] = env_name
        env_kwargs = env_args.setdefault("env_kwargs", {})
        env_kwargs["env_name"] = env_name
        if layout_id is not None and style_id is not None:
            env_kwargs["layout_and_style_ids"] = [[layout_id, style_id]]
            env_kwargs["layout_ids"] = None
            env_kwargs["style_ids"] = None
        f["data"].attrs["env_args"] = json.dumps(env_args)


def add_datagen_info(
    input_hdf5: Path,
    output_hdf5: Path,
    *,
    env_interface: str,
    python_bin: Path,
) -> None:
    env = build_subprocess_env(python_bin)
    cmd = [
        str(python_bin),
        str(REPO_ROOT / "analysis" / "add_datagen_info.py"),
        "--dataset",
        str(input_hdf5),
        "--output",
        str(output_hdf5),
        "--env-interface",
        env_interface,
    ]
    run(cmd, cwd=REPO_ROOT, env=env)
