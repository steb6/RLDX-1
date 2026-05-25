from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_CAMERA_NAMES,
    DEFAULT_DATASETS_ROOT,
    DEFAULT_MIMICGEN_PYTHON,
    MIMICGEN_ROOT,
    add_datagen_info,
    build_mimicgen_config,
    build_subprocess_env,
    canonical_task_name,
    convert_lerobot_to_source_hdf5,
    patch_generation_env_args,
    run,
    task_spec,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and optionally run a RoboCasa MimicGen job with image observations."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--lerobot-root", type=Path, help="Input RoboCasa LeRobot task dataset")
    source_group.add_argument("--source-hdf5", type=Path, help="Input robomimic source HDF5")
    parser.add_argument("--task", type=str, required=True, help="Canonical RoboCasa task name")
    parser.add_argument("--layout", type=int, required=True, help="Target layout id")
    parser.add_argument("--style", type=int, required=True, help="Target style id")
    parser.add_argument("--num-episodes", type=int, required=True, help="Target number of generated successes")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_DATASETS_ROOT / "robocasa_mimicgen_runs",
        help="Root folder for the generated run directory",
    )
    parser.add_argument("--max-source-episodes", type=int, default=None)
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--camera-names", nargs="+", default=DEFAULT_CAMERA_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-failures", type=int, default=500)
    parser.add_argument("--run", action="store_true", help="Launch MimicGen immediately")
    parser.add_argument("--mimicgen-python", type=Path, default=DEFAULT_MIMICGEN_PYTHON)
    parser.add_argument("--template-hdf5", type=Path, default=None)
    args = parser.parse_args()

    task_name = canonical_task_name(args.task)
    spec = task_spec(task_name)
    run_dir = args.output_root / f"{task_name}_layout{args.layout}_style{args.style}"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.source_hdf5 is not None:
        source_hdf5 = args.source_hdf5.resolve()
    else:
        source_hdf5 = run_dir / "source_from_lerobot.hdf5"
        convert_lerobot_to_source_hdf5(
            args.lerobot_root.resolve(),
            source_hdf5,
            env_name=spec["env_name"],
            template_hdf5=args.template_hdf5,
            max_episodes=args.max_source_episodes,
        )

    prepared_hdf5 = run_dir / "source_prepared.hdf5"
    generation_hdf5 = run_dir / "source_generation.hdf5"
    config_path = run_dir / "mg_config.json"

    add_datagen_info(
        source_hdf5,
        prepared_hdf5,
        env_interface=spec["env_interface"],
        python_bin=args.mimicgen_python,
    )
    patch_generation_env_args(
        prepared_hdf5,
        generation_hdf5,
        env_name=spec["env_name"],
        layout_id=args.layout,
        style_id=args.style,
    )
    config = build_mimicgen_config(
        mg_name=spec["mg_name"],
        source_hdf5=generation_hdf5,
        output_root=run_dir,
        num_trials=args.num_episodes,
        max_num_failures=args.max_num_failures,
        env_name=spec["env_name"],
        env_interface=spec["env_interface"],
        camera_names=args.camera_names,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        seed=args.seed,
    )
    write_json(config_path, config)

    print(f"Prepared MimicGen run in {run_dir}")
    print(f"  source:   {source_hdf5}")
    print(f"  prepared: {prepared_hdf5}")
    print(f"  env:      {generation_hdf5}")
    print(f"  config:   {config_path}")

    if args.run:
        run(
            [
                str(args.mimicgen_python),
                str(Path(__file__).resolve().parents[1] / "run_mimicgen_generate_dataset.py"),
                "--config",
                str(config_path),
            ],
            cwd=MIMICGEN_ROOT,
            env=build_subprocess_env(args.mimicgen_python),
        )
    else:
        print("Next step:")
        print(
            f"  {args.mimicgen_python} "
            f"{Path(__file__).resolve().parents[1] / 'run_mimicgen_generate_dataset.py'} "
            f"--config {config_path}"
        )


if __name__ == "__main__":
    main()
