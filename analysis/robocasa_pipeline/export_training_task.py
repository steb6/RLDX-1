from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    DEFAULT_DATASETS_ROOT,
    canonical_task_name,
    default_repo_name,
    rebuild_lerobot_subset,
    upload_folder_to_hub,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage a single-task RoboCasa LeRobot dataset and optionally push it to HF."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input LeRobot task dataset root")
    parser.add_argument("--task", type=str, required=True, help="Canonical RoboCasa task name")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output LeRobot dataset root (default: ~/datasets/<derived-name>)",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optionally keep only the first N episodes and renumber them.",
    )
    parser.add_argument("--skip-extras", action="store_true", help="Do not copy extras/")
    parser.add_argument("--skip-videos", action="store_true", help="Do not copy videos/")
    parser.add_argument("--hf-repo-id", type=str, default=None, help="Optional HF dataset repo to upload")
    args = parser.parse_args()

    task_name = canonical_task_name(args.task)
    default_output = DEFAULT_DATASETS_ROOT / default_repo_name(task_name, "train")
    output_root = args.output or default_output

    selected = rebuild_lerobot_subset(
        args.input.resolve(),
        output_root.resolve(),
        max_episodes=args.max_episodes,
        copy_extras=not args.skip_extras,
        copy_videos=not args.skip_videos,
    )

    print(f"Created {output_root}")
    print(f"Task: {task_name}")
    print(f"Episodes copied: {len(selected)}")

    if args.hf_repo_id:
        upload_folder_to_hub(
            output_root,
            args.hf_repo_id,
            commit_message=f"Upload staged RoboCasa training dataset for {task_name}",
        )
        print(f"Pushed to hf.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()
