"""
Add datagen_info to a robosuite HDF5, making it usable as MimicGen source.
Thin wrapper around MimicGen's official prepare_src_dataset.py.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIMICGEN_ROOT = Path(
    os.environ.get("RLDX_MIMICGEN_ROOT", str(Path.home() / "mimicgen"))
).expanduser()
ROBOCASA_ROOT = Path(
    os.environ.get(
        "RLDX_ROBOCASA_ROOT",
        str(REPO_ROOT / "external_dependencies" / "robocasa"),
    )
).expanduser()
ROBOMIMIC = Path(
    os.environ.get("RLDX_ROBOMIMIC_ROOT", str(MIMICGEN_ROOT / "external" / "robomimic"))
).expanduser()
if str(MIMICGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMICGEN_ROOT))
if str(ROBOCASA_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOCASA_ROOT))
if str(ROBOMIMIC) not in sys.path:
    sys.path.insert(0, str(ROBOMIMIC))

from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset


def _pick_compat_geom_name(env, requested_name: str) -> str:
    geom_names = [str(name) for name in env.sim.model.geom_names]
    geom_name_set = set(geom_names)
    if requested_name in geom_name_set:
        return requested_name

    suffix_aliases = {
        "_bottom": ["_bottom", "_bottom_visual", "_g0", "_main"],
        "_tray": ["_tray", "_tray_visual", "_g0", "_main"],
        "_handle": ["_handle_main", "_handle_g0", "_g0", "_main"],
    }
    for suffix, replacements in suffix_aliases.items():
        if not requested_name.endswith(suffix):
            continue
        base = requested_name[: -len(suffix)]
        for replacement in replacements:
            candidate = f"{base}{replacement}"
            if candidate in geom_name_set:
                return candidate
        prefix_matches = sorted(name for name in geom_names if name.startswith(base))
        if prefix_matches:
            return prefix_matches[0]

    stem = requested_name
    while "_" in stem:
        stem = stem.rsplit("_", 1)[0]
        prefix_matches = sorted(name for name in geom_names if name.startswith(stem))
        if prefix_matches:
            return prefix_matches[0]

    raise ValueError(
        f'No compatible geom found for "{requested_name}". '
        f"Available prefix sample: {geom_names[:20]}"
    )


def install_geom_name_compat_patch() -> None:
    from mimicgen.env_interfaces.robosuite import RobosuiteInterface

    original_get_object_pose = RobosuiteInterface.get_object_pose

    def patched_get_object_pose(self, obj_name, obj_type):
        if obj_type != "geom":
            return original_get_object_pose(self, obj_name, obj_type)
        try:
            return original_get_object_pose(self, obj_name, obj_type)
        except ValueError:
            compat_name = _pick_compat_geom_name(self.env, obj_name)
            return original_get_object_pose(self, compat_name, obj_type)

    RobosuiteInterface.get_object_pose = patched_get_object_pose


def main():
    parser = argparse.ArgumentParser(
        description="Add datagen_info to source HDF5 using MimicGen's official pipeline"
    )
    parser.add_argument("--dataset", required=True, help="Input HDF5 with states+actions")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--env-interface", required=True, 
                        help="Env interface name, e.g. MG_PnPCounterToCab")
    parser.add_argument("--env-interface-type", default="robosuite",
                        help="Env interface type (default: robosuite)")
    parser.add_argument("--filter-key", default=None)
    parser.add_argument("--n", type=int, default=None, help="Max demos to process")
    args = parser.parse_args()

    install_geom_name_compat_patch()

    prepare_src_dataset(
        dataset_path=args.dataset,
        env_interface_name=args.env_interface,
        env_interface_type=args.env_interface_type,
        filter_key=args.filter_key,
        n=args.n,
        output_path=args.output,
    )
    print(f"\n✅ Done. Output: {args.output}")


if __name__ == "__main__":
    main()
