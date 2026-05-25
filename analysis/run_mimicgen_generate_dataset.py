"""
Launch MimicGen dataset generation with RoboCasa geom-name compatibility patches.
"""
import os
from pathlib import Path
import runpy
import sys

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
ROBOMIMIC_ROOT = Path(
    os.environ.get("RLDX_ROBOMIMIC_ROOT", str(MIMICGEN_ROOT / "external" / "robomimic"))
).expanduser()

for root in [REPO_ROOT, MIMICGEN_ROOT, ROBOCASA_ROOT, ROBOMIMIC_ROOT]:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

from analysis.add_datagen_info import install_geom_name_compat_patch


if __name__ == "__main__":
    install_geom_name_compat_patch()
    runpy.run_path(
        str(MIMICGEN_ROOT / "mimicgen" / "scripts" / "generate_dataset.py"),
        run_name="__main__",
    )
