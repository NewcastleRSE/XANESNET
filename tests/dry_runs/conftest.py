# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   ...

"""Shared fixtures and helpers for XANESNET dry-run tests."""

import shutil
from pathlib import Path

import yaml

from xanesnet.utils.prompts import set_auto_yes

# Suppress interactive prompts during automated testing.
set_auto_yes(True)

TESTS_DIR = Path(__file__).resolve().parent
TRAIN_DIR = TESTS_DIR / "train"
INFER_DIR = TESTS_DIR / "infer"
ANALYZE_DIR = TESTS_DIR / "analyze"


def collect_configs(subdir: str) -> list[Path]:
    """Collect YAML configurations from a dry-run subdirectory.

    Args:
        subdir: Subdirectory name, such as ``"train"`` or ``"infer"``.

    Returns:
        Sorted configuration paths.  An absent subdirectory produces an empty
        list.
    """
    directory = TESTS_DIR / subdir
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.yaml"))


def find_checkpoint(run_dir: Path) -> Path:
    """Locate the deployment checkpoint produced by a training run.

    Returns ``models/final.pth`` when it exists (the canonical deployment
    checkpoint carrying the full signature).  Falls back to the most recent
    epoch checkpoint under ``checkpoints/``.

    Args:
        run_dir: Root directory of a completed training run.

    Returns:
        Path to the checkpoint ``.pth`` file.

    Raises:
        FileNotFoundError: If no checkpoint file can be found.
    """
    final = run_dir / "models" / "final.pth"
    if final.exists():
        return final
    checkpoint_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found in {run_dir}")
    return checkpoints[-1]


def collect_model_pairs() -> list[tuple[Path, Path]]:
    """Collect train/infer configurations with matching stems.

    Returns:
        Sorted ``(train_config, infer_config)`` pairs.
    """
    train_configs = collect_configs("train")
    infer_configs = {path.stem: path for path in collect_configs("infer")}
    return [
        (train_path, infer_configs[train_path.stem]) for train_path in train_configs if train_path.stem in infer_configs
    ]


def cleanup_processed_data(config_path: Path) -> None:
    """Remove the processed data directory referenced by a train config.

    Reads ``dataset.root`` from the YAML file at *config_path* and deletes
    the corresponding directory tree.  Errors during parsing or removal are
    silently ignored so that cleanup never masks a test failure.

    Args:
        config_path: Path to a train-mode YAML config file.
    """
    if not config_path.exists():
        return
    try:
        cfg = yaml.safe_load(config_path.read_text())
        root = cfg.get("dataset", {}).get("root")
        if root:
            root_path = Path(root)
            if root_path.exists():
                shutil.rmtree(root_path)
    except Exception:
        pass
