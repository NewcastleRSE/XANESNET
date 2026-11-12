# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Shared fixtures and helpers for XANESNET dry-run tests."""

import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

from xanesnet.utils.prompts import set_auto_yes

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[1]
TRAIN_DIR = TESTS_DIR / "train"
INFER_DIR = TESTS_DIR / "infer"
ANALYZE_DIR = TESTS_DIR / "analyze"


@pytest.fixture(autouse=True)
def _run_from_repo_root() -> Iterator[None]:
    """Configs use paths like ``./data/toy_data`` relative to the repository root.

    Dry-runs therefore need that working directory even when pytest was started from ``tests/``.
    """
    previous = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


@pytest.fixture(autouse=True)
def _auto_yes_prompts() -> Iterator[None]:
    """Suppress interactive prompts during dry-runs without leaking into other suites."""
    set_auto_yes(True)
    try:
        yield
    finally:
        set_auto_yes(False)


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
    """Locate the minimal deployment checkpoint produced by a training run.

    Args:
        run_dir: Root directory of a completed training run.

    Returns:
        Path to ``models/final.pth``, which carries the signature and model states.

    Raises:
        FileNotFoundError: If the final checkpoint does not exist.
    """
    final = run_dir / "models" / "final.pth"
    if not final.exists():
        raise FileNotFoundError(f"No final checkpoint found in {run_dir}")
    return final


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
            if not root_path.is_absolute():
                root_path = REPO_ROOT / root_path
            if root_path.exists():
                shutil.rmtree(root_path)
    except Exception:
        pass
