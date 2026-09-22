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

"""Parametrized dry-run tests for paired train and inference workflows."""

import logging
from pathlib import Path

import pytest

from xanesnet import infer as infer_cli
from xanesnet import train as train_cli

from .conftest import cleanup_processed_data, collect_model_pairs, find_checkpoint

MODEL_PAIRS = collect_model_pairs()
MODEL_IDS = [train_path.stem for train_path, _ in MODEL_PAIRS]


def _run_train(train_path: Path, out_dir: Path) -> Path:
    """Run training and return the created run directory.

    Args:
        train_path: Path to a training configuration.
        out_dir: Parent directory for the training output.

    Returns:
        The created training run directory.
    """
    train_cli.main(
        [
            "-i",
            str(train_path),
            "-o",
            str(out_dir),
            "-n",
            "test",
            "--yes",
        ]
    )
    run_dirs = sorted(out_dir.glob("train_test_*"))
    assert run_dirs, f"No training run directory created under {out_dir}"
    return run_dirs[-1]


def _run_infer(infer_path: Path, checkpoint_path: Path, out_dir: Path) -> Path:
    """Run inference and return the created run directory.

    Args:
        infer_path: Path to an inference configuration.
        checkpoint_path: Path to the trained deployment checkpoint.
        out_dir: Parent directory for the inference output.

    Returns:
        The created inference run directory.
    """
    infer_cli.main(
        [
            "-i",
            str(infer_path),
            "-m",
            str(checkpoint_path),
            "-o",
            str(out_dir),
            "-n",
            "test",
            "--yes",
        ]
    )
    run_dirs = sorted(out_dir.glob("infer_test_*"))
    assert run_dirs, f"No inference run directory created under {out_dir}"
    return run_dirs[-1]


@pytest.mark.slow
@pytest.mark.parametrize("train_path, infer_path", MODEL_PAIRS, ids=MODEL_IDS)
def test_train_and_infer(train_path: Path, infer_path: Path, tmp_path: Path) -> None:
    """Train each configured model and run inference from its checkpoint.

    Args:
        train_path: Path to the paired training configuration.
        infer_path: Path to the paired inference configuration.
        tmp_path: Pytest temporary directory for run outputs.
    """
    logging.info("Train and infer dry run: %s", train_path.stem)

    try:
        train_run_dir = _run_train(train_path, tmp_path / "train")
        checkpoint_path = find_checkpoint(train_run_dir)
        infer_run_dir = _run_infer(infer_path, checkpoint_path, tmp_path / "infer")

        predictions_dir = infer_run_dir / "predictions"
        assert predictions_dir.is_dir()
        assert (predictions_dir / "predictions.h5").exists()
        assert (predictions_dir / "WRITER_INFO.txt").exists()
    finally:
        cleanup_processed_data(train_path)
