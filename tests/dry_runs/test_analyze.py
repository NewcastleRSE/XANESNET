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

"""End-to-end dry-run test: train, infer, analyze."""

import logging
from pathlib import Path

import pytest

from xanesnet import analyze as analyze_cli
from xanesnet import infer as infer_cli
from xanesnet import train as train_cli

from .conftest import (
    ANALYZE_DIR,
    INFER_DIR,
    TRAIN_DIR,
    cleanup_processed_data,
    find_checkpoint,
)

PIPELINE_TRAIN = TRAIN_DIR / "test_schnet.yaml"
PIPELINE_INFER = INFER_DIR / "test_schnet.yaml"
PIPELINE_ANALYZE = ANALYZE_DIR / "test_analyze.yaml"


@pytest.mark.slow
def test_full_pipeline(tmp_path: Path) -> None:
    """Run the complete train, infer, analyze workflow.

    Uses the SchNet model pair as a representative pipeline.  All
    intermediate outputs are written under *tmp_path* and cleaned up
    automatically by pytest.

    Args:
        tmp_path: Pytest temporary directory (auto-cleaned).
    """
    logging.info("Full pipeline dry run: SchNet + toy data")

    try:
        # Train
        train_cli.main(
            [
                "-i",
                str(PIPELINE_TRAIN),
                "-o",
                str(tmp_path / "train"),
                "-n",
                "test",
                "--yes",
            ]
        )
        train_run_dirs = sorted((tmp_path / "train").glob("train_test_*"))
        assert train_run_dirs
        train_run_dir = train_run_dirs[-1]
        assert (train_run_dir / "software.info").is_file()
        assert (train_run_dir / "hardware.info").is_file()

        ckpt_path = find_checkpoint(train_run_dir)

        # Infer
        infer_cli.main(
            [
                "-i",
                str(PIPELINE_INFER),
                "-m",
                str(ckpt_path),
                "-o",
                str(tmp_path / "infer"),
                "-n",
                "test",
                "--yes",
            ]
        )
        infer_run_dirs = sorted((tmp_path / "infer").glob("infer_test_*"))
        assert infer_run_dirs
        infer_run_dir = infer_run_dirs[-1]

        predictions_dir = infer_run_dir / "predictions"
        assert predictions_dir.is_dir()
        assert (predictions_dir / "predictions.h5").exists()
        assert (infer_run_dir / "validated_infer_config.yaml").is_file()
        assert (infer_run_dir / "software.info").is_file()
        assert (infer_run_dir / "hardware.info").is_file()

        # Analyze
        analyze_cli.main(
            [
                "-i",
                str(PIPELINE_ANALYZE),
                "-r",
                str(infer_run_dir),
                "--prediction-names",
                "MLP",
                "-o",
                str(tmp_path / "analyze"),
                "-n",
                "test",
                "--yes",
            ]
        )
        analyze_run_dirs = sorted((tmp_path / "analyze").glob("analyze_test_*"))
        assert analyze_run_dirs
        analyze_run_dir = analyze_run_dirs[-1]

        assert (analyze_run_dir / "reports").is_dir()
        assert (analyze_run_dir / "plots").is_dir()
        assert (analyze_run_dir / "software.info").is_file()
        assert (analyze_run_dir / "hardware.info").is_file()
    finally:
        cleanup_processed_data(PIPELINE_TRAIN)
