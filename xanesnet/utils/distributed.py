"""Utilities for coordinating XANESNET application state across DDP processes."""

import os
from pathlib import Path

XANESNET_DDP_RUN_DIR = "XANESNET_DDP_RUN_DIR"
XANESNET_DDP_REUSE_DATASET = "XANESNET_DDP_REUSE_DATASET"


def is_ddp_child() -> bool:
    """Return whether the current process is a XANESNET DDP child."""
    return os.environ.get(XANESNET_DDP_RUN_DIR) is not None


def set_ddp_run_state(save_dir: Path) -> None:
    """Make the prepared run directory available to DDP child processes."""
    os.environ[XANESNET_DDP_RUN_DIR] = str(save_dir)
    os.environ[XANESNET_DDP_REUSE_DATASET] = "1"


def get_ddp_run_dir() -> Path | None:
    """Return the parent run directory when running as a DDP child."""
    value = os.environ.get(XANESNET_DDP_RUN_DIR)
    return Path(value) if value else None


def should_reuse_dataset() -> bool:
    """Return whether the DDP child should reuse prepared dataset files."""
    return os.environ.get(XANESNET_DDP_REUSE_DATASET) == "1"
