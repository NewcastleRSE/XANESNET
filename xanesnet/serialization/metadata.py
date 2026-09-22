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

"""Run metadata collection and serialization helpers."""

import os
import platform
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
import torch_geometric

from xanesnet import __version__

UNKNOWN = "unknown"


def write_run_metadata(
    save_dir: str | Path,
    *,
    mode: str,
    command_line_args: Sequence[str] | None = None,
) -> tuple[Path, Path]:
    """Write software and hardware metadata files for a run.

    Args:
        save_dir: Root directory of the run.
        mode: Workflow that created the run directory.
        command_line_args: Raw arguments passed to the workflow entry point.
            If omitted, the resulting software metadata records
            ``"unknown"``.

    Returns:
        Paths to the written ``software.info`` and ``hardware.info`` files.

    Raises:
        FileNotFoundError: If ``save_dir`` does not exist or is not a directory.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {save_dir}")

    software_info = get_software_info(mode, command_line_args)
    hardware_info = get_hardware_info()

    software_info_path = _write_info_file(save_dir / "software.info", software_info)
    hardware_info_path = _write_info_file(save_dir / "hardware.info", hardware_info)
    return software_info_path, hardware_info_path


def get_software_info(mode: str, command_line_args: Sequence[str] | None = None) -> dict[str, str]:
    """Collect software and execution information for a run.

    Args:
        mode: Workflow that is being executed.
        command_line_args: Raw arguments passed to the workflow entry point.
            If omitted, the command line is recorded as ``"unknown"``.

    Returns:
        A mapping containing XANESNET, Git, Python, PyTorch, PyTorch Geometric,
        CUDA runtime, and command-line information. Unavailable values are
        reported as ``"unknown"``.
    """
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    torch_geometric_version = getattr(torch_geometric, "__version__", None)
    command_line = shlex.join(command_line_args) if command_line_args else UNKNOWN

    software_info = {
        "package": "XANESNET",
        "version": __version__,
        "workflow": mode,
        "python_version": platform.python_version() or UNKNOWN,
        "pytorch_version": torch.__version__ or UNKNOWN,
        "torch_geometric_version": str(torch_geometric_version) if torch_geometric_version else UNKNOWN,
        "cuda_runtime_version": str(cuda_version) if cuda_version else UNKNOWN,
        "command_line_args": command_line,
    }
    software_info.update(_get_git_info())
    return software_info


def get_hardware_info() -> dict[str, str]:
    """Collect hardware information for a run.

    Returns:
        A mapping containing operating system, kernel, platform, CPU, memory,
        CUDA availability, and GPU information. Unavailable values are
        reported as ``"unknown"``.
    """
    cpu_count = os.cpu_count()
    info = {
        "os": platform.system() or UNKNOWN,
        "kernel": platform.release() or UNKNOWN,
        "platform": platform.platform() or UNKNOWN,
        "architecture": platform.machine() or UNKNOWN,
        "cpu_model": _get_cpu_model(),
        "cpu_count": str(cpu_count) if cpu_count is not None else UNKNOWN,
        "ram_bytes": _get_system_memory(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", UNKNOWN),
    }
    info.update(_get_cuda_hardware_info())
    return info


def _get_cpu_model() -> str:
    """Return the CPU model from platform information or Linux procfs.

    Returns:
        The CPU model, or ``"unknown"`` when it cannot be determined.
    """
    processor = platform.processor().strip()
    if processor:
        return processor

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return UNKNOWN

    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().lower() in {"model name", "hardware", "processor"}:
            model = value.strip()
            if model:
                return model

    return UNKNOWN


def _get_system_memory() -> str:
    """Return the total physical memory in bytes.

    Returns:
        Total physical memory as a string, or ``"unknown"`` when the platform
        does not provide the required system values.
    """
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return UNKNOWN

    return str(page_size * page_count)


def _get_cuda_hardware_info() -> dict[str, str]:
    """Collect CUDA device properties without making CUDA mandatory.

    Returns:
        A mapping containing CUDA availability, GPU count, names, memory
        capacity, and compute capability. Values are reported as ``"unknown"``
        if CUDA information cannot be collected.
    """
    try:
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        info = {
            "cuda_available": "true" if cuda_available else "false",
            "gpu_count": str(device_count),
        }
        for index in range(device_count):
            properties = torch.cuda.get_device_properties(index)
            info[f"gpu_{index}_name"] = str(properties.name)
            info[f"gpu_{index}_memory_bytes"] = str(properties.total_memory)
            info[f"gpu_{index}_compute_capability"] = f"{properties.major}.{properties.minor}"
        return info
    except Exception:
        return {
            "cuda_available": UNKNOWN,
            "gpu_count": UNKNOWN,
        }


def _write_info_file(path: Path, values: Mapping[str, str]) -> Path:
    """Write a mapping as a readable key-value info file.

    Args:
        path: Destination path for the info file.
        values: String key-value pairs to serialize.

    Returns:
        The destination path.
    """
    contents = "".join(f"{key} = {value}\n" for key, value in values.items())
    path.write_text(contents, encoding="utf-8")
    return path


def _get_git_info() -> dict[str, str]:
    """Return Git revision information when the source is in a repository.

    Returns:
        A mapping containing the Git commit, branch, and dirty state. If Git
        information cannot be collected, unavailable values are reported as
        ``"unknown"``.
    """
    repository_dir = Path(__file__).resolve().parents[2]
    try:
        commit = _run_git(repository_dir, "rev-parse", "HEAD")
        branch = _run_git(repository_dir, "branch", "--show-current") or "detached"
        status = _run_git(repository_dir, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {
            "git_commit": UNKNOWN,
            "git_branch": UNKNOWN,
            "git_dirty": UNKNOWN,
        }

    return {
        "git_commit": commit or UNKNOWN,
        "git_branch": branch,
        "git_dirty": "true" if status else "false",
    }


def _run_git(repository_dir: Path, *arguments: str) -> str:
    """Run a Git command and return its stripped standard output.

    Args:
        repository_dir: Directory in which to run the Git command.
        arguments: Arguments passed to the Git executable.

    Returns:
        The command's standard output without leading or trailing whitespace.
    """
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_dir,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return result.stdout.strip()
