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

"""Filesystem utilities for XANESNET (directory listing, run-directory creation, file I/O)."""

import shutil
from datetime import datetime
from pathlib import Path

###############################################################################
################################### HELPERS ###################################
###############################################################################


def list_files(path: Path, with_ext: bool = True, suffixes: str | tuple[str, ...] | None = None) -> list[Path]:
    """List files in a directory, optionally filtered by extension.

    Hidden files (names starting with ``'.'``) are always excluded.

    Args:
        path: Directory to list.
        with_ext: If ``True`` (default), return paths with their file extension.
            If ``False``, return paths with the extension stripped.
        suffixes: If given, only files whose suffix is in ``suffixes`` are
            returned. A single string or a tuple of strings is accepted.

    Returns:
        List of ``Path`` objects for each matching non-hidden file in ``path``.
    """
    if isinstance(suffixes, str):
        suffixes = (suffixes,)

    return [
        (f if with_ext else f.with_suffix(""))
        for f in path.iterdir()
        if f.is_file() and not f.stem.startswith(".") and (suffixes is None or f.suffix in suffixes)
    ]


def list_filestems(d: Path, suffixes: str | tuple[str, ...] | None = None) -> list[str]:
    """List file stems (names without extension) in a directory.

    Hidden files (names starting with ``'.'``) are always excluded.

    Args:
        d: Directory to list.
        suffixes: If given, only files whose suffix is in ``suffixes`` are
            returned.

    Returns:
        List of file-stem strings for each non-hidden file in ``d``.
    """
    return [f.stem for f in list_files(d, suffixes=suffixes)]


def list_subdirs(path: Path) -> list[Path]:
    """List immediate subdirectories of a directory.

    Hidden directories (names starting with ``'.'``) are always excluded.

    Args:
        path: Directory to list.

    Returns:
        List of ``Path`` objects for each non-hidden subdirectory in ``path``.
    """
    return [d for d in path.iterdir() if d.is_dir() and not d.name.startswith(".")]


def list_subdir_stems(path: Path) -> list[str]:
    """List the names of immediate subdirectories of a directory.

    Hidden directories (names starting with ``'.'``) are always excluded.

    Args:
        path: Directory to list.

    Returns:
        Sorted list of directory-name strings for each non-hidden subdirectory.
    """
    return sorted(d.stem for d in path.iterdir() if d.is_dir() and not d.name.startswith("."))


###############################################################################
################################### CREATION ##################################
###############################################################################


def create_run_dir(
    base_dir: str | Path = "./runs",
    name: str | None = None,
    mode: str = "count",
) -> Path:
    """Create a uniquely named run directory.

    Two naming modes:
    - ``"count (default)"``:
        ``<name>_<N>`` where *N* is the highest existing integer suffix
        for ``<name>_`` in ``base_dir``, incremented by 1.

    - ``"time"``: ``<timestamp>_<name>``

    Args:
        base_dir: Parent directory under which the run directory is created.
            Defaults to ``"./runs"``.
        name: Optional suffix appended to the timestamp, separated by
            ``'_'``.
        mode: Naming mode, either ``"count"`` or ``"time"``.

    Returns:
        Path to the newly created run directory.
    """
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    if mode == "count":  # COUNT
        if name is None:
            name = "run"

        highest: int = -1
        prefix = f"{name}_"
        if base_dir.exists() and base_dir.is_dir():
            for entry in base_dir.iterdir():
                if entry.is_dir() and entry.name.startswith(prefix):
                    suffix = entry.name[len(prefix) :]
                    try:
                        n = int(suffix)
                        if n > highest:
                            highest = n
                    except ValueError:
                        pass

        folder_name = f"{name}_{highest + 1:03d}"

    elif mode == "time":  # TIME
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{timestamp}"
        if name:
            folder_name += f"_{name}"

    else:
        raise ValueError(f"Invalid mode: {mode}")

    run_dir = base_dir / folder_name

    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def create_subfolders(parent_dir: str | Path, subfolder_names: list[str]) -> dict[str, Path]:
    """Create multiple subfolders under an existing directory.

    Args:
        parent_dir: Existing parent directory.
        subfolder_names: Names of the subfolders to create.

    Returns:
        Mapping from subfolder name to its ``Path``.

    Raises:
        FileNotFoundError: If ``parent_dir`` does not exist or is not a
            directory.
    """
    if not isinstance(parent_dir, Path):
        parent_dir = Path(parent_dir)

    if not parent_dir.exists() or not parent_dir.is_dir():
        raise FileNotFoundError(f"Parent directory does not exist: {parent_dir}")

    paths = {}
    for name in subfolder_names:
        subfolder = parent_dir / name
        subfolder.mkdir(exist_ok=True)
        paths[name] = subfolder

    return paths


###############################################################################
#################################### OTHER ####################################
###############################################################################


def copy_file(
    src: str | Path,
    dst_dir: str | Path,
    new_name: str | None = None,
    allowed_suffixes: set[str] | None = None,
) -> Path:
    """Copy a file to a destination directory.

    Args:
        src: Source file path.
        dst_dir: Destination directory.
        new_name: If given, the copied file is renamed to this name.
            Otherwise, the original filename is preserved.
        allowed_suffixes: If given, the copy is rejected when ``src``'s
            suffix is not in this set.

    Returns:
        Path to the copied file in ``dst_dir``.

    Raises:
        FileNotFoundError: If ``src`` does not exist, is not a file, or
            ``dst_dir`` does not exist or is not a directory.
        ValueError: If ``src``'s suffix is not in ``allowed_suffixes``.
    """
    src = Path(src)
    dst_dir = Path(dst_dir)

    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Source file does not exist: {src}")
    if allowed_suffixes and src.suffix not in allowed_suffixes:
        raise ValueError(f"File suffix not allowed: {src.suffix}")
    if not dst_dir.exists() or not dst_dir.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {dst_dir}")

    filename = new_name if new_name else src.name
    dst_file = dst_dir / filename

    shutil.copy(src, dst_file)
    return dst_file
