"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import shutil
from datetime import datetime
from pathlib import Path

###############################################################################
################################### HELPERS ###################################
###############################################################################


def list_files(path: Path, with_ext: bool = True) -> list[Path]:
    # returns a list of files (as POSIX paths) found in a directory (`d`);
    # 'hidden' files are always omitted and, if with_ext == False, file
    # extensions are also omitted

    return [
        (f if with_ext else f.with_suffix("")) for f in path.iterdir() if f.is_file() and not f.stem.startswith(".")
    ]


def list_filestems(d: Path) -> list[str]:
    # returns a list of file stems (as strings) found in a directory (`d`);
    # 'hidden' files are always omitted
    return [f.stem for f in list_files(d)]


###############################################################################
################################### CREATION ##################################
###############################################################################


def create_run_dir(base_dir: str | Path = "./runs", name: str | None = None) -> Path:
    """
    Create a unique run directory under `base_dir` with timestamp.
    Optionally appends a custom `name` to the directory.
    """
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{timestamp}"
    if name:
        folder_name += f"_{name}"

    run_dir = base_dir / folder_name

    # Ensure uniqueness by appending a counter if needed
    counter = 1
    unique_dir = run_dir
    while unique_dir.exists():
        unique_dir = run_dir.with_name(f"{run_dir.name}_{counter}")
        counter += 1

    unique_dir.mkdir(parents=True, exist_ok=False)
    return unique_dir


def create_subfolders(parent_dir: str | Path, subfolder_names: list[str]) -> dict[str, Path]:
    """
    Create subfolders under an existing parent directory.
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
    """
    Copies a file from `src` `dst_dir`.
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
