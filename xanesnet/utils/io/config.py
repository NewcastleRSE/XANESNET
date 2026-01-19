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
from pathlib import Path

import yaml


def copy_yaml(yaml_path, dst_dir, new_name=None):
    """
    Copy a .yaml file from `yaml_path` to `dst_dir`.
    """
    yaml_path = Path(yaml_path)
    dst_dir = Path(dst_dir)

    if not yaml_path.exists():
        raise FileNotFoundError(f"Source file does not exist: {yaml_path}")
    if yaml_path.suffix not in {".yaml", ".yml"}:
        raise ValueError(f"Source file must be a YAML file: {yaml_path}")
    if not dst_dir.exists() or not dst_dir.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {dst_dir}")

    # Use new_name if provided, else keep original name
    filename = new_name if new_name else yaml_path.name
    dst_file = dst_dir / filename

    shutil.copy(yaml_path, dst_file)
    return dst_file


def save_dict_as_yaml(config: dict, dst_dir, name: str):
    """
    Save a dictionary as a YAML file in the destination folder.
    """
    dst_dir = Path(dst_dir)
    if not dst_dir.exists() or not dst_dir.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {dst_dir}")

    if not name.endswith((".yaml", ".yml")):
        name += ".yaml"

    dst_file = dst_dir / name

    with open(dst_file, "w") as f:
        yaml.dump(config, f, sort_keys=False)

    return dst_file
