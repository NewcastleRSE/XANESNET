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


from enum import Enum


class Mode(Enum):
    XYZ_TO_XANES = ["train_xyz", "predict_xanes"]
    XANES_TO_XYZ = ["train_xanes", "predict_xyz"]
    BIDIRECTIONAL = ["train_all", "predict_all"]


_MODE_LOOKUP = {s: mode for mode in Mode for s in mode.value}


def get_mode(mode_str: str) -> Mode:
    try:
        return _MODE_LOOKUP[mode_str]
    except KeyError:
        raise ValueError(f"'{mode_str}' is not a valid mode.")
