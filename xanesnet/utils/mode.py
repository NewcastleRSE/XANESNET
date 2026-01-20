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
    FORWARD = ["forward"]
    REVERSE = ["reverse"]
    BIDIRECTIONAL = ["bidirectional"]

    def __str__(self) -> str:
        return self.value[0]


def get_mode(mode_str: str | None) -> Mode | None:
    if mode_str is None:
        return None
    mode_str = mode_str.lower()
    for mode in Mode:
        if mode_str in mode.value:
            return mode
    raise ValueError(f"'{mode_str}' is not a valid mode.")
