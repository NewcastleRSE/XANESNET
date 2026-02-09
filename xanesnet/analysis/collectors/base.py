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

from abc import ABC, abstractmethod
from typing import Any


class Collector(ABC):
    """
    Base class for collectors.
    """

    def __init__(
        self,
        collector_type: str,
    ) -> None:
        self.collector_type = collector_type

    @abstractmethod
    def process(self, sample: dict[str, Any]) -> dict[str, Any]:
        """
        Process a single sample and return values.
        """
        ...
