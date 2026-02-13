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


class BatchProcessor(ABC):
    """
    Base class for batch processors.
    Converts a dataset batch into the correct inputs and targets for a specific model.
    """

    @abstractmethod
    def input_preparation(self, batch: Any) -> dict[str, Any]:
        """
        Prepares the model inputs from a batch.
        """
        ...

    @abstractmethod
    def input_preparation_single(self, sample: Any) -> dict[str, Any]:
        """
        Prepares the model inputs from a single sample.
        """
        ...

    @abstractmethod
    def target_preparation(self, batch: Any) -> Any:
        """
        Prepares the model targets from a batch.
        """
        ...

    @abstractmethod
    def target_preparation_single(self, sample: Any) -> Any:
        """
        Prepares the model targets from a single sample.
        """
        ...

    @abstractmethod
    def sample_id_extraction(self, batch: Any) -> Any:
        """
        Extracts sample IDs from a batch.
        """
        ...
