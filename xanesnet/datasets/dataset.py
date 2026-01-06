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

from xanesnet.datasources import DataSource
from xanesnet.utils.mode import Mode


class Dataset(ABC):
    """
    Abstract base class for datasets.
    All dataset classes should inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        type: str,
        datasource: DataSource,
        root: str,
        mode: Mode,
        params: dict,
    ):
        self.type = type
        self.datasource = datasource
        self.root = root
        self.mode = mode
        self.params = params

        # TODO check if already processed

    @abstractmethod
    def process(self):
        """
        Process the raw data and prepare it for use in the model.
        This method should be implemented by all subclasses.
        """
        pass
