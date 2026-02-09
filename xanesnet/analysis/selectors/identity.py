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

from typing import Any, Iterable, Iterator

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("all")
@SelectorRegistry.register("none")
class IdentitySelector(Selector):
    """
    Return all samples unchanged (identity selector).
    """

    def __init__(
        self,
        selector_type: str,
        data_source: Iterable[dict[str, Any]],
    ) -> None:
        super().__init__(selector_type, data_source)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.data_source
