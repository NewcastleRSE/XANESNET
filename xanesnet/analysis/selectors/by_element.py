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

"""Selector that keeps predictions for configured target elements."""

from collections.abc import Iterator
from typing import cast

from pymatgen.core import Element, Molecule, Structure
from tqdm import tqdm

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import (
    PredictionReader,
    PredictionSample,
)
from xanesnet.utils.exceptions import ConfigError

from .base import Selector
from .registry import SelectorRegistry


@SelectorRegistry.register("element")
class ElementSelector(Selector):
    """Select predictions whose target site has one of the configured elements.

    Requires:
        Matched raw structures: provided by a structure-matched prediction reader.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader to select samples from.
        elements: Chemical symbols of target-site elements to keep.

    Raises:
        ConfigError: If the reader is not structure-matched or an element
            symbol is invalid.
    """

    def __init__(self, selector_type: str, data_source: PredictionReader, elements: list[str]) -> None:
        """Initialize a target-element selector and collect matching indices.

        Raises:
            ConfigError: If ``data_source`` does not attach matched raw
                structures or an element symbol is invalid.
        """
        super().__init__(selector_type, data_source)

        if not data_source.provides_structures():
            raise ConfigError("ElementSelector requires predictions matched to raw structures.")

        self.elements = elements
        try:
            self._atomic_numbers = {Element(element).Z for element in elements}
        except ValueError as exc:
            raise ConfigError(f"Invalid element symbol in ElementSelector: {exc}") from exc

        self._selected_indices: list[int] = []
        for index, sample in tqdm(enumerate(self.data_source), total=len(self.data_source), desc="Getting indices"):
            target_site_index = cast(int, sample["target_site_index"])
            structure = cast(Molecule | Structure, sample.get("structure"))
            atomic_number = structure.atomic_numbers[target_site_index]
            if atomic_number in self._atomic_numbers:
                self._selected_indices.append(index)

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield samples whose target-site element is configured.

        Returns:
            Iterator over selected prediction samples.
        """
        for index in self._selected_indices:
            yield self.data_source[index]

    def __len__(self) -> int:
        """Return the number of selected samples.

        Returns:
            Number of selected prediction samples.
        """
        return len(self._selected_indices)

    @property
    def signature(self) -> Config:
        """Return the selector signature.

        Returns:
            Configuration values needed to recreate this selector.
        """
        signature = super().signature
        signature.update_with_dict({"elements": self.elements})
        return signature

    def __str__(self) -> str:
        """Return the short display label of this selector."""
        return f"{', '.join(self.elements)}"
