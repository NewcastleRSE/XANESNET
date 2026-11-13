# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Fixtures local to shared dataset-base tests."""

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
from pymatgen.core import Molecule

from xanesnet.datasources import DataSource


def water(sample_id: str) -> Molecule:
    """Create a small water molecule carrying a sample identifier.

    Args:
        sample_id: Identifier to store in molecule properties.

    Returns:
        A three-site water molecule.
    """
    molecule = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1], [0.0, 0.76, -0.47], [0.1, -0.76, -0.47]],
        properties={"sample_id": sample_id},
    )
    return molecule


class ListSource(DataSource):
    """In-memory data source returning molecules in list order.

    Args:
        molecules: Molecules exposed by the data source.
    """

    def __init__(self, molecules: list[Molecule]) -> None:
        super().__init__("list")
        self.molecules = molecules

    def __iter__(self) -> Iterator[Molecule]:
        return iter(self.molecules)

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Molecule:
        return self.molecules[idx]


@pytest.fixture(autouse=True)
def confirm(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace dataset confirmation prompts with a controllable response.

    Args:
        monkeypatch: Pytest fixture used to replace the confirmation function.

    Returns:
        Mutable state containing the response and prompt count.
    """
    state = SimpleNamespace(answer=True, asked=0)

    def fake_confirm(message: str, default_yes: bool = True) -> bool:
        state.asked += 1
        return state.answer

    monkeypatch.setattr("xanesnet.datasets.base.base.confirm_yes_no", fake_confirm)
    return state
