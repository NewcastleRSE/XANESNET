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

"""Fixtures shared by the dataset tests."""

from types import SimpleNamespace

import pytest
from pymatgen.core import Molecule


def spectrum(scale: float = 1.0) -> dict:
    return {"energies": [1.0, 2.0, 3.0], "intensities": [0.1 * scale, 0.5 * scale, 0.2 * scale]}


def water(sample_id: str, spectra: list[dict | None] | None, **properties) -> Molecule:
    """Water molecule with ``spectra`` attached to its sites (``None`` = no spectrum there)."""
    molecule = Molecule(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1], [0.0, 0.76, -0.47], [0.1, -0.76, -0.47]],
        properties={"sample_id": sample_id, **properties},
    )
    if spectra is not None:
        molecule.add_site_property("spectrum", spectra)
    return molecule


class ListSource:
    """Data source backed by a list of in-memory molecules."""

    def __init__(self, molecules: list[Molecule]) -> None:
        self.molecules = molecules

    def __len__(self) -> int:
        return len(self.molecules)

    def __getitem__(self, idx: int) -> Molecule:
        return self.molecules[idx]


@pytest.fixture
def make_source():
    return ListSource


@pytest.fixture(autouse=True)
def confirm(monkeypatch) -> SimpleNamespace:
    """Answer the "clear the processed directory?" prompt without waiting for input."""
    state = SimpleNamespace(answer=True, asked=0)

    def fake_confirm(message: str, default_yes: bool = True) -> bool:
        state.asked += 1
        return state.answer

    monkeypatch.setattr("xanesnet.datasets.base.base.confirm_yes_no", fake_confirm)
    return state
