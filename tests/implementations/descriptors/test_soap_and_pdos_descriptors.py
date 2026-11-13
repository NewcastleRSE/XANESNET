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

"""Unit tests for xanesnet's own handling in the ``soap`` and ``pdos`` descriptors.

The numerical output of DScribe, tblite and pySCF is not tested here.
"""

import numpy as np
import pytest
from ase import Atoms

import xanesnet.descriptors.pdos as pdos_module
import xanesnet.descriptors.soap as soap_module
from xanesnet.descriptors import DescriptorRegistry
from xanesnet.descriptors.pdos import atoms_to_pyscf, spectrum


def soap(**overrides):
    kwargs = {
        "r_cut": 3.0,
        "n_max": 2,
        "l_max": 1,
        "sigma": 0.5,
        "species": [1, 8],
        "average": "off",
        "compression_mode": "off",
        "compression_species_weighting": None,
        "use_charge": False,
        "use_spin": False,
    }
    kwargs.update(overrides)
    return DescriptorRegistry.create("soap", descriptor_type="soap", **kwargs)


def pdos(**overrides):
    kwargs = {
        "code": "xtb",
        "method": "GFN2-xTB",
        "e_min": -10.0,
        "e_max": 10.0,
        "sigma": 1.0,
        "orb_type": "p",
        "quad_orb_type": "d",
        "num_points": 5,
        "basis": "sto-3g",
        "init_guess": "minao",
        "max_cycles": 10,
        "use_charge": False,
        "use_spin": False,
        "use_quad": False,
        "use_occupied": True,
        "verbosity": 0,
    }
    kwargs.update(overrides)
    return DescriptorRegistry.create("pdos", descriptor_type="pdos", **kwargs)


def water() -> Atoms:
    return Atoms("OHH", positions=[[0.0, 0.0, 0.1], [0.0, 0.76, -0.47], [0.0, -0.76, -0.47]])


###############################################################################
#################################### SOAP #####################################
###############################################################################


def test_soap_returns_one_row_per_requested_site():
    descriptor = soap()

    everything = descriptor.transform(water(), site_index=None)

    assert everything.shape[0] == 3
    np.testing.assert_allclose(descriptor.transform(water(), site_index=1), everything[[1]])
    np.testing.assert_allclose(descriptor.transform(water(), site_index=[2, 0]), everything[[2, 0]])


def test_soap_appends_spin_then_charge_with_zero_defaults():
    system = water()
    plain = soap().transform(system, site_index=[0, 1])

    default_extras = soap(use_spin=True, use_charge=True).transform(system, site_index=[0, 1])
    system.info.update(S=2, q=-1)
    explicit_extras = soap(use_spin=True, use_charge=True).transform(system, site_index=[0, 1])

    assert default_extras.shape[1] == plain.shape[1] + 2
    np.testing.assert_allclose(default_extras[:, :-2], plain)
    assert default_extras[:, -2:].tolist() == [[0.0, 0.0]] * 2
    assert explicit_extras[:, -2:].tolist() == [[2.0, -1.0]] * 2


@pytest.fixture
def dscribe_calls(monkeypatch):
    """Record how xanesnet configures DScribe's SOAP."""
    calls: list[dict] = []
    original = soap_module.DscribeSOAP

    def recording(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(soap_module, "DscribeSOAP", recording)
    return calls


def test_soap_builds_one_calculator_per_boundary_condition_and_reuses_it(dscribe_calls):
    descriptor = soap()
    molecule = water()
    crystal = Atoms("H", positions=[[0, 0, 0]], cell=[3, 3, 3], pbc=True)

    descriptor.transform(molecule)
    descriptor.transform(molecule)
    descriptor.transform(crystal)
    descriptor.transform(crystal)

    assert [call["periodic"] for call in dscribe_calls] == [False, True]


def test_soap_passes_its_settings_and_covers_every_element_by_default(monkeypatch):
    calls: list[dict] = []

    class StubSOAP:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def create(self, system):
            return np.zeros((len(system), 4))

    monkeypatch.setattr(soap_module, "DscribeSOAP", StubSOAP)

    soap(species=None, compression_mode="mu1nu1", compression_species_weighting={"1": 2.0}).transform(water())

    (call,) = calls
    assert call["species"] == list(range(1, 104))
    assert (call["r_cut"], call["n_max"], call["l_max"], call["sigma"]) == (3.0, 2, 1, 0.5)
    assert call["compression"] == {"mode": "mu1nu1", "species_weighting": {"1": 2.0}}


###############################################################################
#################################### PDOS #####################################
###############################################################################


def test_pdos_cannot_describe_all_sites_at_once():
    with pytest.raises(NotImplementedError, match="does not support computing descriptors for all sites"):
        pdos().transform(water(), site_index=None)


def test_pdos_rejects_unknown_codes():
    with pytest.raises(ValueError, match="Unknown code: orca"):
        pdos(code="orca").transform(water(), site_index=0)


def test_pdos_without_tblite_reports_the_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(pdos_module, "Calculator", None)

    with pytest.raises(ImportError, match="tblite"):
        pdos().transform(water(), site_index=0)


@pytest.mark.parametrize("options", [{"use_spin": True}, {"use_charge": True}])
def test_charge_and_spin_must_be_used_together(monkeypatch, options):
    monkeypatch.setattr(pdos_module, "Calculator", object)

    with pytest.raises(NotImplementedError, match="Both should be included simultaneously"):
        pdos(**options).transform(water(), site_index=0)


@pytest.mark.parametrize(("charge", "spin"), [(0, 1), (1, 0)])
def test_the_spin_must_match_the_electron_parity(monkeypatch, charge, spin):
    monkeypatch.setattr(pdos_module, "Calculator", object)
    system = water()  # 10 electrons
    system.info.update(q=charge, s=spin)

    with pytest.raises(ValueError, match="inconsistent with the spin state"):
        pdos(use_spin=True, use_charge=True).transform(system, site_index=0)


def test_gaussian_broadening_sums_the_weighted_levels():
    x = np.array([0.0, 1.0])

    values = spectrum(np.array([0.0, 2.0]), np.array([2.0, 3.0]), sigma=1.0, x=x)

    np.testing.assert_allclose(
        values,
        [2.0 + 3.0 * np.exp(-4.0), 2.0 * np.exp(-1.0) + 3.0 * np.exp(-1.0)],
    )


def test_atoms_are_converted_to_the_pyscf_format():
    expected = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.5))]
    assert atoms_to_pyscf(Atoms("OH", positions=[[0, 0, 0], [0, 0, 1.5]])) == expected
