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

"""Unit tests for the ``xyzspec`` and ``multixyzspec`` data sources (``.xyz`` structures + ``.txt`` spectra)."""

from pathlib import Path

import numpy as np
import pytest

from xanesnet.datasources import DataSourceRegistry
from xanesnet.utils.exceptions import ResourceError

XYZ = """3
water {name}
O 0.0 0.0 0.1
H 0.0 0.7 -0.4
H 0.0 -0.7 -0.4
"""

SPECTRUM = """# header line one
# header line two
{e0} {i0}
{e1} {i1}
{e2} {i2}
"""


def write_sample(xyz_dir: Path, spectra_dir: Path, name: str, scale: float = 1.0) -> None:
    xyz_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)
    (xyz_dir / f"{name}.xyz").write_text(XYZ.format(name=name))
    (spectra_dir / f"{name}.txt").write_text(
        SPECTRUM.format(e0=7100.0, i0=0.1 * scale, e1=7101.0, i1=0.5 * scale, e2=7102.0, i2=0.2 * scale)
    )


###############################################################################
################################### XYZSPEC ###################################
###############################################################################


@pytest.fixture
def flat(tmp_path):
    for name in ("b", "a", "c"):
        write_sample(tmp_path / "xyz", tmp_path / "spectra", name)
    return DataSourceRegistry.create(
        "xyzspec", datasource_type="xyzspec", xyz_path=str(tmp_path / "xyz"), spectra_path=str(tmp_path / "spectra")
    )


def test_samples_are_the_sorted_names_present_in_both_directories(tmp_path):
    write_sample(tmp_path / "xyz", tmp_path / "spectra", "shared")
    (tmp_path / "xyz" / "only_xyz.xyz").write_text(XYZ.format(name="x"))
    (tmp_path / "spectra" / "only_spectrum.txt").write_text(SPECTRUM)

    source = DataSourceRegistry.create(
        "xyzspec", datasource_type="xyzspec", xyz_path=str(tmp_path / "xyz"), spectra_path=str(tmp_path / "spectra")
    )

    assert len(source) == 1
    assert source.sample_ids == ["shared"]


def test_samples_are_ordered_by_name(flat):
    assert flat.sample_ids == ["a", "b", "c"]
    assert [molecule.properties["sample_id"] for molecule in flat] == ["a", "b", "c"]


def test_a_sample_is_the_molecule_with_the_spectrum_attached_to_the_first_site(flat):
    molecule = flat[0]

    assert [site.specie.symbol for site in molecule] == ["O", "H", "H"]
    np.testing.assert_allclose(molecule.cart_coords[1], [0.0, 0.7, -0.4], atol=1e-6)
    assert molecule.properties["comment"] == "water a"
    spectra = molecule.site_properties["spectrum"]
    assert spectra[1] is None and spectra[2] is None
    np.testing.assert_allclose(spectra[0]["energies"], [7100.0, 7101.0, 7102.0])
    np.testing.assert_allclose(spectra[0]["intensities"], [0.1, 0.5, 0.2], rtol=1e-6)
    assert spectra[0]["energies"].dtype == np.float32


def test_iteration_matches_indexing(flat):
    assert [m.properties["sample_id"] for m in flat] == [flat[i].properties["sample_id"] for i in range(len(flat))]


@pytest.mark.parametrize("missing", ["xyz_path", "spectra_path"])
def test_missing_directories_are_reported(tmp_path, missing):
    write_sample(tmp_path / "xyz", tmp_path / "spectra", "a")
    paths = {"xyz_path": str(tmp_path / "xyz"), "spectra_path": str(tmp_path / "spectra")}
    paths[missing] = str(tmp_path / "nowhere")

    with pytest.raises(ResourceError, match="does not exist"):
        DataSourceRegistry.create("xyzspec", datasource_type="xyzspec", **paths)


def test_directories_without_matching_pairs_are_rejected(tmp_path):
    (tmp_path / "xyz").mkdir()
    (tmp_path / "spectra").mkdir()
    (tmp_path / "xyz" / "a.xyz").write_text(XYZ.format(name="a"))

    with pytest.raises(ResourceError, match="No matching"):
        DataSourceRegistry.create(
            "xyzspec", datasource_type="xyzspec", xyz_path=str(tmp_path / "xyz"), spectra_path=str(tmp_path / "spectra")
        )


###############################################################################
################################ MULTIXYZSPEC #################################
###############################################################################


@pytest.fixture
def multi(tmp_path):
    for subdir, names in {"set_b": ["y", "x"], "set_a": ["z"]}.items():
        for name in names:
            write_sample(tmp_path / subdir / "xyz", tmp_path / subdir / "spectra", name)
    return DataSourceRegistry.create("multixyzspec", datasource_type="multixyzspec", root_path=str(tmp_path))


def test_samples_are_grouped_by_subdirectory_with_a_stable_subdirectory_id(multi):
    samples = list(multi)

    assert len(multi) == 3
    assert [(m.properties["subdir_name"], m.properties["sample_id"]) for m in samples] == [
        ("set_a", "z"),
        ("set_b", "x"),
        ("set_b", "y"),
    ]
    assert [m.properties["subdir_id"] for m in samples] == [0, 1, 1]


def test_multi_samples_carry_structure_and_spectrum(multi):
    molecule = multi[0]

    assert len(molecule) == 3
    np.testing.assert_allclose(molecule.site_properties["spectrum"][0]["intensities"], [0.1, 0.5, 0.2], rtol=1e-6)


def test_subdirectories_without_pairs_are_skipped_with_a_warning(tmp_path, caplog):
    write_sample(tmp_path / "good" / "xyz", tmp_path / "good" / "spectra", "a")
    (tmp_path / "empty" / "xyz").mkdir(parents=True)
    (tmp_path / "empty" / "spectra").mkdir(parents=True)

    with caplog.at_level("WARNING"):
        source = DataSourceRegistry.create("multixyzspec", datasource_type="multixyzspec", root_path=str(tmp_path))

    assert list(source.sample_ids) == ["good"]
    assert "empty" in caplog.text


def test_a_subdirectory_must_have_xyz_and_spectra_folders(tmp_path):
    write_sample(tmp_path / "good" / "xyz", tmp_path / "good" / "spectra", "a")
    (tmp_path / "bad").mkdir()

    with pytest.raises(ResourceError, match="must contain 'xyz' and 'spectra'"):
        DataSourceRegistry.create("multixyzspec", datasource_type="multixyzspec", root_path=str(tmp_path))


@pytest.mark.parametrize("layout", ["missing_root", "no_subdirectories", "no_pairs_anywhere"])
def test_unusable_roots_are_rejected(tmp_path, layout):
    root = tmp_path / "root"
    if layout != "missing_root":
        root.mkdir()
    if layout == "no_pairs_anywhere":
        (root / "s" / "xyz").mkdir(parents=True)
        (root / "s" / "spectra").mkdir(parents=True)

    with pytest.raises(ResourceError):
        DataSourceRegistry.create("multixyzspec", datasource_type="multixyzspec", root_path=str(root))
