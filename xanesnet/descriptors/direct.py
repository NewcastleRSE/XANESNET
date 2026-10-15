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

"""Pass-through descriptor that reads pre-computed features from disk."""

from pathlib import Path

import numpy as np
from ase import Atoms

from .base import Descriptor
from .registry import DescriptorRegistry


@DescriptorRegistry.register("direct")
class DIRECT(Descriptor):
    """Reads pre-computed descriptor vectors from ``.txt`` files on disk.

    Expects one ``.txt`` file per structure in ``source_dir``, named
    ``{file_stem}.txt`` where ``file_stem`` is the structure's
    ``info["sample_id"]`` (set by the datasource).  Each file contains
    whitespace-delimited floats with one row per site.

    Args:
        descriptor_type: Identifier string for this descriptor type.
        source_dir: Path to the directory holding ``.txt`` descriptor
            files (absolute or relative to cwd).
        preload: If ``True``, load all files into memory at init time.
    """

    def __init__(
        self,
        descriptor_type: str,
        source_dir: str,
        preload: bool,
    ) -> None:
        super().__init__(descriptor_type)
        self.source_dir = Path(source_dir)

        self._cache: dict[str, np.ndarray] = {}
        if preload:
            for path in sorted(self.source_dir.glob("*.txt")):
                self._cache[path.stem] = np.loadtxt(path)

    def transform(
        self,
        system: Atoms,
        site_index: int | list[int] | None = 0,
    ) -> np.ndarray:
        """Read pre-computed features for one or more sites.

        Args:
            system: The atomic system.  Must carry ``info["sample_id"]``.
            site_index: Site index, list of site indices, or ``None`` for
                all sites.  Defaults to ``0`` (the target site).

        Returns:
            Descriptor array ``(S, F)`` with one row per selected site.

        Raises:
            KeyError: If ``info["sample_id"]`` is missing.
            FileNotFoundError: If the ``.txt`` file does not exist.
        """
        stem = system.info["sample_id"]
        features = self._cache.get(stem)
        if features is None:
            path = self.source_dir / f"{stem}.txt"
            features = np.loadtxt(path)

        features = np.atleast_2d(features)
        if isinstance(site_index, int):
            site_index = [site_index]
        if site_index is not None:
            features = features[site_index, :]
        return features
