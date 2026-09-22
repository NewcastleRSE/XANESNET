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

"""Cached structure descriptor embedding shared by selectors and collectors."""

from typing import Any, cast

import numpy as np
from pymatgen.core import Molecule, Structure
from tqdm import tqdm

from xanesnet.descriptors import DescriptorRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

# Descriptor vectors are cached per (descriptor config, sample id, target site).
_DESCRIPTOR_CACHE: dict[tuple[Any, ...], np.ndarray] = {}

# Descriptor instances are cached per (descriptor config,).
_DESCRIPTOR_OBJ_CACHE: dict[str, Any] = {}


def descriptor_vector(descriptor: Config, sample: PredictionSample) -> np.ndarray:
    """Return the cached descriptor vector for one sample's structure and site.

    Args:
        descriptor: Descriptor configuration used to embed the structure.
        sample: Prediction sample carrying a matched raw ``structure``.

    Returns:
        The raveled descriptor vector of the sample's structure, averaged over
        the leading dimension when the descriptor returns a matrix.
    """
    config_key = repr(sorted(descriptor.as_dict().items()))
    if config_key not in _DESCRIPTOR_OBJ_CACHE:
        _DESCRIPTOR_OBJ_CACHE[config_key] = DescriptorRegistry.create(
            descriptor.get_str("descriptor_type"), **descriptor.as_kwargs()
        )
    descriptor_obj = _DESCRIPTOR_OBJ_CACHE[config_key]
    structure = cast(Molecule | Structure, sample.get("structure"))
    site_index = sample.get("target_site_index")
    key = (config_key, sample["sample_id"], site_index)
    if key not in _DESCRIPTOR_CACHE:
        vector = np.asarray(descriptor_obj.transform_pmg(structure, site_index=site_index), dtype=float)
        if vector.ndim == 2:
            vector = vector.mean(axis=0)
        _DESCRIPTOR_CACHE[key] = vector.ravel()
    return _DESCRIPTOR_CACHE[key]


def descriptor_matrix(descriptor: Config, data_source: PredictionReader) -> np.ndarray:
    """Return stacked descriptor vectors for every sample of one reader.

    Args:
        descriptor: Descriptor configuration used to embed each structure.
        data_source: Prediction reader whose samples provide the structures.

    Returns:
        Stacked descriptor matrix with one row per sample, computed under a
        progress bar and cached per sample.
    """
    return np.stack(
        [
            descriptor_vector(descriptor, sample)
            for sample in tqdm(data_source, desc="Computing descriptors", total=len(data_source))
        ]
    )
