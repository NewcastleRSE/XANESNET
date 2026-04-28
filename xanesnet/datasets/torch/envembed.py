# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either Version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

"""Environment-embedding tensor dataset implementation."""

import logging
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from pymatgen.core import Molecule, Structure
from torch.nn.utils.rnn import pad_sequence

from xanesnet.datasources import DataSource
from xanesnet.descriptors import Descriptor, DescriptorRegistry
from xanesnet.serialization.config import Config
from xanesnet.utils.math import SpectralBasis, gaussian_fit

from ..base import SavePathFn, TorchDataset
from ..registry import DatasetRegistry

SPECTRUM_KEYS = ["XANES", "XANES_K"]  # TODO maybe put this somewhere more central?


@dataclass
class EnvEmbedData:
    """Container for one environment-embedding sample or batch.

    Attributes:
        descriptor_features: Per-site descriptor tensor with shape ``(n_sites, n_features)``
            or ``(batch, max_sites, n_features)``.
        distance_features: Per-site absorber distance tensor with shape ``(n_sites,)`` or
            ``(batch, max_sites)`` in **Angstrom**.
        intensities: Spectrum intensity tensor with shape ``(n_energies,)`` or ``(batch, n_energies)``.
        energies: Energy grid tensor with shape ``(n_energies,)`` or ``(batch, n_energies)``.
        c_star: Gaussian basis coefficient tensor.
        lengths: Original per-sample site counts for padded batches.
        file_name: Source file name metadata for one sample or a batch.
        basis: Spectral basis attached at runtime and excluded from saved state.
    """

    descriptor_features: torch.Tensor | None = None
    distance_features: torch.Tensor | None = None
    intensities: torch.Tensor | None = None
    energies: torch.Tensor | None = None
    c_star: torch.Tensor | None = None
    lengths: torch.Tensor | None = None
    file_name: str | list[Any] | None = None
    basis: SpectralBasis | None = None  # not saved in state dict

    def to(self, device: str | torch.device) -> "EnvEmbedData":
        """Move tensor-like attributes to ``device`` in place.

        Args:
            device: Target device accepted by ``torch.Tensor.to``.

        Returns:
            This data object after moving tensor-like attributes.
        """
        for attr in [
            "descriptor_features",
            "distance_features",
            "intensities",
            "energies",
            "c_star",
            "lengths",
            "basis",
        ]:
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))
        return self

    def to_state_dict(self) -> dict[str, Any]:
        """Serialize this sample to a torch-saveable state dictionary.

        Returns:
            Dictionary containing tensor and metadata fields.
        """
        return {
            "descriptor_features": self.descriptor_features,
            "distance_features": self.distance_features,
            "intensities": self.intensities,
            "energies": self.energies,
            "c_star": self.c_star,
            "lengths": self.lengths,
            "file_name": self.file_name,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "EnvEmbedData":
        """Create data from a state dictionary.

        Args:
            state: State dictionary produced by ``to_state_dict``.

        Returns:
            Reconstructed environment-embedding data object.
        """
        return cls(
            descriptor_features=state.get("descriptor_features"),
            distance_features=state.get("distance_features"),
            intensities=state.get("intensities"),
            energies=state.get("energies"),
            c_star=state.get("c_star"),
            lengths=state.get("lengths"),
            file_name=state.get("file_name"),
            basis=None,
        )

    def save(self, path: str) -> str:
        """Save this data object to disk.

        Args:
            path: Destination ``.pth`` path.

        Returns:
            The destination path.
        """
        torch.save(self.to_state_dict(), path)
        return path

    @classmethod
    def load(cls, path: str) -> "EnvEmbedData":
        """Load environment-embedding data from disk.

        Args:
            path: Source ``.pth`` path.

        Returns:
            Loaded environment-embedding data object.
        """
        state = torch.load(path, weights_only=True)
        return cls.from_state_dict(state)


@DatasetRegistry.register("envembed")
class EnvEmbedDataset(TorchDataset):
    """Dataset that creates absorber-centered environment embeddings."""

    def __init__(
        self,
        dataset_type: str,
        datasource: DataSource,
        root: str,
        preload: bool,
        skip_prepare: bool,
        split_ratios: list[float] | None,
        split_indexfile: str | None,
        # params:
        widths_eV: list[float],
        basis_stride: int,
        basis_path: str | None,
        env_radius: float | None,
        # descriptors
        descriptors: list[Config],
    ) -> None:
        """Initialize the environment-embedding dataset.

        Args:
            dataset_type: Registered dataset type name.
            datasource: Raw datasource of pymatgen structures or molecules.
            root: Directory that stores processed ``.pth`` files.
            preload: Whether to preload processed samples.
            skip_prepare: Whether to reuse existing processed files.
            split_ratios: Optional split ratios.
            split_indexfile: Optional path to split indices.
            widths_eV: Gaussian basis widths in **eV**.
            basis_stride: Energy-grid stride used when creating a Gaussian basis.
            basis_path: Optional serialized spectral basis path.
            env_radius: Periodic-neighbor cutoff in **Angstrom** for structures.
            descriptors: Descriptor configuration objects.
        """
        super().__init__(dataset_type, datasource, root, preload, skip_prepare, split_ratios, split_indexfile)

        self.widths_eV = widths_eV
        self.basis_stride = basis_stride
        self.basis_path = basis_path
        self.env_radius = env_radius

        # Create descriptors
        self.descriptor_configs = descriptors
        self.descriptor_list: list[Descriptor] = []
        descriptor_types = ", ".join(d.get_str("descriptor_type") for d in descriptors)
        logging.info(f"Initialising descriptors: {descriptor_types}")
        for descriptor_config in descriptors:
            descriptor_type = descriptor_config.get_str("descriptor_type")
            descriptor = DescriptorRegistry.get(descriptor_type)(**descriptor_config.as_kwargs())
            self.descriptor_list.append(descriptor)

        # Setup spectral basis
        self.basis: SpectralBasis | None = None
        self._setup_spectral_basis()

    def _prepare_single(self, idx: int, save_path_fn: SavePathFn) -> int:
        """Process one datasource item into environment-embedding samples.

        Args:
            idx: Datasource index to process.
            save_path_fn: Callback that maps per-item sample sequence numbers to output paths.

        Returns:
            Number of processed absorber samples written.
        """
        assert self.basis is not None, "Spectral basis must be set up successfully."

        pmg_obj = self.datasource[idx]
        for key in SPECTRUM_KEYS:
            if key in pmg_obj.site_properties.keys():
                break
        else:
            logging.warning(f"No XANES spectrum found for sample {idx} ({pmg_obj.properties['file_name']}); skipping.")
            return 0

        xanes = np.array(pmg_obj.site_properties[key], dtype=object)
        xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()  # noqa: E711

        # Compute descriptor features (all sites for env embedding)
        descriptor_features_list = []
        for descriptor in self.descriptor_list:
            feature = descriptor.transform_pmg(pmg_obj, site_index=None)
            descriptor_features_list.append(feature)
        descriptor_features_np = np.concatenate(descriptor_features_list, axis=1)
        descriptor_features = torch.tensor(descriptor_features_np, dtype=torch.float32)

        seq = 0
        for site_idx in xanes_idxs:
            # XANES
            spectrum = pmg_obj.site_properties[key][site_idx]
            energies = torch.tensor(spectrum["energies"], dtype=torch.float32)
            intensities = torch.tensor(spectrum["intensities"], dtype=torch.float32)

            # Build per-site environment: descriptor features + distance features
            # For periodic structures with env_radius, this finds all neighbors
            # (incl. periodic images) within the radius. For molecules, unchanged.
            site_descs, site_dists = self._build_site_environment(
                pmg_obj, absorber_idx=site_idx, all_descriptors=descriptor_features
            )

            # Gaussian
            c_star = gaussian_fit(basis=self.basis, xanes=intensities)

            # Create Data object
            data = EnvEmbedData(
                descriptor_features=site_descs,
                distance_features=site_dists,
                intensities=intensities,
                energies=energies,
                c_star=c_star,
                file_name=pmg_obj.properties["file_name"],
                basis=self.basis,
            )

            # Save processed data
            data.save(save_path_fn(seq))
            seq += 1

        return seq

    def _setup_spectral_basis(self) -> None:
        """Load or create the spectral basis used by Gaussian target features."""
        if self.basis_path is not None:
            # TODO never tested this.
            self.basis = torch.load(self.basis_path)  # TODO: still uses torch.load without weights_only=True
            logging.info(f"Loaded spectral basis from file @ {self.basis_path}")
        else:
            logging.info("Creating spectral basis from datasource")
            first_data = next(iter(self.datasource))
            # TODO requires same energy grid for all samples!
            for key in SPECTRUM_KEYS:
                if key in first_data.site_properties.keys():
                    break
            else:
                raise ValueError("No XANES spectrum found in datasource to set up spectral basis.")

            xanes = np.array(first_data.site_properties[key], dtype=object)
            xanes_idxs: list[int] = np.where(xanes != None)[0].tolist()
            energies = torch.tensor(first_data.site_properties[key][xanes_idxs[0]]["energies"], dtype=torch.float32)
            self.basis = SpectralBasis(
                energies=energies,
                widths_eV=self.widths_eV,
                normalize_atoms=True,
                stride=self.basis_stride,
            )

    def _build_site_environment(
        self,
        pmg_obj: Molecule | Structure,
        absorber_idx: int,
        all_descriptors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build descriptor and distance features for one absorber environment.

        For periodic structures with ``env_radius`` set, this uses
        ``Structure.get_neighbors`` to find atoms, including periodic images,
        within the configured radius in **Angstrom**. Returned descriptors and
        distances place the absorber at index 0.

        For molecules, or when ``env_radius`` is ``None``, the method returns
        the original all-atom descriptors and distances unchanged.

        Args:
            pmg_obj: Structure or molecule that contains the absorber.
            absorber_idx: Index of the absorbing atom.
            all_descriptors: Descriptor tensor for all sites with shape ``(n_sites, n_features)``.

        Returns:
            Pair of descriptor features and distances for the selected environment.
        """
        if self.env_radius is not None and isinstance(pmg_obj, Structure):
            neighbors = pmg_obj.get_neighbors(pmg_obj[absorber_idx], r=self.env_radius)

            # Absorber at index 0
            absorber_desc = all_descriptors[absorber_idx].unsqueeze(0)  # (1, H)
            absorber_dist = torch.zeros(1, dtype=torch.float32)

            if len(neighbors) > 0:
                # Sort by distance for deterministic ordering
                neighbors.sort(key=lambda n: n.nn_distance)
                neighbor_indices = [n.index for n in neighbors]
                neighbor_dists = torch.tensor([n.nn_distance for n in neighbors], dtype=torch.float32)
                neighbor_descs = all_descriptors[neighbor_indices]  # (N_neighbors, H)

                desc = torch.cat([absorber_desc, neighbor_descs], dim=0)
                dist = torch.cat([absorber_dist, neighbor_dists], dim=0)
            else:
                desc = absorber_desc
                dist = absorber_dist

            return desc, dist
        else:
            # Molecule or no env_radius: preserve original behavior
            return all_descriptors, self._distances_to_absorber(pmg_obj, absorber_idx=absorber_idx)

    @staticmethod
    def _distances_to_absorber(data: Molecule | Structure, absorber_idx: int) -> torch.Tensor:
        """Compute all-site distances to one absorber.

        Args:
            data: Structure or molecule with Cartesian coordinates.
            absorber_idx: Index of the absorbing atom.

        Returns:
            Distance tensor with shape ``(n_sites,)`` in **Angstrom**.
        """
        pos = data.cart_coords
        ref = pos[absorber_idx]
        d = np.linalg.norm(pos - ref, axis=1)
        return torch.tensor(d, dtype=torch.float32)

    def collate_fn(self, batch: list[EnvEmbedData]) -> EnvEmbedData:
        """Collate variable-length environment samples into a padded batch.

        Args:
            batch: Environment-embedding samples loaded by ``__getitem__``.

        Returns:
            Batched data with padded environment tensors and ``lengths`` metadata.
        """
        desc_list = cast(list[torch.Tensor], [sample.descriptor_features for sample in batch])
        dist_list = cast(list[torch.Tensor], [sample.distance_features for sample in batch])
        intensities_list = cast(list[torch.Tensor], [sample.intensities for sample in batch])
        energies_list = cast(list[torch.Tensor], [sample.energies for sample in batch])
        c_list = cast(list[torch.Tensor], [sample.c_star for sample in batch])
        lengths = torch.tensor([d.size(0) for d in desc_list], dtype=torch.long)
        file_name_list = [sample.file_name for sample in batch]

        intensities = torch.stack([inten.to(dtype=torch.float32) for inten in intensities_list], dim=0)
        energies = torch.stack([en.to(dtype=torch.float32) for en in energies_list], dim=0)
        c_star = torch.stack([c.to(dtype=torch.float32) for c in c_list], dim=0)
        descriptor_features = pad_sequence(desc_list, batch_first=True, padding_value=0.0)
        distance_features = pad_sequence(dist_list, batch_first=True, padding_value=0.0)

        return EnvEmbedData(
            descriptor_features=descriptor_features,
            distance_features=distance_features,
            intensities=intensities,
            energies=energies,
            c_star=c_star,
            lengths=lengths,
            file_name=file_name_list,
            basis=batch[0].basis,
        )

    def _load_item(self, path: str) -> EnvEmbedData:
        """Load one processed environment-embedding sample.

        Args:
            path: Path to a processed ``.pth`` file.

        Returns:
            Loaded environment-embedding data object with the active basis attached.
        """
        data = EnvEmbedData.load(path)
        assert self.basis is not None, "Spectral basis must be set before loading data items."
        data.basis = self.basis
        return data

    @property
    def signature(self) -> Config:
        """Dataset configuration signature.

        Returns:
            Configuration values that identify this environment-embedding dataset.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "descriptors": self.descriptor_configs,
                "widths_eV": self.widths_eV,
                "basis_stride": self.basis_stride,
                "basis_path": self.basis_path,
                "env_radius": self.env_radius,
            }
        )
        return signature
