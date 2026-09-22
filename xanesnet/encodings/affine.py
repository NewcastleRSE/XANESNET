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

"""Affine spectra-encoding base for XANESNET."""

import torch

from xanesnet.utils.exceptions import ConfigError

from .base import SpectraEncoding


def build_parameter_rows(encoding_type: str, name: str, rows: list[list[float]]) -> torch.Tensor:
    """Build a rectangular ``(E, D)`` parameter tensor from per-element rows.

    Args:
        encoding_type: Identifier of the calling encoding, used in error
            messages.
        name: Name of the parameter being built, used in error messages.
        rows: Per-element parameter rows; one inner list per element.

    Returns:
        Float32 tensor ``(E, D)`` with one row per element.

    Raises:
        ConfigError: If ``rows`` is empty or its inner lists are empty or of
            unequal length.
    """
    if not rows:
        raise ConfigError(f"{encoding_type} requires a non-empty '{name}'.")
    width = len(rows[0])
    if width == 0 or any(len(row) != width for row in rows):
        raise ConfigError(f"{encoding_type} '{name}' rows must be non-empty and of equal length.")
    return torch.tensor(rows, dtype=torch.float32)


class AffineEncoding(SpectraEncoding):
    """Affine spectra-encoding base with optional per-element parameters.

    Implements the invertible affine transform shared by the standardization,
    normalization, centering, and scaling encodings::

        encode(x) = (x - shift) / scale
        decode(y) = y * scale + shift

    ``shift`` and ``scale`` are broadcast against the ``(B, N)`` input. Two
    parameterizations are supported:

    * **Shared** (``per_element=False``): ``shift`` and ``scale`` are 1-D
      tensors - per-point vectors of length ``N`` for point-by-point transforms
      or length-one tensors for a single global transform - applied to every
    sample regardless of its target-site element.
    * **Element-aware** (``per_element=True``): ``shift`` and ``scale`` are 2-D
      ``(E, D)`` lookup tables with one row per known element; the row applied
    to each sample is selected from the sample's target-site atomic number. Rows
      have length ``N`` for point-by-point transforms or length one for a single
      global transform per element.

    Concrete subclasses compute the parameters from their own configuration and
    forward them here, so the encode/decode mechanics, element lookup, and
    device handling live in one place. Internal tensors are moved to the device
    of the input tensor inside :meth:`encode` and :meth:`decode`.

    Args:
        encoding_type: Identifier string for the concrete encoding type.
        shift: Additive offset removed during encoding. A 1-D tensor when
            ``per_element`` is false or an ``(E, D)`` table aligned row-wise
            with ``elements`` when true.
        scale: Divisor applied during encoding, shaped like ``shift``.
            Subclasses should ensure scale values are non-zero (typically by
            clamping near-zero entries to ``1.0``).
        per_element: Whether ``shift`` and ``scale`` are selected per target-site
            element (``True``) or shared across all samples (``False``).
        elements: Atomic numbers aligned row-wise with ``shift`` and ``scale``
            when ``per_element`` is true; ignored otherwise.

    Raises:
        ConfigError: If ``per_element`` is true and ``elements`` is empty,
            contains duplicates, or does not match the number of parameter rows.
    """

    def __init__(
        self,
        encoding_type: str,
        shift: torch.Tensor,
        scale: torch.Tensor,
        per_element: bool,
        elements: list[int] | None,
    ) -> None:
        """Initialize ``AffineEncoding``."""
        super().__init__(encoding_type)

        self.per_element = per_element
        self.shift = shift
        self.scale = scale
        self.elements = list(elements) if elements is not None else None
        self._lookup: torch.Tensor | None = None

        if per_element:
            self._lookup = self._build_lookup(encoding_type, self.elements, shift, scale)

    def output_size(self, input_size: int) -> int:
        """Return the unchanged width of an affine encoding.

        Args:
            input_size: Number of points in the input spectrum.

        Returns:
            Number of points in the encoded representation.
        """
        return input_size

    @staticmethod
    def _build_lookup(
        encoding_type: str,
        elements: list[int] | None,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Validate element rows and build the atomic-number row lookup.

        Args:
            encoding_type: Identifier string used in error messages.
            elements: Atomic numbers aligned row-wise with ``shift`` and
                ``scale``.
            shift: Per-element additive offsets ``(E, D)``.
            scale: Per-element divisors ``(E, D)``.

        Returns:
            Dense atomic-number to row lookup with ``-1`` marking unknown
            elements.

        Raises:
            ConfigError: If ``elements`` is empty, contains duplicates, or does
                not match the number of parameter rows.
        """
        if not elements:
            raise ConfigError(f"{encoding_type} requires a non-empty list of elements when per_element is true.")
        if len(set(elements)) != len(elements):
            raise ConfigError(f"{encoding_type} elements must be unique, got {elements}.")
        if shift.shape[0] != len(elements) or scale.shape[0] != len(elements):
            raise ConfigError(
                f"{encoding_type} expects one parameter row per element: "
                f"{len(elements)} elements vs shift {tuple(shift.shape)} / scale {tuple(scale.shape)}."
            )

        lookup = torch.full((max(elements) + 1,), -1, dtype=torch.long)
        for row, atomic_number in enumerate(elements):
            lookup[atomic_number] = row
        return lookup

    def _resolve(self, elements: torch.Tensor | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shift and scale tensors to apply on ``device``.

        Args:
            elements: Per-sample target-site atomic numbers ``(B,)``; required only
                when this encoding is element-aware.
            device: Device the returned parameter tensors should live on.

        Returns:
            Tuple ``(shift, scale)`` on ``device``. When element-aware these are
            ``(B, D)`` tensors gathered per sample; otherwise the shared
            broadcastable parameters.

        Raises:
            ConfigError: If element-aware and ``elements`` is ``None`` or
                contains an atomic number without configured parameters.
        """
        if not self.per_element:
            return self.shift.to(device), self.scale.to(device)
        return self._gather(elements, device)

    def _gather(self, elements: torch.Tensor | None, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Select per-sample shift and scale rows from the element lookup.

        Args:
            elements: Per-sample target-site atomic numbers ``(B,)``.
            device: Device the returned parameter tensors should live on.

        Returns:
            Tuple ``(shift, scale)`` of ``(B, D)`` tensors on ``device``.

        Raises:
            ConfigError: If ``elements`` is ``None`` or contains an atomic
                number without configured parameters.
        """
        assert self._lookup is not None
        if elements is None:
            raise ConfigError(
                f"Encoding '{self.encoding_type}' is element-aware and requires per-sample atomic numbers, "
                "but none were provided by the batch."
            )

        lookup = self._lookup.to(device)
        indices = elements.to(device=device, dtype=torch.long)
        if torch.any(indices >= lookup.shape[0]) or torch.any(indices < 0):
            raise ConfigError(f"Encoding '{self.encoding_type}' received an out-of-range atomic number.")

        rows = lookup[indices]
        if torch.any(rows < 0):
            unknown = sorted({int(z) for z, row in zip(indices.tolist(), rows.tolist()) if row < 0})
            raise ConfigError(
                f"Encoding '{self.encoding_type}' has no parameters for element(s) {unknown}; "
                f"known elements are {self.elements}."
            )

        shift = self.shift.to(device)[rows]
        scale = self.scale.to(device)[rows]
        return shift, scale

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Apply the affine transform to target spectra.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Per-sample target-site atomic numbers ``(B,)``. Required when
                this encoding is element-aware and ignored otherwise.

        Returns:
            Affinely transformed targets ``(B, N)``.

        Raises:
            ConfigError: If element-aware and ``elements`` is ``None`` or
                contains an unknown atomic number.
        """
        shift, scale = self._resolve(elements, targets.device)
        return (targets - shift) / scale

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Invert the affine transform applied by :meth:`encode`.

        Args:
            predictions: Model predictions in the transformed space ``(B, N)``.
            elements: Per-sample target-site atomic numbers ``(B,)``. Required when
                this encoding is element-aware and ignored otherwise.

        Returns:
            Reconstructed spectra ``(B, N)``.

        Raises:
            ConfigError: If element-aware and ``elements`` is ``None`` or
                contains an unknown atomic number.
        """
        shift, scale = self._resolve(elements, predictions.device)
        return predictions * scale + shift
