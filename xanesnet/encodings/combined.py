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

"""Sequential composition of multiple XANESNET spectra encodings."""

import torch

from xanesnet.serialization.config import Config

from .base import SpectraEncoding
from .registry import SpectraEncodingRegistry


class CombinedEncoding(SpectraEncoding):
    """Sequential composition of multiple spectra encodings.

    Encoding applies the component encodings in order, so
    ``encode(x) = enc[n-1](... enc[1](enc[0](x)))``. Decoding applies the
    component decoders in reverse order, so the composition is invertible up to
    each component's numerical accuracy.

    A combined encoding is always used as the single entry point for training
    and inference, even when only one component encoding is configured (the
    list then has length one). Build instances from configuration with
    :meth:`from_configs`, then call :meth:`prepare` with the raw spectral
    width before encoding or decoding.

    Args:
        encodings: Ordered, non-empty list of :class:`SpectraEncoding` modules
            to compose.

    Raises:
        ValueError: If ``encodings`` is empty.
    """

    def __init__(self, encodings: list[SpectraEncoding]) -> None:
        """Initialize ``CombinedEncoding``."""
        super().__init__(encoding_type="combined")

        if len(encodings) == 0:
            raise ValueError("CombinedEncoding requires at least one component encoding.")

        self.encodings = encodings

    @classmethod
    def from_configs(cls, encoding_configs: list[Config]) -> "CombinedEncoding":
        """Build a combined encoding from a list of encoding configurations.

        Each entry must provide ``encoding_type`` and any additional keyword
        arguments expected by the corresponding registered encoding. The
        components are instantiated via :class:`SpectraEncodingRegistry` and
        composed, in order, into a single :class:`CombinedEncoding`.

        Args:
            encoding_configs: Non-empty, ordered list of encoding
                configurations.

        Returns:
            A :class:`CombinedEncoding` wrapping the configured component
            encodings.

        Raises:
            ValueError: If ``encoding_configs`` is empty.
        """
        if not encoding_configs:
            raise ValueError("Encoding config list is required but was not provided.")

        encodings: list[SpectraEncoding] = []
        for item_config in encoding_configs:
            encoding_type = item_config.get_str("encoding_type")
            encodings.append(SpectraEncodingRegistry.create(encoding_type, **item_config.as_kwargs()))

        return cls(encodings)

    def prepare(self, input_size: int) -> None:
        """Prepare each component for its input width in sequence.

        Args:
            input_size: Number of points in the raw input representation.
        """
        for encoding in self.encodings:
            encoding.prepare(input_size)
            input_size = encoding.output_size(input_size)

    def output_size(self, input_size: int) -> int:
        """Return the output width after sequentially applying all components.

        Args:
            input_size: Number of points in the input representation.

        Returns:
            Number of points in the final encoded representation.
        """
        for encoding in self.encodings:
            input_size = encoding.output_size(input_size)
        return input_size

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Encode target spectra through every component in order.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``,
                forwarded to each component encoding.

        Returns:
            Encoded targets ``(B, M)`` after applying all component encoders.
        """
        for encoding in self.encodings:
            targets = encoding.encode(targets, elements)
        return targets

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Decode predictions through every component in reverse order.

        Args:
            predictions: Model predictions in the encoded space ``(B, M)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``,
                forwarded to each component decoding.

        Returns:
            Decoded spectra ``(B, N)`` after applying all component decoders.
        """
        for encoding in reversed(self.encodings):
            predictions = encoding.decode(predictions, elements)
        return predictions

    @property
    def signature(self) -> list[Config]:
        """Return the ordered per-component encoding signatures.

        The returned list mirrors the ``encodings`` configuration section and
        can be passed straight back to :meth:`from_configs` to reconstruct this
        combined encoding.

        Returns:
            Ordered list of component encoding signatures.
        """
        result: list[Config] = []
        for encoding in self.encodings:
            result.extend(encoding.signature)
        return result
