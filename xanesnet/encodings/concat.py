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

"""Concatenation spectra encoding for XANESNET."""

import torch

from xanesnet.serialization.config import Config

from .base import SpectraEncoding
from .combined import CombinedEncoding
from .registry import SpectraEncodingRegistry


@SpectraEncodingRegistry.register("concat")
class ConcatEncoding(SpectraEncoding):
    """Concatenation spectra encoding.

    Encodes spectra by applying multiple independent sub-encodings in parallel
    and concatenating their outputs along the last dimension::

        encode(x) = concat(e_0(x), e_1(x), ..., e_n(x))

    During decoding, the prediction is split into per-encoding chunks, each
    chunk is decoded by its corresponding sub-decoder, and the resulting
    spectra are averaged element-wise.

    Args:
        encoding_type: Identifier string for this encoding type.
        encodings: Non-empty list of :class:`Config` objects, each describing a
            sub-encoding with an ``encoding_type`` key.

    Raises:
        ValueError: If ``encodings`` is empty.
    """

    def __init__(
        self,
        encoding_type: str,
        encodings: list[Config],
    ) -> None:
        """Initialize ``ConcatEncoding``."""
        super().__init__(encoding_type)

        if not encodings:
            raise ValueError("ConcatEncoding requires at least one sub-encoding.")

        self.encodings: list[SpectraEncoding] = CombinedEncoding.from_configs(encodings).encodings
        self._split_sizes: list[int] = []
        self._input_size: int | None = None

    def prepare(self, input_size: int) -> None:
        """Prepare sub-encodings and record their deterministic output widths.

        Every sub-encoding receives the same input width because concat
        applies them independently. The resulting widths are stored for
        decoding, which means decoding no longer depends on a previous call to
        :meth:`encode`.

        Args:
            input_size: Number of points in the raw spectrum.
        """
        self._split_sizes = []
        self._input_size = None

        split_sizes: list[int] = []
        for encoding in self.encodings:
            encoding.prepare(input_size)
            split_sizes.append(encoding.output_size(input_size))

        self._split_sizes = split_sizes
        self._input_size = input_size

    def output_size(self, input_size: int) -> int:
        """Return the concatenated width of all sub-encoding outputs.

        Args:
            input_size: Number of points in the shared input spectrum.

        Returns:
            Sum of the output widths of all sub-encodings.
        """
        return sum(encoding.output_size(input_size) for encoding in self.encodings)

    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Encode target spectra through every sub-encoding and concatenate.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``,
                forwarded to each sub-encoding.

        Returns:
            Encoded targets ``(B, sum(M_i))``, the concatenation of all
            sub-encoding outputs.

        Raises:
            RuntimeError: If the encoding has not been prepared for the input
                spectrum width.
        """
        if not self._split_sizes or self._input_size is None:
            raise RuntimeError("ConcatEncoding.encode called before prepare; call prepare with the raw spectrum width.")
        if targets.shape[-1] != self._input_size:
            raise ValueError(f"ConcatEncoding expected input width {self._input_size}, got {targets.shape[-1]}.")

        encoded_parts: list[torch.Tensor] = []
        for encoding, split_size in zip(self.encodings, self._split_sizes):
            encoded = encoding.encode(targets, elements)
            encoded_parts.append(encoded)
            if encoded.shape[-1] != split_size:
                raise RuntimeError(
                    f"Sub-encoding '{encoding.encoding_type}' returned width {encoded.shape[-1]}, "
                    f"but prepare predicted {split_size}."
                )
        return torch.cat(encoded_parts, dim=-1)

    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Decode predictions by splitting, decoding, and averaging.

        Splits the concatenated prediction tensor into per-encoding chunks
        according to the sizes determined by :meth:`prepare`, decodes each
        chunk through its corresponding sub-decoder, and averages the
        resulting spectra element-wise.

        Args:
            predictions: Model predictions in the concatenated encoded space
                ``(B, sum(M_i))``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``,
                forwarded to each sub-decoding.

        Returns:
            Decoded spectra ``(B, N)`` in the original space, averaged across
            all sub-decoders.

        Raises:
            RuntimeError: If ``decode`` is called before :meth:`prepare`.
            ValueError: If the prediction size does not match the sum of the
                expected split sizes.
        """
        if not self._split_sizes or self._input_size is None:
            raise RuntimeError("ConcatEncoding.decode called before prepare; call prepare with the raw spectrum width.")
        expected_size = sum(self._split_sizes)
        if predictions.shape[-1] != expected_size:
            raise ValueError(f"ConcatEncoding expected prediction width {expected_size}, got {predictions.shape[-1]}.")
        chunks = torch.split(predictions, self._split_sizes, dim=-1)
        decoded_parts: list[torch.Tensor] = []
        for chunk, encoding in zip(chunks, self.encodings):
            decoded_parts.append(encoding.decode(chunk, elements))
        return torch.stack(decoded_parts, dim=0).mean(dim=0)

    @property
    def signature(self) -> list[Config]:
        """Return the concatenation-encoding signature.

        Returns:
            Single-element list with the concat encoding configuration.
        """
        sub_sigs: list[dict] = []
        for encoding in self.encodings:
            for sig in encoding.signature:
                sub_sigs.append(sig.as_dict())
        sig = Config(
            {
                "encoding_type": self.encoding_type,
                "encodings": sub_sigs,
            }
        )
        return [sig]
