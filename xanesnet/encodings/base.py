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

"""Abstract base class for all XANESNET spectra encodings."""

from abc import ABC, abstractmethod

import torch

from xanesnet.serialization.config import Config


class SpectraEncoding(ABC):
    """Abstract base class for all XANESNET spectra encodings.

    A spectra encoding is an invertible transformation applied to spectra so
    that training operates in an encoded space rather than on raw spectra.
    During training, target spectra are passed through :meth:`encode` before the
    loss is computed; the model therefore learns to predict in the encoded
    space. During inference, model predictions are passed through :meth:`decode`
    to recover spectra in the original space.

    Concrete subclasses must implement :meth:`encode`, :meth:`decode`, and
    :meth:`signature` so that ``decode(encode(x))`` reconstructs ``x`` up to the
    encoding's numerical accuracy. Implementations must be device-agnostic: any
    internal tensors must be moved to the device of the input tensor inside
    :meth:`encode` and :meth:`decode`.

    Call :meth:`prepare` once the raw spectral width is known, before using
    :meth:`encode` or :meth:`decode`. :meth:`output_size` provides the
    deterministic encoded width used for model construction.

    Both :meth:`encode` and :meth:`decode` accept an optional ``elements``
    tensor carrying the per-sample target-site atomic numbers ``(B,)``. Encodings
    that do not depend on the target-site element ignore it; element-aware
    encodings use it to select per-element parameters.

    Args:
        encoding_type: Identifier string for the concrete encoding type.
    """

    def __init__(
        self,
        encoding_type: str,
    ) -> None:
        """Initialize ``SpectraEncoding``."""
        self.encoding_type = encoding_type

    def prepare(self, input_size: int) -> None:
        """Prepare the encoding for spectra with a given input width.

        Encodings whose shape depends on the input width may override this
        method to bind their internal state.

        Args:
            input_size: Number of points in the raw spectrum.
        """
        pass

    @abstractmethod
    def output_size(self, input_size: int) -> int:
        """Return the encoded width for a given input width.

        Args:
            input_size: Number of points in the input representation.

        Returns:
            Number of points in the encoded representation.
        """
        ...

    @abstractmethod
    def encode(self, targets: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Encode target spectra into the encoded space.

        Args:
            targets: Ground-truth target spectra ``(B, N)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored by element-independent encodings and required by
                element-aware encodings.

        Returns:
            Encoded targets ``(B, M)``, where ``M`` is the encoded dimension.
        """
        ...

    @abstractmethod
    def decode(self, predictions: torch.Tensor, elements: torch.Tensor | None = None) -> torch.Tensor:
        """Decode encoded predictions back into the original spectrum space.

        Args:
            predictions: Model predictions in the encoded space ``(B, M)``.
            elements: Optional per-sample target-site atomic numbers ``(B,)``.
                Ignored by element-independent encodings and required by
                element-aware encodings.

        Returns:
            Decoded spectra ``(B, N)`` in the original space.
        """
        ...

    @property
    @abstractmethod
    def signature(self) -> list[Config]:
        """Return the encoding signature as a list.

        Every encoding returns a :class:`list` of :class:`Config` objects:
        a single-element list for leaf encodings, a multi-element list for
        :class:`~xanesnet.encodings.CombinedEncoding`, or a single-element
        list whose value contains nested sub-encoding configs for
        :class:`~xanesnet.encodings.ConcatEncoding`.
        The list format mirrors
        the ``encodings`` configuration section so that the signature can be
        stored directly under the ``"encodings"`` key in a checkpoint.

        Returns:
            Ordered list of configuration values needed to recreate this
            encoding.
        """
        return [
            Config(
                {
                    "encoding_type": self.encoding_type,
                }
            )
        ]
