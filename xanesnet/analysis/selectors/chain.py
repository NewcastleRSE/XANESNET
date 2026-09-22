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

"""Selector that applies a configured chain of selectors in order."""

from collections.abc import Iterator
from copy import copy

from xanesnet.serialization.config import Config
from xanesnet.serialization.prediction_readers import PredictionReader, PredictionSample

from ..utils import SampleKey, sample_key
from .base import Selector
from .registry import SelectorRegistry


class _SubsetPredictionReader(PredictionReader):
    """Indexed prediction-reader view over a subset of another reader.

    Args:
        parent: Reader containing the complete set of prediction samples.
        indices: Parent-reader indices exposed by this view, in iteration order.
    """

    def __init__(self, parent: PredictionReader, indices: tuple[int, ...]) -> None:
        """Initialize a subset prediction-reader view."""
        super().__init__(parent.path)
        self.parent = parent
        self.indices = indices

    def _validate_path(self) -> None:
        """Rely on the parent reader having already validated its path."""

    def __len__(self) -> int:
        """Return the number of records in the subset.

        Returns:
            Number of records exposed by this view.
        """
        return len(self.indices)

    def __getitem__(self, index: int) -> PredictionSample:
        """Return one record from the subset.

        Args:
            index: Zero-based index relative to this subset view.

        Returns:
            The corresponding prediction sample from the parent reader.
        """
        return self.parent[self.indices[index]]

    def provides_structures(self) -> bool:
        """Return whether the parent reader provides matched structures.

        Returns:
            ``True`` when parent samples include matched structures, otherwise
            ``False``.
        """
        return self.parent.provides_structures()


@SelectorRegistry.register("chain")
class ChainSelector(Selector):
    """Apply a sequence of selectors to the same samples in order.

    Each stage receives the subset produced by the preceding stage. If a stage
    expands into multiple selectors, the chain produces one selector per
    resulting branch while retaining indices relative to the original reader.

    Args:
        selector_type: Registered selector name from the analysis configuration.
        data_source: Prediction reader containing the original samples.
        chain: Ordered selector configurations. Schema validation supplies the
            default ``all`` stage when the property is omitted.
    """

    def __init__(
        self,
        selector_type: str,
        data_source: PredictionReader,
        chain: list[Config],
    ) -> None:
        """Initialize a chain and materialize its resolved branches."""
        super().__init__(selector_type, data_source)
        self.chain = tuple(chain)
        self._selected_indices: list[int] = []
        self._stages: tuple[Selector, ...] = ()
        self._expanded: tuple[ChainSelector, ...] = ()

        resolved = self._resolve()
        if len(resolved) == 1:
            # A non-expanding chain can be used directly as a normal selector.
            indices, self._stages = resolved[0]
            self._selected_indices = list(indices)
            self._expanded = (self,)
        else:
            self._expanded = tuple(self._resolved_copy(*branch) for branch in resolved)

    def _resolve(self) -> list[tuple[tuple[int, ...], tuple[Selector, ...]]]:
        """Resolve every stage and build final original-reader indices.

        Returns:
            One ``(indices, stages)`` pair for each expansion branch. ``indices``
            always refer to ``data_source`` rather than an intermediate subset
            reader.
        """
        states: list[tuple[PredictionReader, tuple[int, ...], tuple[Selector, ...]]] = [
            (self.data_source, tuple(range(len(self.data_source))), ())
        ]

        for stage_config in self.chain:
            next_states: list[tuple[PredictionReader, tuple[int, ...], tuple[Selector, ...]]] = []
            for source, original_indices, stages in states:
                if len(source) == 0:
                    # A later selector cannot add samples to an empty branch.
                    # Keep the branch so that the chain still behaves like a
                    # normal empty selector instead of asking selectors such as
                    # structure clustering to operate on too few samples.
                    next_states.append((source, original_indices, stages))
                    continue
                kwargs = stage_config.as_kwargs()
                stage = SelectorRegistry.create(
                    stage_config.get_str("selector_type"),
                    **kwargs,
                    data_source=source,
                )
                for expanded_stage in stage.expand_selectors():
                    positions = _selected_positions(source, expanded_stage)
                    next_states.append(
                        (
                            _SubsetPredictionReader(source, positions),
                            tuple(original_indices[position] for position in positions),
                            (*stages, expanded_stage),
                        )
                    )
            states = next_states

        return [(original_indices, stages) for _, original_indices, stages in states]

    def _resolved_copy(
        self,
        indices: tuple[int, ...],
        stages: tuple[Selector, ...],
    ) -> "ChainSelector":
        """Create a normal selector copy with materialized chain results.

        Args:
            indices: Final indices relative to the original prediction reader.
            stages: Resolved selector instances used to produce ``indices``.

        Returns:
            A chain selector that yields exactly the samples at ``indices``.
        """
        resolved = copy(self)
        resolved._selected_indices = list(indices)
        resolved._stages = stages
        resolved._expanded = (resolved,)
        return resolved

    def __iter__(self) -> Iterator[PredictionSample]:
        """Yield samples selected by the resolved chain.

        Returns:
            Iterator over selected prediction samples.
        """
        for index in self._selected_indices:
            yield self.data_source[index]

    def __len__(self) -> int:
        """Return the number of samples selected by the resolved chain.

        Returns:
            Number of selected samples.
        """
        return len(self._selected_indices)

    def expand_selectors(self) -> list[Selector]:
        """Return the materialized selectors for this chain.

        A non-expanding chain returns itself. A chain with an expanding stage
        returns one resolved selector per branch.

        Returns:
            Resolved selectors for the chain, or an empty list when a stage
            produced no valid branch.
        """
        return list(self._expanded)

    @property
    def signature(self) -> Config:
        """Return the configured or resolved chain signature.

        Returns:
            Configuration containing the chain stages. Expanded stages use
            their resolved signatures, including explicit cluster IDs.
        """
        signature = super().signature
        stages = self._stages
        chain = (
            [stage.signature.as_dict() for stage in stages]
            if len(stages) == len(self.chain)
            else [stage.as_dict() for stage in self.chain]
        )
        signature.update_with_dict({"chain": chain})
        return signature

    def __str__(self) -> str:
        """Return the ordered stage labels as a compact display name."""
        if len(self._stages) == len(self.chain):
            return " -> ".join(str(stage) for stage in self._stages)
        return " -> ".join(stage.get_str("selector_type") for stage in self.chain)


def _selected_positions(source: PredictionReader, selector: Selector) -> tuple[int, ...]:
    """Map selected samples to positions in their source reader.

    Args:
        source: Reader from which ``selector`` selects samples.
        selector: Selector whose samples should be mapped to source positions.

    Returns:
        Source-relative positions in the order yielded by ``selector``.
    """
    positions_by_key: dict[SampleKey, int] = {sample_key(source[position]): position for position in range(len(source))}
    return tuple(positions_by_key[sample_key(sample)] for sample in selector)
