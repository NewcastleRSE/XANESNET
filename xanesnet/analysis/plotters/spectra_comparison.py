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

"""Plotter for best/worst spectra comparisons across methods."""

import logging
from itertools import repeat
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample
from xanesnet.utils.exceptions import ConfigError

from ..result import AnalysisResults
from ..selectors import Selector
from ..utils import (
    SampleKey,
    is_scalar_value,
    one_line_label,
    sample_key,
    sample_key_sort_key,
)
from .base import Plotter
from .common.formatting import format_decimal
from .common.layout import save_figure
from .common.spectra_pages import (
    combined_spectra_page_figure,
    spectra_page_figure,
    spectra_structure_page_figure,
)
from .common.style import PlotSize
from .registry import PlotterRegistry

_Entry = tuple[PredictionSample, float]


@PlotterRegistry.register("spectra_comparison")
class SpectraComparisonPlotter(Plotter):
    """Plot the N best and N worst spectra per method.

    Ranked by a scalar error value, ``n_samples`` best and worst samples are
    written as multi-page PDFs per method, with structure panels when matched.
    With two or more readers, combined PDFs are generated for each method's
    ranking: the samples common to all readers are ranked by that method's
    error, and every reader's prediction is overlaid.

    Requires:
        Per-sample ranking error: provided by a scalar collector emitting ``sort_key``.
        Matched raw structures (optional): shown alongside the spectra.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        latex_font: Render figures in a LaTeX-style serif font when ``True``.
        n_samples: Number of best and worst samples plotted per method.
        sort_key: Scalar collector key used to rank samples. Must be produced
            by a configured collector.
        legend_position: Place spectra legends ``"inside"`` the axes or
            ``"outside"`` them on the right.
        plot_size: Shared figure size profile: ``"small"`` or ``"default"``.
    """

    def __init__(
        self,
        plotter_type: str,
        n_samples: int,
        sort_key: str,
        legend_position: str,
        latex_font: bool,
        plot_size: PlotSize,
    ) -> None:
        """Initialize a best/worst spectra comparison plotter."""
        super().__init__(plotter_type, latex_font=latex_font, plot_size=plot_size)
        self.n_samples = n_samples
        self.sort_key = sort_key
        self.legend_position = legend_position

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write best/worst spectra PDFs with structure panels where available.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``spectra_comparison`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "spectra_comparison"
        metric = self.sort_key

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                label = results.method_label(reader_idx, sel_idx)
                stream = results.collector_stream(reader_idx, sel_idx)

                entries = self._collect_entries(selector, stream)
                if not entries:
                    continue

                by_error = sorted(entries, key=lambda entry: entry[1])
                best = by_error[: self.n_samples]
                worst = list(reversed(by_error[-self.n_samples :]))

                combo_dir = root / label.dir_name
                combo_dir.mkdir(parents=True, exist_ok=True)

                self._write_pages(best, label.lines, combo_dir / "best.pdf", "best", metric)
                self._write_pages(worst, label.lines, combo_dir / "worst.pdf", "worst", metric)

        self._plot_combined(results, root)

    def _collect_entries(self, selector: Selector, stream: JSONLStream | None) -> list[_Entry]:
        """Collect per-sample entries with their ranking error value.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.

        Returns:
            ``(sample, error)`` entries in selector order.

        Raises:
            ConfigError: If ``sort_key`` is missing or non-scalar for a sample.
        """
        entries: list[_Entry] = []
        for sample, record in zip(selector, stream if stream is not None else repeat({})):
            value = record.get(self.sort_key)
            if not is_scalar_value(value):
                raise ConfigError(
                    f"Sort key '{self.sort_key}' is missing or not a scalar for sample "
                    f"'{sample['sample_id']}'. Configure a scalar collector that produces this key."
                )
            entries.append((sample, float(value)))
        return entries

    def _write_pages(
        self,
        entries: list[_Entry],
        label_lines: list[str],
        pdf_path: Path,
        direction: str,
        metric: str,
    ) -> None:
        """Write one multi-page PDF for the best or worst entries of one method.

        Args:
            entries: Ranked ``(sample, error)`` entries, best or worst first.
            label_lines: Method label lines used for the page subtitle.
            pdf_path: Destination PDF path.
            direction: ``"best"`` or ``"worst"`` for the page subtitle.
            metric: Scalar key used for ranking.
        """
        total = len(entries)
        with PdfPages(pdf_path) as pdf:
            for rank, (sample, error) in enumerate(entries, start=1):
                subtitle = (
                    f"{direction} #{rank} of {total}  |  {metric}={format_decimal(error, 4)}  |  "
                    + one_line_label(label_lines)
                )
                if sample.get("structure") is not None:
                    fig = spectra_structure_page_figure(
                        sample,
                        subtitle,
                        self.style,
                        legend_position=self.legend_position,
                    )
                else:
                    fig = spectra_page_figure(
                        sample,
                        subtitle,
                        self.style,
                        legend_position=self.legend_position,
                    )
                save_figure(fig, pdf, self.style)

    def _plot_combined(self, results: AnalysisResults, root: Path) -> None:
        """Write combined best/worst pages ranked independently by each reader.

        For every selector index shared by all prediction readers, each
        reader's error ranks the samples common to all readers. The best and
        worst ``n_samples`` for that reader then overlay every reader's
        prediction against the shared target. Nothing is written with fewer
        than two prediction readers.

        Args:
            results: Analysis pipeline outputs to plot.
            root: Root ``spectra_comparison`` directory; combined PDFs are
                written under ``<root>/combined/``.

        Raises:
            ConfigError: If a ranking value is invalid or duplicate records
                share a compound sample identity.
        """
        if len(results.selectors) < 2:
            return

        n_common_selectors = min(len(reader_selectors) for reader_selectors in results.selectors)
        combined_root = root / "combined"

        for sel_idx in range(n_common_selectors):
            by_reader: list[dict[SampleKey, _Entry]] = []
            for reader_idx in range(len(results.selectors)):
                entries: dict[SampleKey, _Entry] = {}
                for entry in self._collect_entries(
                    results.selectors[reader_idx][sel_idx], results.collector_stream(reader_idx, sel_idx)
                ):
                    identity = sample_key(entry[0])
                    if identity in entries:
                        raise ConfigError(f"Duplicate prediction record for sample identity {identity!r}.")
                    entries[identity] = entry
                by_reader.append(entries)

            common_keys = set.intersection(*(set(entries) for entries in by_reader))
            if len(common_keys) < 2:
                continue

            selector_type = results.selectors[0][sel_idx].selector_type
            selector_str = str(results.selectors[0][sel_idx])
            selector_dir = combined_root / f"sel_{sel_idx:03d}_{selector_type}"

            for ranking_reader_idx, ranking_entries in enumerate(by_reader):
                ranking_error = {identity: ranking_entries[identity][1] for identity in common_keys}
                ranked = sorted(
                    common_keys,
                    key=lambda identity: (ranking_error[identity], sample_key_sort_key(identity)),
                )
                n = min(self.n_samples, len(ranked))
                best = ranked[:n]
                worst = list(reversed(ranked[-n:]))

                ranking_method = results.method_label(ranking_reader_idx, sel_idx)
                combo_dir = selector_dir / f"method_{ranking_reader_idx:03d}"
                combo_dir.mkdir(parents=True, exist_ok=True)

                self._write_combined_pages(
                    by_reader,
                    results.prediction_names,
                    best,
                    ranking_error,
                    self.sort_key,
                    ranking_method.lines[0],
                    selector_str,
                    combo_dir / "best.pdf",
                    "best",
                )
                self._write_combined_pages(
                    by_reader,
                    results.prediction_names,
                    worst,
                    ranking_error,
                    self.sort_key,
                    ranking_method.lines[0],
                    selector_str,
                    combo_dir / "worst.pdf",
                    "worst",
                )

    def _write_combined_pages(
        self,
        by_reader: list[dict[SampleKey, _Entry]],
        method_labels: list[str],
        sample_keys: list[SampleKey],
        ranking_error: dict[SampleKey, float],
        metric: str,
        ranking_method_label: str,
        selector_str: str,
        pdf_path: Path,
        direction: str,
    ) -> None:
        """Write one combined multi-page PDF for shared best or worst samples.

        Args:
            by_reader: Per-reader mapping from compound sample identity to its
                ``(sample, error)`` entry.
            method_labels: Display name per prediction reader.
            sample_keys: Ranked compound sample identities to render, best or worst first.
            ranking_error: Ranking method's error keyed by compound sample identity.
            metric: Scalar key used for ranking.
            ranking_method_label: Display label of the method used for ranking.
            selector_str: Shared selector description for the page subtitle.
            pdf_path: Destination PDF path.
            direction: ``"best"`` or ``"worst"`` for the page subtitle.
        """
        total = len(sample_keys)
        with PdfPages(pdf_path) as pdf:
            for rank, identity in enumerate(sample_keys, start=1):
                sample = by_reader[0][identity][0]
                target = np.asarray(sample["target"]).ravel()
                predictions = [np.asarray(entries[identity][0]["prediction"]).ravel() for entries in by_reader]
                subtitle = (
                    f"{direction} #{rank} of {total}  |  {ranking_method_label} {metric}="
                    f"{format_decimal(ranking_error[identity], 4)}  |  {selector_str}"
                )
                fig = combined_spectra_page_figure(
                    sample,
                    method_labels,
                    predictions,
                    target,
                    subtitle,
                    self.style,
                    legend_position=self.legend_position,
                )
                save_figure(fig, pdf, self.style)

    @property
    def signature(self) -> Config:
        """Return the spectra comparison plotter signature.

        Returns:
            Configuration values needed to recreate this plotter.
        """
        signature = super().signature
        signature.update_with_dict(
            {
                "n_samples": self.n_samples,
                "sort_key": self.sort_key,
                "legend_position": self.legend_position,
            }
        )
        return signature
