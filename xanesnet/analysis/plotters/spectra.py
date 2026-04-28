# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
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

"""Plotter for predicted and target spectra comparisons."""

import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from xanesnet.serialization.jsonl_stream import JSONLStream
from xanesnet.serialization.prediction_readers import PredictionSample

from ..reporters.base import selector_label
from ..result import AnalysisResults
from ..selectors import Selector
from .base import Plotter
from .registry import PlotterRegistry
from .utils import ScalarValue, is_scalar


@PlotterRegistry.register("spectra")
class SpectraPlotter(Plotter):
    """Plot predicted and target spectra for every selected sample.

    Args:
        plotter_type: Registered plotter name from the analysis configuration.
        sort_by_value: Whether to sort PDF pages by a scalar value.
        sort_key: Scalar key to sort by. Values are read from collector output first, then samples.
        sort_ascending: Whether sorted pages should use ascending order.
        max_pages: Maximum number of pages per PDF. ``None`` writes all selected samples.
    """

    def __init__(
        self,
        plotter_type: str,
        sort_by_value: bool = False,
        sort_key: str | None = None,
        sort_ascending: bool = True,
        max_pages: int | None = None,
    ) -> None:
        """Initialize a spectra comparison plotter."""
        super().__init__(plotter_type)
        self.sort_by_value = sort_by_value
        self.sort_key = sort_key
        self.sort_ascending = sort_ascending
        self.max_pages = max_pages

    def plot(self, results: AnalysisResults, output_dir: Path) -> None:
        """Write one multi-page spectra PDF per prediction-reader/selector pair.

        Args:
            results: Analysis pipeline outputs to plot.
            output_dir: Directory where the ``spectra_plots`` tree should be written.
        """
        if not results.selectors:
            logging.info("    No selectors available, skipping.")
            return

        root = output_dir / "spectra_plots"
        root.mkdir(parents=True, exist_ok=True)

        for reader_idx, reader_selectors in enumerate(results.selectors):
            logging.info(f"    Predictions {reader_idx + 1}/{len(results.selectors)}.")

            for sel_idx, selector in enumerate(reader_selectors):
                logging.info(f"      Selector {sel_idx + 1}/{len(reader_selectors)}.")
                sel_label_str = selector_label(results.selectors_config, sel_idx)
                sel_cfg = results.selectors_config[sel_idx] if sel_idx < len(results.selectors_config) else {}

                stream: JSONLStream | None = None
                if reader_idx < len(results.collector_results) and sel_idx < len(results.collector_results[reader_idx]):
                    stream = results.collector_results[reader_idx][sel_idx]

                combo_label = f"pred_{reader_idx:03d}__sel_{sel_idx:03d}_{sel_label_str}"
                subtitle = _subtitle(sel_cfg, reader_idx)
                pdf_path = root / f"{combo_label}.pdf"

                self._plot_to_pdf(selector, stream, pdf_path, subtitle)

    def _plot_to_pdf(
        self,
        selector: Selector,
        stream: JSONLStream | None,
        pdf_path: Path,
        subtitle: str,
    ) -> None:
        """Write selected spectra comparisons into a multi-page PDF.

        Args:
            selector: Selector over prediction samples for one prediction reader and selector pair.
            stream: Optional collector result stream aligned with ``selector``.
            pdf_path: Destination PDF path.
            subtitle: Subtitle text describing prediction and selector context.
        """
        entries: list[tuple[PredictionSample, dict[str, Any]]] = []
        if stream is not None:
            for sel_sample, col_sample in zip(selector, stream):
                entries.append((sel_sample, col_sample))
        else:
            for sel_sample in selector:
                entries.append((sel_sample, {}))

        if not entries:
            return

        if self.sort_by_value and self.sort_key:
            key = self.sort_key

            def _sort_val(entry: tuple[PredictionSample, dict[str, Any]]) -> float:
                """Return the scalar value used to order one PDF page entry.

                Args:
                    entry: Pair of prediction sample and collector scalar dictionary.

                Returns:
                    Sort value for the configured key, or ``0.0`` when unavailable/non-scalar.
                """
                sel_s, col_s = entry
                v = col_s.get(key, sel_s.get(key, 0.0))
                return cast(float, v) if is_scalar(v) else 0.0

            entries.sort(key=_sort_val, reverse=not self.sort_ascending)

        if self.max_pages is not None:
            entries = entries[: self.max_pages]

        with PdfPages(pdf_path) as pdf:
            for sample, col_scalars in entries:
                fig = self._plot_single(sample, col_scalars, subtitle)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

    @staticmethod
    def _plot_single(
        sample: PredictionSample,
        col_scalars: dict[str, Any],
        subtitle: str,
    ) -> Figure:
        """Create a spectra comparison figure for one prediction sample.

        Args:
            sample: Prediction sample containing ``prediction`` and ``target`` spectra. Values are
                flattened to one-dimensional arrays with shape ``(N,)``.
            col_scalars: Collector scalar values aligned with ``sample``.
            subtitle: Subtitle text describing prediction and selector context.

        Returns:
            Matplotlib figure with spectra and residual panels.
        """
        pred = np.asarray(sample["prediction"]).ravel()
        target = np.asarray(sample["target"]).ravel()
        residual = pred - target
        x = np.arange(len(pred))
        sample_id = sample.get("sample_id", "?")

        fig, (ax_spec, ax_res) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(10, 5),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        ax_spec.plot(x, target, label="Target", linewidth=2.0, color="#019cd8")
        ax_spec.plot(x, pred, label="Prediction", linewidth=2.0, color="#005186", linestyle="--")
        ax_spec.set_ylabel("Intensity")
        ax_spec.set_title(f"Sample: {sample_id}")
        ax_spec.legend(fontsize=10, loc="upper right")

        scalars: dict[str, ScalarValue] = {}
        for key, value in sample.items():
            if key not in ("prediction", "target", "sample_id") and is_scalar(value):
                scalars[key] = cast(ScalarValue, value)
        for key, value in col_scalars.items():
            if key != "sample_id" and is_scalar(value):
                scalars[key] = cast(ScalarValue, value)

        if scalars:
            text = "\n".join(f"{k}: {v:.4g}" for k, v in scalars.items())
            ax_spec.text(
                0.01,
                0.97,
                text,
                transform=ax_spec.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        ax_res.plot(x, residual, color="#e85651", linewidth=2.0)
        ax_res.axhline(0, color="black", linewidth=1.0, linestyle=":")
        ax_res.set_xlabel("Channel")
        ax_res.set_ylabel("Residual")

        fig.text(0.5, -0.01, subtitle, ha="center", va="top", fontsize=7, color="gray")
        fig.tight_layout()
        return fig


def _subtitle(sel_cfg: dict[str, Any], reader_idx: int) -> str:
    """Build the spectra plot subtitle for one prediction-reader/selector pair.

    Args:
        sel_cfg: Selector configuration dictionary for this selector index.
        reader_idx: Zero-based prediction reader index.

    Returns:
        Human-readable subtitle string.
    """
    parts = [f"predictions={reader_idx}"]
    sel_type = sel_cfg.get("selector_type", "?")
    parts.append(f"selector={sel_type}")
    extras = {k: v for k, v in sel_cfg.items() if k != "selector_type"}
    if extras:
        parts.append(" ".join(f"{k}={v}" for k, v in extras.items()))
    return "  |  ".join(parts)
