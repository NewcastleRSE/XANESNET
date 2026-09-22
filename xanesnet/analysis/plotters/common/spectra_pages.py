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

"""Single-sample spectra pages."""

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xanesnet.serialization.prediction_readers import PredictionSample

from ...utils import one_line_label, sample_label
from .formatting import shorten_label
from .layout import pad_figure
from .structures import add_structure_legend, draw_structure
from .style import (
    COLOR_ACCENT_RED,
    COLOR_PREDICTION,
    COLOR_TARGET,
    PlotStyle,
    add_legend,
    method_color,
    set_context_title,
    style_axis,
)

_STRUCTURE_PAGE_ADJUST = dict(left=0.04, right=0.99, top=0.86, bottom=0.10, wspace=0.10, hspace=0.16)
_SPECTRA_PAGE_ADJUST = dict(left=0.08, right=0.98, top=0.86, bottom=0.16, hspace=0.25)


def spectra_page_figure(
    sample: PredictionSample,
    subtitle: str,
    style: PlotStyle,
    legend_position: str = "inside",
) -> Figure:
    """Create a spectra comparison figure for one prediction sample.

    Args:
        sample: Prediction sample containing prediction and target spectra.
        subtitle: Context text identifying the prediction and selector.
        style: Rendering style controlling figure geometry and typography.
        legend_position: Whether the spectra legend is inside the axes or
            outside it on the right.

    Returns:
        Figure with spectra and residual panels.
    """
    fig, ax_spec, ax_res = _create_spectra_page_axes(style, legend_position)

    _draw_sample_page(sample, ax_spec, ax_res, style, legend_position)
    _finish_page(fig, ax_spec, subtitle, sample_label(sample), style, has_structure=False)
    return fig


def spectra_structure_page_figure(
    sample: PredictionSample,
    subtitle: str,
    style: PlotStyle,
    legend_position: str = "inside",
) -> Figure:
    """Create a spectra comparison figure with the matched structure alongside.

    Args:
        sample: Prediction sample containing spectra and an optional structure.
        subtitle: Context text identifying the prediction and selector.
        style: Rendering style controlling figure geometry and typography.
        legend_position: Whether the spectra legend is inside the axes or
            outside it on the right.

    Returns:
        Figure with structure, spectra, and residual panels.
    """
    fig, ax_struct, ax_spec, ax_res = _create_structure_page_axes(
        style,
        legend_position,
    )

    _draw_sample_page(sample, ax_spec, ax_res, style, legend_position)
    _finish_page(
        fig,
        ax_spec,
        subtitle,
        sample_label(sample),
        style,
        has_structure=True,
    )
    _draw_structure_panel(
        fig,
        ax_struct,
        sample.get("structure"),
        style,
        sample.get("target_site_index"),
        legend_position,
    )

    return fig


def combined_spectra_page_figure(
    sample: PredictionSample,
    method_labels: list[str],
    predictions: list[np.ndarray],
    target: np.ndarray,
    subtitle: str,
    style: PlotStyle,
    legend_position: str = "inside",
) -> Figure:
    """Create a spectra figure overlaying several methods on one sample.

    Args:
        sample: Reference prediction sample containing the target and structure.
        method_labels: Display label for each prediction series.
        predictions: Prediction spectra in the same order as ``method_labels``.
        target: Target spectrum shared by all methods.
        subtitle: Context text identifying the prediction set and selector.
        style: Rendering style controlling figure geometry and typography.
        legend_position: Whether the spectra legend is inside the axes or
            outside it on the right.

    Returns:
        Figure with overlaid spectra and residual panels.
    """
    sample_id = str(sample["sample_id"])
    structure = sample.get("structure")

    if structure is not None:
        fig, ax_struct, ax_spec, ax_res = _create_structure_page_axes(
            style,
            legend_position,
        )
    else:
        fig, ax_spec, ax_res = _create_spectra_page_axes(style, legend_position)
        ax_struct = None

    _draw_combined_spectra_panels(
        ax_spec,
        ax_res,
        method_labels,
        predictions,
        target,
        style,
        legend_position,
    )
    _finish_page(
        fig,
        ax_spec,
        subtitle,
        sample_id,
        style,
        has_structure=ax_struct is not None,
    )
    if ax_struct is not None:
        _draw_structure_panel(
            fig,
            ax_struct,
            structure,
            style,
            sample.get("target_site_index"),
            legend_position,
        )

    return fig


def _create_spectra_page_axes(
    style: PlotStyle,
    legend_position: str,
) -> tuple[Figure, Axes, Axes]:
    """Create the spectra and residual axes for a regular page."""
    fig, (ax_spec, ax_res) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=style.figure_size(
            "spectra",
            width_margin=style.legend_margin(legend_position),
        ),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    return fig, ax_spec, ax_res


def _create_structure_page_axes(
    style: PlotStyle,
    legend_position: str,
) -> tuple[Figure, Axes, Axes, Axes]:
    """Create the structure, spectra, and residual axes for a composite page."""
    fig, axes = plt.subplot_mosaic(
        [["structure", "spectra"], ["structure", "residual"]],
        figsize=style.figure_size(
            "structure_page",
            width_margin=style.legend_margin(legend_position),
        ),
        width_ratios=[1.0, 2.4],
        height_ratios=[3, 1],
        gridspec_kw={
            "wspace": _STRUCTURE_PAGE_ADJUST["wspace"],
            "hspace": _STRUCTURE_PAGE_ADJUST["hspace"],
        },
    )
    ax_struct = axes["structure"]
    ax_spec = axes["spectra"]
    ax_res = axes["residual"]
    ax_res.sharex(ax_spec)
    # The residual panel is the shared x-axis owner. Keep energy tick labels
    # off the upper spectrum panel to avoid displaying the same values twice.
    ax_spec.tick_params(axis="x", labelbottom=False)
    return fig, ax_struct, ax_spec, ax_res


def _finish_page(
    fig: Figure,
    ax_spec: Axes,
    subtitle: str,
    sample_text: str,
    style: PlotStyle,
    *,
    has_structure: bool,
) -> None:
    """Apply the shared layout, padding, and context title to a page."""
    fig.subplots_adjust(
        **(_STRUCTURE_PAGE_ADJUST if has_structure else _SPECTRA_PAGE_ADJUST),
    )
    setattr(fig, "_xanesnet_fixed_layout", True)
    pad_figure(fig, style)
    set_context_title(
        ax_spec,
        one_line_label([subtitle, f"Sample: {sample_text}"]),
        style,
        location="left",
    )


def _draw_structure_panel(
    fig: Figure,
    ax_struct: Axes,
    structure: Any,
    style: PlotStyle,
    target_site_index: int | None,
    legend_position: str,
) -> None:
    """Draw a structure into the measured composite-page frame."""
    if structure is None:
        ax_struct.axis("off")
        return

    fig.canvas.draw()
    bbox = ax_struct.get_position()
    frame_ratio = (bbox.height * fig.get_figheight()) / (bbox.width * fig.get_figwidth())
    draw_structure(
        ax_struct,
        structure,
        style,
        target_site_index=target_site_index,
        frame_ratio=frame_ratio,
        legend_role="legend",
        show_legend=legend_position == "inside",
    )
    if legend_position == "outside":
        add_structure_legend(
            fig,
            [(structure, target_site_index)],
            style,
            outside_anchor=(1.06, 0.55),
        )
    ax_struct.set_title(
        "Structure",
        loc="left",
        fontsize=style.fontsize("title"),
        pad=style.spacing("title_pad"),
    )


def _draw_combined_spectra_panels(
    ax_spec: Axes,
    ax_res: Axes,
    method_labels: list[str],
    predictions: list[np.ndarray],
    target: np.ndarray,
    style: PlotStyle,
    legend_position: str,
) -> None:
    """Draw target, predictions, residuals, and legend.

    Args:
        ax_spec: Spectra axes.
        ax_res: Residual axes.
        method_labels: Display label for each prediction series.
        predictions: Prediction spectra in the same order as ``method_labels``.
        target: Target spectrum shared by all methods.
        style: Rendering style controlling line and legend appearance.
        legend_position: Whether the spectra legend is inside the axes or
            outside it on the right.
    """
    x = np.arange(len(target))
    ax_spec.plot(x, target, label="Target", linewidth=style.linewidth("main"), color=COLOR_TARGET)
    for index, (label, pred) in enumerate(zip(method_labels, predictions)):
        color = method_color(index)
        ax_spec.plot(
            x,
            pred,
            label=shorten_label([label], width=24),
            linewidth=style.linewidth("secondary"),
            linestyle="--",
            color=color,
        )
        ax_res.plot(x, pred - target, linewidth=style.linewidth("secondary"), color=color)

    ax_res.axhline(0, color="black", linewidth=style.linewidth("reference"), linestyle=":")
    ax_res.set_xlabel("Energy")
    add_legend(ax_spec, style, position=legend_position)
    style_axis(ax_spec, style)
    style_axis(ax_res, style)


def _draw_sample_page(
    sample: PredictionSample,
    ax_spec: Axes,
    ax_res: Axes,
    style: PlotStyle,
    legend_position: str,
) -> None:
    """Draw target, prediction, residual, and sample title.

    Args:
        sample: Prediction sample containing prediction and target spectra.
        ax_spec: Spectra axes.
        ax_res: Residual axes.
        style: Rendering style controlling line appearance.
        legend_position: Whether the spectra legend is inside the axes or
            outside it on the right.
    """
    pred = np.asarray(sample["prediction"]).ravel()
    target = np.asarray(sample["target"]).ravel()
    pred_std_value = sample.get("prediction_std")
    pred_std = np.asarray(pred_std_value).ravel() if pred_std_value is not None else None

    _draw_spectra_panels(ax_spec, ax_res, pred, target, pred_std, style, legend_position)


def _draw_spectra_panels(
    ax_spec: Axes,
    ax_res: Axes,
    pred: np.ndarray,
    target: np.ndarray,
    pred_std: np.ndarray | None,
    style: PlotStyle,
    legend_position: str,
) -> None:
    """Draw target, prediction, uncertainty, residual, and legend.

    Args:
        ax_spec: Spectra axes.
        ax_res: Residual axes.
        pred: Prediction spectrum.
        target: Target spectrum.
        pred_std: Optional prediction standard-deviation spectrum.
        style: Rendering style controlling line and legend appearance.
        legend_position: Whether the spectra legend is inside the axes or
            outside it on the right.
    """
    x = np.arange(len(pred))
    ax_spec.plot(x, target, label="Target", linewidth=style.linewidth("main"), color=COLOR_TARGET)
    if pred_std is not None:
        if pred_std.shape == pred.shape:
            ax_spec.fill_between(
                x,
                pred - pred_std,
                pred + pred_std,
                label="Prediction +/- 1 std",
                color=COLOR_PREDICTION,
                alpha=0.18,
                linewidth=0,
            )
        else:
            logging.warning(
                "Skipping prediction_std shading because shape %s does not match prediction shape %s.",
                pred_std.shape,
                pred.shape,
            )
    ax_spec.plot(x, pred, label="Prediction", linewidth=style.linewidth("main"), color=COLOR_PREDICTION, linestyle="--")
    add_legend(ax_spec, style, position=legend_position)
    ax_res.plot(x, pred - target, color=COLOR_ACCENT_RED, linewidth=style.linewidth("main"))
    ax_res.axhline(0, color="black", linewidth=style.linewidth("reference"), linestyle=":")
    ax_res.set_xlabel("Energy")
    style_axis(ax_spec, style)
    style_axis(ax_res, style)
