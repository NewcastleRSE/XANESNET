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

"""Rendering of atomic structures into a constant-size drawing frame."""

from collections.abc import Iterable
from typing import Any, cast

import numpy as np
from ase.data import chemical_symbols
from ase.io.utils import PlottingVariables
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from pymatgen.core import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .formatting import format_decimal
from .style import PlotStyle, add_figure_legend, add_legend

_TARGET_SITE_RADIUS: float = 0.75
_DEFAULT_SITE_RADIUS: float = 0.45
_RING_COLOR: str = "#111111"
_STRUCTURE_FRAME_HALF: float = 0.5
_STRUCTURE_FILL: float = 0.85
_SCALE_BAR_LENGTH: float = 0.25
_RING_SCALE: float = 1.6
_OUTSIDE_LEGEND_ANCHOR = (1.06, 0.75)


def draw_structure(
    ax: Axes,
    structure: Molecule | Structure,
    style: PlotStyle,
    legend_role: str = "legend",
    target_site_index: int | None = None,
    frame_ratio: float = 1.0,
    show_scale: bool = True,
    show_legend: bool = True,
) -> None:
    """Draw one structure as a ball projection into a fixed-size frame.

    A compact legend lists the element colors (plus the target-site marker
    when present). When ``target_site_index`` is given, that atom is enlarged
    and circled with a bold ring. A scale bar at the bottom right shows the
    physical size of the frame in Angstrom.

    Args:
        ax: Matplotlib axis to draw on.
        structure: Pymatgen structure or molecule.
        style: Rendering style for the structure figure.
        legend_role: Font-size role used for the element legend.
        target_site_index: Index of the target site within ``structure``, or
            ``None`` when unknown.
        frame_ratio: Height/width ratio of the drawing frame. ``1.0`` draws a
            square frame (and enforces equal aspect); other values fill a
            rectangular frame with that ratio while keeping the structure
            undistorted.
        show_scale: Whether to draw the Angstrom scale bar at the bottom
            right of the frame.
        show_legend: Whether to draw the element and target-site legend.
    """
    atoms, positions, colors, atom_radii, highlight = _project_structure(structure, target_site_index)
    fit_radii = atom_radii.copy()
    if highlight:
        assert target_site_index is not None
        fit_radii[target_site_index] *= _RING_SCALE
    pos, scale = _fit_frame(positions, fit_radii, frame_ratio)
    radii_scaled = atom_radii * scale
    for (x, y), radius, color in zip(pos, radii_scaled, colors):
        ax.add_patch(
            Circle(
                (float(x), float(y)),
                float(radius),
                facecolor=tuple(float(v) for v in color),
                edgecolor="black",
                linewidth=style.linewidth("atom"),
                zorder=2,
            )
        )
    if highlight:
        assert target_site_index is not None
        _add_target_site_ring(
            ax,
            pos[target_site_index],
            float(radii_scaled[target_site_index] * _RING_SCALE),
            style,
        )
    _frame_axes(ax, frame_ratio, style)
    if show_legend:
        _add_element_legend(ax, atoms, colors, highlight, style, legend_role)
    if show_scale:
        _add_scale_bar(ax, scale, frame_ratio, style)


def _project_structure(
    structure: Molecule | Structure,
    target_site_index: int | None,
) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Project one structure and return the data shared by drawing and legends.

    Args:
        structure: Pymatgen structure or molecule.
        target_site_index: Index of the target site, or ``None`` when unknown.

    Returns:
        ASE atoms, projected positions, per-atom colors, projected radii, and
        a flag indicating whether the target site is valid.
    """
    atoms: Any = AseAtomsAdaptor().get_atoms(structure)
    radii, highlight = _site_style(len(atoms), target_site_index)
    writer = PlottingVariables(atoms, rotation="10x,20y,0z", radii=radii, scale=1, show_unit_cell=0)
    positions = np.asarray(writer.positions[:, :2], dtype=float)
    colors = np.asarray(writer.colors, dtype=float)
    atom_radii = np.asarray(writer.d, dtype=float) / 2.0
    return atoms, positions, colors, atom_radii, highlight


def _frame_axes(ax: Axes, frame_ratio: float, style: PlotStyle) -> None:
    """Set the constant drawing frame and its visible box on one axes.

    Args:
        ax: Matplotlib axis holding the structure.
        frame_ratio: Height/width ratio of the drawing frame.
        style: Rendering style for the structure figure.
    """
    half = _STRUCTURE_FRAME_HALF
    ax.set_xlim(-half, half)
    ax.set_ylim(-half * frame_ratio, half * frame_ratio)
    if frame_ratio == 1.0:
        ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(style.linewidth("spine"))
        spine.set_color("black")


def _site_style(n_atoms: int, target_site_index: int | None) -> tuple[list[float], bool]:
    """Build per-atom radii and highlight flags for one structure.

    Args:
        n_atoms: Number of atoms in the structure.
        target_site_index: Index of the target site, or ``None`` when unknown.

    Returns:
        ``(radii, highlight)`` where ``highlight`` is ``True`` when the
        target site index is valid and should be marked.
    """
    radii = [_DEFAULT_SITE_RADIUS] * n_atoms
    highlight = target_site_index is not None and 0 <= target_site_index < n_atoms
    if highlight:
        assert target_site_index is not None
        radii[target_site_index] = _TARGET_SITE_RADIUS
    return radii, highlight


def _fit_frame(positions: np.ndarray, radii: np.ndarray, frame_ratio: float = 1.0) -> tuple[np.ndarray, float]:
    """Center and scale projected positions into a constant drawing frame.

    Args:
        positions: Projected atom positions with shape ``(N, 2)``.
        radii: Per-atom radii, including any ring allowance, with shape ``(N,)``.
        frame_ratio: Height/width ratio of the frame to fill.

    Returns:
        ``(scaled_positions, scale)`` where the union of atom extents fills a
        fixed fraction of the frame, preserving the aspect ratio.
    """
    lo = (positions - radii[:, None]).min(axis=0)
    hi = (positions + radii[:, None]).max(axis=0)
    center = (lo + hi) / 2.0
    span = hi - lo
    frame_w = 2.0 * _STRUCTURE_FRAME_HALF * _STRUCTURE_FILL
    frame_h = frame_w * frame_ratio
    scale_w = frame_w / span[0] if span[0] > 0 else np.inf
    scale_h = frame_h / span[1] if span[1] > 0 else np.inf
    scale = float(min(scale_w, scale_h))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return (positions - center) * scale, scale


def _add_scale_bar(ax: Axes, scale: float, frame_ratio: float, style: PlotStyle) -> None:
    """Draw a length annotation in Angstrom at the bottom right of a frame.

    Projected positions are in Angstrom and scaled into the fixed frame by
    ``scale``, so one frame unit corresponds to ``1 / scale`` Angstrom. The
    bar spans a fixed fraction of the frame width and is labelled with the
    corresponding physical length. Both sit in the bottom margin that the
    frame fit leaves empty, so they never cover an atom.

    Args:
        ax: Matplotlib axis holding the structure.
        scale: Frame units per Angstrom returned by :func:`_fit_frame`.
        frame_ratio: Height/width ratio of the frame.
        style: Rendering style for the structure figure.
    """
    if not np.isfinite(scale) or scale <= 0.0:
        return
    label = f"{format_decimal(_SCALE_BAR_LENGTH / scale, 3)} Angstrom"
    half = _STRUCTURE_FRAME_HALF
    x1 = half - 0.05
    y = -half * frame_ratio + (1.0 - _STRUCTURE_FILL) * half * frame_ratio / 2.0
    ax.plot(
        [x1 - _SCALE_BAR_LENGTH, x1],
        [y, y],
        color="black",
        linewidth=style.linewidth("scale_bar"),
        clip_on=False,
        zorder=20,
    )
    ax.text(
        x1 - _SCALE_BAR_LENGTH - 0.02,
        y,
        label,
        ha="right",
        va="center",
        fontsize=style.fontsize("annotation"),
        zorder=20,
    )


def _add_element_legend(
    ax: Axes,
    atoms: Any,
    colors: np.ndarray,
    has_target: bool,
    style: PlotStyle,
    legend_role: str,
) -> None:
    """Add a compact legend explaining the element colors.

    Args:
        ax: Matplotlib axis holding the structure.
        atoms: ASE ``Atoms`` object whose elements should be listed.
        colors: Per-atom RGB color triples with shape ``(N, 3)``.
        has_target: Whether to append a target-site marker entry.
        style: Rendering style for the structure figure.
        legend_role: Font-size role used for the element legend.
    """
    handles, labels = _legend_entries(atoms, colors, has_target, style, legend_role, set())
    if handles:
        add_legend(ax, style, handles=handles, labels=labels, loc="upper left", framealpha=0.8)


def add_structure_legend(
    fig: Figure,
    structures: Iterable[tuple[Molecule | Structure, int | None]],
    style: PlotStyle,
    *,
    legend_role: str = "legend",
    outside_anchor: tuple[float, float] = _OUTSIDE_LEGEND_ANCHOR,
    outside_loc: str = "upper left",
) -> Legend | None:
    """Add one external legend for the unique elements in several structures.

    The target-site marker is included once when at least one structure has a
    valid target site. The legend is placed below the regular external
    spectra legend so both can coexist on composite figures.

    Args:
        fig: Matplotlib figure receiving the legend.
        structures: Structures paired with their optional target-site indices.
        style: Rendering style for the structure figure.
        legend_role: Font-size role used for the legend markers.
        outside_anchor: Figure-relative anchor for the external legend.
        outside_loc: Matplotlib location used with ``outside_anchor``.

    Returns:
        The created legend, or ``None`` when no structures were supplied.
    """
    handles: list[Line2D] = []
    labels: list[str] = []
    seen: set[str] = set()
    has_target = False
    for structure, target_site_index in structures:
        atoms, _, colors, _, highlight = _project_structure(structure, target_site_index)
        has_target = has_target or highlight
        structure_handles, structure_labels = _legend_entries(
            atoms,
            colors,
            False,
            style,
            legend_role,
            seen,
        )
        handles.extend(structure_handles)
        labels.extend(structure_labels)
    if has_target:
        handles.append(_target_site_handle(markersize=1.2 * style.fontsize(legend_role)))
        labels.append("X")
    if not handles:
        return None
    return add_figure_legend(
        fig,
        handles,
        labels,
        style,
        position="outside",
        loc=outside_loc,
        bbox_to_anchor=outside_anchor,
    )


def _legend_entries(
    atoms: Any,
    colors: np.ndarray,
    has_target: bool,
    style: PlotStyle,
    legend_role: str,
    seen: set[str],
) -> tuple[list[Line2D], list[str]]:
    """Build unique element and target-site legend entries."""
    handles: list[Line2D] = []
    labels: list[str] = []
    for number, color in zip(atoms.get_atomic_numbers(), colors):
        symbol = chemical_symbols[int(number)]
        if symbol in seen:
            continue
        seen.add(symbol)
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cast(tuple[float, float, float], tuple(float(v) for v in color[:3])),
                markeredgecolor="black",
                markersize=0.9 * style.fontsize(legend_role),
                linestyle="",
            )
        )
        labels.append(symbol)
    if has_target and "X" not in seen:
        seen.add("X")
        handles.append(_target_site_handle(markersize=1.2 * style.fontsize(legend_role)))
        labels.append("X")
    return handles, labels


def _add_target_site_ring(ax: Axes, xy: np.ndarray, ring_radius: float, style: PlotStyle) -> None:
    """Draw a bold ring around the rendered target-site atom.

    Args:
        ax: Matplotlib axis holding the structure.
        xy: Projected image-plane position of the target site.
        ring_radius: Radius of the ring in frame units.
        style: Rendering style for the structure figure.
    """
    ring = Circle(
        (float(xy[0]), float(xy[1])),
        radius=ring_radius,
        fill=False,
        edgecolor=_RING_COLOR,
        linewidth=style.linewidth("target_ring"),
        zorder=10,
    )
    ax.add_patch(ring)


def _target_site_handle(markersize: float) -> Line2D:
    """Build the open-ring legend proxy for the target-site marker.

    Args:
        markersize: Marker size of the proxy.

    Returns:
        Legend handle matching the drawn target-site ring.
    """
    return Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="none",
        markeredgecolor=_RING_COLOR,
        markersize=markersize,
        linestyle="",
    )
