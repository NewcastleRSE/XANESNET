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

"""Visualize all spectra encodings built from a real dataset via a training config.

Loads the datasource and dataset described in a YAML training config, resolves
any ``auto`` encoding fields from the training split, then applies each
configured encoding to a sample of spectra and produces a rich visualization.

Every encoding produces four base panels:

1. Raw spectra (per-element coloring when element info is available).
2. Encoded representation -- line plot when the encoding preserves the
   spectrum length, or a heat-map / bar chart otherwise.
3. Round-trip: raw vs. decode(encode(x)) overlaid.
4. Point-wise reconstruction error |decode(encode(x)) - x|.

Encoding-specific panels are appended automatically:

* **z_score** -- mean and standard-deviation curves.
* **min_max** -- minimum- and maximum-value curves.
* **scale** -- per-point factor curve.
* **gaussian** -- individual Gaussian basis contributions below the spectrum,
  showing how each basis function participates in the reconstruction.
* **subtract_average** -- per-element average spectra (or the single global
  average when ``per_element`` is false).

Usage::

    python scripts/encodings_tester.py --config configs/in_mlp.yaml \\
        [--max-samples 8] [--save my_plot.pdf] [--no-show]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xanesnet.datasets import DatasetRegistry
from xanesnet.datasources import DataSourceRegistry
from xanesnet.encodings import (
    AffineEncoding,
    CombinedEncoding,
    GaussianEncoding,
    SpectraEncoding,
    SubtractAverageEncoding,
)
from xanesnet.serialization.auto_config import resolve_auto_encoding_config
from xanesnet.serialization.config import Config, load_raw_config
from xanesnet.serialization.schema_validation import validate_config_schema
from xanesnet.utils.exceptions import ConfigError

###############################################################################
# CLI
###############################################################################


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the encoding visualizer.

    Returns:
        Populated argument namespace.
    """
    p = argparse.ArgumentParser(description="Visualize spectrum encodings on real data.")
    p.add_argument(
        "--config",
        default="configs/in_mlp.yaml",
        help="Path to a training YAML config (default: configs/in_mlp.yaml).",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=8,
        help="Maximum number of spectra to draw per plot (default: 8).",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override datasource.json_path in the config (useful for quick tests).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="Do not call plt.show() (useful for headless runs).",
    )
    p.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="If given, save the figure to this file (e.g. encodings.pdf).",
    )
    return p.parse_args()


###############################################################################
# Pipeline helpers
###############################################################################


def _build_datasource_and_dataset(config: Config) -> tuple[Any, Any]:
    """Instantiate, prepare, and split the dataset from *config*.

    Args:
        config: Validated training config (already passed through
            ``validate_config_train``; all schema defaults have been applied).

    Returns:
        ``(datasource, dataset)`` pair ready for use.
    """
    # --- datasource ---
    ds_config = config.section("datasource")
    ds_type = ds_config.get_str("datasource_type")
    datasource = DataSourceRegistry.create(ds_type, **ds_config.as_kwargs())

    # --- dataset ---
    dataset_config = config.section("dataset")
    dataset_type = dataset_config.get_str("dataset_type")
    dataset = DatasetRegistry.create(dataset_type, **dataset_config.as_kwargs(), datasource=datasource)
    dataset.prepare()
    dataset.setup_splits()
    dataset.check_preload()
    return datasource, dataset


def _collect_spectra(dataset: Any, max_samples: int) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Collect up to *max_samples* target spectra from the training split.

    Args:
        dataset: Prepared dataset with optional ``train_subset``.
        max_samples: Upper bound on the number of spectra returned.

    Returns:
        A pair ``(spectra, elements)`` where ``spectra`` is a float32 tensor
        ``(N, L)`` of raw target spectra and ``elements`` is an int64 tensor
        ``(N,)`` of target-site atomic numbers, or ``None`` when the dataset
        carries no element information.
    """
    subset = dataset.train_subset
    indices = list(subset.indices) if subset is not None else list(range(len(dataset)))
    indices = indices[:max_samples]

    spectra_list: list[torch.Tensor] = []
    element_list: list[torch.Tensor] = []
    have_elements = True
    for idx in indices:
        sample = dataset[idx]
        batch = dataset.collate_fn([sample])
        y = batch.y
        if y is not None:
            spectra_list.append(y)
            element = getattr(batch, "element", None)
            if element is None:
                have_elements = False
            else:
                element_list.append(element)

    if not spectra_list:
        raise RuntimeError("Could not collect any spectra from the dataset.")

    spectra = torch.cat(spectra_list, dim=0).float()  # (N, L)
    elements = torch.cat(element_list, dim=0) if have_elements and element_list else None
    return spectra, elements


###############################################################################
# Element utilities
###############################################################################

#: Atomic-number -> IUPAC symbol lookup for target-site elements.
_ELEMENT_SYMBOLS: dict[int, str] = {
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    40: "Zr",
    42: "Mo",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
}

#: Qualitative color palette for up to 10 distinct elements.
_ELEMENT_COLORS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def _element_symbol(z: int) -> str:
    """Return the IUPAC symbol for atomic number *z*, or ``"Z=..."`` if unknown.

    Args:
        z: Atomic number.

    Returns:
        Human-readable element label.
    """
    return _ELEMENT_SYMBOLS.get(z, f"Z={z}")


def _build_element_groups(
    elements: np.ndarray,
) -> tuple[list[int], dict[int, list[int]], dict[int, str], dict[int, str]]:
    """Group sample indices by unique atomic number.

    Args:
        elements: ``(N,)`` int array of atomic numbers.

    Returns:
        Tuple ``(unique_z, groups, labels, colors)`` where:

        * ``unique_z`` -- sorted list of distinct atomic numbers present.
        * ``groups`` -- ``{z: [sample_indices]}`` mapping.
        * ``labels`` -- ``{z: "Symbol"}`` legend labels.
        * ``colors`` -- ``{z: "#hex"}`` per-element colors.
    """
    unique_z = sorted(set(elements.tolist()))
    groups: dict[int, list[int]] = {z: [] for z in unique_z}
    for i, z in enumerate(elements):
        groups[int(z)].append(i)

    labels: dict[int, str] = {}
    colors: dict[int, str] = {}
    for j, z in enumerate(unique_z):
        labels[z] = _element_symbol(z)
        colors[z] = _ELEMENT_COLORS[j % len(_ELEMENT_COLORS)]

    return unique_z, groups, labels, colors


###############################################################################
# Color / style constants
###############################################################################

_ALPHA = 0.55
_COLOR_RAW = "#2c7bb6"
_COLOR_DECODED = "#d7191c"
_COLOR_ERROR = "#fdae61"
_COLOR_MEAN_ERROR = "#000000"
_COLOR_BASIS_GRID = "#a6cee3"

#: Per-width hues for Gaussian basis contributions.
_GAUSS_WIDTH_COLORS: list[str] = [
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
]


def _energy_axis(n_points: int) -> np.ndarray:
    """Return a 0-indexed grid for the x-axis.

    Args:
        n_points: Number of spectral grid points.

    Returns:
        Float32 array ``(n_points,)``.
    """
    return np.arange(n_points, dtype=np.float32)


###############################################################################
# Low-level plot functions
###############################################################################


def _plot_raw(
    ax: Axes,
    spectra: np.ndarray,
    elements: np.ndarray | None = None,
    title: str = "Raw spectra",
) -> None:
    """Overlay raw spectra on *ax*, optionally coloring by element.

    Args:
        ax: Matplotlib axes to draw on.
        spectra: ``(N, L)`` raw intensity array.
        elements: Optional ``(N,)`` atomic-number array for per-element
            coloring.
        title: Panel title.
    """
    energy = _energy_axis(spectra.shape[1])
    if elements is not None:
        _, groups, labels, colors = _build_element_groups(elements)
        for z, idxs in groups.items():
            for i in idxs:
                ax.plot(
                    energy,
                    spectra[i],
                    color=colors[z],
                    alpha=_ALPHA,
                    linewidth=0.8,
                    label=labels[z] if i == idxs[0] else None,
                )
        ax.legend(fontsize=6, loc="upper right", ncol=2)
    else:
        for row in spectra:
            ax.plot(energy, row, color=_COLOR_RAW, alpha=_ALPHA, linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("Energy grid point")
    ax.set_ylabel("Intensity")


def _plot_encoded(
    ax: Axes,
    encoded: np.ndarray,
    encoding_name: str,
    n_points: int | None = None,
    elements: np.ndarray | None = None,
) -> None:
    """Plot the encoded representation on *ax*.

    When the encoded dimension matches the original spectrum length
    (``k == n_points``) the coefficients are drawn as overlaid line spectra
    with per-element coloring.  Otherwise a heat-map (or bar chart for a
    single sample) is used.

    Args:
        ax: Matplotlib axes to draw on.
        encoded: ``(N, K)`` encoded coefficients.
        encoding_name: Human-readable encoding label.
        n_points: Original spectrum length; when equal to *K* a line-plot
            is produced instead of a heat-map.
        elements: Optional ``(N,)`` atomic-number array for per-element
            coloring or y-tick labels.
    """
    n_samples, k = encoded.shape
    title = f"Encoded ({encoding_name})  dim={k}"

    # -- line-plot mode (same length as input spectrum) --
    if n_points is not None and k == n_points:
        energy = _energy_axis(k)
        if elements is not None:
            _, groups, labels, colors = _build_element_groups(elements)
            for z, idxs in groups.items():
                for i in idxs:
                    ax.plot(
                        energy,
                        encoded[i],
                        color=colors[z],
                        alpha=_ALPHA,
                        linewidth=0.8,
                        label=labels[z] if i == idxs[0] else None,
                    )
            ax.legend(fontsize=6, loc="upper right", ncol=2)
        else:
            for row in encoded:
                ax.plot(energy, row, color=_COLOR_RAW, alpha=_ALPHA, linewidth=0.8)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Energy grid point")
        ax.set_ylabel("Encoded value")
        return

    # -- heat-map / bar-chart mode --
    if n_samples == 1:
        ax.bar(np.arange(k), encoded[0], color=_COLOR_BASIS_GRID, edgecolor="none")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Coefficient index")
        ax.set_ylabel("Value")
    else:
        vmax = float(np.abs(encoded).max() or 1.0)
        ax.imshow(
            encoded,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Coefficient index")
        if elements is not None:
            tick_positions = np.arange(n_samples)
            tick_labels = [_element_symbol(int(z)) for z in elements]
            ax.set_yticks(tick_positions)
            ax.set_yticklabels(tick_labels, fontsize=6)
        ax.set_ylabel("Sample index")


def _plot_roundtrip(
    ax: Axes,
    spectra: np.ndarray,
    decoded: np.ndarray,
    encoding_name: str,
    elements: np.ndarray | None = None,
) -> None:
    """Overlay raw and decoded spectra, optionally per-element.

    Args:
        ax: Matplotlib axes to draw on.
        spectra: ``(N, L)`` raw intensities.
        decoded: ``(N, L)`` reconstructed intensities.
        encoding_name: Human-readable encoding label.
        elements: Optional ``(N,)`` atomic-number array for per-element
            coloring.
    """
    energy = _energy_axis(spectra.shape[1])
    if elements is not None:
        _, groups, labels, colors = _build_element_groups(elements)
        for z, idxs in groups.items():
            for i in idxs:
                ax.plot(
                    energy,
                    spectra[i],
                    color=colors[z],
                    alpha=_ALPHA,
                    linewidth=0.8,
                    linestyle="-",
                    label=f"{labels[z]} raw" if i == idxs[0] else None,
                )
                ax.plot(
                    energy,
                    decoded[i],
                    color=colors[z],
                    alpha=_ALPHA,
                    linewidth=0.8,
                    linestyle="--",
                    label=f"{labels[z]} dec" if i == idxs[0] else None,
                )
        ax.legend(fontsize=5, loc="upper right", ncol=2)
    else:
        for i, (raw, dec) in enumerate(zip(spectra, decoded)):
            ax.plot(
                energy,
                raw,
                color=_COLOR_RAW,
                alpha=_ALPHA,
                linewidth=0.8,
                label="Raw" if i == 0 else None,
            )
            ax.plot(
                energy,
                dec,
                color=_COLOR_DECODED,
                alpha=_ALPHA,
                linewidth=0.8,
                label="Decoded" if i == 0 else None,
            )
        ax.legend(fontsize=7, loc="upper right")
    ax.set_title(f"Round-trip ({encoding_name})", fontsize=9)
    ax.set_xlabel("Energy grid point")
    ax.set_ylabel("Intensity")


def _plot_error(
    ax: Axes,
    spectra: np.ndarray,
    decoded: np.ndarray,
    encoding_name: str,
    elements: np.ndarray | None = None,
) -> None:
    """Plot per-point absolute reconstruction error.

    Args:
        ax: Matplotlib axes to draw on.
        spectra: ``(N, L)`` raw intensities.
        decoded: ``(N, L)`` reconstructed intensities.
        encoding_name: Human-readable encoding label.
        elements: Optional ``(N,)`` atomic-number array for per-element
            coloring.
    """
    error = np.abs(decoded - spectra)
    energy = _energy_axis(spectra.shape[1])
    if elements is not None:
        _, groups, _, colors = _build_element_groups(elements)
        for z, idxs in groups.items():
            for i in idxs:
                ax.plot(
                    energy,
                    error[i],
                    color=colors[z],
                    alpha=_ALPHA * 0.7,
                    linewidth=0.5,
                )
    else:
        for row in error:
            ax.plot(energy, row, color=_COLOR_ERROR, alpha=_ALPHA * 0.8, linewidth=0.6)
    ax.plot(
        energy,
        error.mean(axis=0),
        color=_COLOR_MEAN_ERROR,
        linewidth=1.2,
        label="Mean error",
    )
    ax.set_title(f"|decode(encode(x)) - x|  ({encoding_name})", fontsize=9)
    ax.set_xlabel("Energy grid point")
    ax.set_ylabel("|Error|")
    ax.legend(fontsize=7, loc="upper right")


###############################################################################
# Encoding-specific panels
###############################################################################


def _plot_gaussian_decomposition(
    ax: Axes,
    enc: GaussianEncoding,
    spectrum: np.ndarray,
    sample_idx: int = 0,
) -> None:
    """Decompose one spectrum into its individual Gaussian basis contributions.

    Each basis function ``Phi[:, k]`` is multiplied by its fitted coefficient
    and drawn as a faint filled curve.  Contributions are grouped and colored
    by width.  The raw spectrum and the full reconstruction are overlaid as
    thicker lines.

    Args:
        ax: Matplotlib axes to draw on.
        enc: Configured Gaussian encoding.
        spectrum: ``(L,)`` raw spectrum to decompose (only the first row is
            used if multiple rows are passed).
        sample_idx: Index of the spectrum (used in the title).
    """
    if spectrum.ndim > 1:
        spectrum = spectrum[0]

    energy = _energy_axis(len(spectrum))
    basis_np = enc.basis.Phi.detach().cpu().numpy()  # (L, K)

    t = torch.tensor(spectrum[None, :], dtype=torch.float32)
    coeffs = enc.encode(t).detach().cpu().numpy()[0]  # (K,)

    n_widths = len(enc.widths)
    stride_count = (enc.num_points + enc.basis_stride - 1) // enc.basis_stride

    # Draw individual scaled basis contributions, grouped by width.
    for w_idx in range(n_widths):
        color = _GAUSS_WIDTH_COLORS[w_idx % len(_GAUSS_WIDTH_COLORS)]
        start_col = w_idx * stride_count
        end_col = (w_idx + 1) * stride_count
        for k in range(start_col, min(end_col, basis_np.shape[1])):
            contrib = coeffs[k] * basis_np[:, k]
            ax.fill_between(
                energy,
                0,
                contrib,
                color=to_rgba(color, 0.12),
                linewidth=0,
                label=f"width={enc.widths[w_idx]}" if k == start_col else None,
            )

    # Raw spectrum.
    ax.plot(
        energy,
        spectrum,
        color=_COLOR_RAW,
        linewidth=1.3,
        alpha=0.95,
        label="Raw",
    )

    # Full reconstruction.
    recon = enc.decode(torch.tensor(coeffs[None, :])).detach().cpu().numpy()[0]
    ax.plot(
        energy,
        recon,
        color=_COLOR_DECODED,
        linewidth=1.3,
        alpha=0.95,
        linestyle="--",
        label="Reconstructed",
    )

    ax.set_title(
        f"Gaussian decomposition  widths={enc.widths}  "
        f"stride={enc.basis_stride}  K={basis_np.shape[1]}  sample={sample_idx}",
        fontsize=9,
    )
    ax.set_xlabel("Energy grid point")
    ax.set_ylabel("Amplitude / contribution")
    ax.legend(fontsize=6, loc="upper right", ncol=2)


def _plot_subtract_average_panel(
    ax: Axes,
    enc: SubtractAverageEncoding,
    spectra: np.ndarray,
    elements: np.ndarray | None = None,
) -> None:
    """Show the average spectrum (or per-element averages) that are subtracted.

    Args:
        ax: Matplotlib axes to draw on.
        enc: Configured subtract_average encoding.
        spectra: ``(N, L)`` raw spectra for context (faint overlay).
        elements: Optional ``(N,)`` atomic-number array; used to label
            per-element averages when ``enc.per_element`` is true.
    """
    energy = _energy_axis(spectra.shape[1])
    shift_np = enc.shift.detach().cpu().numpy()  # (L,) or (E, L)

    # Faint raw spectra for context.
    for row in spectra[:8]:
        ax.plot(energy, row, color=_COLOR_RAW, alpha=0.12, linewidth=0.5)

    if enc.per_element and shift_np.ndim == 2:
        n_elem = shift_np.shape[0]
        elem_list = enc.elements if enc.elements is not None else list(range(n_elem))
        for i in range(n_elem):
            z = elem_list[i] if i < len(elem_list) else i
            label = _element_symbol(z)
            color = _ELEMENT_COLORS[i % len(_ELEMENT_COLORS)]
            ax.plot(
                energy,
                shift_np[i],
                color=color,
                linewidth=1.5,
                alpha=0.9,
                label=label,
            )
        ax.legend(fontsize=6, loc="upper right", ncol=2)
        ax.set_title("Per-element average spectra (subtract_average)", fontsize=9)
    else:
        avg = shift_np if shift_np.ndim == 1 else shift_np[0]
        ax.plot(
            energy,
            avg,
            color=_COLOR_DECODED,
            linewidth=1.5,
            alpha=0.9,
            label="Average",
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title("Average spectrum (subtract_average)", fontsize=9)
    ax.set_xlabel("Energy grid point")
    ax.set_ylabel("Intensity")


def _plot_affine_parameters(
    ax: Axes,
    enc: AffineEncoding,
    n_points: int,
) -> None:
    """Plot the shift and scale parameters of an affine encoding.

    For **z_score** this draws the mean (shift) and standard deviation
    (scale).  For **min_max** it draws the minimum (shift) and maximum
    (shift + scale).  For **scale** it draws the factor (scale).

    Per-element curves are drawn when ``enc.per_element`` is true.

    Args:
        ax: Matplotlib axes to draw on.
        enc: An affine encoding (z_score, min_max, or scale; not
            subtract_average).
        n_points: Number of spectral grid points (used for the x-axis).
    """
    energy = _energy_axis(n_points)
    shift_np = enc.shift.detach().cpu().numpy()  # (D,) or (E, D)
    scale_np = enc.scale.detach().cpu().numpy()

    etype = enc.encoding_type

    # Choose labels and decide which curves to draw.
    if etype == "z_score":
        label_a, label_b = "mean", "std"
        curve_a, curve_b = shift_np, scale_np
    elif etype == "min_max":
        label_a, label_b = "min", "max"
        curve_a, curve_b = shift_np, shift_np + scale_np
    elif etype == "scale":
        label_a, label_b = "factor", None
        curve_a, curve_b = scale_np, None
    else:
        return  # Should not be reached for non-affine or subtract_average.

    _draw_parameter_curves(ax, energy, curve_a, curve_b, label_a, label_b, enc, etype)


def _draw_parameter_curves(
    ax: Axes,
    energy: np.ndarray,
    curve_a: np.ndarray,
    curve_b: np.ndarray | None,
    label_a: str,
    label_b: str | None,
    enc: AffineEncoding,
    etype: str,
) -> None:
    """Draw one or two parameter curves, handling per-element grouping.

    When a curve's last dimension is 1 (scalar parameter, i.e.
    ``per_point`` is false) the value is broadcast to a horizontal line
    spanning the full energy grid.

    Args:
        ax: Matplotlib axes.
        energy: ``(D,)`` x-axis grid.
        curve_a: ``(D,)``, ``(1,)``, ``(E, D)`` or ``(E, 1)`` first
            parameter array.
        curve_b: Optional second parameter array with the same shape
            conventions as *curve_a*.
        label_a: Legend label for the first curve.
        label_b: Legend label for the second curve (may be ``None``).
        enc: The affine encoding (for ``per_element`` and ``elements``).
        etype: Encoding type string (used in the title).
    """
    n_energy = len(energy)
    color_a = "#2c7bb6"
    color_b = "#d7191c"

    # Broadcast scalar parameters (per_point=false) to horizontal lines.
    def _ensure_length(arr: np.ndarray) -> np.ndarray:
        if arr.shape[-1] == n_energy:
            return arr
        if arr.shape[-1] == 1:
            return np.broadcast_to(arr, arr.shape[:-1] + (n_energy,)).copy()
        return arr

    curve_a = _ensure_length(curve_a)
    if curve_b is not None:
        curve_b = _ensure_length(curve_b)

    if enc.per_element and curve_a.ndim == 2:
        n_elem = curve_a.shape[0]
        elem_list = enc.elements if enc.elements is not None else list(range(n_elem))
        for i in range(n_elem):
            z = elem_list[i] if i < len(elem_list) else i
            sym = _element_symbol(z)
            ec = _ELEMENT_COLORS[i % len(_ELEMENT_COLORS)]
            ax.plot(
                energy,
                curve_a[i],
                color=ec,
                linewidth=1.2,
                alpha=0.85,
                linestyle="-",
                label=f"{sym} {label_a}",
            )
            if curve_b is not None:
                ax.plot(
                    energy,
                    curve_b[i],
                    color=ec,
                    linewidth=1.2,
                    alpha=0.85,
                    linestyle="--",
                    label=f"{sym} {label_b}",
                )
        ax.legend(fontsize=5, loc="upper right", ncol=2)
    else:
        a = curve_a if curve_a.ndim == 1 else curve_a[0]
        ax.plot(energy, a, color=color_a, linewidth=1.3, alpha=0.9, label=label_a)
        if curve_b is not None:
            b = curve_b if curve_b.ndim == 1 else curve_b[0]
            ax.plot(energy, b, color=color_b, linewidth=1.3, alpha=0.9, linestyle="--", label=label_b)
        ax.legend(fontsize=7, loc="upper right")

    ax.set_title(f"{etype} parameters", fontsize=9)
    ax.set_xlabel("Energy grid point")
    ax.set_ylabel("Value")


###############################################################################
# Per-encoding figure builder
###############################################################################


def _figure_for_encoding(
    enc: SpectraEncoding,
    spectra_t: torch.Tensor,
    label: str,
    max_samples: int,
    elements_t: torch.Tensor | None = None,
) -> Figure:
    """Build a multi-panel diagnostic figure for a single encoding.

    Every figure includes four base rows (raw, encoded, round-trip, error).
    Additional rows are appended for specific encoding types:

    * **z_score, min_max, scale** -- affine parameter panel (mean/std,
      min/max, or factor).
    * **gaussian** -- basis decomposition panel.
    * **subtract_average** -- per-element (or global) average panel.

    Spectra are colored by target-site element when element information is
    available.  The encoded panel uses a line plot when the encoding
    preserves the spectrum length, and a heat-map otherwise.

    Args:
        enc: A single (leaf) encoding instance.
        spectra_t: Raw spectra tensor ``(N, L)``.
        label: Short descriptive title for the encoding (e.g. ``"z_score"``).
        max_samples: Maximum number of spectra included.
        elements_t: Optional target-site atomic numbers ``(N,)`` forwarded to
            element-aware encodings.

    Returns:
        Matplotlib figure.
    """
    spectra_t = spectra_t[:max_samples]
    elements_t = None if elements_t is None else elements_t[:max_samples]
    encoded_t = enc.encode(spectra_t, elements_t)
    decoded_t = enc.decode(encoded_t, elements_t)

    spectra = spectra_t.detach().cpu().numpy()
    encoded = encoded_t.detach().cpu().numpy()
    decoded = decoded_t.detach().cpu().numpy()
    elements = elements_t.detach().cpu().numpy() if elements_t is not None else None
    n_points = spectra.shape[1]

    # Determine number of rows.
    n_rows = 4
    if isinstance(enc, GaussianEncoding):
        n_rows += 1
    if isinstance(enc, SubtractAverageEncoding):
        n_rows += 1
    if isinstance(enc, AffineEncoding) and not isinstance(enc, SubtractAverageEncoding):
        n_rows += 1

    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 3.0 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle(f"Encoding: {label}", fontsize=11, fontweight="bold")

    row = 0
    _plot_raw(axes[row], spectra, elements=elements)
    row += 1
    _plot_encoded(axes[row], encoded, label, n_points=n_points, elements=elements)
    row += 1
    _plot_roundtrip(axes[row], spectra, decoded, label, elements=elements)
    row += 1
    _plot_error(axes[row], spectra, decoded, label, elements=elements)
    row += 1

    if isinstance(enc, GaussianEncoding):
        _plot_gaussian_decomposition(axes[row], enc, spectra)
        row += 1
    if isinstance(enc, SubtractAverageEncoding):
        _plot_subtract_average_panel(axes[row], enc, spectra, elements)
        row += 1
    if isinstance(enc, AffineEncoding) and not isinstance(enc, SubtractAverageEncoding):
        _plot_affine_parameters(axes[row], enc, n_points)
        row += 1

    fig.tight_layout()
    return fig


###############################################################################
# Combined-encoding overview figure
###############################################################################


def _figure_overview(
    combined: CombinedEncoding,
    spectra_t: torch.Tensor,
    max_samples: int,
    elements_t: torch.Tensor | None = None,
) -> Figure:
    """Summary figure for the full composed encoding pipeline.

    Produces the four base panels (raw, encoded, round-trip, error) for the
    complete ``combined`` pipeline.  Per-element coloring is applied when
    element data is available.

    Args:
        combined: Complete composed encoding.
        spectra_t: Raw spectra tensor ``(N, L)``.
        max_samples: Maximum number of spectra included.
        elements_t: Optional target-site atomic numbers ``(N,)`` forwarded to
            element-aware encodings.

    Returns:
        Matplotlib figure.
    """
    spectra_t = spectra_t[:max_samples]
    elements_t = None if elements_t is None else elements_t[:max_samples]
    encoded_t = combined.encode(spectra_t, elements_t)
    decoded_t = combined.decode(encoded_t, elements_t)

    spectra = spectra_t.detach().cpu().numpy()
    encoded = encoded_t.detach().cpu().numpy()
    decoded = decoded_t.detach().cpu().numpy()
    elements = elements_t.detach().cpu().numpy() if elements_t is not None else None

    n_encodings = len(combined.encodings)
    label = " -> ".join(e.encoding_type for e in combined.encodings)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle(
        f"Full pipeline: {label}\n({n_encodings} encodings composed)",
        fontsize=10,
        fontweight="bold",
    )

    _plot_raw(axes[0], spectra, elements=elements, title="Raw spectra")
    _plot_encoded(axes[1], encoded, label, n_points=spectra.shape[1], elements=elements)
    _plot_roundtrip(axes[2], spectra, decoded, label, elements=elements)
    _plot_error(axes[3], spectra, decoded, label, elements=elements)

    fig.tight_layout()
    return fig


###############################################################################
# Main
###############################################################################


def _make_error_figure(message: str, config_path: str) -> Figure:
    """Build a single-panel figure displaying an error message.

    Args:
        message: Error description text.
        config_path: Path to the config file that caused the error.

    Returns:
        Matplotlib figure with the error rendered as text.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.axis("off")
    lines = [
        f"Encoding test failed for: {config_path}",
        "",
        message,
        "",
        "This typically happens when per-point statistics (std, min, max)",
        "evaluate to zero in flat regions of the spectrum.",
        "Consider using per_point=false or adding a small epsilon to the",
        "auto-resolved parameters in the core encoding code.",
    ]
    ax.text(
        0.5,
        0.5,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=10,
        ha="center",
        va="center",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "#fff3f3", "edgecolor": "#cc0000"},
    )
    fig.tight_layout()
    return fig


###############################################################################
# Main
###############################################################################


def main() -> None:
    """Parse args, load data, build encodings, and generate diagnostic figures."""
    args = _parse_args()

    # 1. Load and validate config.
    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")
    raw = load_raw_config(config_path)
    if args.data_dir is not None:
        raw["datasource"]["json_path"] = args.data_dir
    config = Config(validate_config_schema(raw, "train"))

    # 2. Build datasource + dataset.
    print(f"Loading dataset from config: {config_path}")
    _, dataset = _build_datasource_and_dataset(config)
    print(f"Dataset size: {len(dataset)} samples")

    # 3. Resolve auto encoding fields.
    print("Resolving auto encoding fields ...")
    config = resolve_auto_encoding_config(config, dataset)

    # 4. Build combined encoding (may fail for scale/min_max with flat data).
    encoding_configs = config.get_config_list("encodings")
    try:
        combined = CombinedEncoding.from_configs(encoding_configs)
    except ConfigError as exc:
        print(f"ERROR: Encoding construction failed: {exc}", file=sys.stderr)
        print("       (per-point statistics contain zeros in flat spectral regions)", file=sys.stderr)
        fig = _make_error_figure(str(exc), str(config_path))
        _save_or_show([fig], args)
        raise SystemExit(1)
    print(f"Encoding pipeline: {' -> '.join(e.encoding_type for e in combined.encodings)}")

    # 5. Collect raw spectra.
    print(f"Collecting up to {args.max_samples} spectra ...")
    spectra_t, elements_t = _collect_spectra(dataset, args.max_samples)
    print(f"Collected spectra tensor: {tuple(spectra_t.shape)}")
    if elements_t is not None:
        unique = torch.unique(elements_t).tolist()
        print(f"Elements present: {[_element_symbol(z) for z in unique]}")

    # 6. Generate one figure per leaf encoding + one overview figure.
    figures: list[Figure] = []

    for enc in combined.encodings:
        label = enc.encoding_type
        print(f"  Plotting encoding: {label}")
        fig = _figure_for_encoding(enc, spectra_t, label, args.max_samples, elements_t)
        figures.append(fig)

    if len(combined.encodings) > 1:
        print("  Plotting full-pipeline overview ...")
        fig_overview = _figure_overview(combined, spectra_t, args.max_samples, elements_t)
        figures.append(fig_overview)

    # 7. Save / show.
    _save_or_show(figures, args)


def _save_or_show(figures: list[Figure], args: argparse.Namespace) -> None:
    """Save figures to PDF or display them interactively.

    Args:
        figures: List of matplotlib figures to output.
        args: Parsed command-line arguments.
    """
    if args.save:
        from matplotlib.backends.backend_pdf import PdfPages

        out_path = Path(args.save)
        print(f"Saving {len(figures)} figure(s) to: {out_path}")
        with PdfPages(str(out_path)) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                plt.close(fig)
        print("Saved.")
    elif not args.no_show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
        print("Done (--no-show; figures not displayed).")


if __name__ == "__main__":
    main()
