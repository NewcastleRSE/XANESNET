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

"""Number and display-label formatting helpers for analysis output."""

import math
from collections.abc import Iterable
from textwrap import shorten

from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter


def format_decimal(value: float, precision: int) -> str:
    """Format a number with significant digits and fixed-point notation.

    The number of decimal places adapts to the magnitude so small values remain
    visible without using scientific notation. Trailing zeros after the decimal
    point are removed, matching the compactness of general-format output.

    Args:
        value: Numeric value to format.
        precision: Number of significant digits to retain.

    Returns:
        A plain decimal string that contains no exponent notation.

    Raises:
        ValueError: If ``precision`` is less than one.
    """
    if precision < 1:
        raise ValueError("precision must be at least 1")

    number = float(value)
    if not math.isfinite(number):
        return str(number)
    if number == 0.0:
        return "0"

    exponent = math.floor(math.log10(abs(number)))
    decimal_places = max(0, precision - exponent - 1)
    text = f"{number:.{decimal_places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def truncate_text(value: str, width: int) -> str:
    """Truncate text with an ellipsis when it exceeds ``width`` characters.

    Args:
        value: Text to truncate.
        width: Maximum number of characters in the returned text.

    Returns:
        The original text when it fits, otherwise a prefix ending in ``...``.
        For widths below four, only the available number of dots is returned.

    Raises:
        ValueError: If ``width`` is negative.
    """
    if width < 0:
        raise ValueError("width must be non-negative")
    if len(value) <= width:
        return value
    if width <= 3:
        return "." * width
    return value[: max(width - 3, 1)].rstrip() + "..."


def shorten_label(lines: Iterable[str], width: int = 28) -> str:
    """Join and shorten label parts at word boundaries.

    This is intended for plot titles and compact cell labels. Use
    :func:`truncate_text` when a hard character limit is required instead.

    Args:
        lines: Label parts in display order.
        width: Maximum label width.

    Returns:
        One shortened display label.
    """
    label = " | ".join(line.strip() for line in lines if line.strip())
    return shorten(label, width=width, placeholder="...")


def apply_decimal_tick_format(ax: Axes, precision: int = 4) -> None:
    """Use plain fixed-point labels for the major ticks on an axes.

    This explicitly disables Matplotlib's scientific and offset-style display
    for analysis figures, including logarithmic axes whose major ticks would
    otherwise be rendered as powers of ten.

    Args:
        ax: Matplotlib axes whose x and y major ticks should be formatted.
        precision: Significant digits retained in each tick label.
    """

    def format_tick(value: float, _: int) -> str:
        """Format one tick value with the shared decimal precision.

        Args:
            value: Numeric tick value.
            _: Matplotlib's unused tick position argument.

        Returns:
            Formatted tick label.
        """
        return format_decimal(value, precision)

    formatter = FuncFormatter(format_tick)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
