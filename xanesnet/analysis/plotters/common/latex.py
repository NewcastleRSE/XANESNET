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

"""Helpers for emitting standalone LaTeX documents and compiling them."""

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .formatting import format_decimal

LATEX_PREAMBLE: str = r"""\documentclass{article}
\usepackage[table]{xcolor}
\usepackage{multirow}
\pagestyle{empty}
\begin{document}
"""

LATEX_FOOTER: str = "\\end{document}\n"
LATEX_MARK_COLORS: dict[str, str] = {"best": "green!15", "worst": "red!15"}
LATEX_TABLE_FONT: str = r"\footnotesize"
_PROCESS_TIMEOUT: int = 30


def escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text.

    Args:
        text: Plain text to escape.

    Returns:
        LaTeX-safe representation of ``text``.
    """
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
        "_": r"\_",
        "^": r"\textasciicircum{}",
        "~": r"\textasciitilde{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def escape_label_line(text: str) -> str:
    """Escape a table label, using a stacked label for pipe-separated parts.

    Args:
        text: Plain text label line.

    Returns:
        LaTeX-safe label. Labels such as ``"SchNet | all"`` become a left-
        aligned ``shortstack`` with one part per line.
    """
    parts = [part.strip() for part in text.split("|")]
    if len(parts) == 1:
        return escape_latex(text)
    lines = r"\\".join(escape_latex(part) for part in parts)
    return rf"\shortstack[l]{{{lines}}}"


def format_cell(value: float | None, mark: str | None, precision: int) -> str:
    """Format one table cell for LaTeX.

    Args:
        value: Numeric cell value; missing values are rendered as a dash.
        mark: Optional ``"best"``/``"worst"`` mark controlling color highlighting.
        precision: Number of significant digits.

    Returns:
        LaTeX cell content with an optional ``\\cellcolor`` prefix.
    """
    text = format_decimal(value, precision) if value is not None else "-"
    color = LATEX_MARK_COLORS[mark] if mark is not None else ""
    return rf"\cellcolor{{{color}}} {text}" if color else text


def sanitize_label(text: str) -> str:
    """Convert a value key into a safe LaTeX label suffix.

    Args:
        text: Value key to sanitize.

    Returns:
        Sanitized suffix containing only letters, digits, and underscores.
    """
    return re.sub(r"[^0-9A-Za-z_]", "_", text)


def write_document(tex_path: Path, body: str) -> None:
    """Write a standalone LaTeX document around one table body.

    Args:
        tex_path: Destination path of the ``.tex`` document.
        body: LaTeX table float to wrap in the shared preamble and footer.
    """
    tex_path.write_text(f"{LATEX_PREAMBLE}{body}\n{LATEX_FOOTER}", encoding="utf-8")


def compile_pdf(tex_path: Path, out_pdf: Path) -> bool:
    """Compile a standalone LaTeX document to a cropped PDF.

    Compilation runs in a temporary directory so no auxiliary files are left
    behind. The compiled page is cropped to its content with ``pdfcrop`` when
    that tool is available. Failures are logged and reported, never raised,
    because a missing LaTeX installation must not abort an analysis run.

    Args:
        tex_path: Path to the standalone LaTeX document.
        out_pdf: Destination path for the compiled PDF.

    Returns:
        True when the PDF was compiled and copied to ``out_pdf``.
    """
    exe = shutil.which("pdflatex")
    if exe is None:
        logging.info("    pdflatex not found, skipping LaTeX rendering: %s", tex_path.name)
        return False

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        work_tex = work_dir / tex_path.name
        shutil.copy(tex_path, work_tex)
        proc = _run_tool([exe, "-interaction=nonstopmode", "-halt-on-error", work_tex.name], work_dir)
        compiled_pdf = work_dir / f"{work_tex.stem}.pdf"
        if proc is None or proc.returncode != 0 or not compiled_pdf.exists():
            logging.warning("    LaTeX compilation failed for %s; no PDF was produced.", tex_path.name)
            return False

        crop_exe = shutil.which("pdfcrop")
        if crop_exe is not None:
            cropped_pdf = work_dir / f"{work_tex.stem}_crop.pdf"
            crop_proc = _run_tool(
                [crop_exe, "--margins", "2", compiled_pdf.name, cropped_pdf.name],
                work_dir,
            )
            if crop_proc is not None and crop_proc.returncode == 0 and cropped_pdf.exists():
                shutil.copy(cropped_pdf, out_pdf)
                return True
        shutil.copy(compiled_pdf, out_pdf)
        return True


def _run_tool(command: list[str], work_dir: Path) -> subprocess.CompletedProcess[bytes] | None:
    """Run one external LaTeX toolchain command inside a working directory.

    Args:
        command: Executable and arguments to run.
        work_dir: Directory to run the command in.

    Returns:
        The completed process, or ``None`` when the command timed out or could
        not be started.
    """
    try:
        return subprocess.run(
            command,
            cwd=work_dir,
            capture_output=True,
            timeout=_PROCESS_TIMEOUT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logging.warning("    '%s' timed out after %d s.", command[0], _PROCESS_TIMEOUT)
        return None
    except OSError as exc:
        logging.warning("    '%s' could not be started: %s", command[0], exc)
        return None
