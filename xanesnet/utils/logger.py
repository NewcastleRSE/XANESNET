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

"""Logging configuration and utilities for XANESNET."""

import logging
import sys
from os.path import join
from pathlib import Path
from typing import Any


def setup_logging(log_level: int = logging.INFO) -> None:
    """Configure the root logger with per-level formatters and console handlers.

    Clears any existing handlers on the root logger before installing new
    ones. Each log level (DEBUG through CRITICAL) gets a dedicated
    ``StreamHandler`` with its own format string.

    Args:
        log_level: Minimum log level to capture. Defaults to
            ``logging.INFO``.
    """
    # Remove all existing handlers from the root logger.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the base logging level.
    root_logger.setLevel(log_level)

    # Define formatters for different levels.
    formatter_debug = logging.Formatter("%(levelname)s \t\t- %(message)s (%(filename)s:%(lineno)d)")
    formatter_info = logging.Formatter("%(levelname)s \t\t- %(message)s")
    formatter_warning = logging.Formatter("%(levelname)s \t- %(message)s (%(filename)s:%(lineno)d)")
    formatter_error = logging.Formatter("%(levelname)s \t\t- %(message)s (%(filename)s:%(lineno)d)")
    formatter_critical = logging.Formatter("%(levelname)s \t- %(message)s (%(filename)s:%(lineno)d)")

    # Create handlers for console output.
    debug_handler = logging.StreamHandler(sys.stdout)
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(LevelFilter(logging.DEBUG))
    debug_handler.setFormatter(formatter_debug)

    info_handler = logging.StreamHandler(sys.stdout)
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(LevelFilter(logging.INFO))
    info_handler.setFormatter(formatter_info)

    warning_handler = logging.StreamHandler(sys.stdout)
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(LevelFilter(logging.WARNING))
    warning_handler.setFormatter(formatter_warning)

    error_handler = logging.StreamHandler(sys.stderr)
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(LevelFilter(logging.ERROR))
    error_handler.setFormatter(formatter_error)

    critical_handler = logging.StreamHandler(sys.stderr)
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.addFilter(LevelFilter(logging.CRITICAL))
    critical_handler.setFormatter(formatter_critical)

    # Add console handlers to the root logger.
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(warning_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(critical_handler)


def setup_file_logging(out_dir: str | Path) -> None:
    """Add file-based log handlers and redirect stdout/stderr to a log file.

    Opens ``<out_dir>/out.txt`` in append mode and registers file handlers
    for all log levels (DEBUG through CRITICAL). Additionally replaces
    ``sys.stdout`` and ``sys.stderr`` with a ``TeeLogger`` that writes to
    both the original stream and the log file.

    Args:
        out_dir: Directory in which to create ``out.txt``.
    """
    log_file = join(out_dir, "out.txt")

    root_logger = logging.getLogger()

    # Define formatters for different levels.
    formatter_debug = logging.Formatter("%(levelname)s \t\t- %(message)s")
    formatter_info = logging.Formatter("%(levelname)s \t\t- %(message)s")
    formatter_warning = logging.Formatter("%(levelname)s \t- %(message)s (%(filename)s:%(lineno)d)")
    formatter_error = logging.Formatter("%(levelname)s \t\t- %(message)s (%(filename)s:%(lineno)d)")
    formatter_critical = logging.Formatter("%(levelname)s \t- %(message)s (%(filename)s:%(lineno)d)")

    # Create handlers for console output.
    debug_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.addFilter(LevelFilter(logging.DEBUG))
    debug_handler.setFormatter(formatter_debug)

    info_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    info_handler.setLevel(logging.INFO)
    info_handler.addFilter(LevelFilter(logging.INFO))
    info_handler.setFormatter(formatter_info)

    warning_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    warning_handler.setLevel(logging.WARNING)
    warning_handler.addFilter(LevelFilter(logging.WARNING))
    warning_handler.setFormatter(formatter_warning)

    error_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.addFilter(LevelFilter(logging.ERROR))
    error_handler.setFormatter(formatter_error)

    critical_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    critical_handler.setLevel(logging.CRITICAL)
    critical_handler.addFilter(LevelFilter(logging.CRITICAL))
    critical_handler.setFormatter(formatter_critical)

    # Add console handlers to the root logger.
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(warning_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(critical_handler)

    # Redirect stdout and stderr to write to both console and file.
    stdout_tee = TeeLogger(log_file, sys.stdout)
    stderr_tee = TeeLogger(log_file, sys.stderr)
    sys.stdout = stdout_tee
    sys.stderr = stderr_tee


class LevelFilter(logging.Filter):
    """Logging filter that passes only records at exactly the specified level.

    Args:
        level: The exact log level to pass through (e.g. ``logging.INFO``).
    """

    def __init__(self, level: int) -> None:
        """Initialize the level filter."""
        self.level = level
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        """Return ``True`` only if the record's level matches the filter level.

        Args:
            record: Log record being evaluated.

        Returns:
            ``True`` when ``record.levelno == self.level``, else ``False``.
        """
        return record.levelno == self.level


class TeeLogger:
    """Duplicates writes to both a stream (e.g. stdout) and a file.

    Args:
        filename: Path to the log file (opened in append mode).
        stream: Original stream to also write to (e.g. ``sys.stdout``).
    """

    def __init__(self, filename: str | Path, stream: Any) -> None:
        """Open ``filename`` in append mode alongside ``stream``."""
        self.file = open(filename, "a")
        self.stream = stream

    def write(self, message: str) -> None:
        """Write ``message`` to both the stream and the file.

        Args:
            message: Text chunk to append to both outputs.
        """
        self.stream.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        """Flush both the stream and the file."""
        self.stream.flush()
        self.file.flush()
