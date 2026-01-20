"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either Version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import sys
from os.path import join
from pathlib import Path
from typing import Any


def setup_logging(log_level: int = logging.INFO) -> None:
    """
    Set up logging with different formats for various levels and a critical exit handler.
    """
    # Remove all existing handlers from the root logger.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set the base logging level.
    root_logger.setLevel(log_level)

    # Define formatters for different levels.
    formatter_debug = logging.Formatter("%(levelname)s \t\t- %(message)s")
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

    critical_handler = CriticalExitHandler(sys.stderr)
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
    log_file = join(out_dir, "out.txt")

    root_logger = logging.getLogger()

    # Remove critical handler
    for handler in root_logger.handlers[:]:
        if isinstance(handler, CriticalExitHandler):
            root_logger.removeHandler(handler)

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

    critical_exit_handler = CriticalExitHandler(sys.stderr)
    critical_exit_handler.setLevel(logging.CRITICAL)
    critical_exit_handler.addFilter(LevelFilter(logging.CRITICAL))
    critical_exit_handler.setFormatter(formatter_critical)

    # Add console handlers to the root logger.
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(info_handler)
    root_logger.addHandler(warning_handler)
    root_logger.addHandler(error_handler)
    root_logger.addHandler(critical_handler)
    root_logger.addHandler(critical_exit_handler)

    # Set up exception hook to log uncaught exceptions.
    def handle_exception(exc_type, exc_value, exc_traceback) -> None:
        logging.critical("Uncaught Exception:", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    # Redirect stdout and stderr to write to both console and file.
    tee = TeeLogger(log_file, sys.stdout)
    sys.stdout = tee
    sys.stderr = tee


class CriticalExitHandler(logging.StreamHandler):
    """
    Custom handler that terminates the program on CRITICAL logs.
    """

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        if record.levelno >= logging.CRITICAL:
            self.flush()
            print("TERMINATE EXECUTION!")
            sys.exit(1)


class LevelFilter(logging.Filter):
    """
    Only pass log records that match the specified level.
    """

    def __init__(self, level: int) -> None:
        self.level = level
        super().__init__()

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == self.level


class TeeLogger:
    """
    Redirects stdout and stderr to both console and a file.
    """

    def __init__(self, filename: str | Path, stream: Any) -> None:
        self.file = open(filename, "a")
        self.stream = stream

    def write(self, message: str) -> None:
        self.stream.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()
