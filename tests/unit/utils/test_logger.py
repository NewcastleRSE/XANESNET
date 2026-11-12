# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Unit tests for logging setup, level filtering and stdout/stderr tee-ing."""

import io
import logging
import sys

import pytest

from xanesnet.utils.logger import LevelFilter, TeeLogger, setup_file_logging, setup_logging


def _record(level: int) -> logging.LogRecord:
    return logging.LogRecord("t", level, __file__, 1, "msg", None, None)


###############################################################################
################################## LevelFilter ################################
###############################################################################


@pytest.mark.parametrize(
    "filter_level, record_level, passes",
    [
        (logging.INFO, logging.INFO, True),
        (logging.INFO, logging.WARNING, False),
        (logging.INFO, logging.DEBUG, False),
        (logging.ERROR, logging.CRITICAL, False),
    ],
)
def test_level_filter_passes_only_the_exact_level(filter_level, record_level, passes):
    assert LevelFilter(filter_level).filter(_record(record_level)) is passes


###############################################################################
################################# setup_logging ###############################
###############################################################################


def test_setup_logging_replaces_existing_root_handlers():
    root = logging.getLogger()
    stale = logging.NullHandler()
    root.addHandler(stale)

    setup_logging(logging.INFO)

    assert stale not in root.handlers


def test_setup_logging_sets_the_root_level():
    setup_logging(logging.WARNING)
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_is_idempotent():
    setup_logging()
    count = len(logging.getLogger().handlers)
    setup_logging()
    assert len(logging.getLogger().handlers) == count


def test_each_level_is_emitted_exactly_once_to_the_right_stream(capsys):
    setup_logging(logging.DEBUG)
    logging.debug("d-msg")
    logging.info("i-msg")
    logging.warning("w-msg")
    logging.error("e-msg")
    logging.critical("c-msg")

    captured = capsys.readouterr()
    for message in ("d-msg", "i-msg", "w-msg"):
        assert captured.out.count(message) == 1
        assert message not in captured.err
    for message in ("e-msg", "c-msg"):
        assert captured.err.count(message) == 1
        assert message not in captured.out


def test_messages_below_the_configured_level_are_suppressed(capsys):
    setup_logging(logging.WARNING)
    logging.info("quiet")
    logging.warning("loud")
    captured = capsys.readouterr()
    assert "quiet" not in captured.out + captured.err
    assert "loud" in captured.out


###############################################################################
################################### TeeLogger #################################
###############################################################################


def test_tee_logger_writes_to_stream_and_file(tmp_path):
    stream = io.StringIO()
    tee = TeeLogger(tmp_path / "log.txt", stream)

    tee.write("hello ")
    tee.write("world")
    tee.flush()

    assert stream.getvalue() == "hello world"
    assert (tmp_path / "log.txt").read_text() == "hello world"


def test_tee_logger_appends_to_an_existing_file(tmp_path):
    path = tmp_path / "log.txt"
    path.write_text("old\n")
    TeeLogger(path, io.StringIO()).write("new\n")
    assert path.read_text() == "old\nnew\n"


###############################################################################
############################### setup_file_logging ############################
###############################################################################


def test_file_logging_mirrors_logs_and_prints_into_out_txt(tmp_path, capsys):
    setup_logging(logging.DEBUG)
    setup_file_logging(tmp_path)

    logging.info("logged-info")
    logging.error("logged-error")
    print("plain-print")
    sys.stdout.flush()

    content = (tmp_path / "out.txt").read_text()
    assert "logged-info" in content
    assert "logged-error" in content
    assert "plain-print" in content


def test_file_logging_keeps_console_output(tmp_path, capsys):
    setup_logging(logging.DEBUG)
    setup_file_logging(tmp_path)

    print("still-visible")

    assert "still-visible" in capsys.readouterr().out


def test_file_logging_replaces_std_streams_with_tee_loggers(tmp_path):
    setup_file_logging(tmp_path)
    assert isinstance(sys.stdout, TeeLogger)
    assert isinstance(sys.stderr, TeeLogger)
