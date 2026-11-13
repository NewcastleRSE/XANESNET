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

"""Unit tests for the ``core_analyze.analyze`` pipeline, run against registered fakes."""

import json
import logging
from argparse import Namespace
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from xanesnet import core_analyze
from xanesnet.analysis.aggregators import (
    Aggregator,
    AggregatorRegistry,
    AggregatorResult,
)
from xanesnet.analysis.collectors import Collector, CollectorRegistry
from xanesnet.analysis.custom_modules import CustomModule, CustomModuleRegistry
from xanesnet.analysis.plotters import Plotter, PlotterRegistry
from xanesnet.analysis.reporters import Reporter, ReporterRegistry
from xanesnet.analysis.result import AnalysisResults
from xanesnet.serialization.config import Config, load_raw_config
from xanesnet.serialization.prediction_readers import PredictionSample
from xanesnet.utils.exceptions import ConfigError

###############################################################################
################################# TEST DOUBLES ################################
###############################################################################

SAMPLES = [
    {"sample_id": "a", "target_site_index": 0},
    {"sample_id": "b", "target_site_index": 2},
    {"sample_id": "c"},
]

CALLS: list[tuple[str, Any]] = []
"""Chronological ``(event, payload)`` log of everything the fakes were asked to do."""


class FakeReader:
    def __init__(self, path: Any, preload: bool) -> None:
        self.path, self.preload, self.closed = path, preload, False

    def close(self) -> None:
        self.closed = True
        CALLS.append(("close", self.path))


class FakeSelector:
    selector_type = "fake"

    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self._samples = samples

    def __iter__(self):
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)

    def __str__(self) -> str:
        return "fake_selector"


@CollectorRegistry.register("unit_test_length")
class LengthCollector(Collector):
    def process(self, sample: PredictionSample) -> dict[str, Any]:
        return {"id_length": len(sample["sample_id"])}


@CollectorRegistry.register("unit_test_constant")
class ConstantCollector(Collector):
    def __init__(self, collector_type: str, key: str = "constant", value: Any = 1) -> None:
        super().__init__(collector_type)
        self.key, self.value = key, value

    def process(self, sample: PredictionSample) -> dict[str, Any]:
        return {self.key: self.value}


@AggregatorRegistry.register("unit_test_count")
class CountAggregator(Aggregator):
    def aggregate(self, selector, per_sample_values, index):
        CALLS.append(("aggregate", (index, None if per_sample_values is None else list(per_sample_values))))
        return AggregatorResult(self.aggregator_type, index, {"n": len(selector)})


@ReporterRegistry.register("unit_test_report")
class FakeReporter(Reporter):
    def report(self, results: AnalysisResults, output_dir: Path) -> None:
        CALLS.append(("report", (results, output_dir)))
        assert output_dir.is_dir(), "the pipeline creates the output directory"


@PlotterRegistry.register("unit_test_plot")
class FakePlotter(Plotter):
    def __init__(self, plotter_type: str, latex_font: bool = False, plot_size: str = "default") -> None:
        super().__init__(plotter_type, latex_font, plot_size)  # type: ignore[arg-type]

    def _plot(self, results: AnalysisResults, output_dir: Path) -> None:
        CALLS.append(("plot", (results, output_dir)))
        assert output_dir.is_dir()


@CustomModuleRegistry.register("unit_test_module")
class FakeModule(CustomModule):
    def run(self, inference_run_dirs, prediction_names, output_dir):
        CALLS.append(("custom", (inference_run_dirs, prediction_names, output_dir)))
        assert output_dir.is_dir()


@pytest.fixture(autouse=True)
def calls() -> Iterator[list[tuple[str, Any]]]:
    """Isolate the custom analysis-component call log for each test.

    Yields:
        The shared call log, empty before and after the test.
    """
    CALLS.clear()
    yield CALLS
    CALLS.clear()


@pytest.fixture
def readers(monkeypatch: pytest.MonkeyPatch) -> list[FakeReader]:
    """Replace prediction-reader and selector construction with fakes.

    Args:
        monkeypatch: Pytest fixture used to replace analysis setup functions.

    Returns:
        Readers created by the fake setup function.
    """
    created: list[FakeReader] = []

    def make_readers(inference_run_dirs, preload):
        created.extend(FakeReader(path, preload) for path in inference_run_dirs)
        return created

    def make_selectors(config, predictions_readers):
        selectors = [[FakeSelector(SAMPLES)] for _ in predictions_readers]
        return selectors, Config({"selectors": [{"selector_type": "fake"}]})

    monkeypatch.setattr(core_analyze, "_setup_predictions_readers", make_readers)
    monkeypatch.setattr(core_analyze, "_setup_selectors", make_selectors)
    return created


def make_config(**sections: list[dict[str, Any]]) -> Config:
    """Build an analysis config with empty defaults for unspecified sections.

    Args:
        sections: Analysis sections that override the empty defaults.

    Returns:
        Config containing the supplied analysis sections.
    """
    raw: dict[str, Any] = {name: [] for name in ("selectors", "collectors", "aggregators", "reporters", "plotters")}
    raw.update({"custom_modules": [], "preload": False})
    raw.update(sections)
    return Config(raw)


def make_args(*runs: str, names: list[str] | None = None) -> Namespace:
    """Build parsed analysis arguments for the requested inference runs.

    Args:
        runs: Inference run directories passed to analysis.
        names: Optional names assigned to the corresponding prediction sets.

    Returns:
        Namespace matching the arguments consumed by the analysis pipeline.
    """
    return Namespace(inference_runs=list(runs), prediction_names=names)


def calls_named(name: str) -> list[Any]:
    """Return payloads recorded for a named analysis event.

    Args:
        name: Event name to filter.

    Returns:
        Payloads for matching events in recorded order.
    """
    return [payload for event, payload in CALLS if event == name]


###############################################################################
############################# CUSTOM-MODULES-ONLY RUN #########################
###############################################################################


class TestCustomModulesOnly:
    @pytest.fixture
    def config(self) -> Config:
        return make_config(custom_modules=[{"custom_module_type": "unit_test_module"}])

    def test_modules_run_without_building_any_reader(self, config, tmp_path, monkeypatch):
        def forbidden(*args):
            raise AssertionError("readers must not be created for a custom-modules-only run")

        monkeypatch.setattr(core_analyze, "_setup_predictions_readers", forbidden)

        core_analyze.analyze(config, make_args("runs/one", "runs/two"), tmp_path)

        ((inference_runs, names, output_dir),) = calls_named("custom")
        assert inference_runs == [Path("runs/one"), Path("runs/two")]
        assert names == ["one", "two"]
        assert output_dir == tmp_path / "custom_modules" / "unit_test_module_000"

    def test_empty_pipeline_sections_are_recorded(self, config, tmp_path):
        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        for section in ("selectors", "collectors", "aggregators", "reporters", "plotters"):
            assert load_raw_config(tmp_path / f"{section}.yaml") == {section: []}
        assert load_raw_config(tmp_path / "custom_modules.yaml") == {
            "custom_modules": [{"custom_module_type": "unit_test_module"}]
        }

    def test_each_module_gets_its_own_numbered_directory(self, tmp_path):
        config = make_config(custom_modules=[{"custom_module_type": "unit_test_module"}] * 2)

        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        directories = [output_dir.name for _, _, output_dir in calls_named("custom")]
        assert directories == ["unit_test_module_000", "unit_test_module_001"]

    def test_nothing_configured_at_all_is_a_quiet_no_op(self, tmp_path, readers):
        core_analyze.analyze(make_config(), make_args("runs/one"), tmp_path)

        assert CALLS == []
        assert readers == []

    def test_unknown_module_type_is_reported(self, tmp_path):
        config = make_config(custom_modules=[{"custom_module_type": "does_not_exist"}])

        with pytest.raises(KeyError, match="does_not_exist"):
            core_analyze.analyze(config, make_args("runs/one"), tmp_path)


###############################################################################
############################### PREDICTION NAMES ##############################
###############################################################################


@pytest.mark.parametrize(
    ("runs", "names", "expected"),
    [
        (("runs/one", "runs/two"), None, ["one", "two"]),
        (("runs/one", "runs/two"), [], ["one", "two"]),
        (("runs/one", "runs/two"), ["A", "B"], ["A", "B"]),
        pytest.param(
            ("runs/one", "runs/two"),
            ["A"],
            ["A", "two"],
            marks=pytest.mark.xfail(
                strict=True,
                reason="Known bug: padding uses run_dir_names[:n] instead of run_dir_names[len(names):], "
                "so the second reader is labelled 'one' instead of its own directory name 'two'.",
            ),
        ),
        (("runs/one", "runs/two"), ["A", "B", "C"], ["A", "B"]),
        (("runs/nested/one/",), None, ["one"]),
    ],
    ids=["default", "empty", "exact", "too-few", "too-many", "trailing-slash"],
)
def test_prediction_names_fall_back_to_run_directory_names(tmp_path, runs, names, expected):
    config = make_config(custom_modules=[{"custom_module_type": "unit_test_module"}])

    core_analyze.analyze(config, make_args(*runs, names=names), tmp_path)

    assert calls_named("custom")[0][1] == expected


@pytest.mark.parametrize("names", [["A"], ["A", "B", "C"]])
def test_mismatched_name_count_is_logged(tmp_path, caplog, names):
    config = make_config(custom_modules=[{"custom_module_type": "unit_test_module"}])

    with caplog.at_level(logging.WARNING):
        core_analyze.analyze(config, make_args("runs/one", "runs/two", names=names), tmp_path)

    assert "does not match the number of inference runs" in caplog.text


###############################################################################
################################ FULL PIPELINE ################################
###############################################################################


class TestFullPipeline:
    @pytest.fixture
    def config(self) -> Config:
        return make_config(
            collectors=[{"collector_type": "unit_test_length"}, {"collector_type": "unit_test_constant", "value": 7}],
            aggregators=[{"aggregator_type": "unit_test_count"}],
            reporters=[{"reporter_type": "unit_test_report"}],
            plotters=[{"plotter_type": "unit_test_plot"}],
            custom_modules=[{"custom_module_type": "unit_test_module"}],
        )

    def test_stage_order(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        order = [event for event, _ in CALLS]
        assert order == ["aggregate", "report", "plot", "custom", "close"]

    def test_collector_records_are_written_per_sample(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        aux = tmp_path / "aux" / "predictions_000"
        records = [json.loads(line) for line in (aux / "000.jsonl").read_text().splitlines()]
        assert records == [
            {"id_length": 1, "constant": 7, "sample_id": "a", "target_site_index": 0},
            {"id_length": 1, "constant": 7, "sample_id": "b", "target_site_index": 2},
            {"id_length": 1, "constant": 7, "sample_id": "c", "target_site_index": None},
        ]
        assert json.loads((aux / "000.meta.json").read_text()) == {"count": 3}

    def test_every_reader_gets_its_own_aux_directory(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one", "runs/two"), tmp_path)

        assert sorted(path.name for path in (tmp_path / "aux").iterdir()) == ["predictions_000", "predictions_001"]

    def test_aggregators_receive_the_collected_values(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        ((index, values),) = calls_named("aggregate")
        assert index == 0
        assert [record["sample_id"] for record in values] == ["a", "b", "c"]

    def test_reporters_and_plotters_receive_the_aggregated_results(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one", names=["First"]), tmp_path)

        ((results, report_dir),) = calls_named("report")
        ((plot_results, plot_dir),) = calls_named("plot")
        assert plot_results is results
        assert report_dir == tmp_path / "reports" / "unit_test_report_000"
        assert plot_dir == tmp_path / "plots" / "unit_test_plot_000"
        assert results.prediction_names == ["First"]
        assert results.aggregation(0, 0, "unit_test_count").data == {"n": 3}
        assert results.collector_stream(0, 0) is not None

    def test_component_configuration_is_saved(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        assert load_raw_config(tmp_path / "collectors.yaml")["collectors"][1] == {
            "collector_type": "unit_test_constant",
            "value": 7,
        }
        assert load_raw_config(tmp_path / "aggregators.yaml") == {
            "aggregators": [{"aggregator_type": "unit_test_count"}]
        }
        assert load_raw_config(tmp_path / "selectors.yaml") == {"selectors": [{"selector_type": "fake"}]}
        for section in ("reporters", "plotters", "custom_modules"):
            assert (tmp_path / f"{section}.yaml").is_file()

    @pytest.mark.parametrize("preload", [True, False])
    def test_preload_flag_reaches_the_readers(self, config, tmp_path, readers, preload):
        raw = config.as_dict()
        raw["preload"] = preload

        core_analyze.analyze(Config(raw), make_args("runs/one", "runs/two"), tmp_path)

        assert [reader.preload for reader in readers] == [preload, preload]

    def test_readers_are_closed_after_success(self, config, tmp_path, readers):
        core_analyze.analyze(config, make_args("runs/one", "runs/two"), tmp_path)

        assert [reader.closed for reader in readers] == [True, True]

    def test_readers_are_closed_after_a_failure(self, tmp_path, readers):
        config = make_config(collectors=[{"collector_type": "does_not_exist"}])

        with pytest.raises(KeyError, match="does_not_exist"):
            core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        assert [reader.closed for reader in readers] == [True]


class TestPartialPipelines:
    def test_aggregators_without_collectors_get_no_per_sample_values(self, tmp_path, readers):
        config = make_config(aggregators=[{"aggregator_type": "unit_test_count"}])

        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        assert calls_named("aggregate") == [(0, None)]
        assert not (tmp_path / "aux").exists()

    def test_collectors_alone_still_persist_their_results(self, tmp_path, readers):
        config = make_config(collectors=[{"collector_type": "unit_test_length"}])

        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        assert (tmp_path / "aux" / "predictions_000" / "000.jsonl").is_file()
        assert not (tmp_path / "reports").exists()
        assert not (tmp_path / "plots").exists()

    def test_reporters_alone_see_empty_collector_and_aggregator_results(self, tmp_path, readers):
        config = make_config(reporters=[{"reporter_type": "unit_test_report"}])

        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        ((results, _),) = calls_named("report")
        assert results.collector_results == []
        assert results.aggregator_results == []

    def test_several_components_of_one_kind_get_numbered_directories(self, tmp_path, readers):
        config = make_config(reporters=[{"reporter_type": "unit_test_report"}] * 2)

        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        assert [d.name for _, d in calls_named("report")] == ["unit_test_report_000", "unit_test_report_001"]


class TestCollectorKeyContract:
    def test_duplicate_output_keys_are_rejected(self, tmp_path, readers):
        config = make_config(
            collectors=[{"collector_type": "unit_test_constant"}, {"collector_type": "unit_test_constant"}]
        )

        with pytest.raises(ConfigError, match="Duplicate collector key 'constant' for sample a"):
            core_analyze.analyze(config, make_args("runs/one"), tmp_path)

    def test_reserved_metadata_keys_cannot_be_overridden(self, tmp_path, readers):
        config = make_config(
            collectors=[{"collector_type": "unit_test_constant", "key": "sample_id", "value": "hijacked"}]
        )

        core_analyze.analyze(config, make_args("runs/one"), tmp_path)

        records = [json.loads(line) for line in (tmp_path / "aux/predictions_000/000.jsonl").read_text().splitlines()]
        assert [record["sample_id"] for record in records] == ["a", "b", "c"]


###############################################################################
################################### HELPERS ###################################
###############################################################################


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0.0s"), (4.24, "4.2s"), (59.94, "59.9s"), (60, "1m 00s"), (63, "1m 03s"), (3725, "62m 05s")],
)
def test_duration_formatting(seconds, expected):
    assert core_analyze._format_duration(seconds) == expected
