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

"""Core analysis pipeline: setup, collection, aggregation, reporting, and plotting."""

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from xanesnet.analysis.aggregators import (
    Aggregator,
    AggregatorRegistry,
    AggregatorResult,
)
from xanesnet.analysis.collectors import Collector, CollectorRegistry
from xanesnet.analysis.plotters import Plotter, PlotterRegistry
from xanesnet.analysis.reporters import Reporter, ReporterRegistry
from xanesnet.analysis.result import AnalysisResults
from xanesnet.analysis.selectors import Selector, SelectorRegistry
from xanesnet.serialization.config import Config
from xanesnet.serialization.jsonl_stream import JSONLStream, json_friendly
from xanesnet.serialization.prediction_readers import (
    PredictionReader,
    detect_prediction_format,
)

###############################################################################
################################### ANALYZE ###################################
###############################################################################


def analyze(config: Config, args_namespace: Namespace, save_dir: Path) -> None:
    """Run the complete analysis pipeline.

    Sets up readers, selectors, collectors, aggregators, reporters, and
    plotters from ``config``; executes the pipeline; and writes all outputs
    under ``save_dir``.

    Args:
        config: Validated analysis configuration.
        args_namespace: Parsed CLI arguments (must contain ``predictions``).
        save_dir: Root directory for all analysis outputs.
    """
    logging.info("Analysis.")

    predictions_dirs = args_namespace.predictions
    logging.info(f"You provided {len(predictions_dirs)} predictions directories:")

    predictions_readers = _setup_predictions_readers(predictions_dirs)
    try:
        selectors, selectors_config = _setup_selectors(config, predictions_readers)
        collectors, collectors_config = _setup_collectors(config)
        aggregators, aggregators_config = _setup_aggregators(config)

        reporters, reporters_config = _setup_reporters(config)
        plotters, plotters_config = _setup_plotters(config)

        selectors_config.save(save_dir / "selectors.yaml")
        collectors_config.save(save_dir / "collectors.yaml")
        aggregators_config.save(save_dir / "aggregators.yaml")
        reporters_config.save(save_dir / "reporters.yaml")
        plotters_config.save(save_dir / "plotters.yaml")

        # Run collectors
        logging.info("Running collectors.")
        collector_results = _run_collectors(collectors, selectors, save_dir)

        # Run aggregators
        logging.info("Running aggregators.")
        aggregator_results = _run_aggregators(aggregators, selectors, collector_results)

        results = AnalysisResults(
            selectors=selectors,
            collector_results=collector_results,
            aggregator_results=aggregator_results,
            selectors_config=[cfg.as_dict() for cfg in selectors_config.get_config_list("selectors")],
            collectors_config=[cfg.as_dict() for cfg in collectors_config.get_config_list("collectors")],
            aggregators_config=[cfg.as_dict() for cfg in aggregators_config.get_config_list("aggregators")],
        )

        # Run reporters
        logging.info("Running reporters.")
        _run_reporters(reporters, results, save_dir)

        # Run plotters
        logging.info("Running plotters.")
        _run_plotters(plotters, results, save_dir)
    finally:
        for predictions_reader in predictions_readers:
            predictions_reader.close()

    # Summary
    logging.info("Analysis completed!")


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_predictions_readers(predictions_dirs: list[str] | list[Path]) -> list[PredictionReader]:
    """Create a prediction reader for each supplied directory.

    Auto-detects the prediction format from the directory layout.

    Args:
        predictions_dirs: Paths to directories containing prediction files.

    Returns:
        One ``PredictionReader`` per directory.
    """
    readers: list[PredictionReader] = []
    for predictions_dir in predictions_dirs:
        reader_class = detect_prediction_format(predictions_dir)
        logging.info(f"Detected format for {predictions_dir}: {reader_class.__name__}")
        reader = reader_class(predictions_dir)
        readers.append(reader)

    return readers


def _setup_selectors(
    config: Config, predictions_readers: list[PredictionReader]
) -> tuple[list[list[Selector]], Config]:
    """Instantiate selectors for every prediction reader.

    If no selectors are specified in ``config``, an ``'all'`` selector is
    created for each reader. Otherwise one selector per configured type is
    instantiated per reader.

    Args:
        config: Validated analysis configuration.
        predictions_readers: Readers to attach selectors to.

    Returns:
        A ``(selectors, selectors_config)`` tuple where ``selectors`` is a
        list of per-reader selector lists.
    """
    selectors_config = config.get_config_list("selectors")

    # If no selectors are configured, use 'all' selector for each predictions reader
    if len(selectors_config) == 0:
        logging.info("No selectors configured, using 'all' selector for each predictions reader")
        selector_config = Config({"selector_type": "all"})
        selectors = [
            [SelectorRegistry.get("all")(**selector_config.as_kwargs(), data_source=reader)]
            for reader in predictions_readers
        ]
        return selectors, Config({"selectors": [selector_config]})

    # Initializing configured selectors
    selectors: list[list[Selector]] = []
    for selector_config in selectors_config:
        selector_type = selector_config.get_str("selector_type")

        logging.info(f"Initializing selector: {selector_type}")
        selector_list: list[Selector] = []
        for reader in predictions_readers:
            selector = SelectorRegistry.get(selector_type)(**selector_config.as_kwargs(), data_source=reader)
            selector_list.append(selector)
        selectors.append(selector_list)

    return selectors, Config({"selectors": selectors_config})


def _setup_collectors(config: Config) -> tuple[list[Collector], Config]:
    """Instantiate collectors from the configuration.

    Args:
        config: Validated analysis configuration.

    Returns:
        A ``(collectors, collectors_config)`` tuple.
    """
    collectors_config = config.get_config_list("collectors")
    assert isinstance(collectors_config, list)

    if len(collectors_config) == 0:
        logging.warning("No collectors configured.")
        return [], Config({"collectors": []})

    collectors: list[Collector] = []
    for collector_config in collectors_config:
        collector_type = collector_config.get_str("collector_type")

        logging.info(f"Initializing collector: {collector_type}")
        collector = CollectorRegistry.get(collector_type)(**collector_config.as_kwargs())
        collectors.append(collector)

    return collectors, Config({"collectors": collectors_config})


def _setup_aggregators(config: Config) -> tuple[list[Aggregator], Config]:
    """Instantiate aggregators from the configuration.

    Args:
        config: Validated analysis configuration.

    Returns:
        A ``(aggregators, aggregators_config)`` tuple.
    """
    aggregators_config = config.get_config_list("aggregators")

    if len(aggregators_config) == 0:
        logging.warning("No aggregators configured.")
        return [], Config({"aggregators": []})

    aggregators: list[Aggregator] = []
    for aggregator_config in aggregators_config:
        aggregator_type = aggregator_config.get_str("aggregator_type")

        logging.info(f"Initializing aggregator: {aggregator_type}")
        aggregator = AggregatorRegistry.get(aggregator_type)(**aggregator_config.as_kwargs())
        aggregators.append(aggregator)

    return aggregators, Config({"aggregators": aggregators_config})


def _setup_reporters(config: Config) -> tuple[list[Reporter], Config]:
    """Instantiate reporters from the configuration.

    Args:
        config: Validated analysis configuration.

    Returns:
        A ``(reporters, reporters_config)`` tuple.
    """
    reporters_config = config.get_config_list("reporters")

    if len(reporters_config) == 0:
        logging.warning("No reporters configured.")
        return [], Config({"reporters": []})

    reporters: list[Reporter] = []
    for reporter_config in reporters_config:
        reporter_type = reporter_config.get_str("reporter_type")

        logging.info(f"Initializing reporter: {reporter_type}")
        reporter = ReporterRegistry.get(reporter_type)(**reporter_config.as_kwargs())
        reporters.append(reporter)

    return reporters, Config({"reporters": reporters_config})


def _setup_plotters(config: Config) -> tuple[list[Plotter], Config]:
    """Instantiate plotters from the configuration.

    Args:
        config: Validated analysis configuration.

    Returns:
        A ``(plotters, plotters_config)`` tuple.
    """
    plotters_config = config.get_config_list("plotters")

    if len(plotters_config) == 0:
        logging.warning("No plotters configured.")
        return [], Config({"plotters": []})

    plotters: list[Plotter] = []
    for plotter_config in plotters_config:
        plotter_type = plotter_config.get_str("plotter_type")

        logging.info(f"Initializing plotter: {plotter_type}")
        plotter = PlotterRegistry.get(plotter_type)(**plotter_config.as_kwargs())
        plotters.append(plotter)

    return plotters, Config({"plotters": plotters_config})


###############################################################################
############################ ANALYSIS PIPELINE ################################
###############################################################################


def _run_collectors(
    collectors: list[Collector], selectors: list[list[Selector]], save_dir: Path
) -> list[list[JSONLStream]]:
    """Execute all collectors for each selector and persist results to disk.

    Results are written as JSONL files under ``<save_dir>/aux/``. Each sample
    record contains a ``"sample_id"`` key plus one entry per collector output key.

    Args:
        collectors: Collector instances to run on each sample.
        selectors: Per-reader lists of selectors providing sample iterators.
        save_dir: Root output directory; JSONL files are written under
            ``aux/predictions_<NNN>/<selector_idx>.jsonl``.

    Returns:
        Results indexed by ``[predictions_idx][selector_idx]``.
    """
    if not collectors:
        return []

    aux_root = save_dir / "aux"

    # Iterating over predictions selectors
    all_results: list[list[JSONLStream]] = []
    for predictions_idx, predictions_selectors in enumerate(selectors):
        # Create auxiliary sub directory for this predictions reader
        aux_subdir = aux_root / f"predictions_{predictions_idx:03d}"
        aux_subdir.mkdir(parents=True, exist_ok=True)

        # Running all collectors for this predictions
        logging.info(f"  Predictions {predictions_idx + 1}/{len(selectors)}.")
        predictions_results: list[JSONLStream] = []
        for selector_idx, selector in enumerate(predictions_selectors):
            logging.info(f"    Selector {selector_idx + 1}/{len(predictions_selectors)}.")

            aux_path = aux_subdir / f"{selector_idx:03d}.jsonl"
            count = 0
            with open(aux_path, "w") as f:
                # Iterating over all samples in selector
                for sample_idx, sample in enumerate(selector):
                    sample_id = sample.get("sample_id", None)
                    if sample_id is None:
                        logging.error(f"Sample {sample_idx} has no 'sample_id'. This is not good!")

                    # Iterating over all collectors
                    sample_result: dict[str, Any] = {"sample_id": sample_id}
                    for collector in collectors:
                        collector_result = collector.process(sample)  # run collector on the sample
                        for key, value in collector_result.items():
                            if key in sample_result:
                                logging.warning(f"Duplicate key '{key}' for sample {sample_id}. Overwriting!")
                            sample_result[key] = json_friendly(value)
                    f.write(json.dumps(sample_result) + "\n")
                    count += 1

            # Save count to meta file for later use when loading the JSONLStream
            meta_path = aux_subdir / f"{selector_idx:03d}.meta.json"
            with open(meta_path, "w") as meta_file:
                json.dump({"count": count}, meta_file)

            predictions_results.append(JSONLStream(aux_path, count=count))

        all_results.append(predictions_results)

    return all_results


def _run_aggregators(
    aggregators: list[Aggregator], selectors: list[list[Selector]], collector_results: list[list[JSONLStream]]
) -> list[list[list[AggregatorResult]]]:
    """Run all aggregators over the per-sample collector results.

    Args:
        aggregators: Aggregator instances to apply.
        selectors: Per-reader lists of selectors (used for loop indexing).
        collector_results: Output of ``_run_collectors``, indexed by
            ``[predictions_idx][selector_idx]``.

    Returns:
        Results indexed by ``[predictions_idx][selector_idx][aggregator_idx]``.
    """
    if not aggregators:
        return []

    # Iterating over predictions selectors
    all_results: list[list[list[AggregatorResult]]] = []
    for predictions_idx, predictions_selectors in enumerate(selectors):
        logging.info(f"  Predictions {predictions_idx + 1}/{len(selectors)}.")

        # Iterating over selectors for this predictions reader
        predictions_results: list[list[AggregatorResult]] = []
        for selector_idx, selector in enumerate(predictions_selectors):
            logging.info(f"    Selector {selector_idx + 1}/{len(predictions_selectors)}.")

            per_sample_values = collector_results[predictions_idx][selector_idx]

            selector_results: list[AggregatorResult] = []
            for aggregator_idx, aggregator in enumerate(aggregators):
                aggregated = aggregator.aggregate(selector, per_sample_values, aggregator_idx)
                selector_results.append(aggregated)

            predictions_results.append(selector_results)

        all_results.append(predictions_results)

    return all_results


def _run_reporters(reporters: list[Reporter], results: AnalysisResults, save_dir: Path) -> None:
    """Write reports for all configured reporters.

    Args:
        reporters: Reporter instances to execute.
        results: Collected and aggregated analysis results.
        save_dir: Root output directory; reports are written under
            ``<save_dir>/reports/``.
    """
    if not reporters:
        return

    report_dir = save_dir / "reports"

    for idx, reporter in enumerate(reporters):
        logging.info(f"  Reporter {idx + 1}/{len(reporters)}: {reporter.reporter_type}")
        reporter.report(results, report_dir)


def _run_plotters(plotters: list[Plotter], results: AnalysisResults, save_dir: Path) -> None:
    """Generate plots for all configured plotters.

    Args:
        plotters: Plotter instances to execute.
        results: Collected and aggregated analysis results.
        save_dir: Root output directory; plots are written under
            ``<save_dir>/plots/``.
    """
    if not plotters:
        return

    plot_dir = save_dir / "plots"

    for idx, plotter in enumerate(plotters):
        logging.info(f"  Plotter {idx + 1}/{len(plotters)}: {plotter.plotter_type}")
        plotter.plot(results, plot_dir)
