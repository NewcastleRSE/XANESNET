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
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import Any

from xanesnet.analysis.aggregators import Aggregator, AggregatorRegistry
from xanesnet.analysis.per_sample import PerSampleModule, PerSampleRegistry
from xanesnet.analysis.plotters import PlotterRegistry
from xanesnet.analysis.reporters import ReporterRegistry
from xanesnet.analysis.selectors import Selector, SelectorRegistry
from xanesnet.serialization import PredictionReader, detect_prediction_format

###############################################################################
################################### ANALYZE ###################################
###############################################################################


def analyze(config: dict[str, Any], args_namespace: Namespace, save_dir: Path) -> None:
    """
    Main analysis entry point.
    """
    logging.info("Starting analysis pipeline.")

    # Load predictions
    predictions_reader = _setup_predictions_reader(args_namespace.predictions)

    # Run analysis
    analysis_time = _run_analysis_pipeline(config, predictions_reader, save_dir)

    # Close reader
    predictions_reader.close()

    # Summary
    logging.info(f"Analysis completed in {str(timedelta(seconds=int(analysis_time)))}")


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_predictions_reader(predictions_path: str | Path) -> PredictionReader:
    """
    Setup the predictions reader from path.
    """
    logging.info(f"Loading predictions from: {predictions_path}")

    reader_class = detect_prediction_format(predictions_path)
    logging.info(f"Detected format: {reader_class.__name__}")

    return reader_class(predictions_path)


###############################################################################
############################ ANALYSIS PIPELINE ################################
###############################################################################


def _run_analysis_pipeline(config: dict[str, Any], predictions_reader: PredictionReader, save_dir: Path) -> float:
    """
    Run the complete analysis pipeline:
    """
    start_time = time.time()

    # Stage 1: Selection
    logging.info("Stage 1: Applying selectors")
    selectors, selector_configs = _apply_selectors(config, predictions_reader)

    # Stage 2: Per-sample processing
    logging.info("Stage 2: Collecting per-sample values")
    per_sample_results, per_sample_configs = _collect_per_sample(config, selectors)

    # Stage 3: Aggregation
    logging.info("Stage 3: Aggregating results")
    aggregated_results, aggregator_configs = _aggregate_results(config, selectors, per_sample_results)

    # Stage 4: Plotting
    logging.info("Stage 4: Generating plots")
    _generate_plots(
        config,
        selectors,
        selector_configs,
        per_sample_results,
        per_sample_configs,
        aggregated_results,
        aggregator_configs,
        save_dir / "plots",
    )

    # Stage 5: Reporting
    logging.info("Stage 5: Generating reports")
    _generate_reports(
        config,
        selectors,
        selector_configs,
        per_sample_results,
        per_sample_configs,
        aggregated_results,
        aggregator_configs,
        save_dir / "reports",
    )

    analysis_time = time.time() - start_time

    return analysis_time


def _apply_selectors(
    config: dict[str, Any],
    predictions_reader: PredictionReader,
) -> tuple[list[Selector], list[dict[str, Any]]]:
    """
    Apply selectors and return a list of Selector objects
    (Iterables that can be iterated multiple times).
    """
    selectors_config = config.get("selectors", [])

    if len(selectors_config) == 0:
        logging.info("No selectors configured, using 'all' selector")
        selector_cfg = {"selector_type": "all"}
        selector = SelectorRegistry.get("all")(**selector_cfg, data_source=predictions_reader)

        return [selector], [selector_cfg]

    selections: list[Selector] = []
    configs: list[dict[str, Any]] = []
    for selector_config in selectors_config:
        selector_type = selector_config["selector_type"]

        logging.info(f"Applying selector: {selector_type}")
        selector = SelectorRegistry.get(selector_type)(**selector_config, data_source=predictions_reader)
        selections.append(selector)
        configs.append(selector_config)

    return selections, configs


def _collect_per_sample(
    config: dict[str, Any],
    selectors: list[Selector],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Collect per-sample values from each selection using per-sample modules.
    """
    per_sample_config = config.get("per_sample", [])

    if len(per_sample_config) == 0:
        logging.warning("No per-sample modules configured, skipping per-sample collection")
        return [], []

    # Initialize per-sample modules
    modules: list[PerSampleModule] = []
    for module_config in per_sample_config:
        per_sample_type = module_config["per_sample_type"]

        logging.info(f"Initializing per-sample module: {per_sample_type}")
        module = PerSampleRegistry.get(per_sample_type)(**module_config)
        modules.append(module)

    # Process each selection
    all_results: list[list[dict[str, Any]]] = []
    for selection_idx, selector in enumerate(selectors):
        logging.info(f"Processing selection {selection_idx + 1}/{len(selectors)}")

        selection_results: list[dict[str, Any]] = []
        for sample in selector:
            value_dict: dict[str, Any] = {}

            # Adding sample_id to value_dict if available, for safe traceability
            sample_id = sample.get("sample_id", None)
            if sample_id is None:
                logging.warning("Sample without 'sample_id' encountered. This is not good!")
            else:
                value_dict["sample_id"] = sample_id

            for module in modules:
                result_dict = module.process(sample)

                for result_name, result_value in result_dict.items():
                    if result_name in value_dict:
                        logging.warning(
                            f"Result name '{result_name}' from module '{module.per_sample_type}' "
                            "conflicts with existing key in value_dict. "
                            "Overwriting existing value."
                        )
                    value_dict[result_name] = result_value

            selection_results.append(value_dict)

        all_results.append(selection_results)

    return all_results, per_sample_config


def _aggregate_results(
    config: dict[str, Any],
    selectors: list[Selector],
    per_sample_results: list[list[dict[str, Any]]],
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    """
    Aggregate per-sample results for each selection.
    """
    aggregators_config = config.get("aggregators", [])

    if len(aggregators_config) == 0:
        logging.warning("No per-sample results to aggregate")
        return [], []

    # Initialize aggregators
    aggregators: list[Aggregator] = []
    for aggregator_config in aggregators_config:
        aggregator_type = aggregator_config["aggregator_type"]

        logging.info(f"Initializing aggregator: {aggregator_type}")
        aggregator = AggregatorRegistry.get(aggregator_type)(**aggregator_config)
        aggregators.append(aggregator)

    # Process each selection
    all_aggregated: list[list[dict[str, Any]]] = []
    for selection_idx, (selector, selection_results) in enumerate(zip(selectors, per_sample_results)):
        logging.info(f"Aggregating selection {selection_idx + 1}/{len(per_sample_results)}")

        aggregated_results: list[dict[str, Any]] = []
        for aggregator in aggregators:
            logging.debug(f"Selection {selection_idx + 1}: Running aggregator: {aggregator.aggregator_type}")
            result = aggregator.aggregate(selector, selection_results)
            aggregated_results.append(result)

        all_aggregated.append(aggregated_results)

    return all_aggregated, aggregators_config


def _generate_plots(
    config: dict[str, Any],
    selectors: list[Selector],
    selector_configs: list[dict[str, Any]],
    per_sample_results: list[list[dict[str, Any]]],
    per_sample_configs: list[dict[str, Any]],
    aggregated_results: list[list[dict[str, Any]]],
    aggregator_configs: list[dict[str, Any]],
    plots_dir: Path,
) -> None:
    """
    Generate plots from results for each selection.
    """
    plotters_config = config.get("plotters", [])

    if len(plotters_config) == 0:
        logging.warning("No plotters configured, skipping plotting")
        return

    if not plots_dir.is_dir():
        raise FileNotFoundError(f"Plots directory does not exist: {plots_dir}")

    # Generate plots for each selection
    for selection_idx, (selector, selector_config, per_sample_result, aggregated_result) in enumerate(
        zip(selectors, selector_configs, per_sample_results, aggregated_results)
    ):
        selection_dir = plots_dir / f"selection_{selection_idx}"
        selection_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating plots for selection {selection_idx + 1}/{len(per_sample_results)}")

        for plotter_config in plotters_config:
            plotter_type = plotter_config["plotter_type"]

            logging.debug(f"Selection {selection_idx + 1}: Generating plot: {plotter_type}")
            plotter = PlotterRegistry.get(plotter_type)(**plotter_config)

            plotter.plot(
                selector,
                selector_config,
                per_sample_result,
                per_sample_configs,
                aggregated_result,
                aggregator_configs,
                selection_dir,
            )


def _generate_reports(
    config: dict[str, Any],
    selectors: list[Selector],
    selector_configs: list[dict[str, Any]],
    per_sample_results: list[list[dict[str, Any]]],
    per_sample_configs: list[dict[str, Any]],
    aggregated_results: list[list[dict[str, Any]]],
    aggregator_configs: list[dict[str, Any]],
    reports_dir: Path,
) -> None:
    """
    Generate reports from results for each selection.
    """
    reporters_config = config.get("reporters", [])

    if not reporters_config:
        logging.warning("No reporters configured")
        return

    reports_dir.mkdir(parents=True, exist_ok=True)

    # Generate reports for each selection
    for selection_idx, (selector, per_sample_res, aggregated_res) in enumerate(
        zip(selectors, per_sample_results, aggregated_results)
    ):
        selection_dir = reports_dir / f"selection_{selection_idx}"
        selection_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Generating reports for selection {selection_idx + 1}/{len(per_sample_results)}")

        for reporter_cfg in reporters_config:
            reporter_type = reporter_cfg["type"]
            reporter_params = {k: v for k, v in reporter_cfg.items() if k != "type"}

            logging.debug(f"Selection {selection_idx + 1}: Generating report: {reporter_type}")
            reporter_class = ReporterRegistry.get(reporter_type)
            reporter = reporter_class(**reporter_params)

            reporter.report(selector, per_sample_res, aggregated_res, selection_dir)
