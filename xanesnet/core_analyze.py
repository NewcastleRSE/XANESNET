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

import json
import logging
from argparse import Namespace
from pathlib import Path
from typing import Any

from xanesnet.analysis.aggregators import Aggregator, AggregatorRegistry
from xanesnet.analysis.collectors import Collector, CollectorRegistry
from xanesnet.analysis.plotters import PlotterRegistry
from xanesnet.analysis.reporters import ReporterRegistry
from xanesnet.analysis.selectors import Selector, SelectorRegistry
from xanesnet.serialization import (
    JSONLStream,
    PredictionReader,
    detect_prediction_format,
    json_friendly,
    save_dict_as_yaml,
)

###############################################################################
################################### ANALYZE ###################################
###############################################################################


def analyze(config: dict[str, Any], args_namespace: Namespace, save_dir: Path) -> None:
    """
    Main analysis entry point.
    """
    logging.info("Starting analysis pipeline.")

    predictions_dirs = args_namespace.predictions
    logging.info(f"You provided {len(predictions_dirs)} predictions directories:")

    predictions_readers = _setup_predictions_readers(predictions_dirs)
    selectors, selectors_config = _setup_selectors(config, predictions_readers)
    collectors, collectors_config = _setup_collectors(config)
    aggregators, aggregators_config = _setup_aggregators(config)

    save_dict_as_yaml({"selectors": selectors_config}, save_dir, "selectors")
    save_dict_as_yaml({"collectors": collectors_config}, save_dir, "collectors")
    save_dict_as_yaml({"aggregators": aggregators_config}, save_dir, "aggregators")

    # Run collectors
    collector_results = _run_collectors(collectors, selectors, save_dir)

    # Run aggregators
    aggregator_results = _run_aggregators(aggregators, selectors, collector_results)

    # TODO now we should have finished the main analysis pipeline and have selectors, per-sample values and aggregated results.
    # TODO time for reporting and plotting!

    # TODO Then I want to run 1. reporters, 2. plotters
    # TODO important for reporters and plotters is that they always include necessary config information in their output
    # TODO (e.g. in the filename or in the report text) to ensure traceability of what was done and how to reproduce it.
    # TODO reporters should save results to disk
    # TODO there should exist reporters that report usable files (.json, .csv, .yaml) and also human readable reports (.pdf, .txt, .md)
    # TODO plotters should then plot to disk

    _run_reporters()
    _run_plotters()

    # Close readers
    for predictions_reader in predictions_readers:
        predictions_reader.close()

    # Summary
    logging.info(f"Analysis completed!")


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_predictions_readers(
    predictions_dirs: list[str] | list[Path],
) -> list[PredictionReader]:
    """
    Setup the predictions readers from a list of directories.
    """
    readers: list[PredictionReader] = []
    for predictions_dir in predictions_dirs:
        reader_class = detect_prediction_format(predictions_dir)
        logging.info(f"Detected format for {predictions_dir}: {reader_class.__name__}")
        reader = reader_class(predictions_dir)
        readers.append(reader)

    return readers


def _setup_selectors(
    config: dict[str, Any],
    predictions_readers: list[PredictionReader],
) -> tuple[list[list[Selector]], list[dict[str, Any]]]:
    """
    Setup selectors for each predictions reader based on the configuration.
    If no selectors are configured, use a default 'all' selector for each reader.
    Return a list of lists of selectors (one list per predictions reader) and their corresponding configs.
    """
    selectors_config = config.get("selectors", [])
    assert isinstance(selectors_config, list)

    if len(selectors_config) == 0:
        logging.info("No selectors configured, using 'all' selector for each predictions reader")
        selector_config = {"selector_type": "all"}
        selectors = [
            [SelectorRegistry.get("all")(**selector_config, data_source=reader)] for reader in predictions_readers
        ]
        return selectors, [selector_config]

    selectors: list[list[Selector]] = []
    for selector_config in selectors_config:
        selector_type = selector_config["selector_type"]

        logging.info(f"Initializing selector: {selector_type}")
        selector_list: list[Selector] = []
        for reader in predictions_readers:
            selector = SelectorRegistry.get(selector_type)(**selector_config, data_source=reader)
            selector_list.append(selector)
        selectors.append(selector_list)

    return selectors, selectors_config


def _setup_collectors(config: dict[str, Any]) -> tuple[list[Collector], list[dict[str, Any]]]:
    """
    Setup collectors based on the configuration.
    """
    collectors_config = config.get("collectors", [])
    assert isinstance(collectors_config, list)

    if len(collectors_config) == 0:
        logging.warning("No collectors configured.")
        return [], []

    collectors: list[Collector] = []
    for collector_config in collectors_config:
        collector_type = collector_config["collector_type"]

        logging.info(f"Initializing collector: {collector_type}")
        collector = CollectorRegistry.get(collector_type)(**collector_config)
        collectors.append(collector)

    return collectors, collectors_config


def _setup_aggregators(config: dict[str, Any]) -> tuple[list[Aggregator], list[dict[str, Any]]]:
    """
    Setup aggregators based on the configuration.
    """
    aggregators_config = config.get("aggregators", [])
    assert isinstance(aggregators_config, list)

    if len(aggregators_config) == 0:
        logging.warning("No aggregators configured.")
        return [], []

    aggregators: list[Aggregator] = []
    for aggregator_config in aggregators_config:
        aggregator_type = aggregator_config["aggregator_type"]

        logging.info(f"Initializing aggregator: {aggregator_type}")
        aggregator = AggregatorRegistry.get(aggregator_type)(**aggregator_config)
        aggregators.append(aggregator)
    return aggregators, aggregators_config


###############################################################################
############################ ANALYSIS PIPELINE ################################
###############################################################################


def _run_collectors(
    collectors: list[Collector],
    selectors: list[list[Selector]],
    save_dir: Path,
) -> list[list[JSONLStream]]:
    if not collectors:
        return []

    # Create auxiliary output directories
    aux_root = save_dir / "aux"
    aux_root.mkdir(parents=True, exist_ok=True)

    all_results: list[list[JSONLStream]] = []
    for predictions_idx, predictions_selectors in enumerate(selectors):
        # Create auxiliary sub directory for this predictions reader
        aux_subdir = aux_root / f"predictions_{predictions_idx:03d}"
        aux_subdir.mkdir(parents=True, exist_ok=True)

        # Running all collectors for this predictions
        logging.info(f"Running collectors for predictions {predictions_idx + 1}/{len(selectors)}")
        predictions_results: list[JSONLStream] = []
        for selector_idx, selector in enumerate(predictions_selectors):
            logging.info(f"Selector {selector_idx + 1}/{len(predictions_selectors)}.")

            aux_path = aux_subdir / f"{selector_idx:03d}.jsonl"
            count = 0
            with open(aux_path, "w") as f:
                # Iterating over all samples in selector
                for sample_idx, sample in enumerate(selector):
                    sample_id = sample.get("sample_id", None)
                    if sample_id is None:
                        logging.error(f"Sample {sample_idx} has no 'sample_id'. This is not good!")
                    sample_id = json_friendly(sample_id)

                    # Iterating over all collectors
                    sample_result: dict[str, Any] = {"sample_id": sample_id}
                    for collector in collectors:
                        collector_result = collector.process(sample)
                        for key, value in collector_result.items():
                            if key in sample_result:
                                logging.warning(f"Duplicate key '{key}' for sample {sample_id}. Overwriting!")
                            sample_result[key] = json_friendly(value)
                    f.write(json.dumps(sample_result) + "\n")
                    count += 1

            meta_path = aux_subdir / f"{selector_idx:03d}.meta.json"
            with open(meta_path, "w") as meta_file:
                json.dump({"count": count}, meta_file)

            predictions_results.append(JSONLStream(aux_path, count=count))

        all_results.append(predictions_results)

    return all_results


def _run_aggregators(
    aggregators: list[Aggregator],
    selectors: list[list[Selector]],
    collector_results: list[list[JSONLStream]],
) -> list[list[list[dict[str, Any]]]]:
    if not aggregators:
        return []

    all_results: list[list[list[dict[str, Any]]]] = []

    for predictions_idx, predictions_selectors in enumerate(selectors):
        logging.info(f"Running aggregators for predictions {predictions_idx + 1}/{len(selectors)}")

        predictions_results: list[list[dict[str, Any]]] = []
        for selector_idx, selector in enumerate(predictions_selectors):
            logging.info(f"Selector {selector_idx + 1}/{len(predictions_selectors)}.")

            per_sample_values = collector_results[predictions_idx][selector_idx]

            selector_results: list[dict[str, Any]] = []
            for aggregator in aggregators:
                try:
                    aggregated = aggregator.aggregate(selector, per_sample_values)
                except Exception as e:
                    logging.warning(f"Aggregator '{aggregator.aggregator_type}' failed on selector {selector_idx}: {e}")
                    aggregated = {}
                selector_results.append(aggregated)

            predictions_results.append(selector_results)

        all_results.append(predictions_results)

    return all_results


def _run_reporters() -> None:
    # TODO Implement
    pass


def _run_plotters() -> None:
    # TODO Implement
    pass
