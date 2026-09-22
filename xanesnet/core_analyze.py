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

"""Core analysis pipeline: setup, collection, aggregation, reporting, and plotting."""

import json
import logging
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, TypeVar

from tqdm import tqdm

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
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.serialization.config import Config, load_raw_config
from xanesnet.serialization.jsonl_stream import JSONLStream, json_friendly
from xanesnet.serialization.prediction_readers import (
    PredictionReader,
    build_prediction_reader,
)
from xanesnet.utils.exceptions import ConfigError, ResourceError
from xanesnet.utils.registry import Registry

# Any component instantiated from a configuration section by ``_setup_components``.
_ComponentT = TypeVar("_ComponentT")

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
        args_namespace: Parsed CLI arguments (must contain ``inference_runs``
            and, optionally, ``prediction_names``).
        save_dir: Root directory for all analysis outputs.
    """
    logging.info("Analysis.")
    started = time.perf_counter()

    preload = config.get_bool("preload")
    logging.info(f"Preload predictions and structures: {preload}")

    inference_run_dirs = args_namespace.inference_runs
    logging.info(f"You provided {len(inference_run_dirs)} inference run directories:")

    run_dir_names: list[str] = [Path(d).name for d in inference_run_dirs]
    prediction_names: list[str] = list(args_namespace.prediction_names or [])
    if not prediction_names:
        logging.info("No prediction names provided; using inference run directory names as labels.")
        prediction_names = run_dir_names
    elif len(prediction_names) != len(run_dir_names):
        if len(prediction_names) > len(run_dir_names):
            mismatch_note = "surplus names are ignored."
        else:
            mismatch_note = "remaining readers fall back to inference run directory names."
        logging.warning(
            f"Number of prediction names ({len(prediction_names)}) does not match the number of "
            f"inference runs ({len(run_dir_names)}); {mismatch_note}"
        )
    names_padded = (prediction_names + run_dir_names)[: len(run_dir_names)]

    predictions_readers = _setup_predictions_readers(inference_run_dirs, preload)
    try:
        logging.info("Setup")
        selectors, selectors_config = _setup_selectors(config, predictions_readers)
        collectors, collectors_config = _setup_components(config, "collectors", CollectorRegistry)
        aggregators, aggregators_config = _setup_components(config, "aggregators", AggregatorRegistry)
        reporters, reporters_config = _setup_components(config, "reporters", ReporterRegistry)
        plotters, plotters_config = _setup_components(config, "plotters", PlotterRegistry)

        n_combinations = len(selectors) * len(selectors[0]) if selectors else 0
        logging.info(
            f"  Pipeline: {len(selectors)} predictions x {len(selectors[0]) if selectors else 0} selectors "
            f"= {n_combinations} method(s); {len(collectors)} collector(s), {len(aggregators)} aggregator(s), "
            f"{len(reporters)} reporter(s), {len(plotters)} plotter(s)."
        )

        selectors_config.save(save_dir / "selectors.yaml")
        collectors_config.save(save_dir / "collectors.yaml")
        aggregators_config.save(save_dir / "aggregators.yaml")
        reporters_config.save(save_dir / "reporters.yaml")
        plotters_config.save(save_dir / "plotters.yaml")

        logging.info(f"Running Collectors ({len(collectors)})")
        collector_results = _run_collectors(collectors, selectors, save_dir)

        logging.info(f"Running Aggregators ({len(aggregators)})")
        aggregator_results = _run_aggregators(aggregators, selectors, collector_results)

        results = AnalysisResults(
            selectors=selectors,
            collector_results=collector_results,
            aggregator_results=aggregator_results,
            prediction_names=names_padded,
        )

        logging.info(f"Running Reporters ({len(reporters)})")
        _run_reporters(reporters, results, save_dir)

        logging.info(f"Running Plotters ({len(plotters)})")
        _run_plotters(plotters, results, save_dir)
    finally:
        for predictions_reader in predictions_readers:
            predictions_reader.close()

    logging.info(f"Analysis completed in {_format_duration(time.perf_counter() - started)}. Output: '{save_dir}'")


###############################################################################
############################### SETUP FUNCTIONS ###############################
###############################################################################


def _setup_datasource(config: Config) -> DataSource:
    """Instantiate the data source from config.

    Args:
        config: Validated configuration containing a ``datasource`` section.

    Returns:
        Initialized data source.
    """
    datasource_config = config.section("datasource")
    datasource_type = datasource_config.get_str("datasource_type")
    logging.info(f"Initializing data source: {datasource_type}")
    datasource = DataSourceRegistry.create(datasource_type, **datasource_config.as_kwargs())

    return datasource


def _setup_predictions_readers(inference_run_dirs: list[str] | list[Path], preload: bool) -> list[PredictionReader]:
    """Create prediction readers from inference run directories.

    Auto-detects the format in each run's ``predictions`` directory. When the
    run also contains a readable ``validated_infer_config.yaml`` and its
    datasource can be loaded, the returned reader attaches matching
    structures. With ``preload`` enabled, prediction records and matched
    structures are materialized into memory during setup.

    Args:
        inference_run_dirs: Paths to inference run directories.
        preload: Whether to preload predictions and structures into memory.

    Returns:
        One plain or structure-enriched ``PredictionReader`` per inference run.
    """
    readers: list[PredictionReader] = []
    for inference_run_dir in inference_run_dirs:
        inference_run_path = Path(inference_run_dir)
        predictions_dir = inference_run_path / "predictions"

        datasource: DataSource | None = None
        config_path = inference_run_path / "validated_infer_config.yaml"
        try:
            logging.info(f"Loading inference configuration: {config_path}")
            inference_config = Config(load_raw_config(config_path))
            datasource = _setup_datasource(inference_config)
        except (ConfigError, ResourceError, OSError, KeyError, ValueError) as exc:
            logging.warning(
                f"Could not load the data source declared in {config_path}: {exc!r}. Continuing with prediction data only."
            )
        readers.append(build_prediction_reader(predictions_dir, datasource=datasource, preload=preload))

    return readers


def _setup_components(
    config: Config,
    section: str,
    registry: Registry[type[_ComponentT], str],
) -> tuple[list[_ComponentT], Config]:
    """Instantiate every component of one pipeline section.

    Each entry of the ``<section>`` list selects a registered component by its
    ``<singular>_type`` key; the remaining keys are forwarded to the component
    constructor.

    Args:
        config: Validated analysis configuration.
        section: Configuration section name, for example ``"collectors"``.
        registry: Registry that resolves the component type keys of ``section``.

    Returns:
        A ``(components, section_config)`` tuple where ``section_config`` wraps
        the raw entries for saving alongside the results.
    """
    type_key = f"{section[:-1]}_type"
    entries = config.get_config_list(section)

    if len(entries) == 0:
        logging.warning(f"No {section} configured.")
        return [], Config({section: []})

    components: list[_ComponentT] = []
    for entry in entries:
        component_type = entry.get_str(type_key)
        component = registry.create(component_type, **entry.as_kwargs())
        components.append(component)
        logging.info(f"  Initializing {section[:-1]}: {component!r}")

    return components, Config({section: entries})


def _setup_selectors(
    config: Config, predictions_readers: list[PredictionReader]
) -> tuple[list[list[Selector]], Config]:
    """Instantiate selectors for every prediction reader.

    If no selectors are specified in ``config``, an ``'all'`` selector is
    created for each reader. Otherwise every configured selector is
    instantiated for every reader. Selectors may expand into several
    selectors.

    Args:
        config: Validated analysis configuration.
        predictions_readers: Readers to attach selectors to.

    Returns:
        A ``(selectors, selectors_config)`` tuple where ``selectors`` is a
        prediction-first list indexed by ``[predictions_idx][selector_idx]``.
    """
    selectors_config = config.get_config_list("selectors")

    if len(selectors_config) == 0:
        logging.warning("  No selectors configured, using the 'all' selector for each predictions reader.")
        selectors_config = [Config({"selector_type": "all"})]

    selectors: list[list[Selector]] = []
    # Selector expansion is intentionally assumed symmetric:
    # all selector expansions applied to each reader must yield the same number of selectors.
    # If that invariant changes, selector indices must become reader-specific.
    for reader in predictions_readers:
        predictions_selectors: list[Selector] = []
        for selector_config in selectors_config:
            selector_kwargs = selector_config.as_kwargs()
            base = SelectorRegistry.create(selector_kwargs["selector_type"], **selector_kwargs, data_source=reader)
            predictions_selectors.extend(base.expand_selectors())
        selectors.append(predictions_selectors)

    for selector_idx, selector in enumerate(selectors[0]):
        sizes = ", ".join(f"({len(ps[selector_idx])})" for ps in selectors)
        logging.info(f"  Initializing selector: {selector!r} ({sizes})")

    return selectors, Config({"selectors": [selector.signature for selector in selectors[0]]})


###############################################################################
############################ ANALYSIS PIPELINE ################################
###############################################################################


def _run_collectors(
    collectors: list[Collector], selectors: list[list[Selector]], save_dir: Path
) -> list[list[JSONLStream]]:
    """Execute all collectors for each selector and persist results to disk.

    Results are written as JSONL files under ``<save_dir>/aux/``. Each sample
    record contains ``"sample_id"`` and ``"target_site_index"`` plus one
    entry per collector output key.

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

    all_results: list[list[JSONLStream]] = []
    for predictions_idx, predictions_selectors in enumerate(selectors):
        aux_subdir = aux_root / f"predictions_{predictions_idx:03d}"
        aux_subdir.mkdir(parents=True, exist_ok=True)

        logging.info(f"  Predictions {predictions_idx + 1}/{len(selectors)}.")
        predictions_results: list[JSONLStream] = []
        for selector_idx, selector in enumerate(predictions_selectors):
            started = time.perf_counter()
            aux_path = aux_subdir / f"{selector_idx:03d}.jsonl"
            count = _write_collector_records(collectors, selector, aux_path)

            # Persist the count so the stream length is known without a re-read.
            meta_path = aux_subdir / f"{selector_idx:03d}.meta.json"
            with open(meta_path, "w") as meta_file:
                json.dump({"count": count}, meta_file)

            predictions_results.append(JSONLStream(aux_path, count=count))
            logging.info(
                f"    Selector {selector_idx + 1}/{len(predictions_selectors)}: "
                f"{count} sample(s) in {_format_duration(time.perf_counter() - started)}."
            )

        all_results.append(predictions_results)

    return all_results


def _write_collector_records(collectors: list[Collector], selector: Selector, aux_path: Path) -> int:
    """Run every collector on every selected sample and write one JSONL record per sample.

    Collectors are expected to use distinct output keys. Duplicate output keys
    are rejected because overwriting a value would corrupt the analysis.

    Args:
        collectors: Collector instances to run on each sample.
        selector: Selector providing the samples to process.
        aux_path: Destination JSONL path.

    Returns:
        Number of records written.
    """
    count = 0
    with open(aux_path, "w") as f:
        for sample in tqdm(selector, desc="Collecting", total=len(selector)):
            sample_id = sample["sample_id"]
            target_site_index = sample.get("target_site_index")
            sample_result: dict[str, Any] = {}
            for collector in collectors:
                for key, value in collector.process(sample).items():
                    if key in sample_result:
                        raise ConfigError(f"Duplicate collector key '{key}' for sample {sample_id}.")
                    sample_result[key] = json_friendly(value)
            # Keep the source identity authoritative if a custom collector
            # attempted to emit a reserved metadata key.
            sample_result["sample_id"] = sample_id
            sample_result["target_site_index"] = json_friendly(target_site_index)
            f.write(json.dumps(sample_result) + "\n")
            count += 1
    return count


def _run_aggregators(
    aggregators: list[Aggregator], selectors: list[list[Selector]], collector_results: list[list[JSONLStream]]
) -> list[list[list[AggregatorResult]]]:
    """Run all aggregators over the per-sample collector results.

    Args:
        aggregators: Aggregator instances to apply.
        selectors: Per-reader lists of selectors (used for loop indexing).
        collector_results: Output of ``_run_collectors``, indexed by
            ``[predictions_idx][selector_idx]``. May be empty when no collectors
            were configured.

    Returns:
        Results indexed by ``[predictions_idx][selector_idx][aggregator_idx]``.
    """
    if not aggregators:
        return []

    all_results: list[list[list[AggregatorResult]]] = []
    for predictions_idx, predictions_selectors in enumerate(selectors):
        logging.info(f"  Predictions {predictions_idx + 1}/{len(selectors)}.")

        predictions_results: list[list[AggregatorResult]] = []
        for selector_idx, selector in enumerate(predictions_selectors):
            logging.info(f"    Selector {selector_idx + 1}/{len(predictions_selectors)}.")

            per_sample_values: JSONLStream | None = None
            if predictions_idx < len(collector_results) and selector_idx < len(collector_results[predictions_idx]):
                per_sample_values = collector_results[predictions_idx][selector_idx]

            selector_results: list[AggregatorResult] = []
            for aggregator_idx, aggregator in enumerate(aggregators):
                started = time.perf_counter()
                selector_results.append(aggregator.aggregate(selector, per_sample_values, aggregator_idx))
                logging.info(f"      {aggregator!r} ({_format_duration(time.perf_counter() - started)}).")

            predictions_results.append(selector_results)

        all_results.append(predictions_results)

    return all_results


def _run_reporters(reporters: list[Reporter], results: AnalysisResults, save_dir: Path) -> None:
    """Write reports for all configured reporters.

    Args:
        reporters: Reporter instances to execute.
        results: Collected and aggregated analysis results.
        save_dir: Root output directory; each reporter receives a unique
            directory under ``<save_dir>/reports/`` named after its string
            label and configuration order, for example ``statistics_000``.
    """
    if not reporters:
        return

    report_dir = save_dir / "reports"

    for idx, reporter in enumerate(reporters):
        started = time.perf_counter()
        logging.info(f"  Reporter {idx + 1}/{len(reporters)}: {reporter!r}")
        reporter_dir = report_dir / f"{str(reporter)}_{idx:03d}"
        reporter_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"    Output directory: '{reporter_dir}'.")
        reporter.report(results, reporter_dir)
        logging.info(f"    Done in {_format_duration(time.perf_counter() - started)}.")


def _run_plotters(plotters: list[Plotter], results: AnalysisResults, save_dir: Path) -> None:
    """Generate plots for all configured plotters.

    Args:
        plotters: Plotter instances to execute.
        results: Collected and aggregated analysis results.
        save_dir: Root output directory; each plotter receives a unique
            directory under ``<save_dir>/plots/`` named after its string label
            and configuration order, for example ``scalar_000``.
    """
    if not plotters:
        return

    plot_dir = save_dir / "plots"

    for idx, plotter in enumerate(plotters):
        started = time.perf_counter()
        logging.info(f"  Plotter {idx + 1}/{len(plotters)}: {plotter!r}")
        plotter_dir = plot_dir / f"{str(plotter)}_{idx:03d}"
        plotter_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"    Output directory: '{plotter_dir}'.")
        plotter.plot(results, plotter_dir)
        logging.info(f"    Done in {_format_duration(time.perf_counter() - started)}.")


def _format_duration(seconds: float) -> str:
    """Format an elapsed duration for console output.

    Args:
        seconds: Elapsed wall-clock time in seconds.

    Returns:
        Compact ``"1m 03s"`` or ``"4.2s"`` style string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"
