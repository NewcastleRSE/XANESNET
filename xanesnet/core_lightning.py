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

"""Lightning training through ``train_initial`` and ``train_worker`` entry points."""

import importlib
import json
import logging
import os
from argparse import Namespace
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from xanesnet.core_train import (
    _run_training,
    _save_results,
    _setup_dataset,
    _setup_datasource,
    _setup_strategy,
)
from xanesnet.serialization.auto_config import resolve_auto_model_config
from xanesnet.serialization.config import Config
from xanesnet.serialization.model_profile import reset_peak_memory_stats
from xanesnet.serialization.splits import load_splits, save_splits
from xanesnet.utils.exceptions import ConfigError
from xanesnet.utils.lightning import LIGHTNING_CONTEXT_ENV, LIGHTNING_LAUNCH_ENVIRONMENT
from xanesnet.utils.logger import setup_file_logging
from xanesnet.utils.random import set_global_seed

_CONTEXT_NAME = "lightning_context.json"


def _register_lightning_trainer() -> None:
    """Import the Lightning trainer module to register it in this process.

    Raises:
        ConfigError: If the optional Lightning dependency is not installed.
    """
    try:
        importlib.import_module("xanesnet.runners.trainers.lightning")
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("lightning"):
            raise
        raise ConfigError("Lightning training requires `python -m pip install 'xanesnet[lightning]'`.") from exc


###############################################################################
################################## LIGHTNING ##################################
###############################################################################


def train_initial(config: Config, args_namespace: Namespace, save_dir: Path) -> None:
    """Prepare one Lightning run, executes training, and exports its models.

    Builds the data source and dataset, lets the strategy prepare splits and
    encodings, resolves model fields, and publishes the run context before
    ``fit()`` starts workers.
    Only this initial process exports shared models and reports.

    Args:
        config: Validated training configuration with the effective seed set
            by the CLI, possibly containing supported ``"auto"`` fields.
        args_namespace: Parsed CLI arguments (must contain ``tensorboard``;
            must contain ``dry_run`` to select profile output).
        save_dir: Root directory for all training outputs.
    """
    logging.info("Lightning training.")

    _register_lightning_trainer()

    save_dir = save_dir.resolve()
    config_raw = config.as_dict()
    config_raw["dataset"]["root"] = str(Path(config_raw["dataset"]["root"]).resolve())
    config = Config(config_raw)

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)

    strategy = _setup_strategy(config, dataset, save_dir, args_namespace.tensorboard)
    model_splits = strategy.setup_model_splits()
    encoding = strategy.setup_encoding(config)

    logging.info("Resolving 'auto' model fields...")
    config = resolve_auto_model_config(
        config,
        dataset,
        encoding,
    )
    resolved_config_raw = config.as_dict()
    resolved_config_raw["encodings"] = [item.as_dict() for item in encoding.signature]
    config = Config(resolved_config_raw)
    resolved_config_save_path = config.save(save_dir / "resolved_train_config.yaml")
    logging.info(f"Resolved training config saved to: {resolved_config_save_path}.")

    strategy.setup_models(config.section("model"))
    strategy.setup_checkpointer()
    strategy.init_model_weights()
    strategy.setup_trainers(config.get_str("device"))

    # Save signature
    signature = Config(
        {
            "dataset": dataset.signature,
            "model": strategy.model_signature,
            "strategy": strategy.signature,
            "encodings": encoding.signature,
        }
    )
    signature_save_path = signature.save(save_dir / "models" / "signature.yaml")
    logging.info(f"Signature saved to: {signature_save_path}")

    if strategy.checkpointer is not None:
        strategy.checkpointer.set_signature(signature)

    split_indices_save_path = save_dir / "split_indices.json"
    save_splits(split_indices_save_path, model_splits)
    logging.info(f"Split indices saved to: {split_indices_save_path}")

    # Build Lightning context
    context = Config({"config": config.as_dict(), "dry_run": args_namespace.dry_run})

    # Main training
    with _published_context(context, save_dir):
        if args_namespace.dry_run:
            reset_peak_memory_stats(config.get_str("device"))

        model_list, train_time = _run_training(strategy)

        _save_results(
            model_list,
            train_time,
            config,
            dataset,
            encoding,
            signature,
            save_dir,
            args_namespace.dry_run,
        )


def train_worker(context_path: Path) -> None:
    """Start worker for a prepared Lightning run from the given context path.

    Args:
        context_path: Prepared context path detected by the CLI dispatcher.

    Raises:
        ConfigError: If the run context, process role, or prepared data is missing or inconsistent.
    """
    context = _load_worker_context(context_path)

    _register_lightning_trainer()

    config = context.section("config")
    worker_config_raw = config.as_dict()
    worker_config_raw["dataset"]["skip_prepare"] = True
    worker_config = Config(worker_config_raw)

    save_dir = context_path.parent
    rank = os.environ["LOCAL_RANK"]
    worker_log_dir = save_dir / "workers" / f"rank_{rank}"
    worker_log_dir.mkdir(parents=True, exist_ok=True)
    setup_file_logging(worker_log_dir)

    logging.info(f"Lightning worker {rank} loading prepared run from {save_dir}.")

    # Setting the global random seed for reproducibility
    set_global_seed(config.get_int("seed"))

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(worker_config, datasource)

    strategy = _setup_strategy(config, dataset, save_dir, False)
    model_splits = load_splits(save_dir / "split_indices.json")
    strategy.setup_model_splits(model_splits)
    strategy.setup_encoding(config)

    strategy.setup_models(config.section("model"))
    strategy.setup_checkpointer()
    strategy.init_model_weights()
    strategy.setup_trainers(config.get_str("device"))

    # Main training
    if context.get_bool("dry_run"):
        reset_peak_memory_stats(config.get_str("device"))

    _run_training(strategy)


###############################################################################
############################### WORKER CONTEXT ################################
###############################################################################


def _load_worker_context(path: Path) -> Config:
    """Load the prepared configuration and validate the worker launch role.

    Returns:
        Prepared configuration belonging to the current self-launched worker.

    Raises:
        ConfigError: If the context is invalid or the process is not a worker in the expected single-node launch.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigError(f"Cannot load Lightning context: {path}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Lightning context must contain an object.")

    context = Config(payload)
    try:
        rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise ConfigError("Lightning worker requires an integer LOCAL_RANK.") from exc
    if rank <= 0:
        raise ConfigError("Lightning context does not identify this process as a worker.")
    return context


@contextmanager
def _published_context(context: Config, save_dir: Path) -> Generator[None, None, None]:
    """Publish worker context and restore launcher state on every exit.

    Args:
        context: Worker context to publish before ``fit()``.
        save_dir: Run directory containing the context file.

    Yields:
        Control while workers may inherit the context marker.
    """
    path = save_dir / _CONTEXT_NAME
    names = (*LIGHTNING_LAUNCH_ENVIRONMENT, LIGHTNING_CONTEXT_ENV)
    previous = {name: os.environ.get(name) for name in names}
    _write_context(path, context)
    try:
        os.environ[LIGHTNING_CONTEXT_ENV] = str(path)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_context(path: Path, context: Config) -> None:
    """Atomically replace a context file after writing complete JSON data.

    Args:
        path: Final context path inside the run directory.
        context: JSON-compatible context values.
    """
    temporary = path.with_suffix(".tmp")
    try:
        temporary.write_text(json.dumps(context.as_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
