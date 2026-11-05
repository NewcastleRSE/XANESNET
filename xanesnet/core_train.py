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

"""Core training pipeline: dataset setup, strategy initialization, and model training."""

import logging
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path
from typing import cast

import torch

from xanesnet.checkpointing import Checkpointer
from xanesnet.components import Scheduler
from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.runners.trainers.base import Trainer
from xanesnet.serialization.auto_config import resolve_auto_model_config
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config
from xanesnet.serialization.model_profile import (
    build_model_profile,
    create_model_summary,
    get_peak_memory_allocated_mb,
    reset_peak_memory_stats,
    save_model_profile,
)
from xanesnet.serialization.splits import save_separate_splits, save_splits
from xanesnet.strategies import Strategy, StrategyRegistry

###############################################################################
#################################### TRAIN ####################################
###############################################################################


def train(config: Config, args_namespace: Namespace, save_dir: Path, checkpoint: Checkpoint | None) -> None:
    """Run the complete training pipeline.

    Builds the data source and dataset, lets the strategy prepare splits and
    encodings, resolves model fields, then runs training and writes artifacts
    under ``save_dir``.

    Args:
        config: Validated training configuration, possibly containing
            ``"auto"`` model fields supported by the resolver.
        args_namespace: Parsed CLI arguments (must contain ``tensorboard``;
            may contain ``dry_run`` to enable profile output).
        save_dir: Root directory for all training outputs.
        checkpoint: Optional checkpoint already loaded by the training CLI.
    """
    logging.info("Training.")

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)

    strategy = _setup_strategy(config, dataset, save_dir, args_namespace.tensorboard)
    model_splits = strategy.setup_model_splits()
    encoding = strategy.setup_encoding(config)

    logging.info("Resolving 'auto' model fields...")
    config = resolve_auto_model_config(config, dataset, encoding)
    resolved_config_raw = config.as_dict()
    resolved_config_raw["encodings"] = [item.as_dict() for item in encoding.signature]
    config = Config(resolved_config_raw)
    resolved_config_save_path = config.save(save_dir / "resolved_train_config.yaml")
    logging.info(f"Resolved training config saved to: {resolved_config_save_path}.")

    strategy.setup_models(config.section("model"))
    checkpointer = strategy.setup_checkpointer()
    strategy.init_model_weights()
    strategy.setup_trainers(config.get_str("device"))

    # Restore checkpoint if provided
    if checkpoint is not None:
        strategy.set_state_dicts(checkpoint.model_states)
        logging.info("Restored model states from checkpoint.")
        if not args_namespace.restart_training_state:
            _restore_optimizer(strategy, checkpoint)
            _restore_scheduler(strategy, checkpoint)
            if checkpoint.training_state is not None:
                logging.info("Restored optimizer and scheduler states from checkpoint.")

    # Save signature
    signature_save_path = checkpointer.signature.save(save_dir / "models" / "signature.yaml")
    logging.info(f"Signature saved to: {signature_save_path}")

    split_indices_save_path = save_dir / "split_indices.json"
    save_splits(split_indices_save_path, model_splits)
    save_separate_splits(save_dir, model_splits)
    logging.info(f"Split indices saved to: {split_indices_save_path}")

    # Main training
    if args_namespace.dry_run:
        reset_peak_memory_stats(config.get_str("device"))

    model_list, train_time = _run_training(strategy)

    _save_results(
        model_list,
        train_time,
        config,
        dataset,
        encoding,
        checkpointer,
        save_dir,
        args_namespace.dry_run,
    )


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


def _setup_dataset(config: Config, datasource: DataSource) -> Dataset:
    """Construct the initial dataset.

    Args:
        config: Validated configuration containing a ``dataset`` section.
        datasource: Data source providing raw data.

    Returns:
        Prepared dataset without initialized splits.
    """
    dataset_config = config.section("dataset")
    dataset_type = dataset_config.get_str("dataset_type")

    logging.info(f"Initializing training dataset: {dataset_type}")
    dataset = DatasetRegistry.create(dataset_type, **dataset_config.as_kwargs(), datasource=datasource)
    dataset.prepare()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_strategy(
    config: Config,
    dataset: Dataset,
    save_dir: Path,
    enable_tensorboard: bool,
) -> Strategy:
    """Instantiate the training strategy from config.

    Args:
        config: Validated configuration containing ``strategy``, ``model``, and ``trainer`` sections.
        dataset: Prepared dataset whose splits have not been initialized.
        save_dir: Root run directory.
        enable_tensorboard: Whether to enable TensorBoard logging.

    Returns:
        Configured strategy instance (models not yet initialized).
    """
    strategy_config = config.section("strategy")
    strategy_type = strategy_config.get_str("strategy_type")

    trainer_config = config.section("trainer")

    logging.info(f"Initializing strategy: {strategy_type}")
    strategy = StrategyRegistry.create(
        strategy_type,
        **strategy_config.as_kwargs(),
        checkpoint_dir=save_dir / "checkpoints",
        tensorboard_dir=save_dir / "tensorboard" if enable_tensorboard else None,
        dataset=dataset,
        trainer_config=trainer_config,
        inferencer_config=None,
    )

    return strategy


###############################################################################
############################## RESTORE FUNCTIONS ##############################
###############################################################################


def _restore_optimizer(
    strategy: Strategy,
    checkpoint: Checkpoint,
) -> None:
    """Restore optimizer state for each strategy member from a checkpoint.

    Args:
        strategy: Prepared strategy with initialized trainers.
        checkpoint: Checkpoint containing per-member optimizer states.
    """
    training_state = checkpoint.training_state
    if training_state is None:
        return

    trainers = cast(list[Trainer], strategy.trainers)
    for trainer, state in zip(trainers, training_state["optimizer_states"]):
        trainer.optimizer.torch_optimizer.load_state_dict(state)


def _restore_scheduler(
    strategy: Strategy,
    checkpoint: Checkpoint,
) -> None:
    """Restore epoch and warm-up scheduler state for each strategy member.

    Args:
        strategy: Prepared strategy with initialized trainers and schedulers.
        checkpoint: Checkpoint containing per-member scheduler states.
    """
    training_state = checkpoint.training_state
    if training_state is None:
        return

    trainers = cast(list[Trainer], strategy.trainers)
    for trainer, epoch_state, warmup_state in zip(
        trainers, training_state["epoch_scheduler_states"], training_state["warmup_scheduler_states"]
    ):
        trainer.epoch_lr_scheduler.load_state_dict(epoch_state)
        if warmup_state is not None:
            cast(Scheduler, trainer.warmup_lr_scheduler).load_state_dict(warmup_state)


###############################################################################
############################## TRAINING STARTER ###############################
###############################################################################


def _run_training(strategy: Strategy) -> tuple[list[Model], float]:
    """Execute training and return trained models with elapsed time.

    Moves all models to CPU after training completes.

    Args:
        strategy: Fully initialized training strategy.

    Returns:
        A ``(model_list, train_time)`` tuple where ``train_time`` is elapsed
        wall-clock time in seconds.
    """
    start_time = time.perf_counter()

    model_list = strategy.run_training()

    train_time = time.perf_counter() - start_time

    # Move to CPU
    for model in model_list:
        model.to(torch.device("cpu"))

    return model_list, train_time


###############################################################################
############################### SUMMARY LOGGING ###############################
###############################################################################


def _save_results(
    model_list: list[Model],
    train_time: float,
    config: Config,
    dataset: Dataset,
    encoding: SpectraEncoding,
    checkpointer: Checkpointer,
    save_dir: Path,
    dry_run: bool,
) -> None:
    """Write the final checkpoints, model summaries, and optional profiling output.

    Args:
        model_list: Trained models on CPU.
        train_time: Elapsed strategy execution time in seconds.
        config: Resolved training configuration.
        dataset: Prepared dataset used for summary inputs.
        encoding: Prepared encoding used during training.
        checkpointer: Checkpointer collecting and saving the final model and training state.
        save_dir: Experiment output directory.
        dry_run: Whether to write the existing model profiling reports.
    """
    logging.info(f"Number of trained models: {len(model_list)}")
    logging.info(f"Training completed in {timedelta(seconds=int(train_time))}")
    if dry_run:
        device = config.get_str("device")
        peak_gpu_memory_allocated_mb = get_peak_memory_allocated_mb(device)
        model_profile = build_model_profile(model_list[0], dataset, device, peak_gpu_memory_allocated_mb, encoding)
        profile_json_path, profile_readable_path = save_model_profile(save_dir, model_profile)
        logging.info(f"Dry-run model profile JSON saved to: {profile_json_path}")
        logging.info(f"Dry-run model profile readable report saved to: {profile_readable_path}")
    try:
        _summary_models(model_list, dataset, encoding)
    except Exception as exc:
        logging.warning(f"Model summary failed and will be skipped: {exc}")
    checkpointer.save(save_dir / "models" / "final.pth")
    logging.info(f"Final checkpoints saved to: {save_dir / 'models'}")


def _summary_models(model_list: list[Model], dataset: Dataset, encoding: SpectraEncoding) -> None:
    """Log a torchinfo summary for each trained model.

    Args:
        model_list: Trained model instances.
        dataset: Dataset used during training (for batch input preparation).
        encoding: Prepared spectra encoding used by the models.
    """
    logging.info("Model Summary")

    for idx, model in enumerate(model_list):
        logging.info(f"Model  {idx}:")
        create_model_summary(model, dataset, encoding=encoding)
