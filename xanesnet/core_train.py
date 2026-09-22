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

"""Core training pipeline: dataset setup, strategy initialization, and model training."""

import logging
import time
from argparse import Namespace
from datetime import timedelta
from pathlib import Path

import torch

from xanesnet.datasets import Dataset, DatasetRegistry
from xanesnet.datasources import DataSource, DataSourceRegistry
from xanesnet.encodings import CombinedEncoding, SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.auto_config import (
    resolve_auto_encoding_config,
    resolve_auto_model_config,
)
from xanesnet.serialization.checkpoints import Checkpoint
from xanesnet.serialization.config import Config
from xanesnet.serialization.model_profile import (
    build_model_profile,
    create_model_summary,
    get_peak_memory_allocated_mb,
    reset_peak_memory_stats,
    save_model_profile,
)
from xanesnet.serialization.models import save_models
from xanesnet.serialization.splits import save_split_indices
from xanesnet.strategies import Strategy, StrategyRegistry

###############################################################################
#################################### TRAIN ####################################
###############################################################################


def train(config: Config, args_namespace: Namespace, save_dir: Path) -> None:
    """Run the complete training pipeline.

    Builds the data source and dataset from ``config``, resolves any automatic
    model fields from the prepared dataset, builds the strategy from the
    resolved config, runs training, and writes models, checkpoints, and metadata
    under ``save_dir``.

    Args:
        config: Validated training configuration, possibly containing
            ``"auto"`` model fields supported by the resolver.
        args_namespace: Parsed CLI arguments (must contain ``tensorboard``;
            may contain ``dry_run`` to enable profile output).
        save_dir: Root directory for all training outputs.
    """
    logging.info("Training.")

    datasource = _setup_datasource(config)
    dataset = _setup_dataset(config, datasource)

    logging.info("Resolving 'auto' encoding fields...")
    config = resolve_auto_encoding_config(config, dataset)

    encoding = _setup_encoding(config, dataset)

    logging.info("Resolving 'auto' model fields...")
    config = resolve_auto_model_config(config, dataset, encoding)
    resolved_config_save_path = config.save(save_dir / "resolved_train_config.yaml")
    logging.info(f"Resolved training config saved to: {resolved_config_save_path}.")

    strategy = _setup_strategy(config, dataset, encoding, save_dir, args_namespace.tensorboard)
    strategy.setup_models()
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

    # Save split indices if they were generated
    split_indices_save_path = save_dir / "split_indices.json"
    save_split_indices(split_indices_save_path, dataset.get_all_subset_indices())
    logging.info(f"Split indices saved to: {split_indices_save_path}")

    # Main training
    if args_namespace.dry_run:
        reset_peak_memory_stats(config.get_str("device"))

    model_list, train_time = _run_training(strategy)

    # Display model summary and training duration
    logging.info(f"Number of trained models: {len(model_list)}")
    logging.info(f"Training completed in {str(timedelta(seconds=int(train_time)))}")
    if args_namespace.dry_run:
        peak_gpu_memory_allocated_mb = get_peak_memory_allocated_mb(config.get_str("device"))
        model_profile = build_model_profile(
            model_list[0],
            dataset,
            config.get_str("device"),
            peak_gpu_memory_allocated_mb,
            encoding,
        )
        profile_json_path, profile_readable_path = save_model_profile(save_dir, model_profile)
        logging.info(f"Dry-run model profile JSON saved to: {profile_json_path}")
        logging.info(f"Dry-run model profile readable report saved to: {profile_readable_path}")
    try:
        _summary_models(model_list, dataset, encoding)
    except Exception as exc:
        logging.warning(f"Model summary failed and will be skipped: {exc}")

    # Save model(s)
    save_models(save_dir / "models", model_list)
    logging.info(f"Trained model(s) saved to: {save_dir / 'models'}")
    final_checkpoint = Checkpoint.build(model_list, signature=signature)
    final_save_path = final_checkpoint.save(save_dir / "models" / "final.pth")
    logging.info(f"Final checkpoint without optimizers and epochs saved @ {final_save_path}")


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
    """Instantiate, prepare, and split the training dataset.

    Args:
        config: Validated configuration containing a ``dataset`` section.
        datasource: Data source providing raw data.

    Returns:
        Dataset with splits set up, ready for training.
    """
    dataset_config = config.section("dataset")
    dataset_type = dataset_config.get_str("dataset_type")

    logging.info(f"Initializing training dataset: {dataset_type}")
    dataset = DatasetRegistry.create(dataset_type, **dataset_config.as_kwargs(), datasource=datasource)
    dataset.prepare()
    dataset.setup_splits()
    dataset.check_preload()  # may preload the dataset into memory

    # Log dataset summary
    logging.info(f"Dataset Summary: # of samples = {len(dataset)}")

    return dataset


def _setup_encoding(config: Config, dataset: Dataset) -> SpectraEncoding:
    """Build and prepare the composed spectra encoding from config.

    Args:
        config: Validated configuration containing an ``encodings`` section.
        dataset: Prepared dataset containing the raw spectral energy grid.

    Returns:
        A :class:`CombinedEncoding` wrapping all configured component
        encodings, prepared for the dataset's spectral width.
    """
    encoding = CombinedEncoding.from_configs(config.get_config_list("encodings"))
    energies = getattr(dataset[0], "energies", None)
    if energies is None or energies.ndim == 0:
        raise ValueError("Prepared dataset sample must provide a spectral energy grid.")
    encoding.prepare(int(energies.shape[-1]))
    return encoding


def _setup_strategy(
    config: Config,
    dataset: Dataset,
    encoding: SpectraEncoding,
    save_dir: Path,
    enable_tensorboard: bool,
) -> Strategy:
    """Instantiate the training strategy from config.

    Args:
        config: Validated configuration containing ``strategy``, ``model``,
            and ``trainer`` sections.
        dataset: Prepared and split dataset.
        encoding: Composed spectra encoding forwarded to the strategy's
            trainers.
        save_dir: Root run directory (used to derive checkpoint and
            TensorBoard paths).
        enable_tensorboard: Whether to enable TensorBoard logging.

    Returns:
        Configured strategy instance (models not yet initialized).
    """
    strategy_config = config.section("strategy")
    strategy_type = strategy_config.get_str("strategy_type")

    model_config = config.section("model")
    trainer_config = config.section("trainer")

    logging.info(f"Initializing strategy: {strategy_type}")
    strategy = StrategyRegistry.create(
        strategy_type,
        **strategy_config.as_kwargs(),
        checkpoint_dir=save_dir / "checkpoints",
        tensorboard_dir=save_dir / "tensorboard" if enable_tensorboard else None,
        dataset=dataset,
        model_config=model_config,
        encoding=encoding,
        trainer_config=trainer_config,
    )

    return strategy


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
    start_time = time.time()

    model_list = strategy.run_training()

    train_time = time.time() - start_time

    # Move to CPU
    for model in model_list:
        model.to(torch.device("cpu"))

    return model_list, train_time


###############################################################################
############################### SUMMARY LOGGING ###############################
###############################################################################


def _summary_models(
    model_list: list[Model], dataset: Dataset, encoding: SpectraEncoding
) -> None:
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
