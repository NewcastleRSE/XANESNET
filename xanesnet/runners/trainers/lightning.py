# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas Pope, C. D. Rankine, B. Li
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
from typing import Any

import lightning as L
import torch
from torch.utils.data import DataLoader

from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.batchprocessors import BatchProcessorRegistry
from xanesnet.losses import LossRegistry, CombinedLoss
from xanesnet.losses.base import Loss
from xanesnet.regularizers import RegularizerRegistry
from xanesnet.regularizers.base import Regularizer
from xanesnet.components import LRSchedulerRegistry, OptimizerRegistry
from xanesnet.serialization.config import Config

class LightningModule(L.LightningModule):
    """Lightning module for training.

    Lightning manages device placement, distributed training, optimization,
    gradient synchronization, and the training/validation loop.

    Args:
        dataset: Dataset used for training and validation.
        model: Model to train.
        encoding: Prepared spectra encoding.
        batch_size: Number of samples per batch per device.
        shuffle: Whether to shuffle the training data.
        drop_last: Whether to drop the last incomplete training batch.
        num_workers: Number of data-loader workers.
        loss: Non-empty list of loss configurations.
        regularizer: Configuration for the regularizer.
        epochs: Total number of training epochs.
        learning_rate: Initial learning rate.
        optimizer: Optimizer name.
        max_norm: Maximum gradient norm, or ``None`` to disable clipping.
        lr_scheduler: Configuration for the learning-rate scheduler.
        validation_interval: Number of epochs between validation runs.
        lr_warmup: Whether to apply per-step linear warm-up.
        warmup_steps: Number of warm-up steps.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        encoding: SpectraEncoding,
######### runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        loss: list[Config],
        regularizer: Config,
######### trainer params:
        epochs: int,
        learning_rate: float,
        optimizer: str,
        max_norm: float | None,
        lr_scheduler: Config,
        validation_interval: int,
    ) -> None:
        """Initialize the Lightning training module."""
        super().__init__()

        self.dataset = dataset
        self.model = model
        self.encoding = encoding

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = num_workers

        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer
        self.max_norm = max_norm
        self.lr_scheduler_config = lr_scheduler
        self.validation_interval = validation_interval

        self.loss_config = loss
        self.regularizer_config = regularizer

        self.batchprocessor = BatchProcessorRegistry.create(
            (self.dataset.dataset_type, self.model.model_type),
            encoding=self.encoding,
        )

        self.loss = self._setup_loss()
        self.regularizer = self._setup_regularizer()

    def forward(self, **inputs: Any) -> Any:
        """Run a forward pass through the XANESNET model."""
        return self.model(**inputs)

##### Training
    def training_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one training batch."""
        loss, regularization, total = self._calculate_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_regularization", regularization, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_total", total, on_step=False, on_epoch=True, sync_dist=True)
        return total

##### Validation
    def validation_step(
        self,
        batch: Any,
        batch_idx: int,
    ) -> torch.Tensor:
        """Run one validation batch."""
        loss, regularization, total = self._calculate_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_regularization", regularization, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_total", total, on_step=False, on_epoch=True, sync_dist=True)
        return total

##### Batch processing / loss

    def _calculate_loss(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate loss, regularization, and total loss for a batch."""
        inputs = self.batchprocessor.input_preparation(batch)
        elements = self.batchprocessor.element_preparation(batch)
        inputs = self.batchprocessor.encode_input(inputs,elements)
        targets = self.batchprocessor.target_preparation(batch)
        targets = self.batchprocessor.encode_target(targets,elements)
        predictions = self.model(**inputs)
        predictions = self.batchprocessor.prediction_preparation(batch,predictions)
        loss = self.loss(predictions,targets)
        regularization = self.regularizer(self.model)
        total = loss + regularization
        return loss, regularization, total

##### Data loaders
    def train_dataloader(self) -> DataLoader:
        if self.dataset.train_subset is None:
            raise ValueError("Training subset is required but was not provided.")
        return self._build_dataloader(
            self.dataset.train_subset,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader | None:
        if self.dataset.valid_subset is None:
            return None
        return self._build_dataloader(
            self.dataset.valid_subset,
            shuffle=False,
            drop_last=False,
        )

    def _build_dataloader(
        self,
        data: Any,
        shuffle: bool,
        drop_last: bool,
    ) -> DataLoader:
        """Build a data loader.

        Lightning handles distributed sampling, so no distributed sampler
        is created manually.
        """

        dataloader_cls = self.dataset.get_dataloader()
        return dataloader_cls(
            data,
            batch_size=self.batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

##### Setups
    def _setup_loss(self) -> Loss:
        loss_configs = self.loss_config
        if not loss_configs:
            raise ValueError("Loss config list is required but was not provided.")
        losses: list[Loss] = []
        raw_weights: list[float] = []

        for item_config in loss_configs:
            loss_type = item_config.get_str("loss_type")
            raw_weight = item_config.get_optional_float("loss_weight")
            raw_weights.append(raw_weight if raw_weight is not None else 1.0)
            kwargs = {
                k: v
                for k, v in item_config.as_kwargs().items()
                if k != "loss_weight"
            }
            losses.append(LossRegistry.create(loss_type, **kwargs))
        return CombinedLoss(losses, raw_weights)

    def _setup_regularizer(self) -> Regularizer:
        regularizer_config = self.regularizer_config
        if regularizer_config is None:
            raise ValueError("Regularizer config is required but was not provided.")
        regularizer_type = regularizer_config.get_str("regularizer_type")
        return RegularizerRegistry.create(regularizer_type, **regularizer_config.as_kwargs())

    def configure_optimizers(self):
        optimizer_cls = OptimizerRegistry.get(self.optimizer_type)
        optimizer = optimizer_cls(self.model.parameters(),lr=self.learning_rate)
    
        lr_scheduler_type = self.lr_scheduler_config.get_str("lr_scheduler_type")
        lr_scheduler_kwargs = self.lr_scheduler_config.as_kwargs()
    
        scheduler_kwargs = {
            k: v
            for k, v in lr_scheduler_kwargs.items()
            if k != "lr_scheduler_type"
        }
    
        scheduler_cls = LRSchedulerRegistry.get(lr_scheduler_type)
        scheduler = scheduler_cls(optimizer,**scheduler_kwargs)
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
