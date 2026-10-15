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

"""Lightning DDP trainer registered like ``BasicTrainer``.

Uses Lightning's standard ``ddp`` launcher. The CLI is re-entered with
``LOCAL_RANK``; run-directory reuse lives in ``xanesnet.utils.distributed``.
"""

from __future__ import annotations

import logging
from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.config import Config

from .base import Trainer
from .registry import TrainerRegistry


class _LightningModule(L.LightningModule):
    """Lightning module for XANESNET training."""
    def __init__(self, trainer: LightningTrainer) -> None:
        super().__init__()
        self.xn = trainer
        self.model = self.xn.model

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        if hasattr(batch, "to"):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, regularization, total = self._calculate_loss(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_regularization", regularization, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train_total", total, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss, regularization, total = self._calculate_loss(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_regularization", regularization, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_total", total, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return total

    def train_dataloader(self) -> Any:
        return self.xn.dataloader

    def val_dataloader(self) -> Any:
        return self.xn.valid_dataloader

    def configure_optimizers(self):
        schedulers: list[dict[str, Any]] = []
        if self.xn.warmup_lr_scheduler is not None:
            schedulers.append({"scheduler": self.xn.warmup_lr_scheduler, "interval": "step"})
        schedulers.append({"scheduler": self.xn.epoch_lr_scheduler, "interval": "epoch"})
        return [self.xn.optimizer], schedulers

    def _calculate_loss(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        processor = self.xn.batchprocessor
        inputs = processor.input_preparation(batch)
        elements = processor.element_preparation(batch)
        inputs = processor.encode_input(inputs, elements)
        targets = processor.target_preparation(batch)
        targets = processor.encode_target(targets, elements)
        predictions = processor.prediction_preparation(batch, self.model(**inputs))
        loss = self.xn.loss(predictions, targets)
        regularization = self.xn.regularizer(self.model)
        return loss, regularization, loss + regularization


@TrainerRegistry.register("lightning")
class LightningTrainer(Trainer):
    """Lightning DDP trainer."""

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
        checkpointer: Checkpointer,
        encoding: SpectraEncoding,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        loss: list[Config],
        regularizer: Config,
        trainer_type: str,
        epochs: int,
        learning_rate: float,
        optimizer: str,
        max_norm: float | None,
        lr_scheduler: Config,
        early_stopper: Config,
        validation_interval: int,
        lr_warmup: bool,
        warmup_steps: int,
        num_gpus: int = 1,
    ) -> None:
        """Initialize the Lightning trainer and inherited XANESNET setup."""
        super().__init__(
            dataset,
            model,
            device,
            checkpointer,
            encoding,
            batch_size,
            shuffle,
            drop_last,
            num_workers,
            loss,
            regularizer,
            trainer_type,
            epochs,
            learning_rate,
            optimizer,
            max_norm,
            lr_scheduler,
            early_stopper,
            validation_interval,
            lr_warmup,
            warmup_steps,
        )
        self.num_gpus = num_gpus

    def train(self) -> float | None:
        """Fit with Lightning."""
        self.model.to(self.device)
        self.loss.to(self.device)

        accelerator, devices, strategy = self._distributed_settings()
        logging.info(
            f"Lightning Trainer: accelerator={accelerator}, devices={devices}, "
            f"strategy={strategy}, use_distributed_sampler=True"
        )

        module = _LightningModule(self)
        pl_trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=self.epochs,
            gradient_clip_val=self.max_norm,
            gradient_clip_algorithm="norm",
            check_val_every_n_epoch=self.validation_interval,
            callbacks=self._callbacks(),
            logger=False,
            enable_checkpointing=self.checkpointer.active,
            use_distributed_sampler=True,
            default_root_dir=str(self.checkpointer.save_dir) if self.checkpointer.active else None,
        )
        pl_trainer.fit(module)
        self._restore_best(pl_trainer, module)
        return None

    def _train_one_epoch(self) -> tuple[float, float, float]:
        raise RuntimeError("LightningTrainer does not run the XANESNET epoch loop.")

    def _validate_one_epoch(self) -> tuple[float, float, float]:
        raise RuntimeError("LightningTrainer does not run the XANESNET epoch loop.")

    def _distributed_settings(self) -> tuple[str, int, str]:
        device = str(self.device).lower()
        if device.startswith("cuda"):
            devices = max(1, self.num_gpus)
            return "gpu", devices, "ddp" if devices > 1 else "auto"
        if device.startswith("mps"):
            return "mps", 1, "auto"
        if self.num_gpus > 1:
            logging.warning(f"num_gpus={self.num_gpus} ignored because device is '{self.device}'.")
        return "cpu", 1, "auto"

    def _callbacks(self) -> list[L.Callback]:
        """Map XANESNET early-stop / checkpoint config to Lightning callbacks."""
        callbacks: list[L.Callback] = []
        monitor = "val_total" if self.valid_dataloader is not None else "train_total"
        stopper_type = self.early_stopper_config.get_str("early_stopper_type").lower()
        if stopper_type == "basic":
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    patience=self.early_stopper_config.get_int("patience"),
                    min_delta=self.early_stopper_config.get_optional_float("min_delta") or 0.0,
                )
            )
        elif stopper_type not in {"none", "no"}:
            logging.warning(f"Lightning trainer ignores early_stopper_type '{stopper_type}'.")

        if self.checkpointer.active:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(self.checkpointer.save_dir),
                    filename="{epoch:04d}",
                    monitor=monitor,
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                    every_n_epochs=self.checkpointer.save_interval,
                )
            )
        return callbacks

    def _restore_best(self, pl_trainer: L.Trainer, module: _LightningModule) -> None:
        """Load the best Lightning checkpoint when ``restore_best`` is set."""
        if not self.early_stopper_config.get_bool("restore_best"):
            return
        callback = getattr(pl_trainer, "checkpoint_callback", None)
        best_path = getattr(callback, "best_model_path", "") if callback is not None else ""
        if not best_path:
            return
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        module.load_state_dict(ckpt["state_dict"])
        logging.info(f"Restored best Lightning checkpoint: {best_path}")
