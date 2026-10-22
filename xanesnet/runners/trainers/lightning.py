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

"""Lightning DDP trainer for XANESNET."""

# TODO: Revisit only if profiling shows a performance bottleneck:
# TODO - Skip target-count reductions where equal per-rank target counts are guaranteed.
# TODO - Skip stop-decision reductions for no-op stoppers or epochs without monitoring.
# TODO - Avoid duplicate final checkpoint saves and make weight logging configurable.
# TODO - Improve custom-batch transfers: pinned-memory/nonblocking transfer path

import logging
from pathlib import Path
from typing import Any, cast

import lightning as L
import torch
import torch.distributed as distributed
from lightning.fabric.plugins.environments.lightning import LightningEnvironment
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    LRSchedulerTypeUnion,
    OptimizerLRScheduler,
)
from torch.utils.data import DistributedSampler

from xanesnet.checkpointing import Checkpointer
from xanesnet.datasets import Dataset
from xanesnet.encodings import SpectraEncoding
from xanesnet.models import Model
from xanesnet.serialization.config import Config
from xanesnet.serialization.tensorboard import tb_logger

from .base import Trainer
from .registry import TrainerRegistry


class _LightningModule(L.LightningModule):
    """Lightning module for XANESNET training.

    Args:
         trainer: Initialized backend containing the model and objective.
    """

    def __init__(self, trainer: "LightningTrainer") -> None:
        """Attach local XANESNET training state to the Lightning module."""
        super().__init__()

        self.xn = trainer
        self.model = self.xn.model
        self.loss = self.xn.loss
        self.regularizer = self.xn.regularizer
        self.last_score: float | None = None
        self._train_totals: torch.Tensor | None = None
        self._validation_totals: torch.Tensor | None = None
        self._validation_metrics: tuple[float, float, float] | None = None
        self._validate_this_epoch = False

    def on_train_epoch_start(self) -> None:
        """Reset local accumulators and select Basic-compatible validation epochs."""
        self._train_totals = torch.zeros(4, device=self.device, dtype=torch.float64)
        self._validation_metrics = None
        self._validate_this_epoch = self.xn.valid_dataloader is not None and (
            self.current_epoch % self.xn.validation_interval == 0 or self.current_epoch == self.xn.epochs - 1
        )
        # check_val_every_n_epoch must be a positive integer: Lightning validates
        # when (current_epoch + 1) % check_val_every_n_epoch == 0. Use 1 to force
        # validation; otherwise use epochs + 1, which never divides into any
        # epoch index of this run, so validation is skipped this epoch.
        self.trainer.check_val_every_n_epoch = 1 if self._validate_this_epoch else self.xn.epochs + 1

    def on_validation_epoch_start(self) -> None:
        """Reset validation sums without registering rank-local state as buffers."""
        self._validation_totals = torch.zeros(4, device=self.device, dtype=torch.float64)

    def on_train_start(self) -> None:
        """Log the model graph once on the global-zero rank, matching Basic."""
        if not self.trainer.is_global_zero:
            return
        sample_batch = next(iter(self.xn.dataloader))
        sample_batch = self.transfer_batch_to_device(sample_batch, self.device, 0)
        sample_inputs = self.xn.batchprocessor.input_preparation(sample_batch)
        tb_logger.log_model_graph(self.model, sample_inputs)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        """Move a custom batch or standard Lightning collection to its device.

        Args:
            batch: Collated dataset batch.
            device: Device assigned by Lightning to this process.
            dataloader_idx: Index of the source data loader.

        Returns:
            Batch after device transfer.
        """
        if hasattr(batch, "to"):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute a differentiable training total and record epoch metrics.

        Args:
            batch: Training batch on the assigned device.
            batch_idx: Batch index within the epoch.

        Returns:
            Scalar loss plus regularization for automatic optimization.
        """
        loss, regularization, total, observations = self._calculate_loss(batch)
        self._accumulate_metrics(self._train_totals, loss, regularization, total, observations)
        if self.trainer.world_size > 1:
            local_count = loss.new_tensor(float(observations))
            global_count = self._reduce_sum(local_count.clone())
            loss = loss * (self.trainer.world_size * local_count / global_count)
        return loss + regularization

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute validation totals and record distributed epoch metrics.

        Args:
            batch: Validation batch on the assigned device.
            batch_idx: Batch index within the validation pass.

        Returns:
            Scalar validation loss plus regularization.
        """
        loss, regularization, total, observations = self._calculate_loss(batch, batch_idx)
        self._accumulate_metrics(self._validation_totals, loss, regularization, total, observations)
        return total

    def on_validation_epoch_end(self) -> None:
        """Publish exact global validation metrics after excluding padded samples."""
        self._validation_metrics = self._epoch_metrics(self._validation_totals, "val")

    def on_train_epoch_end(self) -> None:
        """Log metrics, advance the epoch scheduler, and synchronize stopping."""
        train_metrics = self._epoch_metrics(self._train_totals, "train")
        valid_metrics = self._validation_metrics
        if self.trainer.is_global_zero:
            self.xn._log_epoch_loss(
                *train_metrics,
                self.current_epoch,
                *(valid_metrics if valid_metrics is not None else (None, None, None)),
            )
            tb_logger.log_epoch_metrics(
                self.current_epoch,
                *train_metrics,
                *(valid_metrics if valid_metrics is not None else (None, None, None)),
            )
            tb_logger.log_learning_rate(self.current_epoch, self.xn.optimizer.param_groups[0]["lr"])
            # TODO this step can be expensive for large models -> Maybe make this optional
            tb_logger.log_model_weights(self.current_epoch, self.model)

        if self.xn.warmup_lr_scheduler is None or self.global_step >= self.xn.warmup_steps:
            self.xn.epoch_lr_scheduler.step()

        monitored = train_metrics if self.xn.valid_dataloader is None else valid_metrics
        stopped = False
        if monitored is not None:
            self.last_score = monitored[2]
            stopped = self.xn.early_stopper.step(self.last_score, self.model, self.current_epoch)
        stopped = bool(self._reduce_sum(torch.tensor(int(stopped), device=self.device)).item())
        self.trainer.should_stop = self.trainer.should_stop or stopped
        checkpointer = self.xn.checkpointer
        if self.trainer.is_global_zero and checkpointer.active:
            if self.current_epoch == self.xn.epochs - 1 or stopped:
                saved_checkpoint, checkpoint_name = checkpointer.save_checkpoint(
                    self.current_epoch, self.xn.model, self.xn.optimizer
                )
            else:
                saved_checkpoint, checkpoint_name = checkpointer.step(
                    self.current_epoch, self.xn.model, self.xn.optimizer
                )
            if saved_checkpoint:
                logging.info(f"Saved new checkpoint @ {checkpointer.save_dir}: {checkpoint_name}")

    @staticmethod
    def _accumulate_metrics(
        totals: torch.Tensor | None,
        loss: torch.Tensor,
        regularization: torch.Tensor,
        total: torch.Tensor,
        observations: int,
    ) -> None:
        """Accumulate detached weighted sums without synchronizing each batch.

        Args:
            totals: Local epoch accumulator initialized by the start hook.
            loss: Mean data loss over target rows.
            regularization: Scalar model regularization.
            total: Sum of data loss and regularization.
            observations: Number of target rows contributing to this batch.
        """
        assert totals is not None
        values = torch.stack((loss.detach(), regularization.detach(), total.detach(), loss.new_tensor(1.0)))
        totals.add_(values.to(dtype=totals.dtype) * observations)

    def _reduce_sum(self, values: torch.Tensor) -> torch.Tensor:
        """Sum a tensor in place across DDP ranks without a separate barrier.

        Args:
            values: Local tensor to reduce on this rank's assigned device.

        Returns:
            The same tensor containing the global sum, or unchanged during
            single-process execution.
        """
        if self.trainer.world_size > 1:
            distributed.all_reduce(values, op=distributed.ReduceOp.SUM)
        return values

    def _epoch_metrics(self, totals: torch.Tensor | None, prefix: str) -> tuple[float, float, float]:
        """Reduce global sufficient statistics and publish their weighted means.

        Args:
            totals: Rank-local loss, regularization, total, and observation sums.
            prefix: Metric namespace, either ``"train"`` or ``"val"``.

        Returns:
            Globally weighted data loss, regularization, and total.
        """
        assert totals is not None
        reduced = self._reduce_sum(totals)
        values = (reduced[:3] / reduced[3]).tolist()
        metrics = (float(values[0]), float(values[1]), float(values[2]))
        for name, value in zip(("loss", "regularization", "total"), metrics, strict=True):
            self.log(
                f"{prefix}_{name}",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                prog_bar=name == "total",
            )
        return metrics

    def train_dataloader(self) -> Any:
        """Return the configured training data loader.

        Returns:
            Local loader whose sampling Lightning manages during training.
        """
        return self.xn.dataloader

    def val_dataloader(self) -> Any:
        """Return the configured optional validation data loader.

        Returns:
            Validation loader, or an empty collection when no validation subset exists.
        """
        return self.xn.valid_dataloader if self.xn.valid_dataloader is not None else []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Expose the existing optimizer and scheduler intervals to Lightning.

        Returns:
            Optimizer list and step/epoch scheduler dictionaries. Lightning
            accepts these dictionaries at runtime, although its public tuple
            return alias omits their dictionary type.
        """
        schedulers: list[LRSchedulerConfigType] = []
        if self.xn.warmup_lr_scheduler is not None:
            schedulers.append({"scheduler": self.xn.warmup_lr_scheduler, "interval": "step"})
        schedulers.append({"scheduler": self.xn.epoch_lr_scheduler, "interval": "epoch"})
        return cast(OptimizerLRScheduler, ([self.xn.optimizer], schedulers))

    def lr_scheduler_step(self, scheduler: LRSchedulerTypeUnion, metric: Any) -> None:
        """Step warm-up after optimization and defer epoch scheduling to its hook.

        Args:
            scheduler: Scheduler requested by Lightning's automatic loop.
            metric: Optional scheduler monitor supplied by Lightning.
        """
        warmup_scheduler = self.xn.warmup_lr_scheduler
        if warmup_scheduler is not None and scheduler is warmup_scheduler and self.global_step <= self.xn.warmup_steps:
            warmup_scheduler.step()

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Persist project stopping/selection state alongside Lightning recovery.

        Args:
            checkpoint: Lightning checkpoint being populated before writing.
        """
        checkpoint["xanesnet_stopper"] = vars(self.xn.early_stopper).copy()
        checkpoint["xanesnet_last_score"] = self.last_score

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        """Restore project state when Lightning resumes a recovery checkpoint.

        Args:
            checkpoint: Previously saved complete Lightning checkpoint.
        """
        vars(self.xn.early_stopper).update(checkpoint["xanesnet_stopper"])
        self.last_score = checkpoint["xanesnet_last_score"]

    def _calculate_loss(
        self, batch: Any, validation_batch_idx: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Prepare encoded inputs/targets and evaluate the existing objective.

        Args:
            batch: Dataset batch already transferred to the training device.
            validation_batch_idx: Validation batch position used to exclude
                distributed sampler padding, or ``None`` during training.

        Returns:
            Data loss, regularization, their scalar total, and target-row count.
        """
        processor = self.xn.batchprocessor
        inputs = processor.input_preparation(batch)
        elements = processor.element_preparation(batch)
        inputs = processor.encode_input(inputs, elements)
        targets = processor.target_preparation(batch)
        targets = processor.encode_target(targets, elements)
        predictions = processor.prediction_preparation(batch, self.model(**inputs))
        if validation_batch_idx is not None:
            rows = self._validation_rows(validation_batch_idx, targets.shape[0])
            predictions = predictions[:rows]
            targets = targets[:rows]
        observations = int(targets.shape[0])
        loss = self.loss(predictions, targets) if observations else predictions.sum() * 0.0
        regularization = self.regularizer(self.model)
        return loss, regularization, loss + regularization, observations

    def _validation_rows(self, batch_idx: int, rows: int) -> int:
        """Count real target rows in an evenly padded distributed validation batch.

        Args:
            batch_idx: Local validation batch index.
            rows: Total target rows, including any trailing duplicated samples.

        Returns:
            Target-row count after excluding sampler padding. Every rank still
            executes the same number of model forward passes.
        """
        sampler = getattr(self.trainer.val_dataloaders, "sampler", None)
        if not isinstance(sampler, DistributedSampler):
            return rows
        validation_subset = self.xn.dataset.valid_subset
        assert validation_subset is not None
        dataset_size = len(validation_subset)
        first_sample = batch_idx * self.xn.batch_size
        local_samples = min(self.xn.batch_size, len(sampler) - first_sample)
        for offset in range(local_samples):
            global_position = sampler.rank + (first_sample + offset) * sampler.num_replicas
            if global_position >= dataset_size:
                sample = sampler.dataset[global_position % dataset_size]
                single_batch = self.xn.dataset.collate_fn([sample])
                rows -= int(self.xn.batchprocessor.target_preparation(single_batch).shape[0])
        return rows


@TrainerRegistry.register("lightning")
class LightningTrainer(Trainer):
    """Lightning DDP trainer.

    Args:
        dataset: Dataset providing training and optional validation subsets.
        model: Model trained by the backend.
        device: Configured device identifier.
        checkpointer: Existing checkpoint configuration holder.
        encoding: Prepared spectra encoding used by the batch processor.
        batch_size: Number of samples per local batch.
        shuffle: Whether to shuffle training samples.
        drop_last: Whether to discard incomplete training batches.
        num_workers: Loader workers per process.
        loss: Non-empty component loss configurations.
        regularizer: Regularization configuration.
        trainer_type: Trainer registry identifier.
        epochs: Maximum number of training epochs.
        learning_rate: Optimizer learning rate.
        optimizer: Optimizer registry name.
        max_norm: Gradient clipping norm, or ``None`` to disable.
        lr_scheduler: Per-epoch scheduler configuration.
        early_stopper: Callback stopping and restoration configuration.
        validation_interval: Number of epochs between validation passes.
        lr_warmup: Whether to construct the warm-up scheduler.
        warmup_steps: Configured number of warm-up optimizer steps.
        num_devices: Number of CUDA devices or CPU training processes.
        find_unused_parameters: Whether DDP detects parameters skipped by the model's input-dependent branches.
        static_graph: Whether DDP assumes a stable set of used and unused parameters throughout training.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: Model,
        device: str | torch.device,
        checkpointer: Checkpointer,
        encoding: SpectraEncoding,
        # runner params:
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        num_workers: int,
        loss: list[Config],
        regularizer: Config,
        # trainer params:
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
        # lightning params:
        num_devices: int,
        find_unused_parameters: bool,
        static_graph: bool,
    ) -> None:
        """Initialize shared model/objective setup for Lightning execution."""
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
        self.num_devices = num_devices
        self.find_unused_parameters = find_unused_parameters
        self.static_graph = static_graph

    def train(self) -> float | None:
        """Fit with Lightning and return the selected best or last score.

        Returns:
            Best monitored total when restoring best weights, otherwise the
            last validation or training total. Returns ``None`` without a
            completed monitoring event.

        Raises:
            ValueError: If no training batches were processed.
        """
        self.early_stopper.start()

        accelerator, devices, strategy = self._distributed_settings()
        strategy_name = strategy if isinstance(strategy, str) else "ddp_static_graph"
        logging.info(
            f"Lightning Trainer: accelerator={accelerator}, devices={devices}, "
            f"strategy={strategy_name}, use_distributed_sampler=True"
        )

        module = _LightningModule(self)
        pl_trainer = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            max_epochs=self.epochs,
            gradient_clip_val=self.max_norm,
            gradient_clip_algorithm="norm",
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            callbacks=self._callbacks(),
            logger=False,
            enable_checkpointing=self.checkpointer.active,
            use_distributed_sampler=True,
            plugins=[LightningEnvironment()],
            default_root_dir=str(self.checkpointer.save_dir) if self.checkpointer.active else None,
        )

        pl_trainer.fit(module)

        if pl_trainer.global_step == 0:
            raise ValueError(
                "No training batches were processed. Check batch_size, drop_last, and training subset size."
            )
        callback = pl_trainer.checkpoint_callback
        if isinstance(callback, ModelCheckpoint) and callback.dirpath is not None:
            pl_trainer.save_checkpoint(Path(callback.dirpath) / "last.ckpt")

        score = module.last_score
        if self.early_stopper.restore_best:
            best_score, best_epoch = self.early_stopper.restore(self.model)
            if best_score is not None:
                score = best_score
                logging.info(f"Restored best model from epoch {best_epoch} with score {score}.")
        return score

    def _distributed_settings(self) -> tuple[str, int | list[int], str | DDPStrategy]:
        """Translate configured devices to Lightning accelerator settings.

        Returns:
            Accelerator name, local device count or explicit device indices,
            and strategy identifier or configured DDP strategy.
        """
        device = str(self.device).lower()
        if self.static_graph:
            ddp_strategy: str | DDPStrategy = DDPStrategy(static_graph=True)
        else:
            ddp_strategy = "ddp_find_unused_parameters_true" if self.find_unused_parameters else "ddp"
        # CUDA
        if device.startswith("cuda"):
            count = self.num_devices
            index = torch.device(self.device).index
            devices = count if index is None else list(range(index, index + count))
            return "gpu", devices, ddp_strategy if count > 1 else "auto"
        # CPU
        return "cpu", self.num_devices, ddp_strategy if self.num_devices > 1 else "auto"

    def _callbacks(self) -> list[L.Callback]:
        """Configure periodic recovery saves independently of model selection.

        Returns:
            Recovery checkpoint callback when enabled, otherwise an empty list.
        """
        callbacks: list[L.Callback] = []
        if self.checkpointer.active:
            callbacks.append(
                ModelCheckpoint(
                    dirpath=str(self.checkpointer.save_dir / f"model_{self.checkpointer.model_counter}"),
                    filename="{epoch:04d}",
                    monitor=None,
                    save_top_k=-1,
                    save_last=True,
                    every_n_epochs=self.checkpointer.save_interval,
                    save_on_train_epoch_end=True,
                )
            )
        return callbacks
