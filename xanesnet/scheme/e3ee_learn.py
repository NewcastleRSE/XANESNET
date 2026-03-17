"""
Custom learning scheme for absorber-centred e3nn XANES models.

Adds:
- automatic mixed precision (AMP) on CUDA
- gradient accumulation
- gradient clipping
- optional derivative-spectrum loss term
- same train/train_kfold interface as NNLearn
"""

import copy
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import RepeatedKFold

from xanesnet.scheme.base_learn import Learn, EarlyStopState
from xanesnet.utils.switch import LossSwitch, LossRegSwitch


class E3EELearn(Learn):
    """
    Custom learning scheme for e3nn XANES models.
    """

    def __init__(self, model, dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)

        # extra hyperparameters
        self.use_amp = self.hyper_params.get("use_amp", True)
        self.grad_accum_steps = max(1, int(self.hyper_params.get("grad_accum_steps", 1)))
        self.grad_clip_norm = float(self.hyper_params.get("grad_clip_norm", 0.0))
        self.deriv_lambda = float(self.hyper_params.get("deriv_lambda", 0.0))

        # AMP is only useful on CUDA
        self.amp_enabled = bool(self.use_amp and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp_enabled)

    def tensorboard_layout(self) -> Dict:
        layout = {
            "Losses": {
                "Losses": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        return layout

    def train(self, model, dataset) -> float:
        train_loader, valid_loader = self.setup_dataloaders(dataset)
        optimizer, criterion, regularizer, scheduler = self.setup_components(model)

        model.to(self.device)
        state = EarlyStopState() if self.earlystop_flag else None
        valid_loss = 0.0

        logging.info(f"--- Starting e3nn XANES Training for {self.epochs} epochs ---")

        for epoch in range(self.epochs):
            train_loss = self._run_one_epoch(
                "train",
                train_loader,
                model,
                criterion,
                regularizer,
                optimizer,
            )

            valid_loss = self._run_one_epoch(
                "valid",
                valid_loader,
                model,
                criterion,
                regularizer,
                optimizer=None,
            )

            if self.lr_scheduler:
                scheduler.step()

            self._log_epoch_loss(epoch, train_loss, valid_loss)

            if self.earlystop_flag:
                self._early_stop(valid_loss, state)
                if state.stop:
                    break

        logging.info("--- Training Finished ---")

        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)
            self.log_close()

        return valid_loss

    def train_std(self):
        self.train(self.model, self.dataset)
        return self.model

    def train_kfold(self):
        best_model = None
        best_score = float("inf")
        score_list = {"train_score": [], "test_score": []}

        kfold_splitter = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        criterion = LossSwitch().get(self.loss, **self.loss_params)
        regularizer = LossRegSwitch()

        indices = self.dataset.indices

        for i, (train_index, test_index) in enumerate(kfold_splitter.split(indices)):
            model = copy.deepcopy(self.model)

            train_data = self.dataset[train_index]
            train_score = self.train(model, train_data)

            test_data = self.dataset[test_index]
            test_loader = self._create_loader(test_data)
            test_score = self._run_one_epoch(
                "valid",
                test_loader,
                model,
                criterion,
                regularizer,
                optimizer=None,
            )

            score_list["train_score"].append(train_score)
            score_list["test_score"].append(test_score)

            if test_score < best_score:
                logging.info(
                    f"--- [Fold {i+1}] New best model found with test score: {test_score:.6f} ---"
                )
                best_score = test_score
                best_model = model

        logging.info("--- K-Fold Cross-Validation Finished ---")
        logging.info(
            f"Average Train Score: {np.mean(score_list['train_score']):.6f} +/- "
            f"{np.std(score_list['train_score']):.6f}"
        )
        logging.info(
            f"Average Test Score : {np.mean(score_list['test_score']):.6f} +/- "
            f"{np.std(score_list['test_score']):.6f}"
        )

        return best_model

    def _run_one_epoch(
        self,
        phase,
        loader,
        model,
        criterion,
        regularizer,
        optimizer=None,
    ):
        is_train = phase == "train"
        model.train() if is_train else model.eval()

        running_loss = 0.0
        device = self.device

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            batch.to(device)

            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    input_data = batch if model.batch_flag else batch.x
                    pred = model(input_data)

                    if model.gnn_flag:
                        pred = pred.view(-1)

                    criterion = LossSwitch().get(self.loss, **self.loss_params)
                    loss_spec = criterion(batch.y, pred)
                    loss_aux = F.mse_loss(batch.y, pred)
                    loss = loss_spec + loss_aux

                    if is_train:
                        loss = loss / self.grad_accum_steps

                if is_train:
                    self.scaler.scale(loss).backward()

                    do_step = ((step + 1) % self.grad_accum_steps == 0) or ((step + 1) == len(loader))
                    if do_step:
                        if self.grad_clip_norm > 0.0:
                            self.scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)

                        self.scaler.step(optimizer)
                        self.scaler.update()
                        optimizer.zero_grad(set_to_none=True)

            # log the unscaled batch loss
            running_loss += loss.item() * self.grad_accum_steps if is_train else loss.item()

        return running_loss / len(loader)

    def _log_epoch_loss(self, epoch, train_loss, valid_loss):
        logging.info(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
        )
        self.log_loss("loss/train", train_loss, epoch)
        self.log_loss("loss/validation", valid_loss, epoch)
