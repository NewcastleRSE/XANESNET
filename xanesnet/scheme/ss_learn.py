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

import copy
import logging

import numpy as np
import torch
import torch.nn.functional as F

from collections import defaultdict
from sklearn.model_selection import RepeatedKFold

from xanesnet.models.base_model import Model
from xanesnet.scheme.base_learn import Learn, EarlyStopState
from xanesnet.utils.gaussian import gaussian_inverse
from xanesnet.utils.switch import LossSwitch


class SSLearn(Learn):
    """
    SSLearn: SoftShell training class.

    This class implements training loop for SoftShell Spectra Network.

    Model compatible with this training process includes: SoftShell.
    """

    def __init__(self, model, dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)

        hyper_params = self.hyper_params

        if hyper_params.get("diagnostics", True):
            self._model_diagnostics(self.dataset.gauss_basis, model, dataset)

    def train(self, model, dataset):
        train_loader, valid_loader = self.setup_dataloaders(dataset)

        optimizer, criterion, regularizer, scheduler = self.setup_components(model)
        model.to(self.device)

        state = EarlyStopState() if self.earlystop_flag else None
        valid_loss = 0.0

        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run training phase
            train_loss = self._run_one_epoch_train(
                epoch, train_loader, model, optimizer
            )

            # Run validation phase
            valid_loss = self._run_one_epoch_valid(valid_loader, model)

            # Adjust learning rate if scheduler is used
            if self.lr_scheduler:
                scheduler.step()

            # Logging for the current epoch
            self._log_epoch_loss(epoch, train_loss, valid_loss)

            # Early stopping
            if self.earlystop_flag:
                self._early_stop(valid_loss, state)
                if state.stop:
                    break

        logging.info("--- Training Finished ---")

        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        score = valid_loss

        return score

    def train_std(self):
        """
        Trains model using standard single-run training.

        Returns
        -------
        Model
            The trained model object.
        """
        self.train(self.model, self.dataset)

        return self.model

    def train_kfold(self):
        """
        Trains model using k-fold cross-validation.


        Returns
        -------
        Model
            The (best) trained model object.
        """
        best_model = None
        best_score = float("inf")
        score_list = {"train_score": [], "test_score": []}

        kfold_splitter = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        # indices for k-fold splits
        indices = self.dataset.indices

        for i, (train_index, test_index) in enumerate(kfold_splitter.split(indices)):
            # Deep copy model
            model = copy.deepcopy(self.model)

            #  Train model on the training split
            train_data = self.dataset[train_index]
            train_score = self.train(model, train_data)

            # Evaluate model on the test split
            test_data = self.dataset[test_index]
            test_loader = self._create_loader(test_data)

            test_score = self._run_one_epoch_valid(test_loader, model)

            score_list["train_score"].append(train_score)
            score_list["test_score"].append(test_score)

            if test_score < best_score:
                logging.info(
                    f"--- [Fold {i+1}] New best model found with test score: {test_score:.6f} ---"
                )
                best_score = test_score
                best_model = model

        # Log final averaged results
        logging.info("--- K-Fold Cross-Validation Finished ---")
        logging.info(
            f"Average Train Score: {np.mean(score_list['train_score']):.6f} +/- {np.std(score_list['train_score']):.6f}"
        )
        logging.info(
            f"Average Test Score : {np.mean(score_list['test_score']):.6f} +/- {np.std(score_list['test_score']):.6f}"
        )

        return best_model

    def _run_one_epoch_train(self, epoch, loader, model, optimizer):
        """
        Run one epoch of training.
        """
        model.train()
        device = self.device
        epoch_losses = defaultdict(float)

        model_params = list(model.encoder.parameters()) + list(
            model.coeff_head.parameters()
        )

        # Setup constants
        sigma_max, sigma_min = 9.0, 5.0
        eta_aux_max, eta_aux_min = 3e-3, 3e-4
        T = max(1, self.epochs - 50)  # stop annealing near the end
        lambda_neg = 1e2

        for batch in loader:
            batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            c_pred = model(batch)
            y_pred = self.synthesis.forward_from_coeffs(c_pred)

            neg_part = F.relu(-y_pred)
            loss_neg = (neg_part**2).mean()

            sigma_now = sigma_min + (sigma_max - sigma_min) * max(0, T - epoch) / T
            eta_aux = eta_aux_min + (eta_aux_max - eta_aux_min) * max(0, T - epoch) / T

            if self.loss == "specplus":
                self.loss_params["blur_sigma_bins"] = sigma_now
            criterion = LossSwitch().get(self.loss, **self.loss_params)
            loss_spec = criterion(batch.y, y_pred)
            loss_aux = F.mse_loss(c_pred, batch.c_star)

            loss_total = loss_spec + eta_aux * loss_aux + lambda_neg * loss_neg
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            optimizer.step()

            epoch_losses["spec"] += loss_spec.item()
            epoch_losses["aux"] += loss_aux.item()

        train_losses = {k: v / len(loader) for k, v in epoch_losses.items()}

        return train_losses

    def _run_one_epoch_valid(self, loader, model):
        """Runs a single epoch of training or validation."""
        model.eval()

        running_loss = 0.0
        n_elem = 0
        device = self.device

        with torch.set_grad_enabled(False):
            for batch in loader:
                batch.to(device)

                c_pred = model(batch)
                y_pred = self.synthesis.forward_from_coeffs(c_pred)  # (B,N)

                running_loss += F.mse_loss(y_pred, batch.y, reduction="sum").item()
                n_elem += batch.y.numel()

        return running_loss / max(1, n_elem)

    def tensorboard_layout(self):
        layout = {
            "Losses": {
                "Losses": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        return layout

    def _model_diagnostics(self, basis, model, dataset):
        print("--- Model Diagnostics ---")
        train_loader, valid_loader = self.setup_dataloaders(dataset)

        with torch.no_grad():
            for loader, tag in [(train_loader, "Train"), (valid_loader, "Val")]:
                sse, n_elem = 0.0, 0
                for batch in loader:
                    batch.to(self.device)
                    c_batch = batch.c_star
                    y_batch = batch.y

                    y_gauss = gaussian_inverse(basis, c_batch)  # (B, N)
                    sse += F.mse_loss(y_gauss, y_batch, reduction="sum").item()
                    n_elem += y_batch.numel()
                mse_gauss = sse / max(1, n_elem)

        trainable_e, total_e = self._count_trainable_params(model.encoder)
        trainable_h, total_h = self._count_trainable_params(model.coeff_head)

        logging.info(
            f"Encoder parameters: {trainable_e:,} trainable / {total_e:,} total"
        )
        logging.info(
            f"CoeffHead parameters: {trainable_h:,} trainable / {total_h:,} total"
        )
        logging.info(f"TOTAL trainable parameters: {trainable_e + trainable_h:,}")

    @staticmethod
    def _count_trainable_params(m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        return trainable, total

    def _log_epoch_loss(self, epoch, train_loss, valid_loss):
        logging.info(
            f"Epoch {epoch + 1:03d} | "
            f"Train Lspec={train_loss['spec']:.6f} "
            f"Aux(c*)={train_loss['aux']:.6f} | "
            f"Val spectral MSE={valid_loss:.6f}"
        )

        for key in ["spec", "aux"]:
            self.log_loss(f"loss/{key}", train_loss[key], epoch)

        self.log_loss("loss/SSE", valid_loss, epoch)
