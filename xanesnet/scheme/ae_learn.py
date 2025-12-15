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
import torch
import numpy as np

from collections import defaultdict
from sklearn.model_selection import RepeatedKFold

from xanesnet.scheme.base_learn import Learn, EarlyStopState
from xanesnet.utils.switch import LossSwitch, LossRegSwitch


class AELearn(Learn):
    """
    AELearn: AutoEncoder training class.

    This class implements training loop for autoencoder models.

    It supports models for reconstruction tasks, including AE_MLP, and AE_CNN.
    """

    def train(self, model, dataset):
        """
        Main training loop
        """
        train_loader, valid_loader, eval_loader = self.setup_dataloaders(dataset)

        optimizer, criterion, regularizer, scheduler = self.setup_components(model)
        model.to(self.device)

        state = EarlyStopState() if self.earlystop_flag else None
        valid_losses = {}

        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run the training phase
            train_losses = self._run_one_epoch(
                "train", train_loader, model, criterion, regularizer, optimizer
            )

            # Run the validation phase
            valid_losses = self._run_one_epoch(
                "valid", valid_loader, model, criterion, regularizer, optimizer=None
            )

            # Adjust learning rate if scheduler is used
            if self.lr_scheduler:
                scheduler.step()

            # Logging for the current epoch
            self._log_epoch_loss(epoch, train_losses, valid_losses)

            # Early stopping
            if self.earlystop_flag:
                self._early_stop(valid_losses["total"], state)
                if state.stop:
                    break

        logging.info("--- Training Finished ---")

        # Log model and final evaluation
        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        # The final score is the total loss from the last training epoch
        score = valid_losses["total"]

        return score

    def train_std(self):
        """
        Performs standard training run
        """

        self.train(self.model, self.dataset)

        return self.model

    def train_kfold(self):
        """
        Performs K-fold cross-validation
        """
        best_model = None
        best_score = float("inf")
        score_list = {"train_score": [], "test_score": []}

        kfold_splitter = RepeatedKFold(
            n_splits=self.n_splits,
            n_repeats=self.n_repeats,
            random_state=self.seed_kfold,
        )

        # Initialise loss criterion and regularizer
        criterion = LossSwitch().get(self.loss)
        regularizer = LossRegSwitch()

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

            test_losses = self._run_one_epoch(
                "valid", test_loader, model, criterion, regularizer
            )

            test_score = test_losses["total"]
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

    def _run_one_epoch(
        self, phase, loader, model, criterion, regularizer, optimizer=None
    ):
        """Runs a single epoch of training or validation."""
        is_train = phase == "train"
        model.train() if is_train else model.eval()

        epoch_losses = defaultdict(float)
        device = self.device

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                batch.to(device)

                # Zero the parameter gradients only during training
                if is_train:
                    optimizer.zero_grad()

                # Pass X or batch object to model
                input_data = batch if model.batch_flag else batch.x
                recon, predict = model(input_data)
                # Calculate individual loss
                recon_loss = criterion(recon, batch.x.float())
                predict_loss = criterion(predict, batch.y.float())

                # Calculate total loss
                loss = recon_loss + predict_loss

                if is_train:
                    # Add regularization loss
                    loss_reg = regularizer.loss(model, self.loss_reg, device)
                    loss += self.loss_lambda * loss_reg

                    loss.backward()
                    optimizer.step()

                epoch_losses["total"] += loss.item()
                epoch_losses["recon"] += recon_loss.item()
                epoch_losses["predict"] += predict_loss.item()

        running_losses = {k: v / len(loader) for k, v in epoch_losses.items()}

        return running_losses

    def tensorboard_layout(self):
        layout = {
            "Losses": {
                "Total Losses": (
                    "Multiline",
                    ["total_loss/train", "total_loss/valid"],
                ),
                "Reconstruction Losses": (
                    "Multiline",
                    ["recon_loss/train", "recon_loss/valid"],
                ),
                "Prediction Losses": (
                    "Multiline",
                    ["pred_loss/train", "pred_loss/valid"],
                ),
            },
        }
        return layout

    def _log_epoch_loss(self, epoch, train_loss, valid_loss):
        logging.info(
            f">>> Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_loss['total']:.4f} | "
            f"Valid Loss: {train_loss['total']:.4f}"
        )

        self.log_loss("total_loss/train", train_loss["total"], epoch)
        self.log_loss("recon_loss/train", train_loss["recon"], epoch)
        self.log_loss("pred_loss/train", train_loss["predict"], epoch)

        self.log_loss("total_loss/valid", valid_loss["total"], epoch)
        self.log_loss("recon_loss/valid", valid_loss["recon"], epoch)
        self.log_loss("pred_loss/valid", valid_loss["predict"], epoch)
