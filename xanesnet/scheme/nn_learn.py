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

from sklearn.model_selection import RepeatedKFold

from xanesnet.scheme.base_learn import Learn
from xanesnet.utils.switch import LossSwitch, LossRegSwitch


class NNLearn(Learn):
    """
    NNLearn: Neural Network training class.

    This class implements standard supervised learning training loop for neural network.

    Models compatible with this training process include: MLP, CNN, LSTM, GNN, and Transformer.
    """

    def train(self, model, dataset):
        """
        Main training loop
        """
        train_loader, valid_loader, eval_loader = self.setup_dataloaders(dataset)

        optimizer, criterion, regularizer, scheduler = self.setup_components(model)
        model.to(self.device)

        valid_loss = 0.0
        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run training phase
            train_loss = self._run_one_epoch(
                "train", train_loader, model, criterion, regularizer, optimizer
            )

            # Run validation phase
            valid_loss = self._run_one_epoch(
                "valid", valid_loader, model, criterion, regularizer, optimizer=None
            )

            # Adjust learning rate if scheduler is used
            if self.lr_scheduler:
                scheduler.step()

            # Logging for the current epoch
            logging.info(
                f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}"
            )
            self.log_loss("loss/train", train_loss, epoch)
            self.log_loss("loss/validation", valid_loss, epoch)

        logging.info("--- Training Finished ---")

        # Log model and final evaluation
        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        # The final score is the validation loss from the last epoch
        score = valid_loss

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

            test_score = self._run_one_epoch(
                "valid", test_loader, model, criterion, regularizer
            )

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

        running_loss = 0.0
        device = self.device

        with torch.set_grad_enabled(is_train):
            for batch in loader:
                batch.to(device)

                # Zero the parameter gradients only during training
                if is_train:
                    optimizer.zero_grad()

                # Pass X or batch object to model
                input_data = batch if model.batch_flag else batch.x
                predict = model(input_data)

                if model.gnn_flag:
                    predict = torch.flatten(predict)

                if predict.shape[1] != batch.y.float().shape[1]:
                    loss = criterion(predict, batch.cstar.float())
                else:
                    loss = criterion(predict, batch.y.float())

                # Add regularization loss
                loss_reg = regularizer.loss(model, self.loss_reg, device)
                loss += self.loss_lambda * loss_reg

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

        return running_loss / len(loader)

    def tensorboard_layout(self):
        layout = {
            "Losses": {
                "Losses": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        return layout
