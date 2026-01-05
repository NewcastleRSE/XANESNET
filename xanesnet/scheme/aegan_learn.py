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
import itertools
import logging
import torch
import numpy as np

from collections import defaultdict
from sklearn.model_selection import RepeatedKFold

from xanesnet.scheme.base_learn import Learn, EarlyStopState
from xanesnet.utils.switch import (
    LossSwitch,
    OptimSwitch,
    LossRegSwitch,
    LRSchedulerSwitch,
)


class AEGANLearn(Learn):
    """
    AEGANLearn: Autoencoder - GAN training class.

    This class implements training loop for models combining autoencoders and GANs.
    It supports reconstruction through the autoencoder and adversarial training via the GAN discriminator.

    Compatible model architecture includes AEGAN-MLP.
    """

    def __init__(self, model, dataset, **kwargs):
        # Call the constructor of the parent class
        super().__init__(model, dataset, **kwargs)

        # Unpack AEGAN hyperparameters
        hyper_params = self.hyper_params
        self.lr_gen, self.lr_dis = hyper_params.get("lr", [0.001, 0.00001])
        self.optim_gen, self.optim_dis = hyper_params.get("optimizer", ["adam", "adam"])
        self.loss_gen, self.loss_dis = hyper_params.get("loss", ["mse", "bce"])
        self.loss_reg_gen, self.loss_reg_dis = hyper_params.get(
            "loss_reg", ["None", "None"]
        )
        self.loss_lambda_gen, self.loss_lambda_dis = hyper_params.get(
            "loss_lambda", [0.001, 0.001]
        )

    def train(self, model, dataset):
        """
        Main training loop
        """
        train_loader, valid_loader = self.setup_dataloaders(dataset)

        optimizers, criterion, regularizer, schedulers = self.setup_components(model)
        model.to(self.device)

        valid_losses = {}
        state = EarlyStopState() if self.earlystop_flag else None

        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run the training phase
            train_losses = self._run_one_epoch(
                "train", train_loader, model, criterion, regularizer, optimizers
            )

            # Run the validation phase
            valid_losses = self._run_one_epoch(
                "valid", valid_loader, model, criterion, regularizer, optimizer=None
            )

            # Adjust learning rate if scheduler is used
            if self.lr_scheduler:
                schedulers[0].step()
                schedulers[1].step()

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

        # The final score is the total loss from the last validation epoch
        score = valid_losses["total"]

        return score

    def train_std(self):
        """
        Performs standard training run
        """
        self.train(self.model, self.dataset)

        return self.model

    def train_kfold(self, x_data=None, y_data=None):
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
        criterion = [LossSwitch().get(self.loss_gen), LossSwitch().get(self.loss_dis)]
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

    def setup_components(self, model):
        """Initializes optimizer, loss function, and LR scheduler."""
        # --- Generator Optimizer ---
        params_gen = itertools.chain(
            model.gen_a.parameters(),
            model.gen_b.parameters(),
            model.shared_encoder.parameters(),
            model.shared_decoder.parameters(),
        )
        optimizer_gen = OptimSwitch().get(self.optim_gen)(
            params=params_gen, lr=self.lr_gen
        )

        # --- Discriminator Optimizer ---
        params_dis = itertools.chain(
            model.dis_a.parameters(),
            model.dis_b.parameters(),
        )
        optimizer_dis = OptimSwitch().get(self.optim_dis)(
            params=params_dis, lr=self.lr_dis
        )
        optimizers = [optimizer_gen, optimizer_dis]

        # --- Initialise loss functions ---
        criterion = [LossSwitch().get(self.loss_gen), LossSwitch().get(self.loss_dis)]

        # --- Regularizer ---
        regularizer = LossRegSwitch()

        # --- LR schedulers (optional) ---
        schedulers = []
        if self.lr_scheduler:
            scheduler_gen = LRSchedulerSwitch(
                optimizer_gen,
                self.scheduler_type,
                self.scheduler_params,
            )
            scheduler_dis = LRSchedulerSwitch(
                optimizer_dis,
                self.scheduler_type,
                self.scheduler_params,
            )
            schedulers = [scheduler_gen, scheduler_dis]

        return optimizers, criterion, regularizer, schedulers

    def _run_one_epoch(
        self, phase, loader, model, criterion, regularizer, optimizer=None
    ):
        """Runs a single epoch of training or validation."""
        criterion_gen = criterion[0]
        criterion_dis = criterion[1]

        is_train = phase == "train"
        model.train() if is_train else model.eval()

        epoch_losses = defaultdict(float)
        device = self.device

        with torch.set_grad_enabled(is_train):
            # self.X = a, self.y = b
            for batch in loader:
                inputs_a = batch.x
                inputs_b = batch.y

                inputs_a, inputs_b = (inputs_a.to(device), inputs_b.to(device))

                if is_train:
                    optimizer_gen = optimizer[0]
                    optimizer_dis = optimizer[1]

                    # Update generator and discriminator
                    self._update_generator(
                        inputs_a, inputs_b, model, optimizer_gen, criterion_gen
                    )
                    dis_loss = self._update_discriminator(
                        inputs_a, inputs_b, model, optimizer_dis, criterion_dis
                    )

                # Reconstruct and predict all inputs
                recon_a, recon_b, predict_a, predict_b = model.generate_all(
                    inputs_a, inputs_b
                )

                # Calculate individual loss
                recon_loss_a = criterion_gen(recon_a, inputs_a).mean()
                recon_loss_b = criterion_gen(recon_b, inputs_b).mean()
                predict_loss_a = criterion_gen(predict_a, inputs_a).mean()
                predict_loss_b = criterion_gen(predict_b, inputs_b).mean()

                # Calculate total loss
                gen_loss = recon_loss_a + recon_loss_b + predict_loss_a + predict_loss_b

                # Add regularization loss for generator
                loss_reg_gen_a = regularizer.loss(
                    model.gen_a, self.loss_reg_gen, device
                )
                loss_reg_gen_b = regularizer.loss(
                    model.gen_b, self.loss_reg_gen, device
                )
                gen_loss += self.loss_lambda_gen * loss_reg_gen_a + loss_reg_gen_b

                if is_train:
                    # Add regularization loss for discriminator
                    loss_reg_dis_a = regularizer.loss(
                        model.dis_a, self.loss_reg_dis, device
                    )
                    loss_reg_dis_b = regularizer.loss(
                        model.dis_b, self.loss_reg_dis, device
                    )
                    dis_loss += self.loss_lambda_dis * loss_reg_dis_a + loss_reg_dis_b
                    epoch_losses["dis"] += dis_loss.item()

                epoch_losses["total"] += gen_loss.item()
                epoch_losses["recon_a"] += recon_loss_a.item()
                epoch_losses["recon_b"] += recon_loss_b.item()
                epoch_losses["predict_a"] += predict_loss_a.item()
                epoch_losses["predict_b"] += predict_loss_b.item()

        running_losses = {k: v / len(loader) for k, v in epoch_losses.items()}

        return running_losses

    def _update_generator(self, x_a, x_b, model, optimizer_gen, criterion_gen):
        optimizer_gen.zero_grad()
        x_a_recon, x_b_recon, x_a_predict, x_b_predict = model.generate_all(x_a, x_b)

        # scale loss by mean-maximum value of input
        a_max = torch.max(x_a)
        b_max = torch.max(x_b)

        # reconstruction loss
        loss_recon_a = criterion_gen(x_a_recon, x_a) / a_max
        loss_recon_b = criterion_gen(x_b_recon, x_b) / b_max
        loss_pred_a = criterion_gen(x_a_predict, x_a) / a_max
        loss_pred_b = criterion_gen(x_b_predict, x_b) / b_max

        # total loss
        loss_total = loss_recon_a + loss_recon_b + loss_pred_a + loss_pred_b

        loss_total.backward()
        optimizer_gen.step()

    def _update_discriminator(self, x_a, x_b, model, optimizer_dis, criterion_dis):
        # encode
        optimizer_dis.zero_grad()
        x_a_recon, x_b_recon, x_a_predict, x_b_predict = model.generate_all(x_a, x_b)

        # loss for real inputs
        loss_real_a = self.calc_real_loss(x_a, model.dis_a, criterion_dis)
        loss_real_b = self.calc_real_loss(x_b, model.dis_b, criterion_dis)
        loss_real = loss_real_a + loss_real_b

        # loss for fake inputs
        loss_fake_a = self.calc_fake_loss(x_a_recon, model.dis_a, criterion_dis)
        loss_fake_b = self.calc_fake_loss(x_b_recon, model.dis_b, criterion_dis)
        loss_fake_recon_a = self.calc_fake_loss(x_a_predict, model.dis_a, criterion_dis)
        loss_fake_recon_b = self.calc_fake_loss(x_b_predict, model.dis_b, criterion_dis)
        loss_fake = loss_fake_a + loss_fake_b + loss_fake_recon_a + loss_fake_recon_b

        loss_real = 0.5 * loss_real
        loss_fake = 0.25 * loss_fake

        loss_dis_total = loss_real + loss_fake

        loss_dis_total.backward()
        optimizer_dis.step()

        return loss_dis_total

    def calc_real_loss(self, input_real, model, criterion_dis):
        # Calculate the loss to train dis
        out = model.forward(input_real)
        target = torch.ones_like(out)
        loss = criterion_dis(out, target)
        return loss

    def calc_fake_loss(self, input_fake, model, criterion_dis):
        # Calculate the loss to train gen
        out = model.forward(input_fake)
        target = torch.zeros_like(out)
        loss = criterion_dis(out, target)
        return loss

    def tensorboard_layout(self):
        # Initialise tensorboard writer with custom layout
        layout = {
            "Generator & Discriminator Losses": {
                "Generator Loss": ("Multiline", ["gen_loss/train"]),
                "Discriminator Loss": ("Multiline", ["dis_loss/train"]),
            },
            "Reconstruction Losses": {
                "Reconstruction X": (
                    "Multiline",
                    ["recon_X_loss/train", "recon_X_loss/valid"],
                ),
                "Reconstruction y": (
                    "Multiline",
                    ["recon_y_loss/train", "recon_y_loss/valid"],
                ),
            },
            "Prediction Losses": {
                "Prediction X": (
                    "Multiline",
                    ["pred_X_loss/train", "pred_X_loss/valid"],
                ),
                "Prediction y": (
                    "Multiline",
                    ["pred_y_loss/train", "pred_y_loss/valid"],
                ),
            },
        }
        return layout

    def _log_epoch_loss(self, epoch, train_losses, valid_losses):
        logging.info(
            f">>> Epoch {epoch + 1:03d} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Valid Loss: {valid_losses['total']:.4f}"
        )

        self.log_loss("total_loss/train", train_losses["total"], epoch)
        self.log_loss("dis_loss/train", train_losses["dis"], epoch)
        self.log_loss("recon_X_loss/train", train_losses["recon_a"], epoch)
        self.log_loss("recon_y_loss/train", train_losses["recon_b"], epoch)
        self.log_loss("pred_X_loss/train", train_losses["predict_a"], epoch)
        self.log_loss("pred_y_loss/train", train_losses["predict_b"], epoch)

        self.log_loss("total_loss/valid", valid_losses["total"], epoch)
        self.log_loss("recon_X_loss/valid", valid_losses["recon_a"], epoch)
        self.log_loss("recon_y_loss/valid", valid_losses["recon_b"], epoch)
        self.log_loss("pred_X_loss/valid", valid_losses["predict_a"], epoch)
        self.log_loss("pred_y_loss/valid", valid_losses["predict_b"], epoch)
