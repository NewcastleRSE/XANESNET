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

import logging
import random
from collections import defaultdict

import numpy as np
import torch

import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch.optim.swa_utils import AveragedModel, SWALR

from xanesnet.scheme.base_learn import Learn
from xanesnet.utils.switch import LossSwitch


class SSLearn(Learn):
    """
    SSLearn: SoftShell training class.

    This class implements training loop for SoftShell Spectra Network.

    Model compatible with this training process includes: SoftShell.
    """

    def __init__(self, model, dataset, **kwargs):
        # Call the constructor of the parent class
        super().__init__(model, dataset, **kwargs)

        # Unpack SoftShell hyperparameters
        hyper_params = self.hyper_params
        self.weight_decay = hyper_params.get("weight_decay", 1e-4)
        self.basis_stride = hyper_params.get("basis_stride", 4)

        # SWA parameters
        self.swa_frac = hyper_params.get("swa_frac", 0.8)
        self.swa_lr = hyper_params.get("swa_lr", 5e-5)

        # loss parameters
        self.loss_kwargs = dict(
            alpha=hyper_params.get("loss_alpha", 0.65),
            beta=hyper_params.get("loss_beta", 0.35),
            gamma=hyper_params.get("loss_gamma", 0.25),
            huber_delta=hyper_params.get("loss_huber_delta", 0.01),
            kappa_peak=hyper_params.get("kappa_peak", 0.05),
        )

    # ============================================================
    # Main training loop
    # ============================================================
    def train(self, model, dataset):
        from xanesnet.models.softshell import SpectralPost, SpectralBasis

        train_loader, valid_loader, _ = self.setup_dataloaders(dataset)
        model.to(self.device)

        # Initialise parameter-free spectral "post" component
        eV = dataset[0].e
        widths_bins = self.compute_widths_bins(eV)

        basis = SpectralBasis(
            energies=eV,
            widths_bins=widths_bins,
            normalize_atoms=True,
            stride=self.basis_stride,
        ).to(self.device)

        spectral_post = SpectralPost(basis=basis, nonneg_output=False).to(self.device)
        spectral_post.eval()

        # Model diagnostics
        self.model_diagnostics(spectral_post, model, dataset)

        # Initialise optimizer
        model_param = list(model.encoder.parameters()) + list(
            model.coeff_head.parameters()
        )
        optimizer = torch.optim.AdamW(
            model_param, lr=self.lr, weight_decay=self.weight_decay
        )

        # SWA setup
        swa_start = int(self.swa_frac * self.epochs)
        swa_encoder, swa_coeff, swa_scheduler = self.setup_swa(
            model, optimizer, self.swa_lr
        )
        swa_active = False

        valid_loss = 0.0
        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run training phase
            train_loss = self._run_one_epoch_train(
                epoch, train_loader, model, optimizer, spectral_post
            )

            # Update SWA scheduler
            if epoch >= swa_start:
                swa_active = True
                self.update_swa(swa_encoder, swa_coeff, model, swa_scheduler)

            # Run validation phase
            valid_loss = self._run_one_epoch_valid(
                epoch, valid_loader, model, spectral_post
            )

            # Logging for the current epoch
            self.log_epoch(
                epoch, "SWA" if swa_active else "Base", train_loss, valid_loss
            )

        logging.info("--- Training Finished ---")

        # Log model and final evaluation
        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        if swa_active:
            model.encoder = swa_encoder.module
            model.coeff_head = swa_coeff.module

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
        pass

    def _run_one_epoch_train(self, epoch, loader, model, optimizer, spectral_post):
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
        ETA_AUX_MAX, ETA_AUX_MIN = 3e-3, 3e-4
        T = max(1, self.epochs - 50)  # stop annealing near the end

        for batch in loader:
            batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            # ---- Forward pass ----
            c_pred = model(batch)
            y_pred = spectral_post.forward_from_coeffs(c_pred)

            # ---- Compute losses ----
            sigma_now = sigma_min + (sigma_max - sigma_min) * max(0, T - epoch) / T
            eta_aux = ETA_AUX_MIN + (ETA_AUX_MAX - ETA_AUX_MIN) * max(0, T - epoch) / T

            # Additional parameter pass to loss
            self.loss_kwargs["blur_sigma_bins"] = sigma_now
            criterion = LossSwitch().get(self.loss, **self.loss_kwargs)
            loss_spec, (Lc, Ld, Lg) = criterion(batch.y, y_pred)
            # Tiny auxiliary coefficient regression to ridge c*
            loss_aux = F.mse_loss(c_pred, batch.c_star)

            # ---- total loss and optimize ----
            loss_total = loss_spec + eta_aux * loss_aux
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model_params, 1.0)
            optimizer.step()

            epoch_losses["spec"] += loss_spec.item()
            epoch_losses["aux"] += loss_aux.item()
            epoch_losses["Lc"] += Lc.item()
            epoch_losses["Ld"] += Ld.item()
            epoch_losses["Lg"] += Lg.item()

        train_losses = {k: v / len(loader) for k, v in epoch_losses.items()}

        return train_losses

    def _run_one_epoch_valid(self, epoch, loader, model, spectral_post):
        """Runs a single epoch of training or validation."""
        model.eval()

        epoch_losses = defaultdict(float)
        device = self.device

        n_elem = 0
        with torch.set_grad_enabled(False):
            for batch in loader:
                batch.to(device)

                c_pred = model(batch)
                y_pred = spectral_post.forward_from_coeffs(c_pred)  # (B,N)

                epoch_losses["sse"] += F.mse_loss(
                    y_pred, batch.y, reduction="sum"
                ).item()
                n_elem += batch.y.numel()

        valid_losses = {k: v / max(1, n_elem) for k, v in epoch_losses.items()}

        return valid_losses

    def tensorboard_layout(self):
        layout = {
            "Losses": {
                "Losses": ["Multiline", ["loss/train", "loss/validation"]],
            },
        }
        return layout

    def model_diagnostics(self, spectral_post, model, dataset):
        print("--- Model Diagnostics ---")
        train_loader, valid_loader, _ = self.setup_dataloaders(dataset)

        with torch.no_grad():
            for loader, tag in [(train_loader, "Train"), (valid_loader, "Val")]:
                sse, n_elem = 0.0, 0
                for batch in loader:
                    batch.to(self.device)
                    c_batch = batch.c_star
                    y_batch = batch.y

                    y_gauss = spectral_post.basis.synthesize(c_batch)  # (B, N)
                    sse += F.mse_loss(y_gauss, y_batch, reduction="sum").item()
                    n_elem += y_batch.numel()
                mse_gauss = sse / max(1, n_elem)
                logging.info(
                    f"[Stage 1] Gaussian-only fit MSE ({tag}) = {mse_gauss:.10f}"
                )

        trainable, total = self.count_trainable_params(spectral_post)
        logging.info(
            f"[Stage 1] SpectralPost parameters: {trainable:,} trainable / {total:,} total (no optimizer needed)"
        )

        trainable_e, total_e = self.count_trainable_params(model.encoder)
        trainable_h, total_h = self.count_trainable_params(model.coeff_head)

        logging.info(
            f"[Stage 2 / Encoder parameters: {trainable_e:,} trainable / {total_e:,} total"
        )
        logging.info(
            f"[Stage 2 / CoeffHead parameters: {trainable_h:,} trainable / {total_h:,} total"
        )
        logging.info(
            f"[Stage 2 / TOTAL trainable parameters: {trainable_e + trainable_h:,}"
        )

        return spectral_post

    @staticmethod
    def compute_widths_bins(eV):
        dE = float(eV[1] - eV[0])
        widths_eV = (0.5, 1.0, 2.0, 4.0)
        widths_bins = tuple(max(w / dE, 0.5) for w in widths_eV)

        return widths_bins

    @staticmethod
    def count_trainable_params(m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        return trainable, total

    @staticmethod
    def setup_swa(model, optimizer, swa_lr):
        swa_encoder = AveragedModel(model.encoder)
        swa_coeff = AveragedModel(model.coeff_head)
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr)

        return swa_encoder, swa_coeff, swa_scheduler

    @staticmethod
    def update_swa(swa_encoder, swa_coeff, model, swa_scheduler):
        swa_encoder.update_parameters(model.encoder)
        swa_coeff.update_parameters(model.coeff_head)
        swa_scheduler.step()

    def log_epoch(self, epoch, tag, train_loss, valid_loss):
        logging.info(
            f"[Stage 2 / Epoch {epoch + 1:03d} [{tag}] | "
            f"Train Lspec={train_loss['spec']:.6f} "
            f"(Lc={train_loss['Lc']:.6f}, Ld={train_loss['Ld']:.6f}, Lg={train_loss['Lg']:.6f}) | "
            f"Aux(c*)={train_loss['aux']:.6f} | "
            f"Val spectral MSE={valid_loss['sse']:.6f}"
        )

        for key in ["spec", "Lc", "Ld", "Lg", "aux"]:
            self.log_loss(f"loss/{key}", train_loss[key], epoch)

        self.log_loss("loss/SSE", valid_loss["sse"], epoch)
