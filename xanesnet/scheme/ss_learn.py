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
import torch
import torch.nn.functional as F

from collections import defaultdict

from xanesnet.scheme.base_learn import Learn, EarlyStopState
from xanesnet.utils.switch import LossSwitch
from xanesnet.utils.gaussian import SpectralPost, SpectralBasis


class SSLearn(Learn):
    """
    SSLearn: SoftShell training class.

    This class implements training loop for SoftShell Spectra Network.

    Model compatible with this training process includes: SoftShell.
    """

    def __init__(self, model, dataset, **kwargs):
        super().__init__(model, dataset, **kwargs)

        hyper_params = self.hyper_params
        self.widths_eV = kwargs.get("widths_eV", [0.5, 1.0, 2.0, 4.0])
        self.basis_stride = hyper_params.get("basis_stride", 4)

        self.loss_kwargs = dict(
            alpha=hyper_params.get("loss_alpha", 0.65),
            beta=hyper_params.get("loss_beta", 0.35),
            gamma=hyper_params.get("loss_gamma", 0.25),
            huber_delta=hyper_params.get("loss_huber_delta", 0.01),
            kappa_peak=hyper_params.get("kappa_peak", 0.05),
        )

    def train(self, model, dataset):
        train_loader, valid_loader, _ = self.setup_dataloaders(dataset)

        optimizer, _, _, scheduler = self.setup_components(model)
        model.to(self.device)

        basis = SpectralBasis(
            energies=dataset[0].e,
            widths_eV=self.widths_eV,
            normalize_atoms=True,
            stride=self.basis_stride,
        ).to(self.device)

        spectral_post = SpectralPost(basis=basis, nonneg_output=False).to(self.device)
        spectral_post.eval()
        self._model_diagnostics(spectral_post, model, dataset)

        state = EarlyStopState() if self.earlystop_flag else None
        valid_loss = 0.0

        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run training phase
            train_loss = self._run_one_epoch_train(
                epoch, train_loader, model, optimizer, spectral_post
            )

            # Run validation phase
            valid_loss = self._run_one_epoch_valid(valid_loader, model, spectral_post)

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
        eta_aux_max, eta_aux_min = 3e-3, 3e-4
        T = max(1, self.epochs - 50)  # stop annealing near the end

        for batch in loader:
            batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            c_pred = model(batch)
            y_pred = spectral_post.forward_from_coeffs(c_pred)

            sigma_now = sigma_min + (sigma_max - sigma_min) * max(0, T - epoch) / T
            eta_aux = eta_aux_min + (eta_aux_max - eta_aux_min) * max(0, T - epoch) / T

            self.loss_kwargs["blur_sigma_bins"] = sigma_now
            criterion = LossSwitch().get(self.loss, **self.loss_kwargs)
            loss_spec, (Lc, Ld, Lg) = criterion(batch.y, y_pred)
            loss_aux = F.mse_loss(c_pred, batch.c_star)

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

    def _run_one_epoch_valid(self, loader, model, spectral_post):
        """Runs a single epoch of training or validation."""
        model.eval()

        running_loss = 0.0
        n_elem = 0
        device = self.device

        with torch.set_grad_enabled(False):
            for batch in loader:
                batch.to(device)

                c_pred = model(batch)
                y_pred = spectral_post.forward_from_coeffs(c_pred)  # (B,N)

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

    def _model_diagnostics(self, spectral_post, model, dataset):
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

        trainable, total = self._count_trainable_params(spectral_post)
        trainable_e, total_e = self._count_trainable_params(model.encoder)
        trainable_h, total_h = self._count_trainable_params(model.coeff_head)

        logging.info(
            f"Encoder parameters: {trainable_e:,} trainable / {total_e:,} total"
        )
        logging.info(
            f"CoeffHead parameters: {trainable_h:,} trainable / {total_h:,} total"
        )
        logging.info(f"TOTAL trainable parameters: {trainable_e + trainable_h:,}")

        return spectral_post

    @staticmethod
    def _count_trainable_params(m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        return trainable, total

    def _log_epoch_loss(self, epoch, train_loss, valid_loss):
        logging.info(
            f"Epoch {epoch + 1:03d} | "
            f"Train Lspec={train_loss['spec']:.6f} "
            f"(Lc={train_loss['Lc']:.6f}, Ld={train_loss['Ld']:.6f}, Lg={train_loss['Lg']:.6f}) | "
            f"Aux(c*)={train_loss['aux']:.6f} | "
            f"Val spectral MSE={valid_loss:.6f}"
        )

        for key in ["spec", "Lc", "Ld", "Lg", "aux"]:
            self.log_loss(f"loss/{key}", train_loss[key], epoch)

        self.log_loss("loss/SSE", valid_loss, epoch)
