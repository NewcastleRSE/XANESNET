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
from collections import defaultdict

import torch

import torch.nn.functional as F
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
        self.diagnostics = hyper_params.get("diagnostics", False)
        self.weight_decay = hyper_params.get("weight_decay", 1e-4)

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

    def train(self, model, dataset):
        """
        Main training loop
        """
        train_loader, valid_loader, eval_loader = self.setup_dataloaders(dataset)
        model.to(self.device)

        encoder = model.encoder
        coeff_head = model.coeff_head

        spectral_post = self.model_diagnostics(model, dataset)

        encoder_params = list(encoder.parameters()) + list(coeff_head.parameters())

        optimizer = torch.optim.AdamW(
            encoder_params, lr=self.lr, weight_decay=self.weight_decay
        )

        has_enc_params = any(p.requires_grad for p in encoder.parameters())
        has_head_params = any(p.requires_grad for p in coeff_head.parameters())

        swa_start_epoch = int(self.swa_frac * self.epochs)
        swa_encoder = AveragedModel(encoder) if has_enc_params else None
        swa_coeff = AveragedModel(coeff_head) if has_head_params else None
        swa_scheduler = SWALR(optimizer, swa_lr=self.swa_lr)
        swa_active = False

        valid_loss = 0.0
        logging.info(f"--- Starting Training for {self.epochs} epochs ---")
        for epoch in range(self.epochs):
            # Run training phase
            train_loss = self._run_one_epoch_train(
                epoch, train_loader, model, optimizer, spectral_post
            )

            # SWA scheduler
            if epoch >= swa_start_epoch:
                swa_active = True
                if swa_encoder is not None:
                    swa_encoder.update_parameters(encoder)
                if swa_coeff is not None:
                    swa_coeff.update_parameters(coeff_head)
                swa_scheduler.step()

            # Run validation phase
            valid_loss = self._run_one_epoch_valid(valid_loader, model, spectral_post)

            # # Logging for the current epoch
            tag = "SWA" if swa_active else "Base"
            avg_Lc = train_loss["Lc"]
            avg_Ld = train_loss["Ld"]
            avg_Lg = train_loss["Lg"]
            avg_spec_loss = train_loss["spec"]
            avg_aux_loss = train_loss["aux"]

            val_spectral_mse = valid_loss["sse"]
            logging.info(
                f"[Stage 2 / Epoch {epoch+1:03d} [{tag}] | "
                f"Train Lspec={avg_spec_loss:.6f} (Lc={avg_Lc:.6f}, Ld={avg_Ld:.6f}, Lg={avg_Lg:.6f}) | "
                f"Aux(c*)={avg_aux_loss:.6f} | Val spectral MSE={val_spectral_mse:.6f}"
            )

            self.log_loss("loss/Lspec", avg_spec_loss, epoch)
            self.log_loss("loss/Lc", avg_Lc, epoch)
            self.log_loss("loss/Ld", avg_Ld, epoch)
            self.log_loss("loss/Lg", avg_Lg, epoch)
            self.log_loss("loss/Aux", avg_aux_loss, epoch)
            self.log_loss("loss/SSE", val_spectral_mse, epoch)

        logging.info("--- Training Finished ---")

        # Log model and final evaluation
        if self.mlflow_flag:
            logging.info("\nLogging the trained model as a run artifact...")
            self.log_mlflow(model)

        self.log_close()

        # The final score is the validation loss from the last epoch
        score = valid_loss

        model_list = []
        if swa_active and (swa_encoder is not None):
            model_list.append(swa_encoder)
        else:
            model_list.append(encoder)
        if swa_active and (swa_coeff is not None):
            model_list.append(swa_coeff)
        else:
            model_list.append(coeff_head)

        return model_list, score

    def train_std(self):
        """
        Performs standard training run
        """
        model, _ = self.train(self.model, self.dataset)

        return self.model

    def train_kfold(self):
        """
        Performs K-fold cross-validation
        """
        pass

    def _run_one_epoch_train(self, epoch, loader, model, optimizer, spectral_post):
        """Runs a single epoch of training or validation."""
        model.train()

        epoch_losses = defaultdict(float)
        device = self.device

        encoder = model.encoder
        coeff_head = model.coeff_head
        encoder_params = list(encoder.parameters()) + list(coeff_head.parameters())

        with torch.set_grad_enabled(True):
            for batch in loader:
                batch.to(device)

                # Zero the parameter gradients only during training
                optimizer.zero_grad(set_to_none=True)

                h = encoder(
                    batch.desc, lengths=batch.lengths, dists=batch.dist
                )  # (B,512)
                c_pred = coeff_head(h)  # (B,K)
                # Stage-1 is frozen; forward through it
                y_pred = spectral_post.forward_from_coeffs(c_pred)  # (B,N)

                # initialise spectral loss plus
                sigma_max, sigma_min = 9.0, 5.0  # bins
                T = max(1, self.epochs - 50)  # stop annealing near the end
                sigma_now = sigma_min + (sigma_max - sigma_min) * max(0, T - epoch) / T
                self.loss_kwargs["blur_sigma_bins"] = sigma_now
                criterion = LossSwitch().get(self.loss, **self.loss_kwargs)
                loss_spec, (Lc, Ld, Lg) = criterion(batch.y, y_pred)

                # Tiny auxiliary coefficient regression to ridge c*
                loss_aux = F.mse_loss(c_pred, batch.c_star)

                ETA_AUX_MAX, ETA_AUX_MIN = 3e-3, 3e-4
                eta_aux = (
                    ETA_AUX_MIN + (ETA_AUX_MAX - ETA_AUX_MIN) * max(0, T - epoch) / T
                )

                loss_total = loss_spec + eta_aux * loss_aux
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(encoder_params, 1.0)
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

        epoch_losses = defaultdict(float)
        device = self.device

        encoder = model.encoder
        coeff_head = model.coeff_head

        n_elem = 0
        with torch.set_grad_enabled(False):
            for batch in loader:
                batch.to(device)

                h = encoder(
                    batch.desc, lengths=batch.lengths, dists=batch.dist
                )  # (B,512)
                c_pred = coeff_head(h)  # (B,K)
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

    def model_diagnostics(self, model, dataset):
        print("--- Model Diagnostics ---")
        from xanesnet.models.softshell import SpectralPost, SpectralBasis

        train_loader, valid_loader, _ = self.setup_dataloaders(dataset)

        eV = dataset[0].e
        dE = float(eV[1] - eV[0])
        widths_eV = (0.5, 1.0, 2.0, 4.0)
        widths_bins = tuple(max(w / dE, 0.5) for w in widths_eV)
        basis_stride = 4
        add_constant_column = False

        basis = SpectralBasis(
            energies=eV,
            widths_bins=widths_bins,
            normalize_atoms=True,
            stride=basis_stride,
            add_constant_column=add_constant_column,
        ).to(self.device)

        with torch.no_grad():
            for loader, tag in [(train_loader, "Train"), (valid_loader, "Val")]:
                sse, n_elem = 0.0, 0
                for batch in loader:
                    batch.to(self.device)
                    c_batch = batch.c_star
                    y_batch = batch.y

                    y_gauss = basis.synthesize(c_batch)  # (B, N)
                    sse += F.mse_loss(y_gauss, y_batch, reduction="sum").item()
                    n_elem += y_batch.numel()
                mse_gauss = sse / max(1, n_elem)
                logging.info(
                    f"[Stage 1] Gaussian-only fit MSE ({tag}) = {mse_gauss:.10f}"
                )

        # Parameter-free spectral "post"
        spectral_post = SpectralPost(basis=basis, nonneg_output=False).to(self.device)
        spectral_post.eval()
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

    def count_trainable_params(self, m):
        trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
        total = sum(p.numel() for p in m.parameters())
        return trainable, total
