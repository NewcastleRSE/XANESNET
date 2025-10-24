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

import torch

from xanesnet.scheme.nn_learn import NNLearn


class MHLearn(NNLearn):
    """
    MHLearn: Multi-Head neural network training class.

    This class implements training loop for multihead neural network.

    Models compatible with this training process include: MH_MLP,MH_CNN, and MH_GNN.
    """

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
                # predict shape = (Head, Batch, Feat)
                predict = model(input_data)

                # Rearrange to (Batch, Head, Feat)
                predict = predict.permute(1, 0, 2)
                head_idx = batch.head_idx

                # predict shape = (Batch, Feat)
                predict = predict[torch.arange(batch.x.shape[0]), head_idx]
                y = batch.y.float()
                loss = criterion(predict, y).mean() / y.abs().mean()

                # Add regularization loss
                loss_reg = regularizer.loss(model, self.loss_reg, device)
                loss += self.loss_lambda * loss_reg

                if is_train:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

        return running_loss / len(loader)
