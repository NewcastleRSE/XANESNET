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


class ScaleFactor(torch.nn.Module):
    def __init__(self, initial: float = 1.0) -> None:
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor(float(initial)), requires_grad=False)

    def forward(self, x: torch.Tensor, ref: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        return x * self.scale_factor
