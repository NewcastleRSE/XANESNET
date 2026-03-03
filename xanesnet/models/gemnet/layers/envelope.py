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


class Envelope(torch.nn.Module):
    """
    Envelope function that ensures a smooth cutoff.

    Parameters
    ----------
        p: int
            Exponent of the envelope function.
    """

    def __init__(self, p: int) -> None:
        super().__init__()
        assert p > 0
        self.p = p
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, d_scaled: torch.Tensor) -> torch.Tensor:
        env_val = 1 + self.a * d_scaled**self.p + self.b * d_scaled ** (self.p + 1) + self.c * d_scaled ** (self.p + 2)
        return torch.where(d_scaled < 1, env_val, torch.zeros_like(d_scaled))
