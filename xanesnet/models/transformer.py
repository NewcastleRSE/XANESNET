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

from typing import List

import torch
from torch import nn

from xanesnet.models.base_model import Model
from xanesnet.registry import ModelRegistry, SchemeRegistry


@ModelRegistry.register("transformer")
@SchemeRegistry.register("transformer", scheme_name="nn")
class Transformer(Model):
    def __init__(
        self,
        in_size: List[int],
        out_size: int,
        hidden_size: int = 128,
        dropout=0.1,
        n_heads: int = 8,
        n_self_attn_layers: int = 2,
        n_cross_attn_layers: int = 3,
    ) -> None:
        super().__init__()

        self.nn_flag = 1
        self.batch_flag = 1
        self.attn_weights = []

        # Save model configuration
        self.register_config(locals(), type="transformer")

        mace_size = in_size[0]
        feat_size = in_size[1]

        # Atom encoding
        self.ln_input = nn.LayerNorm(mace_size)
        self.atom_proj = nn.Linear(mace_size, hidden_size)
        self.context_proj = nn.Linear(hidden_size, hidden_size)

        # Learnable energy embedding (L, D)
        self.energy_embedding = nn.Parameter(torch.randn(out_size, hidden_size))

        # PDOS → energy embedding modulator
        self.feat_to_energy = nn.Linear(feat_size, hidden_size)

        # Self-attention layers over atom descriptors
        self.self_blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_size, n_heads, dropout) for _ in range(n_self_attn_layers)]
        )

        self.energy_self_attn = SelfAttentionBlock(hidden_size, n_heads, dropout)

        # Cross-attention layers for energy queries
        self.cross_blocks = nn.ModuleList(
            [CrossAttentionBlock(hidden_size, n_heads, dropout) for _ in range(n_cross_attn_layers)]
        )

        # Final output projection
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )

        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, batch):
        # Unpack attributes from the batch object
        mace, feat, pos, weight, mask = (
            batch.mace,
            batch.desc,
            batch.pos,
            batch.weight,
            batch.mask,
        )
        B, N, _ = mace.shape
        mace_proj = self.atom_proj(mace)
        context = self.context_proj(mace_proj)
        key_padding_mask = ~mask

        # Atom self-attention layers
        for block in self.self_blocks:
            context = block(context, key_padding_mask=key_padding_mask)

        feat_condition = self.feat_to_energy(feat).unsqueeze(1)
        energy_embedding = self.energy_embedding.unsqueeze(0).expand(B, -1, -1)
        energy_input = energy_embedding + feat_condition

        x = energy_input
        attn_weights_list = []
        for block in self.cross_blocks:
            x, attn = block(query=x, context=context, key_padding_mask=key_padding_mask)
            attn_weights_list.append(attn)

        self.attn_weights = attn_weights_list
        x = self.energy_self_attn(x)

        x = self.final_norm(x)
        x = self.final_mlp(x) + x
        out = self.out_proj(x).squeeze(-1)

        return out

    def get_attn_weights(self):
        return self.attn_weights


class SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ff(x))


class CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, query, context, key_padding_mask=None):
        attn_out, attn_weights = self.cross_attn(
            query=query, key=context, value=context, key_padding_mask=key_padding_mask
        )
        x = self.norm1(query + attn_out)
        x = self.norm2(x + self.ff(x))
        return x, attn_weights
