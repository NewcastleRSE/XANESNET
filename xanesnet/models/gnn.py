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

import inspect
from typing import List, Optional

import torch
import torch_geometric.nn as geom_nn
from torch import nn

from xanesnet.models.base_model import Model
from xanesnet.registry import ModelRegistry, SchemeRegistry
from xanesnet.utils.switch import ActivationSwitch

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GATv2": geom_nn.GATv2Conv,
    "GraphConv": geom_nn.GraphConv,
}


@ModelRegistry.register("gnn")
@SchemeRegistry.register("gnn", scheme_name="nn")
class GNN(Model):
    """
    A class for constructing a customisable GNN (Graph Neural Network) model.
    """

    def __init__(
        self,
        in_size: List[int],
        out_size: int,
        layer_name: str = "GATv2",
        layer_params: Optional[dict] = None,
        hidden_size: int = 512,
        dropout: float = 0.2,
        num_hidden_layers: int = 5,
        num_mlp_hidden_layers: int = 3,
        activation: str = "prelu",
    ):
        """
        Args:
            in_size (List[int]): List of input size integers
            out_size (integer): Output size
            layer_name (string): Name of GNN layer (GAT, GATv2, GCN, GraphConv)
            layer_params (dict): parameters pass to GNN layers
            hidden_size (integer): Size of the hidden layer.
            dropout (float): If none-zero, add dropout layer on the outputs
                of each hidden layer with dropout probability equal to dropout.
            num_hidden_layers (integer): Number of hidden layers
                in the network.
            activation (string): Name of activation function applied
                to the hidden layers.
        """
        super().__init__()
        self.nn_flag = 1
        self.gnn_flag = 1
        self.batch_flag = 1

        # Save model configuration
        self.register_config(locals(), type="gnn")

        # Two input sizes: one for the GNN part, one for the MLP part
        gnn_feat_size = in_size[0]
        mlp_feat_size = in_size[1]

        if layer_params is None:
            layer_params = {
                "heads": 2,
                "concat": True,
                "edge_dim": 16,
            }
        heads = layer_params.get("heads")

        act_fn = ActivationSwitch().get(activation)
        gnn_layer = gnn_layer_by_name[layer_name]

        # --- Input and hidden Layers ---
        layers = []
        for i in range(num_hidden_layers):
            layers.append(gnn_layer(in_channels=gnn_feat_size, out_channels=hidden_size, **layer_params))
            layers.append(nn.BatchNorm1d(hidden_size * heads))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))

            gnn_feat_size = hidden_size * heads

        # --- Output Layer ---
        layers.append(gnn_layer(in_channels=gnn_feat_size, out_channels=hidden_size, **layer_params))

        self.gnn_layers = nn.ModuleList(layers)

        # ---- MLP hidden Layers ----
        mlp_in_size = gnn_feat_size + mlp_feat_size
        num_mlp_hidden_size = mlp_in_size * 2

        layers = []
        current_size = mlp_in_size
        for i in range(num_mlp_hidden_layers):
            next_size = num_mlp_hidden_size
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.Dropout(dropout))
            layers.append(act_fn)
            current_size = next_size

        # --- MLP output Layer ---
        layers.append(nn.Linear(current_size, out_size))

        self.mlp_layers = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        """
        Inputs:
            x - Input features per node
            edge_index - List of edge index pairs
            batch_idx - Index of batch element for each node
        """
        # reshape concatenated graph_attr to [batch_size, feat_size]
        nfeats = batch[0].graph_attr.shape[0]
        graph_attr = batch.graph_attr.reshape(len(batch), nfeats)

        # Unpack attributes from the batch object
        x, edge_idx, edge_attr, batch_idx = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        for layer in self.gnn_layers:
            if isinstance(layer, geom_nn.MessagePassing):
                sig = inspect.signature(layer.forward)

                # Check if the layer can accept edge attributes/weights
                if "edge_attr" in sig.parameters:
                    x = layer(x, edge_idx, edge_attr=edge_attr)
                elif "edge_weight" in sig.parameters:
                    x = layer(x, edge_idx, edge_weight=edge_attr)
                else:
                    x = layer(x, edge_idx)
            else:
                x = layer(x)

        # Specific node
        # node_idx = 0
        # node_feature = x[node_idx]
        # out = self.head(node_feature)

        # Average pooling
        x = geom_nn.global_mean_pool(x, batch_idx)

        # Append graph feature before sending to mlp layers
        x = torch.cat((x, graph_attr), dim=1)

        out = self.mlp_layers(x)

        return out
