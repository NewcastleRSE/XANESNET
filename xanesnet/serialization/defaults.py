# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.

"""Default configuration values and required-key specifications for all XANESNET components."""

from typing import Any

###############################################################################
################################## DEFAULTS ###################################
###############################################################################

# These dictionaries contain default config values

DATASOURCE_DEFAULT: dict[str, dict[str, Any]] = {
    "xyzspec": {},
    "multixyzspec": {},
    "pmgjson": {},
}

DATASET_DEFAULT: dict[str, dict[str, Any]] = {
    "descriptor": {
        "preload": True,
        "skip_prepare": False,
        "mode": "forward",
        "split_ratios": [1.0],
        "split_indexfile": None,
        "fourier": False,
        "fourier_concat": False,
        "gaussian": False,
        "widths_eV": [0.5, 1.0, 2.0, 4.0],
        "basis_stride": 4,
        "basis_path": None,
        "descriptors": [{"descriptor_type": "wacsf", "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32}}],
    },
    "e3ee": {
        "preload": True,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "cutoff": 5.0,
        "max_num_neighbors": 32,
        "use_path_branch": False,
        "max_paths_per_structure": 128,
        "graph_method": "radius",
        "min_facet_area": None,
        "cov_radii_scale": 1.5,
        "att_cutoff": 10.0,
        "att_max_num_neighbors": 64,
        "att_graph_method": "radius",
        "att_min_facet_area": None,
        "att_cov_radii_scale": 1.5,
    },
    "e3ee_full": {
        "preload": True,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "cutoff": 5.0,
        "max_num_neighbors": 32,
        "use_path_branch": False,
        "max_paths_per_site": 128,
        "graph_method": "radius",
        "min_facet_area": None,
        "cov_radii_scale": 1.5,
        "att_cutoff": 10.0,
        "att_max_num_neighbors": 64,
        "att_graph_method": "radius",
        "att_min_facet_area": None,
        "att_cov_radii_scale": 1.5,
        "use_absorber_mask": False,
    },
    "envembed": {
        "preload": True,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "widths_eV": [0.2, 1.0, 2.0, 4.0],
        "basis_stride": 4,
        "basis_path": None,
        "env_radius": None,
        "descriptors": [{"descriptor_type": "wacsf", "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32}}],
    },
    "richgraph": {
        "preload": True,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
    },
    "geometrygraph": {
        "preload": True,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "cutoff": 6.0,
        "max_num_neighbors": 32,
        "compute_angles": False,
        "graph_method": "radius",
        "min_facet_area": None,
        "cov_radii_scale": 1.5,
    },
    "gemnet": {
        "preload": False,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "cutoff": 5.0,
        "max_num_neighbors": 50,
        "graph_method": "radius",
        "min_facet_area": None,
        "cov_radii_scale": 1.5,
        "quadruplets": False,
        "int_cutoff": 10.0,
        "int_max_neighbors": None,
        "int_graph_method": None,
        "int_min_facet_area": None,
        "int_cov_radii_scale": None,
        "oc_mode": False,
        "oc_cutoff_aeaint": None,
        "oc_max_neighbors_aeaint": None,
        "oc_graph_method_aeaint": None,
        "oc_min_facet_area_aeaint": None,
        "oc_cov_radii_scale_aeaint": None,
        "oc_cutoff_aint": None,
        "oc_max_neighbors_aint": None,
        "oc_graph_method_aint": None,
        "oc_min_facet_area_aint": None,
        "oc_cov_radii_scale_aint": None,
    },
    "gemnet_oc": {
        "preload": False,
        "skip_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "cutoff": 5.0,
        "max_num_neighbors": 50,
        "graph_method": "radius",
        "min_facet_area": None,
        "cov_radii_scale": 1.5,
        "quadruplets": True,
        "int_cutoff": 10.0,
        "int_max_neighbors": None,
        "int_graph_method": None,
        "int_min_facet_area": None,
        "int_cov_radii_scale": None,
        "oc_mode": True,
        "oc_cutoff_aeaint": 5.0,
        "oc_max_neighbors_aeaint": 50,
        "oc_graph_method_aeaint": None,
        "oc_min_facet_area_aeaint": None,
        "oc_cov_radii_scale_aeaint": None,
        "oc_cutoff_aint": 10.0,
        "oc_max_neighbors_aint": 50,
        "oc_graph_method_aint": None,
        "oc_min_facet_area_aint": None,
        "oc_cov_radii_scale_aint": None,
    },
}

# Multiprocessing dataset variants.
# adds ``num_workers`` knob (None / non-positive means "use os.cpu_count()").
_MP_DATASET_NAMES: set[str] = {
    "descriptor",
    "e3ee",
    "e3ee_full",
    "envembed",
    "geometrygraph",
    "gemnet",
    "gemnet_oc",
}
for _name in _MP_DATASET_NAMES:
    DATASET_DEFAULT[f"{_name}_mp"] = {**DATASET_DEFAULT[_name], "num_workers": None}

MODEL_DEFAULTS: dict[str, dict[str, Any]] = {
    "mlp": {
        "hidden_size": 226,
        "dropout": 0.1,
        "num_hidden_layers": 3,
        "shrink_rate": 0.5,
        "activation": "prelu",
    },
    "envembed": {
        "n_shells": 4,
        "max_radius_angs": 7.0,
        "init_width": 0.8,
        "use_gating": True,
        "head_hidden": 256,
        "head_depth": 3,
        "dropout": 0.1,
    },
    "schnet": {
        "hidden_channels": 128,
        "reduce_channels_1": 64,
        "num_filters": 128,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "mean_spectrum": None,
    },
    "dimenet": {
        "hidden_channels": 128,
        "num_blocks": 6,
        "num_bilinear": 8,
        "num_spherical": 7,
        "num_radial": 6,
        "cutoff": 5.0,
        "envelope_exponent": 5,
        "num_before_skip": 1,
        "num_after_skip": 2,
        "num_output_layers": 3,
        "act": "swish",
        "output_initializer": "zeros",
    },
    "dimenet++": {
        "hidden_channels": 128,
        "num_blocks": 4,
        "int_emb_size": 64,
        "basis_emb_size": 8,
        "out_emb_channels": 256,
        "num_spherical": 7,
        "num_radial": 6,
        "cutoff": 5.0,
        "envelope_exponent": 5,
        "num_before_skip": 1,
        "num_after_skip": 2,
        "num_output_layers": 3,
        "act": "swish",
        "output_initializer": "zeros",
    },
    "gemnet": {
        "num_spherical": 7,
        "num_radial": 6,
        "num_blocks": 4,
        "emb_size_atom": 128,
        "emb_size_edge": 128,
        "emb_size_trip": 64,
        "emb_size_quad": 32,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_sbf": 32,
        "emb_size_bil_quad": 32,
        "emb_size_bil_trip": 64,
        "num_before_skip": 1,
        "num_after_skip": 1,
        "num_concat": 1,
        "num_atom": 2,
        "triplets_only": False,
        "cutoff": 5.0,
        "int_cutoff": 10.0,
        "envelope_exponent": 5,
        "output_init": "HeOrthogonal",
        "activation": "swish",
        "scale_file": None,
        "num_elements": 100,
    },
    "gemnet_oc": {
        "num_spherical": 7,
        "num_radial": 128,
        "num_blocks": 4,
        "emb_size_atom": 256,
        "emb_size_edge": 512,
        "emb_size_trip_in": 64,
        "emb_size_trip_out": 64,
        "emb_size_quad_in": 32,
        "emb_size_quad_out": 32,
        "emb_size_aint_in": 64,
        "emb_size_aint_out": 64,
        "emb_size_rbf": 16,
        "emb_size_cbf": 16,
        "emb_size_sbf": 32,
        "num_before_skip": 2,
        "num_after_skip": 2,
        "num_concat": 1,
        "num_atom": 3,
        "num_output_afteratom": 3,
        "num_atom_emb_layers": 2,
        "num_global_out_layers": 2,
        "cutoff": 12.0,
        "cutoff_qint": 12.0,
        "cutoff_aeaint": 12.0,
        "cutoff_aint": 12.0,
        "rbf": {"name": "gaussian"},
        "rbf_spherical": {"name": "gaussian"},
        "envelope": {"name": "polynomial", "exponent": 5},
        "cbf": {"name": "spherical_harmonics"},
        "sbf": {"name": "spherical_harmonics"},
        "output_init": "HeOrthogonal",
        "activation": "silu",
        "quad_interaction": True,
        "atom_edge_interaction": True,
        "edge_atom_interaction": True,
        "atom_interaction": True,
        "scale_basis": False,
        "num_elements": 100,
        "scale_file": None,
    },
    "e3ee": {
        "max_z": 100,
        "atom_emb_dim": 128,
        "atom_hidden_dim": 128,
        "atom_layers": 3,
        "local_cutoff": 6.0,
        "rbf_dim": 32,
        "energy_rbf_dim": 48,
        "scatter_dim": 128,
        "latent_dim": 128,
        "head_hidden_dim": 128,
        "e3nn_irreps": "64x0e + 32x1o + 16x2e",
        "e3nn_irreps_message": "16x0e + 8x1o + 4x2e",
        "e3nn_lmax": 2,
        "out_mlp_layers": 3,
        "use_invariant_branch": True,
        "use_attention_branch": True,
        "use_eq_attention_branch": False,
        "use_conv_branch": False,
        "use_eq_conv_branch": False,
        "use_equivariant_branch": True,
        "use_path_branch": False,
        "fusion_mode": "cat",
        "residual_scale_init": 0.1,
        "attention_heads": 4,
        "attention_rbf_dim": 16,
        "attention_lmax": 2,
        "attention_irreps": "32x0e + 16x1o + 8x2e",
        "conv_use_gate": True,
    },
    "e3ee_full": {
        "max_z": 100,
        "atom_emb_dim": 128,
        "atom_hidden_dim": 128,
        "atom_layers": 3,
        "local_cutoff": 6.0,
        "rbf_dim": 32,
        "energy_rbf_dim": 48,
        "scatter_dim": 128,
        "latent_dim": 128,
        "head_hidden_dim": 128,
        "e3nn_irreps": "64x0e + 32x1o + 16x2e",
        "e3nn_irreps_message": "16x0e + 8x1o + 4x2e",
        "e3nn_lmax": 2,
        "out_mlp_layers": 3,
        "use_invariant_branch": True,
        "use_attention_branch": True,
        "use_eq_attention_branch": False,
        "use_conv_branch": False,
        "use_eq_conv_branch": False,
        "use_equivariant_branch": True,
        "use_path_branch": False,
        "fusion_mode": "cat",
        "use_absorber_mask": True,
        "residual_scale_init": 0.1,
        "attention_heads": 4,
        "attention_rbf_dim": 16,
        "attention_lmax": 2,
        "attention_irreps": "32x0e + 16x1o + 8x2e",
        "conv_use_gate": True,
    },
}

# Model fields that may be resolved from a prepared training dataset by using
# the registered batch processor for the dataset/model pair.
MODEL_AUTO_FIELDS: dict[str, set[str]] = {
    "mlp": {"in_size", "out_size"},
    "lstm": {"in_size", "out_size"},
    "envembed": {"in_size", "kgroups"},
    "schnet": {"reduce_channels_2"},
    "dimenet": {"out_channels"},
    "dimenet++": {"out_channels"},
    "gemnet": {"num_targets"},
    "gemnet_oc": {"num_targets"},
    "e3ee": {"out_size"},
    "e3ee_full": {"out_size"},
}

TRAINER_DEFAULTS: dict[str, dict[str, Any]] = {
    "basic": {
        "batch_size": 3,
        "shuffle": True,
        "drop_last": False,
        "num_workers": 0,
        "loss.loss_type": "mse",
        "regularizer.regularizer_type": "none",
        "regularizer.weight": 1.0,
        "epochs": 10,
        "learning_rate": 0.001,
        "optimizer": "Adam",
        "max_norm": None,
        "validation_interval": 1,
        "lr_scheduler.lr_scheduler_type": "none",
        "lr_warmup": True,
        "warmup_steps": 500,
        "early_stopper.early_stopper_type": "none",
        "early_stopper.restore_best": True,
    },
}

INFERENCER_DEFAULTS: dict[str, dict[str, Any]] = {
    "basic": {
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
    },
}

STRATEGY_DEFAULTS: dict[str, dict[str, Any]] = {
    "single": {
        "weight_init": "default",
        "weight_init_params": {},
        "bias_init": "zeros",
        "checkpoint_interval": None,
    },
    "bootstrap": {
        "weight_init": "default",
        "weight_init_params": {},
        "bias_init": "zeros",
        "checkpoint_interval": None,
    },
    "ensemble": {
        "weight_init": "default",
        "weight_init_params": {},
        "bias_init": "zeros",
        "checkpoint_interval": None,
    },
    "kfold": {
        "weight_init": "default",
        "weight_init_params": {},
        "bias_init": "zeros",
        "checkpoint_interval": None,
    },
}

###############################################################################
################################## REQUIRED ###################################
###############################################################################

# These dictionaries contain required config keys

DATASOURCE_REQUIRED: dict[str, list[str]] = {
    "xyzspec": ["xyz_path", "xanes_path"],
    "multixyzspec": ["root_path"],
    "pmgjson": ["json_path"],
}

DATASET_REQUIRED: dict[str, list[str]] = {
    "descriptor": ["root"],
    "descriptor_mp": ["root"],
    "envembed": ["root"],
    "envembed_mp": ["root"],
    "geometrygraph": ["root"],
    "geometrygraph_mp": ["root"],
    "gemnet": ["root"],
    "gemnet_mp": ["root"],
    "gemnet_oc": ["root"],
    "gemnet_oc_mp": ["root"],
    "e3ee": ["root"],
    "e3ee_mp": ["root"],
    "e3ee_full": ["root"],
    "e3ee_full_mp": ["root"],
    "richgraph": ["root"],  # TODO Not fully implemented yet.
}

MODEL_REQUIRED: dict[str, list[str]] = {
    "mlp": ["out_size", "in_size"],
    "envembed": ["in_size", "kgroups"],
    "schnet": ["reduce_channels_2"],
    "dimenet": ["out_channels"],
    "dimenet++": ["out_channels"],
    "gemnet": ["num_targets"],
    "gemnet_oc": ["num_targets"],
    "e3ee": ["out_size"],
    "e3ee_full": ["out_size"],
}

TRAINER_REQUIRED: dict[str, list[str]] = {
    "basic": [],
}

INFERENCER_REQUIRED: dict[str, list[str]] = {
    "basic": [],
}

STRATEGY_REQUIRED: dict[str, list[str]] = {
    "single": [],
    "bootstrap": [],
    "ensemble": [],
    "kfold": [],
}
