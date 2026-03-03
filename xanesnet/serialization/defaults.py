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

###############################################################################
################################## DEFAULTS ###################################
###############################################################################

# These dictionaries contain default config values

DATASOURCE_DEFAULT = {
    "xyzspec": {},
}

DATASET_DEFAULT = {
    "xanesx": {
        "preload": True,
        "force_prepare": False,
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
    "geometric": {
        "preload": True,
        "force_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
    },
    "gemset": {
        "preload": True,
        "force_prepare": False,
        "split_ratios": [1.0],
        "split_indexfile": None,
        "cutoff": 5.0,
        "int_cutoff": 10.0,
        "triplets_only": False,
    },
}

MODEL_DEFAULTS = {
    "mlp": {
        "hidden_size": 226,
        "dropout": 0.1,
        "num_hidden_layers": 3,
        "shrink_rate": 0.5,
        "activation": "prelu",
    },
    "schnet": {
        "hidden_channels": 128,
        "reduce_channels_1": 64,
        "num_filters": 128,
        "num_interactions": 6,
        "num_gaussians": 50,
        "cutoff": 10.0,
        "max_num_neighbors": 32,
        "readout": "add",
        "dipole": False,
        "mean": None,
        "std": None,
        "atomref": None,
    },
    "dimenet": {
        "hidden_channels": 128,
        "num_blocks": 6,
        "num_bilinear": 8,
        "num_spherical": 7,
        "num_radial": 6,
        "cutoff": 5.0,
        "max_num_neighbors": 32,
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
        "max_num_neighbors": 32,
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
        "readout": "mean",
        "output_init": "HeOrthogonal",
        "activation": "swish",
        "scale_file": None,
    },
}

TRAINER_DEFAULTS = {
    "nntrainer": {
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
        "validation_interval": 1,
        "lr_scheduler.lr_scheduler_type": "none",
        "early_stopper.early_stopper_type": "none",
        "early_stopper.restore_best": True,
    },
}

INFERENCER_DEFAULTS = {
    "nninferencer": {
        "batch_size": 1,
        "shuffle": False,
        "drop_last": False,
        "num_workers": 0,
    }
}

STRATEGY_DEFAULTS = {
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

DATASOURCE_REQUIRED = {
    "xyzspec": ["xyz_path", "xanes_path"],
}

DATASET_REQUIRED = {
    "xanesx": ["root"],
    "geometric": ["root"],
    "gemset": ["root"],
}

MODEL_REQUIRED = {
    "mlp": ["out_size", "in_size"],
    "schnet": ["reduce_channels_2"],
    "dimenet": ["out_channels"],
    "dimenet++": ["out_channels"],
    "gemnet": ["num_targets"],
}

TRAINER_REQUIRED = {
    "nntrainer": [],
}

INFERENCER_REQUIRED = {
    "nninferencer": [],
}

STRATEGY_REQUIRED = {
    "single": [],
    "bootstrap": [],
    "ensemble": [],
    "kfold": [],
}
