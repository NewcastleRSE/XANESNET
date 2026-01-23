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
        "mode": "forward",
        "fourier": False,
        "fourier_concat": False,
        "gaussian": False,
        "widths_eV": [0.5, 1.0, 2.0, 4.0],
        "basis_stride": 4,
        "basis_path": None,
        "descriptors": [{"descriptor_type": "wacsf", "params": {"r_min": 1.0, "r_max": 6.0, "n_g2": 16, "n_g4": 32}}],
    }
}

MODEL_DEFAULTS = {
    "mlp": {
        "params.hidden_size": 226,
        "params.dropout": 0.1,
        "params.num_hidden_layers": 3,
        "params.shrink_rate": 0.5,
        "params.activation": "prelu",
        "weights_init.weights": "default",
        "weights_init.bias": "zeros",
        "weights_init.weights_params": {},
    },
}

TRAINER_DEFAULTS = {
    "nntrainer": {
        "params.epochs": 10,
        "params.batch_size": 3,
        "params.learning_rate": 0.001,
        "params.optimizer": "Adam",
        "params.loss.loss_type": "mse",
        "params.loss.params": {},
        "params.regularizer.regularizer_type": "none",
        "params.regularizer.params.weight": 1.0,
        "params.shuffle": True,
        "params.drop_last": False,
        "params.num_workers": 0,
        "params.lr_scheduler.lr_scheduler_type": "none",
        "params.lr_scheduler.params": {},
        "params.early_stopper.early_stopper_type": "none",
        "params.early_stopper.params.restore_best": True,
    },
}

INFERENCER_DEFAULTS = {
    "nninferencer": {
        "params.batch_size": 1,
        "params.loss.loss_type": "mse",
        "params.regularizer.regularizer_type": "none",
        "params.shuffle": False,
        "params.drop_last": False,
        "params.num_workers": 0,
    }
}

STRATEGY_DEFAULTS = {
    "single": {"params": {}},
    "bootstrap": {"params": {}},
    "ensemble": {"params": {}},
    "kfold": {"params": {}},
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
}

MODEL_REQUIRED = {
    "mlp": ["params.out_size", "params.in_size"],
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
