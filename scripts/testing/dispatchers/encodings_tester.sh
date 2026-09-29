#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
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
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$REPO_ROOT"

# Path to the training config whose encodings should be visualized.
# The datasource, dataset, and encoding pipeline are taken from this file.
CONFIG="./configs/mlp.yaml"

# Override the datasource json_path from the config (leave empty to use the
# value in the config file).
DATA_DIR="./data/toy_data/"

# Maximum number of spectra drawn per panel (keep small for readability).
MAX_SAMPLES=12

# Optional output file (PDF with one page per encoding + overview).
# Leave empty to display interactively.
SAVE_PATH=""   # e.g. "runs/encoding_check.pdf"

# Set to "true" to suppress plt.show() (useful for headless environments).
NO_SHOW=false


args=(
    "scripts/testing/encodings_tester.py"
    "--config"    "$CONFIG"
    "--max-samples" "$MAX_SAMPLES"
)

[[ -n "$DATA_DIR"   ]] && args+=("--data-dir" "$DATA_DIR")
[[ -n "$SAVE_PATH"  ]] && args+=("--save" "$SAVE_PATH")
[[ "$NO_SHOW" == "true" ]] && args+=("--no-show")

python "${args[@]}"
