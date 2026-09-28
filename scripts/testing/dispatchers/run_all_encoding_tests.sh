#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Tom W. Pope, C. D. Rankine, B. Li
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

# Override the datasource json_path (leave empty to use the value in each config).
DATA_DIR="./data/toy_data/"

# Maximum number of spectra drawn per panel.
MAX_SAMPLES=12

# Output directory for the per-encoding PDF reports.
OUT_DIR="./runs/encoding_tests"

# Set to "true" to suppress plt.show() (headless environments).
NO_SHOW=true

declare -A ENCODING_CONFIGS=(
    ["identity"]="configs/encoding_test_configs/in_identity.yaml"
    ["scale"]="configs/encoding_test_configs/in_scale.yaml"
    ["fourier"]="configs/encoding_test_configs/in_fourier.yaml"
    ["gaussian"]="configs/encoding_test_configs/in_gaussian.yaml"
    ["z_score"]="configs/encoding_test_configs/in_z_score.yaml"
    ["min_max"]="configs/encoding_test_configs/in_min_max.yaml"
    ["subtract_average"]="configs/encoding_test_configs/in_subtract_average.yaml"
    ["concat"]="configs/encoding_test_configs/in_concat.yaml"
    ["z_score_per_element"]="configs/encoding_test_configs/in_mlp.yaml"
)

ENCODING_ORDER=(
    "identity"          # runs first: prepares data
    "scale"
    "fourier"
    "gaussian"
    "z_score"
    "min_max"
    "subtract_average"
    "concat"
    "z_score_per_element"
)

mkdir -p "$OUT_DIR"

for enc_name in "${ENCODING_ORDER[@]}"; do
    config="${ENCODING_CONFIGS[$enc_name]}"
    save_path="${OUT_DIR}/encoding_${enc_name}.pdf"

    echo "============================================================"
    echo "  Encoding: ${enc_name}"
    echo "  Config:   ${config}"
    echo "  Output:   ${save_path}"
    echo "============================================================"

    args=(
        "scripts/testing/encodings_tester.py"
        "--config"       "$config"
        "--max-samples"  "$MAX_SAMPLES"
        "--save"         "$save_path"
    )

    [[ -n "$DATA_DIR" ]] && args+=("--data-dir" "$DATA_DIR")
    [[ "$NO_SHOW" == "true" ]] && args+=("--no-show")

    python "${args[@]}" || {
        echo "WARNING: encodings_tester failed for '${enc_name}' (config: ${config})" >&2
        echo "         Continuing with next encoding type ..." >&2
    }

    echo ""
done

echo "Done. Outputs saved to: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"encoding_*.pdf 2>/dev/null || echo "(no PDFs produced)"
