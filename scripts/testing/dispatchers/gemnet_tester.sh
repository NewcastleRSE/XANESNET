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

# Required
JSON_DIR="./data/toy_data/"

# Exactly one of these should be used. If FILE_STEM is non-empty, it wins.
SAMPLE_INDEX=0
FILE_STEM=""

# Main graph params
CUTOFF=5.0
MAX_NEIGHBORS=50
GRAPH_METHOD="voronoi"       # radius | voronoi | cov_radius
MIN_FACET_AREA="0.01%"           # e.g. "0.25" or "1.0%"
COV_RADII_SCALE=1.5
SHOW_VORONOI=false

# Quadruplet / interaction graph params
ENABLE_QUADRUPLETS=true
INT_CUTOFF=6.0
INT_MAX_NEIGHBORS=50
INT_GRAPH_METHOD="radius"
INT_MIN_FACET_AREA=""
INT_COV_RADII_SCALE=1.5

# GemNet-OC params
ENABLE_OC=true

OC_CUTOFF_AEAINT=5.0
OC_MAX_NEIGHBORS_AEAINT=50
OC_GRAPH_METHOD_AEAINT="voronoi"
OC_MIN_FACET_AREA_AEAINT=""
OC_COV_RADII_SCALE_AEAINT=1.5

OC_CUTOFF_AINT=10.0
OC_MAX_NEIGHBORS_AINT=50
OC_GRAPH_METHOD_AINT="radius"
OC_MIN_FACET_AREA_AINT=""
OC_COV_RADII_SCALE_AINT=1.5

# Drawing / output params
TARGET_SITE_IDX=0
MAX_TRIPLETS_DRAWN=60
MAX_QUADS_DRAWN=60
MAX_MIXED_DRAWN=60
NO_ATOM_LABELS=false
SAVE_PATH=""
NO_SHOW=false

args=(
	"scripts/testing/gemnet_tester.py"
	"--json-dir" "$JSON_DIR"
	"--cutoff" "$CUTOFF"
	"--max-neighbors" "$MAX_NEIGHBORS"
	"--graph-method" "$GRAPH_METHOD"
	"--cov-radii-scale" "$COV_RADII_SCALE"
	"--int-cutoff" "$INT_CUTOFF"
	"--int-max-neighbors" "$INT_MAX_NEIGHBORS"
	"--int-graph-method" "$INT_GRAPH_METHOD"
	"--int-cov-radii-scale" "$INT_COV_RADII_SCALE"
	"--oc-cutoff-aeaint" "$OC_CUTOFF_AEAINT"
	"--oc-cutoff-aint" "$OC_CUTOFF_AINT"
	"--oc-max-neighbors-aeaint" "$OC_MAX_NEIGHBORS_AEAINT"
	"--oc-max-neighbors-aint" "$OC_MAX_NEIGHBORS_AINT"
	"--oc-graph-method-aeaint" "$OC_GRAPH_METHOD_AEAINT"
	"--oc-cov-radii-scale-aeaint" "$OC_COV_RADII_SCALE_AEAINT"
	"--oc-graph-method-aint" "$OC_GRAPH_METHOD_AINT"
	"--oc-cov-radii-scale-aint" "$OC_COV_RADII_SCALE_AINT"
	"--target-site-idx" "$TARGET_SITE_IDX"
	"--max-triplets-drawn" "$MAX_TRIPLETS_DRAWN"
	"--max-quads-drawn" "$MAX_QUADS_DRAWN"
	"--max-mixed-drawn" "$MAX_MIXED_DRAWN"
)

if [[ -n "$FILE_STEM" ]]; then
	args+=("--file" "$FILE_STEM")
else
	args+=("--index" "$SAMPLE_INDEX")
fi

if [[ -n "$MIN_FACET_AREA" ]]; then
	args+=("--min-facet-area" "$MIN_FACET_AREA")
fi

if [[ "$SHOW_VORONOI" == "true" ]]; then
	args+=("--show-voronoi")
fi

if [[ "$ENABLE_QUADRUPLETS" == "true" ]]; then
	args+=("--quadruplets")
fi

if [[ -n "$INT_MIN_FACET_AREA" ]]; then
	args+=("--int-min-facet-area" "$INT_MIN_FACET_AREA")
fi

if [[ "$ENABLE_OC" == "true" ]]; then
	args+=("--oc")
fi

if [[ -n "$OC_MIN_FACET_AREA_AEAINT" ]]; then
	args+=("--oc-min-facet-area-aeaint" "$OC_MIN_FACET_AREA_AEAINT")
fi

if [[ -n "$OC_MIN_FACET_AREA_AINT" ]]; then
	args+=("--oc-min-facet-area-aint" "$OC_MIN_FACET_AREA_AINT")
fi

if [[ "$NO_ATOM_LABELS" == "true" ]]; then
	args+=("--no-atom-labels")
fi

if [[ -n "$SAVE_PATH" ]]; then
	args+=("--save" "$SAVE_PATH")
fi

if [[ "$NO_SHOW" == "true" ]]; then
	args+=("--no-show")
fi

echo "Running: python3 ${args[*]}"
python3 "${args[@]}"
