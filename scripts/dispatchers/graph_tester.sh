#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
# graph_tester dispatcher
# Edit values below, then run:
#   bash scripts/dispatchers/graph_tester.sh
# -----------------------------------------------------------------------------

# Required
JSON_DIR="/media/hendrik/ExternalSSD/final_data/omnixas_test/"

# Exactly one of these should be used. If FILE_STEM is non-empty, it wins.
SAMPLE_INDEX=0
FILE_STEM=""

# Graph parameters
CUTOFF=5.0
MAX_NEIGHBORS=50
GRAPH_METHOD="voronoi"       # radius | voronoi | cov_radius
MIN_FACET_AREA=""           # e.g. "0.25" or "1.0%" (voronoi only)
COV_RADII_SCALE=1.5          # cov_radius only

# Optional toggles
SHOW_VORONOI=false
NO_ATOM_LABELS=false
NO_SHOW=false

# Absorber / sampling
ABSORBER_IDX=0
MAX_TRIPLETS_DRAWN=60
MAX_PATHS_DRAWN=60
MAX_PATHS=128

# Optional output path (empty = do not save)
SAVE_PATH=""

args=(
	"scripts/graph_tester.py"
	"--json-dir" "$JSON_DIR"
	"--cutoff" "$CUTOFF"
	"--max-neighbors" "$MAX_NEIGHBORS"
	"--graph-method" "$GRAPH_METHOD"
	"--cov-radii-scale" "$COV_RADII_SCALE"
	"--absorber-idx" "$ABSORBER_IDX"
	"--max-triplets-drawn" "$MAX_TRIPLETS_DRAWN"
	"--max-paths-drawn" "$MAX_PATHS_DRAWN"
	"--max-paths" "$MAX_PATHS"
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
