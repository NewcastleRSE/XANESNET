#!/bin/bash
# Clean up all generated files
set -e

echo "Cleaning up generated files..."
rm -rf ./models/
rm -rf ./outputs/
rm -rf ./data/fe/processed*
rm -rf ./data/graph-set/processed*
rm -rf ./multihead/processed*
rm -rf ./data/multihead/processed*