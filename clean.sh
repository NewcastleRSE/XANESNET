#!/bin/bash
# Clean up all generated files
set -e

echo "Cleaning up generated files..."
# Remove specific directories
rm -rf ./models/
rm -rf ./outputs/

dirs=(./data .github/workflows/data)

for base_dir in "${dirs[@]}"; do
    if [ -d "$base_dir" ]; then
        # Find all directories starting with "processed" recursively
        find "$base_dir" -type d -name 'processed*' -print0 | while IFS= read -r -d '' dir; do
            echo "Deleting $dir"
            rm -rf "$dir"
        done
    fi
done