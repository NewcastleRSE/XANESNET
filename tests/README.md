# XANESNET Tests

## Quick start

```bash
./run_tests.sh         # run all tests (including slow dry-runs)
./run_tests.sh -q      # quick mode: skip slow dry-run tests
./run_tests.sh -h      # show all options
```

Extra arguments after `--` are forwarded to pytest:

```bash
./run_tests.sh -- -k "schnet" --tb=long
```

## Structure

```
tests/
├── README.md
├── dry_runs/              # complete pipeline smoke tests
├── unit/                  # framework-level unit tests
└── implementations/       # implementation-specific unit tests
```

### `dry_runs/`: pipeline smoke tests

End-to-end tests for the representative train, infer, and analyze workflow using toy data, plus parametrized train-and-infer tests for each matching pair of configurations under `dry_runs/train/` and `dry_runs/infer/`.

All dry-run tests are marked `@pytest.mark.slow` and are skipped in quick mode (`-q`).

To add a train-and-infer test pair, create matching configurations in `train/` and `infer/` with the same stem.

### `unit/`: framework-level unit tests

Tests for general XANESNET framework logic: config parsing and validation, schema enforcement, serialization, CLI dispatching, checkpointing, core pipelines, and shared utilities. These tests are independent of any specific model, dataset, or encoding implementation.

### `implementations/`: implementation-specific unit tests

Tests for individual pluggable components: models, datasets, encodings, losses, descriptors, graphs, regularizers, stoppers, strategies, batch processors, and other registered implementations. Kept separate from framework tests so the general pipeline can be validated independently.

