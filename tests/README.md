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

End-to-end tests for the representative train, infer, and analyze workflow using toy data. Model-specific one-epoch training dry runs are defined by the top-level configurations and run with `scripts/testing/dispatchers/dry_run_all_configs.sh`.

The full pipeline test is marked `@pytest.mark.slow` and is skipped in quick mode (`-q`).

To add a model to the one-epoch dry-run sweep, add its training configuration to the `TRAIN_CONFIGS` list in `scripts/testing/dispatchers/dry_run_all_configs.sh`.

### `unit/`: framework-level unit tests

Tests for general XANESNET framework logic: config parsing and validation, schema enforcement, serialization, CLI dispatching, checkpointing, core pipelines, and shared utilities. These tests are independent of any specific model, dataset, or encoding implementation.

### `implementations/`: implementation-specific unit tests

Tests for individual pluggable components: models, datasets, encodings, losses, descriptors, graphs, regularizers, stoppers, strategies, batch processors, and other registered implementations. Kept separate from framework tests so the general pipeline can be validated independently.

