# XANESNET Tests

## Quick start

```bash
./run_tests.sh         # run all tests (including slow dry-runs)
./run_tests.sh -q      # quick mode: skip slow dry-run tests
./run_tests.sh -h      # show all options
./run_tests.sh -q -- --cov=xanesnet --cov-report=term-missing  # quick suite with coverage
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
├── unit/                  # shared core behavior and base contracts
└── implementations/       # concrete registered module implementations
```

### `dry_runs/`: pipeline smoke tests

End-to-end tests for the representative train, infer, and analyze workflow using toy data, plus parametrized train-and-infer tests for each matching pair of configurations under `dry_runs/train/` and `dry_runs/infer/`.

All dry-run tests are marked `@pytest.mark.slow` and are skipped in quick mode (`-q`).

To add a train-and-infer test pair, create matching configurations in `train/` and `infer/` with the same stem.

### `unit/`: fixed core behavior

Tests for core functionality shared by the framework rather than owned by one registered implementation: orchestration, base classes, shared graph utilities, serialization, and generic registry/schema contracts.

```
unit/
├── conftest.py       # shared test configuration and global-state isolation
├── batchprocessors/  # shared forward/inverse behavior and base defaults
├── checkpointing/    # Checkpointer state collection, saving and restoration
├── cli/              # argument parsing and train / infer / analyze entry points
├── core/             # training, inference, analysis and Lightning worker orchestration
├── datasets/         # shared TorchDataset preparation, caching and splits
├── descriptors/      # Descriptor base contract
├── graphs/           # shared graph and edge utilities
├── serialization/    # config I/O, schemas, checkpoints, splits, readers/writers and auto-config
├── strategies/       # shared Strategy behavior
└── utils/            # registry, filesystem, random seeding, prompts, logging, math, exceptions
```

Useful guards for contributors:

- `unit/serialization/test_registry_schema_sync.py` fails when a registered type is missing from its JSON schema (or the reverse).
- `unit/serialization/test_shipped_configs.py` validates every YAML under `configs/` and `dry_runs/` against the schemas.

### `implementations/`: implementation-specific unit tests

Tests for concrete registry implementations of the XANESNET extension points.

```text
implementations/
├── batchprocessors/   # registered dataset/model compatibility mappings
├── components/        # registered activations, initializers, optimizers and schedulers
├── datasets/          # descriptor and graph datasets, multiprocessing implementations
├── datasources/       # xyzspec, multixyzspec, pmgjson, multipmgjson on temporary files
├── descriptors/       # direct, rdc, wacsf, soap and pdos implementations
├── encodings/         # affine families, combined/concat, fourier, gaussian, auto resolvers
├── graphs/            # radius, cov_radius and voronoi graph builders
├── losses/            # emd, wcc, specplus, msssim, combined
├── models/            # mlp, multihead, schnet, auto-config resolvers
├── regularizers/
├── stoppers/
└── strategies/        # single, deep_ensemble, bootstrap, kfold (with lightweight fakes)
```

## Coverage (last test run)

The full suite passed (1,469 passed, 1 xfailed) with 84% statement coverage.
Remaining gaps are concentrated in analysis plots and selectors, advanced E3EE/E3EE-Full layers, optional MACE/PDOS paths, and multiprocessing helpers.
Tests are constantly in development and will be expanded in the future.