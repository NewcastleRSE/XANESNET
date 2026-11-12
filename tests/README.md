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

Tests for general XANESNET framework logic. They are independent of any specific model, dataset, or encoding implementation: collaborators are replaced with small fakes registered in the public registries under `unit_test_*` keys, so the tests exercise public contracts rather than private details. They run in well under a minute and need no data or GPU.

```
unit/
├── conftest.py     # config factories and an autouse fixture that isolates global state
├── checkpointing/  # Checkpointer: collecting, saving and restoring model/training state
├── cli/            # argument parsing and the train / infer / analyze entry points
├── components/     # activation, initializer, optimizer and LR-scheduler registries
├── core/           # core_train, core_infer, core_analyze and the Lightning worker pipelines (run on fakes)
├── serialization/  # config I/O and accessors, JSON-schema validation, checkpoints, splits,
│                   # prediction readers/writers, TensorBoard, auto-config, shipped example configs,
│                   # schema/registry consistency
└── utils/          # registry, filesystem, random seeding, prompts, logging, math, exceptions
```

Useful guards for contributors:

- `serialization/test_registry_schema_sync.py` fails when a registered type is missing from its JSON schema (or the reverse).
- `serialization/test_shipped_configs.py` validates every YAML under `configs/` and `dry_runs/` against the schemas.

```bash
python -m pytest tests/unit -q                       # run the unit tests
python -m pytest tests/unit --cov=xanesnet           # with coverage
```

### `implementations/`: implementation-specific unit tests

Tests for the pluggable components registered by `xanesnet`. They follow the same rules as `unit/`: they check xanesnet's own behaviour (formulas it defines, validation, signatures, wiring, ordering and branches) and not the correctness of PyTorch, DScribe, pymatgen or other dependencies. They are kept separate from the framework tests so the general pipeline can be validated independently.

```text
implementations/
├── batchprocessors/   # registry, forward/inverse bases, per-model field mapping
├── datasets/          # base dataset (prepare, preload, splits), descriptor and graph datasets, mp helpers
├── datasources/       # xyzspec, multixyzspec, pmgjson, multipmgjson on temporary files
├── descriptors/       # direct, rdc, wacsf, plus xanesnet's handling in soap and pdos
├── encodings/         # affine families, combined/concat, fourier, gaussian, auto resolvers
├── graphs/            # radius, cov_radius, voronoi builders and the edge/triplet utilities
├── losses/            # emd, wcc, specplus, msssim, combined
├── models/            # mlp, multihead, schnet, auto-config resolvers
├── regularizers/
├── stoppers/
└── strategies/        # single, deep_ensemble, bootstrap, kfold (with lightweight fakes)
```