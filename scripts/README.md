# Scripts

Utility and testing scripts for XANESNET.

## Structure

```
scripts/
├── testing/                  # Standalone test/visualisation scripts
│   ├── e3ee_tester.py        #   Inspect E3NN graph builder output
│   ├── encodings_tester.py   #   Visualise encoding pipelines
│   ├── gemnet_tester.py      #   Inspect GemNet graph builder output
│   ├── graph_tester.py       #   Inspect general graph construction
│   ├── test_readers.py       #   Quick datasource reader checks
│   └── dispatchers/          #   Shell wrappers for the testers above
│       ├── run_all_encoding_tests.sh  # Run every encoding tester
│       └── dry_run_all_configs.sh     # One-epoch dry-run on all configs
├── gemnet_scale_fitting.py   # Fit GemNet(-OC) scale factors from data
└── dispatchers/              # Shell wrappers for core scripts
```

## Usage

Most Python scripts under `testing/` have a corresponding shell dispatcher in
`testing/dispatchers/`.  Edit the dispatcher to set paths and parameters, then
run it:

```bash
bash scripts/testing/dispatchers/encodings_tester.sh
bash scripts/testing/dispatchers/graph_tester.sh
```

Additionally:

```bash
bash scripts/testing/dispatchers/run_all_encoding_tests.sh
bash scripts/testing/dispatchers/dry_run_all_configs.sh
```
