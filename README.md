<div align="center">

<img src="logo.png" width="420" alt="XANESNET logo">

# XANESNET

Deep learning for theoretical X-ray spectroscopy.

[User Manual](https://xanesnet.readthedocs.io) | [Setup](#setup) | [Config UI](#config-ui) | [Usage](#usage) | [People](#people-and-attribution) | [Publications](#publications)

</div>

<div align="center">
<strong>This README is a work in progress.</strong>
</div>

## Overview

We present XANESNET, a PyTorch-based, open-source software framework for machine learning in spectroscopy. The framework integrates training, inference, and automated analysis within a plugin-based architecture. Its modular design allows users to compare, combine, and extend different methods without modifying the core codebase, providing a flexible and reusable framework rather than a single-purpose implementation. XANESNET supports forward prediction from structure to spectrum and inverse inference from spectra to structures or properties. A unified data pipeline uniformly handles molecular and periodic systems, while the framework remains agnostic to the spectroscopic technique. We demonstrate its use for learning structure-spectrum relationships in X-ray absorption spectroscopy. By prioritizing extensibility and reproducibility, XANESNET aims to accelerate and make more accessible machine-learning research in spectroscopy.

## Highlights

- Command-line workflows for forward-model training, checkpointed inference, and prediction analysis.
- Interactive browser-based config editor in [tools/config-ui/](tools/config-ui/).
- GPLv3 licensed open-source distribution.

## Setup

### Requirements

- Linux (tested on Ubuntu)
- Python **3.12** (the current known-working environment uses Python **3.12.9**)
- A C/C++ build toolchain if pip needs to compile native PyTorch Geometric extensions
- Optional: NVIDIA GPU and CUDA-compatible PyTorch wheels for `device: cuda`

### Install XANESNET

Install from a local checkout with the project metadata in [pyproject.toml](pyproject.toml):

From the XANESNET directory:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For a non-editable install, use:

```bash
python -m pip install .
```

After installation, the `xanesnet` command should be available:

```bash
xanesnet --help
```

### PyTorch and PyTorch Geometric

XANESNET depends on PyTorch and PyTorch Geometric native extensions. CPU-only Linux installs may work with plain pip resolution. For CUDA installs, or if `torch-scatter`, `torch-sparse`, or `torch-cluster` cannot find wheels automatically, install wheels matching your Python, PyTorch, and CUDA versions first.

Example for PyTorch 2.5 with CUDA 12.4:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install -e .
```

Adjust the wheel URLs for your target platform. The PyTorch and PyTorch Geometric installation guides are the source of truth for the current compatibility matrix.

### Developer and Docs Extras

```bash
python -m pip install -e ".[dev]"   # build and test tooling
python -m pip install -e ".[docs]"  # documentation tooling
```

[frozen.txt](frozen.txt) is kept as a pinned snapshot of one known-working environment. Use it only when you need to reproduce that exact environment; normal installs should use [pyproject.toml](pyproject.toml).

## Config UI

The config UI in [tools/config-ui/](tools/config-ui/) is an interactive React editor for XANESNET `train`, `infer`, and `analyze` YAML files. It is useful both as a guided config builder and as a browsable reference for valid options in the current forward workflow.

It provides:

- Mode-specific forms for Train, Infer, and Analyze workflows.
- YAML import with automatic mode detection.
- Inference `signature.yaml` import for checkpoint-aware infer configs.
- Live YAML preview with defaults materialized and top-level sections ordered for readability.
- Schema-backed defaults and object-shape validation for datasource, dataset, encoding, model, runner, strategy, and analysis choices.

Run it locally:

```bash
cd tools/config-ui
npm install
npm run dev
```

Vite prints the local URL, usually `http://127.0.0.1:5173/` or the next free port. Save the generated YAML into [configs/](configs/) or another working directory, then pass it to the CLI.

For descriptor workflows, set `dataset.dataset_type` to `descriptor` for forward prediction (descriptors -> spectra) or `descriptor_inverse` for inverse prediction (spectra -> descriptors); the multiprocessing variants are `descriptor_mp` and `descriptor_inverse_mp`.

Common config UI commands:

```bash
npm run dev      # Start the Vite development server
npm run lint     # Run ESLint
npm run build    # Type-check and build production assets into dist/
npm run preview  # Preview the production build locally
```

More details are in [tools/config-ui/README.md](tools/config-ui/README.md).

## Usage

XANESNET runs are driven by YAML configuration files. The supported workflow today is forward spectrum prediction: prepare structure/spectrum data, train a model, infer spectra with a saved checkpoint, and analyze prediction files. Examples live in [configs/](configs/), including [configs/in_mlp.yaml](configs/in_mlp.yaml), [configs/in_mlp_infer.yaml](configs/in_mlp_infer.yaml), and [configs/analyze_example.yaml](configs/analyze_example.yaml).

### Train

```bash
xanesnet train \
    -i configs/in_mlp.yaml \
    -n mlp_test \
    -t \
    -y
```

Training outputs are written under `runs/<timestamp>_train_<name>/`. Typical outputs include the copied raw config, validated and resolved configs, split indices, checkpoint signatures, checkpoints, final model weights, logs, and optional TensorBoard files.

Use `--dry-run` with `xanesnet train` to construct the normal training pipeline, run one real epoch, and write `model_profile.json` with the model architecture, parameter counts, model size, and peak CUDA memory for that epoch when CUDA is available.

### Infer

```bash
xanesnet infer \
    -i configs/in_mlp_infer.yaml \
    -m runs/<train_run>/models/final.pth \
    -n mlp_infer \
    -y
```

Inference configs are strictly merged with the checkpoint signature saved during training. If a user-provided value conflicts with the signature, inference fails early. The merged and validated configs are saved in the inference run directory, and predictions are written under `predictions/`.

### Analyze

```bash
xanesnet analyze \
    -i configs/analyze_example.yaml \
    -r runs/<infer_run> \
    -n mlp_analysis \
    -y
```

The `-r` option can be supplied multiple times when comparing or aggregating inference runs. Each run's directory name is used as its display name in tables; override with `-d`. Analysis reads each run's `validated_infer_config.yaml` and, when its datasource is available, supplies selectors with the matched raw `structure`. If this context cannot be loaded, analysis continues with prediction data only. Set `preload: true` in the analysis config to load predictions and matched structures into memory during setup (default: `false`).

### Python Entry Point

For scripts and notebooks, you can call the same dispatcher that backs the CLI:

```python
from xanesnet.cli import main

main(["train", "-i", "configs/in_mlp.yaml", "-n", "mlp_test", "-y"])
```

For most workflows, the installed `xanesnet` command is the recommended interface because it keeps command logging and run directories consistent.

## Configuration

Configuration validation and defaults are implemented with packaged JSON Schema files:

- [xanesnet/serialization/config.py](xanesnet/serialization/config.py)
- [xanesnet/serialization/schema_validation.py](xanesnet/serialization/schema_validation.py)
- [xanesnet/schemas/](xanesnet/schemas/)

At a high level, a config contains:

- `seed` and `device`
- `datasource`: input structure and spectrum source
- `dataset`: preprocessing, storage, split, and descriptor or graph settings; set `dataset_type` to `descriptor` (forward) or `descriptor_inverse` (inverse) for descriptor workflows
- `model`: model family and hyperparameters
- exactly one runner section: `trainer`, `inferencer`, or analysis settings depending on workflow
- `strategy`: single model or ensemble training/inference behavior

The config UI reads the same schemas through [tools/config-ui/src/schemas](tools/config-ui/src/schemas), a symlink to [xanesnet/schemas/](xanesnet/schemas/).

## Testing

See [tests/README.md](tests/README.md) and [run_tests.sh](run_tests.sh).

## People and Attribution

[Hendrik Junkawitsch](https://www.helmholtz-berlin.de/pubbin/vkart.pl?v=yyqxqn&sprache=de), Helmholtz-Zentrum Berlin and Humboldt University of Berlin - main code author and current maintainer

[Prof. Thomas Penfold](https://ncl.ac.uk/nes/people/profile/tompenfold.html), Newcastle University - project lead and supervision

[Dr. Thomas Pope](https://www.ncl.ac.uk/nes/people/profile/thomaspope2.html), Newcastle University

[Dr. Conor Rankine](https://pure.york.ac.uk/portal/en/persons/conor-rankine), York University

[Dr. Bowen Li](https://rse.ncldata.dev/team/bowen-li), Newcastle University

## License

This project is licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for details.

## Publications

### XANESNET
*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

*[Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network](https://doi.org/10.1063/5.0087255)* - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.
 
### Extension to X-ray Emission
*[A deep neural network for valence-to-core X-ray emission spectroscopy](https://doi.org/10.1080/00268976.2022.2123406)* - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.

### Applications
*[On the Analysis of X-ray Absorption Spectra for Polyoxometallates](https://doi.org/10.1016/j.cplett.2021.138893)* - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

*[Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

### Miscellaneous
*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.