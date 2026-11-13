<div align="center">

<img src="logo.png" width="420" alt="XANESNET logo">

# XANESNET

Deep learning for theoretical spectroscopy.

[User Manual](https://xanesnet.readthedocs.io) | [Setup](#setup) | [Config UI](#config-ui) | [Usage](#usage) | [People](#people-and-attribution) | [Publications](#publications)

[![Tests](https://github.com/NewcastleRSE/XANESNET/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/NewcastleRSE/XANESNET/actions/workflows/tests.yml)

</div>

## Overview

We present XANESNET, an open-source, PyTorch-based software package for machine learning in spectroscopy. Specifically, the framework integrates training, inference, and automated analysis for structure--spectrum learning within a plugin-based architecture. Its modular design enables users to compare, combine, and extend workflow and analysis components without modifying the core codebase, providing a flexible and reusable alternative to single-purpose implementations. XANESNET supports both forward prediction from atomic structures to spectra and inverse inference from spectra to structures, properties, or descriptors. Through PyTorch Lightning, we enable multi-device training using Distributed Data Parallel (DDP) across multiple GPUs. A shared data pipeline uniformly handles both molecular and periodic systems. Although the framework is agnostic to spectroscopic techniques, current modules primarily implement atom-centred spectra. We demonstrate its use for learning structure--spectrum relationships in X-ray absorption spectroscopy. To our knowledge, \textsc{XANESNET} is the first general-purpose machine-learning framework for spectroscopy with a central focus on extensibility and reproducibility. This facilitates systematic comparisons and evaluations of machine-learning approaches, lowering the barrier for researchers to develop and benchmark novel methods.

## Highlights

- Command-line workflows for forward and inverse training, checkpointed inference, and automated analysis.
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

From the XANESNET directory, create an environment and install a matching CPU PyTorch/PyG wheel set before XANESNET. This Python 3.12 sequence was tested with PyTorch 2.5.1:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
python -m pip install -e .
```

For a non-editable install, use `python -m pip install .` instead of `-e .`, after installing the matching PyTorch/PyG wheels.

```bash
python -m pip install .
```

After installation, the `xanesnet` command should be available:

```bash
xanesnet --help
```

### PyTorch and PyTorch Geometric

XANESNET depends on PyTorch and PyTorch Geometric native extensions. The PyTorch version and CPU/CUDA build must match the extension wheels. If no matching wheel is installed first, pip may try to compile an extension from source and fail.

For CUDA 12.4 with PyTorch 2.5.1, install the matching wheels before XANESNET:

```bash
python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install -e .
```

For other Python, PyTorch, or CUDA versions, choose the corresponding wheel URLs from the PyTorch and PyTorch Geometric installation guides. Keep the PyTorch and extension versions aligned.

### Lightning, Developer and Docs Extras
After installing the matching PyTorch/PyG wheels and the base package, install optional tooling as needed:


```bash
python -m pip install -e ".[lightning]"      # Lightning training backend
python -m pip install -e ".[dev,lightning]"  # build/test tooling and Lightning tests
python -m pip install -e ".[docs,lightning]" # docs tooling and Lightning API imports
```

[frozen.txt](frozen.txt) is kept as a pinned snapshot of one known-working environment. Use it only when you need to reproduce that exact environment; normal installs should use [pyproject.toml](pyproject.toml).

## Config UI

The config UI in [tools/config-ui/](tools/config-ui/) is an interactive React editor for XANESNET `train`, `infer`, and `analyze` YAML files. It is useful both as a guided config builder and as a browsable reference for valid options in the forward and inverse workflows.

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

XANESNET runs are driven by YAML configuration files. The main example workflow is forward structure-to-spectrum prediction: prepare structure/spectrum data, train a model, infer spectra with a saved checkpoint, and analyze inference-run outputs. The framework also supports inverse inference with registered descriptor and multi-head model pairings. Examples live in [configs/](configs/), including [configs/mlp.yaml](configs/mlp.yaml), [configs/mlp_infer.yaml](configs/mlp_infer.yaml), and [configs/analyze_example.yaml](configs/analyze_example.yaml).

### Train

```bash
xanesnet train \
    -i configs/mlp.yaml \
    -n mlp_test \
    -t \
    -y
```

Training outputs are written under `runs/train_<name>_NNN/` (or `runs/train_000/` without a name). Typical outputs include the copied raw config, validated and resolved configs, split indices, the signature (`models/signature.yaml`), final checkpoints (`models/final.pth` for inference and `models/final_full.pth` with the full training state), periodic checkpoints in `checkpoints/`, logs, and optional TensorBoard files. When TensorBoard is enabled, model weight and gradient histograms are omitted by default; pass `--log-weights` to include them.

Use `--dry-run` with `xanesnet train` to construct the normal training pipeline, run one real epoch, and write `model_profile.json` with the model architecture, parameter counts, model size, and peak CUDA memory for that epoch when CUDA is available.

### Multi-Device Training

XANESNET supports multi-gpu/multi-cpu training through Lightning. Install it with `python -m pip install -e ".[lightning]"`. Use the special Trainer `lightning`, set `trainer.num_devices` to the number of CPU training processes with `device: cpu`, or visible GPUs with `device: cuda`. `device: cuda:N` starts at visible index `N`. MPS devices are currently unsupported and untested. `batch_size` and `num_workers` are per process. Launch once with `xanesnet train` or `python -m xanesnet.cli train`, not `torchrun`, MPI, or multi-task `srun`. Under SLURM, request one task on one node and let Lightning start its workers.



### Infer

```bash
xanesnet infer \
    -i configs/mlp_infer.yaml \
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

main(["train", "-i", "configs/mlp.yaml", "-n", "mlp_test", "-y"])
```

For most workflows, the installed `xanesnet` command is the recommended interface because it keeps command logging and run directories consistent.

## Example Usage Workflow

For the paper's end-to-end usage workflow, including external data preparation, see [scripts/paper_workflow/README.md](scripts/paper_workflow/README.md).

## Configuration

Configuration validation and defaults are implemented with packaged JSON Schema files:

- [xanesnet/serialization/config.py](xanesnet/serialization/config.py)
- [xanesnet/serialization/schema_validation.py](xanesnet/serialization/schema_validation.py)
- [xanesnet/schemas/](xanesnet/schemas/)

At a high level, a config contains:

- `seed` and `device`
- `datasource`: input structure and spectrum source
- `dataset`: preprocessing, storage, split, and descriptor or graph settings;
- `model`: model family and hyperparameters
- exactly one runner section: `trainer`, `inferencer`, or analysis settings depending on workflow
- `strategy`: single-model, deep-ensemble, bootstrap, or k-fold training and inference behavior

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

### Main Reference
*XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy* - H. Junkawitsch, B. Li, T. Pope, A. Bande, and T. Penfold, 2026.

### XANESNET
*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

*[Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network](https://doi.org/10.1063/5.0087255)* - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.

*[Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy](https://doi.org/10.26434/chemrxiv.15004916/v1)* - T. J. Penfold, and T. J. Pope, *ChemRxiv*, 2026.

### Extension to X-ray Emission
*[A deep neural network for valence-to-core X-ray emission spectroscopy](https://doi.org/10.1080/00268976.2022.2123406)* - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.

### Applications
*[On the Analysis of X-ray Absorption Spectra for Polyoxometallates](https://doi.org/10.1016/j.cplett.2021.138893)* - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

*[Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

### Miscellaneous
*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.