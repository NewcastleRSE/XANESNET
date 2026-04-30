<div align="center">

<img src="logo.png" width="420" alt="XANESNET logo">

# XANESNET

Deep learning for theoretical X-ray spectroscopy.

[User Manual](https://xanesnet.readthedocs.io) | [Setup](#setup) | [Config UI](#config-ui) | [Usage](#usage) | [People](#people-and-attribution) | [Publications](#publications)

</div>

<div align="center">
! This README.md is WORK IN PROGRESS !
</div>

## Overview

XANESNET is a Python package for machine-learning simulation and analysis of X-ray absorption near edge structure (XANES) spectra. The current Python codebase supports the forward prediction workflow: train models on structures and spectra, run checkpointed inference, and analyze prediction outputs.

- **Supported now:** forward mapping from molecular or periodic structural inputs to spectra.
- **Planned later:** reverse mapping from spectra back to properties or structural information.

The project combines descriptor-based neural networks, graph neural networks, YAML-driven workflows, checkpoint signatures, and an interactive configuration editor so that forward XANES experiments are reproducible and easier to share.

## Highlights

- Command-line workflows for forward-model training, checkpointed inference, and prediction analysis.
- YAML configuration files with validation, defaults, copied raw configs, resolved train configs, checkpoint signatures, and strict inference merging.
- Model families including MLP, LSTM, SchNet, DimeNet, DimeNet++, GemNet, GemNet-OC, E3EE, E3EEFull, and EnvEmbed.
- Data sources for XYZ/spectrum files, multi-XYZ/spectrum folders, and pymatgen JSON data.
- Descriptor, EnvEmbed, geometry graph, GemNet, E3EE, and E3EEFull datasets, including multiprocessing variants where implemented.
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

```bash
git clone https://github.com/NewcastleRSE/xray-spectroscopy-ml.git
cd xray-spectroscopy-ml
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

### PyTorch And PyTorch Geometric

XANESNET depends on PyTorch and PyTorch Geometric native extensions. CPU-only Linux installs may work with plain pip resolution. For CUDA installs, or if `torch-scatter`, `torch-sparse`, or `torch-cluster` cannot find wheels automatically, install wheels matching your Python, PyTorch, and CUDA versions first.

Example for PyTorch 2.5 with CUDA 12.4:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install -e .
```

Adjust the wheel URLs for your target platform. The PyTorch and PyTorch Geometric installation guides are the source of truth for the current compatibility matrix.

### Developer And Docs Extras

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
- Schema-backed compatibility rules for datasource, dataset, model, batch processor, strategy, runner, and analysis choices.
- Light and dark themes stored in browser state.

Run it locally:

```bash
cd tools/config-ui
npm install
npm run dev
```

Vite prints the local URL, usually `http://127.0.0.1:5173/` or the next free port. Save the generated YAML into [configs/](configs/) or another working directory, then pass it to the CLI.

For supported descriptor workflows, keep `dataset.mode: forward`. Reverse-mode configuration is not part of the current documented workflow and is planned for a later stage.

Common config UI commands:

```bash
npm run dev      # Start the Vite development server
npm run lint     # Run ESLint
npm run build    # Type-check and build production assets into dist/
npm run preview  # Preview the production build locally
```

More details are in [tools/config-ui/README.md](tools/config-ui/README.md).

## Usage

XANESNET runs are driven by YAML configuration files. The supported workflow today is forward XANES prediction: prepare structure/spectrum data, train a model, infer spectra with a saved checkpoint, and analyze prediction files. Examples live in [configs/](configs/), including [configs/in_mlp.yaml](configs/in_mlp.yaml), [configs/in_mlp_infer.yaml](configs/in_mlp_infer.yaml), and [configs/analyze_example.yaml](configs/analyze_example.yaml).

### Train

```bash
xanesnet train \
    -i configs/in_mlp.yaml \
    -n mlp_test \
    -t \
    -y
```

Training outputs are written under `runs/<timestamp>_train_<name>/`. Typical outputs include the copied raw config, validated and resolved configs, split indices, checkpoint signatures, checkpoints, final model weights, logs, and optional TensorBoard files.

### Infer

```bash
xanesnet infer \
    -i configs/in_mlp_infer.yaml \
    -m runs/<train_run>/models/final.pth \
    -n mlp_infer \
    -y
```

Inference configs are strictly merged with the checkpoint signature saved during training. If a user-provided value conflicts with the signature, inference fails early. The merged and validated configs are saved in the new inference run directory, and predictions are written under `predictions/`.

### Analyze

```bash
xanesnet analyze \
    -i configs/analyze_example.yaml \
    -p runs/<infer_run>/predictions \
    -n mlp_analysis \
    -y
```

The `-p` option can be supplied multiple times when comparing or aggregating predictions from several inference runs.

### Python Entry Point

For scripts and notebooks, you can call the same dispatcher that backs the CLI:

```python
from xanesnet.cli import main

main(["train", "-i", "configs/in_mlp.yaml", "-n", "mlp_test", "-y"])
```

For most workflows, the installed `xanesnet` command is the recommended interface because it keeps command logging and run directories consistent.

## Configuration

Configuration validation and defaults are implemented in:

- [xanesnet/serialization/config.py](xanesnet/serialization/config.py)
- [xanesnet/serialization/defaults.py](xanesnet/serialization/defaults.py)

At a high level, a config contains:

- `seed` and `device`
- `datasource`: input structure and spectrum source
- `dataset`: preprocessing, storage, split, and descriptor or graph settings; use `mode: forward` for supported descriptor workflows
- `model`: model family and hyperparameters
- exactly one runner section: `trainer`, `inferencer`, or analysis settings depending on workflow
- `strategy`: single model or ensemble training/inference behavior

The config UI schemas in [tools/config-ui/src/schemas/](tools/config-ui/src/schemas/) mirror this structure for interactive editing.

## People And Attribution

**Main code author and current maintainer:** [Hendrik Junkawitsch](https://www.helmholtz-berlin.de/pubbin/vkart.pl?v=yyqxqn&sprache=de), Helmholtz-Zentrum Berlin.

**Scientific lead:** [Prof. Thomas Penfold](https://ncl.ac.uk/nes/people/profile/tompenfold.html), Newcastle University.

**Original project team:** [Prof. Thomas Penfold](https://ncl.ac.uk/nes/people/profile/tompenfold.html), Newcastle University, and [Dr. Conor Rankine](https://pure.york.ac.uk/portal/en/persons/conor-rankine), York University.

**Research software engineering contributors:** Dr. Bowen Li and Dr. Lorenzo Rossi, Newcastle University RSE team.

## License

This project is licensed under the GPL-3.0 License. See [LICENSE](LICENSE) for details.

## Publications

### The Program

*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

*[Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network](https://doi.org/10.1063/5.0087255)* - C. D. Rankine and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.

### Extension To X-ray Emission

*[A deep neural network for valence-to-core X-ray emission spectroscopy](https://doi.org/10.1080/00268976.2022.2123406)* - T. J. Penfold and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.

### Applications

*[On the Analysis of X-ray Absorption Spectra for Polyoxometallates](https://doi.org/10.1016/j.cplett.2021.138893)* - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

*[Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

### Miscellaneous

*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.
