<table align="center">
<tr><td align="center" width="10000">

<img src = "docs/source/images/xanesnet_graphic.png" width = "380">

# <strong> X A N E S N E T </strong>

<p>
    <a href="http://penfoldgroup.co.uk">Penfold Group </a> @ <a href="https://ncl.ac.uk">Newcastle University </a>
</p>

<p>
    <a href="https://xanesnet.readthedocs.io">User Manual</a> • <a href="#setup">Setup</a> • <a href="#training-quick-guide">Quickstart</a> • <a href="#publications">Publications</a>
</p>

</td></tr></table>

#

We think that the theoretical simulation of X-ray spectroscopy (XS) should be fast, affordable, and accessible to all researchers. 

The popularity of XS is on a steep upward trajectory globally, driven by advances at, and widening access to, high-brilliance light sources such as synchrotrons and X-ray free-electron lasers (XFELs). However, the high resolution of modern X-ray spectra, coupled with ever-increasing data acquisition rates, brings into focus the challenge of accurately and cost-effectively analyzing these data. Decoding the dense information content of modern X-ray spectra demands detailed theoretical calculations that are capable of capturing satisfactorily the complexity of the underlying physics but that are - at the same time - fast, affordable, and accessible enough to appeal to researchers. 

This is a tall order - but we're using deep neural networks to make this a reality. 

Our XANESNET code address two fundamental challenges: the so-called forward (property/structure-to-spectrum) and reverse (spectrum-to-property/structure) mapping problems. The forward mapping appraoch is similar to the appraoch used by computational researchers in the sense that an input structure is used to generate a spectral observable. In this area the objective of XANESNET is to supplement and support analysis provided by first principles quantum mechnanical simulations. The reverse mapping problem is perhaps the more natural of the two, as it has a clear connection to the problem that X-ray spectroscopists face day-to-day in their work: how can a measurement/observable be interpreted? Here we are seeking to provide methodologies in allow the direct extraction of properties from a recorded spectrum. 

XANESNET is under continuous development.

The original Keras implementation is available at https://gitlab.com/team-xnet/xanesnet_keras.

## Features

- GPLv3 licensed open-source distribution
- TODO

## Setup

### Prerequisites

- Linux (tested on Ubuntu)
- Python **3.12** (the current known-working environment uses Python **3.12.9**)
- A C/C++ build toolchain if pip needs to compile native PyTorch Geometric extensions
- Optional: NVIDIA GPU + CUDA matching your PyTorch build if you want `device: cuda`

### Installation

Install from a local checkout with the project metadata in [pyproject.toml](pyproject.toml):

```bash
git clone https://github.com/NewcastleRSE/xray-spectroscopy-ml.git
cd xray-spectroscopy-ml
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

For a non-editable install, replace the last command with:

```bash
python -m pip install .
```

This installs the `xanesnet` Python package and the `xanesnet` command-line entry point:

```bash
xanesnet --help
```

#### PyTorch and PyTorch Geometric wheels

The package depends on PyTorch and PyTorch Geometric native extensions. For CPU-only Linux installs, pip may be enough. For CUDA installs, or if `torch-scatter`, `torch-sparse`, or `torch-cluster` cannot find a wheel automatically, install the PyTorch/PyG wheels that match your Python, PyTorch, and CUDA versions first, then install XANESNET.

Example for a PyTorch 2.5 / CUDA 12.4 environment:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
python -m pip install -e .
```

Adjust the PyTorch and PyG wheel URLs for your target platform. See the official PyTorch and PyTorch Geometric installation guides for the current matrix.

#### Optional extras

```bash
python -m pip install -e ".[dev]"   # test/build tooling
python -m pip install -e ".[docs]"  # documentation tooling
```

[frozen.txt](frozen.txt) is kept as a pinned snapshot of one known-working environment. Use it only when you need to reproduce that exact environment; normal installs should use [pyproject.toml](pyproject.toml).

### Training (quick guide)

Training is driven by a YAML config in the [configs/](configs/) folder. The most complete example is [configs/in_mlp.yaml](configs/in_mlp.yaml).

Run training

```bash
xanesnet train -i configs/in_mlp.yaml -y
```

You get the outputs of your training run under `./runs/...`.


### Inference (minimal notes; WIP)

--> under development

Current invocation:

```bash
xanesnet infer \
    -i configs/in_mlp_infer.yaml \
    -m runs/<run_dir>/models/final.pth \
    -y
```

How it works:

- The inference YAML is **strictly merged** with the checkpoint signature (saved during training). If you specify keys that conflict with the signature, inference will error.
- The merged config is written to `merged_infer_config.yaml` in the new `runs/` folder.

The reference inference config is [configs/in_mlp_infer.yaml](configs/in_mlp_infer.yaml).

## Configuration

Config validation and defaults are defined in the serialization module:

- [xanesnet/serialization/config.py](xanesnet/serialization/config.py)
- [xanesnet/serialization/defaults.py](xanesnet/serialization/defaults.py)

At a high level, a config contains:

- `seed` (optional) and `device` (optional; defaults to `cpu`)
- `datasource`: must include `datasource_type` and its required keys
- `dataset`: must include `dataset_type` and its required keys
- `model`: must include `model_type` and its required keys
- Exactly one of: `trainer` or `inferencer`
- `strategy`: must include `strategy_type`



## Contact

### Project Team

<a href="https://ncl.ac.uk/nes/people/profile/tompenfold.html">Prof. Thomas Penfold </a>, Newcastle University, (tom.penfold@newcastle.ac.uk)\
<a href="https://pure.york.ac.uk/portal/en/persons/conor-rankine">Dr. Conor Rankine </a>, York University (conor.rankine@york.ac.uk)

### RSE Contact
<a href="https://rse.ncldata.dev/team/bowen-li">Dr. Bowen Li </a>, Newcastle University (bowen.li2@newcastle.ac.uk)\
<a href="https://rse.ncldata.dev/team/alex-surtees">Dr. Lorenzo Rossi </a>,  Newcastle University (lorenzo.rossi@newcastle.ac.uk)


## License

This project is licensed under the GPL-3.0 License - see [LICENSE](LICENSE) for details.

## Publications

### The Program:
*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

*[Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network](https://doi.org/10.1063/5.0087255)* - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.
 
#### Extension to X-ray Emission:
*[A deep neural network for valence-to-core X-ray emission spectroscopy](https://doi.org/10.1080/00268976.2022.2123406)* - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.

#### The Applications:
*[On the Analysis of X-ray Absorption Spectra for Polyoxometallates](https://doi.org/10.1016/j.cplett.2021.138893)* - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

*[Enhancing the Anaysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

#### Miscellaneous:
*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.
