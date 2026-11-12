Installing XANESNET
===================

Download
--------

XANESNET can be cloned from GitHub or GitLab:

.. code-block:: bash

   git clone https://github.com/NewcastleRSE/XANESNET.git

or

.. code-block:: bash

   git clone https://gitlab.com/team-xnet/xanesnet.git

The repository contains source code, example configs under ``configs/``, and toy data under ``data/``.

Training datasets for first-row transition-metal X-ray absorption and emission can be obtained separately:

.. code-block:: bash

   git clone https://gitlab.com/team-xnet/training-sets.git

Requirements
------------

* Linux (tested on Ubuntu)
* Python 3.12 and above
* Matching PyTorch and PyTorch Geometric wheels for the chosen Python, PyTorch, and CPU/CUDA versions.
* A C/C++ build toolchain only if pip must compile native PyTorch Geometric extensions from source.
* Optional: NVIDIA GPU and CUDA-compatible PyTorch wheels for ``device: cuda``.

The Lightning trainer is an optional extra. Install it with ``python -m pip install -e ".[lightning]"`` when using ``trainer_type: lightning``.
The tested Lightning environment uses Python 3.12.9 and PyTorch 2.5.1 with CUDA 12.4.
CPU-only training does not require CUDA. MPS devices are currently unsupported and untested.


Installation
------------

The following Python 3.12 CPU-only sequence was tested from the XANESNET repository root. Install PyTorch and its matching PyTorch Geometric extension wheels before the package so pip does not attempt an incompatible source build:

.. code-block:: bash

   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cpu
   python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
   python -m pip install -e .

For a non-editable install, replace ``-e .`` with ``.`` after installing the matching PyTorch/PyG wheels:

.. code-block:: bash

   python -m pip install .

To use the optional Lightning trainer, include its extra:
.. code-block:: bash

   python -m pip install -e ".[lightning]"

To build the API documentation, install the documentation and Lightning extras:

.. code-block:: bash

   python -m pip install -e ".[docs,lightning]"

Verify the installation:

.. code-block:: bash

   xanesnet --help

Continue with :doc:`running` for command-line workflows and :doc:`input` for configuration files.

For the work-in-progress browser-based config editor, see :doc:`config-editor`.

PyTorch and PyTorch Geometric
-----------------------------

XANESNET depends on PyTorch and PyTorch Geometric native extensions. The PyTorch version and CPU/CUDA build must match the extension wheels. The CPU-only sequence above uses PyTorch 2.5.1 and its matching ``pt25cpu`` wheels.

For CUDA, install matching wheels first, then install XANESNET. Example for PyTorch 2.5.1 with CUDA 12.4:


.. code-block:: bash

   python -m pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu124
   python -m pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
   python -m pip install -e .

For other combinations, choose wheel URLs from the PyTorch and PyTorch Geometric installation guides and keep the PyTorch and extension versions aligned.