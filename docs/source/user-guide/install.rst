Installing XANESNET
===================

-----------
Download
-----------

XANESNET can be cloned from GitHub or GitLab:

.. code-block:: bash

   git clone https://github.com/NewcastleRSE/XANESNET.git

or

.. code-block:: bash

   git clone https://gitlab.com/team-xnet/XANESNET.git

The repository contains source code, example configs under ``configs/``, and toy
data under ``data/``.

Training datasets for first-row transition-metal X-ray absorption and emission
can be obtained separately:

.. code-block:: bash

   git clone https://gitlab.com/team-xnet/training-sets.git

------------
Requirements
------------

* Linux (tested on Ubuntu)
* Python 3.12 and above
* A C/C++ build toolchain if pip needs to compile native PyTorch Geometric extensions
* Optional: NVIDIA GPU and CUDA-compatible PyTorch wheels for device: cuda


------------
Installation
------------

Install the package from the XANESNET repository root:

.. code-block:: bash

   python -m pip install -e .
   source .venv/bin/activate 
   python -m pip install --upgrade pip setuptools wheel
   python -m pip install -e .

For a non-editable install:

.. code-block:: bash

   python -m pip install .

Verify the installation:

.. code-block:: bash

   xanesnet --help

For the browser-based config editor, see :doc:`user-interface`.

-----------------------------
PyTorch and PyTorch Geometric
-----------------------------

XANESNET depends on PyTorch and PyTorch Geometric. On Linux with CPU-only
PyTorch, a normal ``pip install`` is often enough.

For CUDA, or if packages such as ``torch-scatter``, ``torch-sparse``, or
``torch-cluster`` fail to install, install matching wheels for your Python,
PyTorch, and CUDA versions first, then install XANESNET.

Example for PyTorch 2.5 with CUDA 12.4:

.. code-block:: bash

   python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
   python -m pip install torch-scatter torch-sparse torch-cluster \
       -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
   python -m pip install -e .

Change the URLs to match your platform. See the
`PyTorch <https://pytorch.org/get-started/locally/>`_ and
`PyTorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
install guides for the latest compatibility matrix.

