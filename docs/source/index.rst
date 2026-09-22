XANESNET Documentation
======================

Welcome to the XANESNET documentation.

We present XANESNET, a PyTorch-based, open-source software framework for machine learning in spectroscopy. The framework integrates training, inference, and automated analysis within a plugin-based architecture. Its modular design allows users to compare, combine, and extend different methods without modifying the core codebase, providing a flexible and reusable framework rather than a single-purpose implementation. XANESNET supports forward prediction from structure to spectrum and inverse inference from spectra to structures or properties. A unified data pipeline uniformly handles molecular and periodic systems, while the framework remains agnostic to the spectroscopic technique. We demonstrate its use for learning structure-spectrum relationships in X-ray absorption spectroscopy. By prioritizing extensibility and reproducibility, XANESNET aims to accelerate and make more accessible machine-learning research in spectroscopy.


The :doc:`overview` summarizes the supported workflows and points to the
project README, configuration examples, and interactive config UI. The full
API reference is generated from in-source Google-style docstrings.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   overview

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   modules

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
