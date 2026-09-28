Introduction
============

XANESNET overview
-----------------

We present XANESNET, a PyTorch-based, open-source software framework for machine learning in spectroscopy.
The framework integrates training, inference, and automated analysis within a plugin-based architecture.
Its modular design enables users to compare, combine, and extend different machine-learning workflow and analysis components
without modifying the core codebase, providing a flexible and reusable framework rather than a single-purpose implementation.
XANESNET supports both forward prediction from structure to spectrum and inverse inference from spectra to structures or properties.
A common data pipeline uniformly handles both molecular and periodic systems. Moreover, the framework remains entirely agnostic to the spectroscopic technique.
We demonstrate its use for learning structure-spectrum relationships in X-ray absorption spectroscopy.
By prioritizing extensibility and reproducibility, XANESNET facilitates systematic comparison and evaluation of machine-learning approaches,
lowering the barrier to research and accelerating the development and reuse of methods in spectroscopy.

XANESNET software paper
-----------------------

Check the accompanying software paper [Junkawitsch2026]_ for design and implementation details of XANESNET.
The paper also provides an example workflow for training, inference, and analysis.

.. [Junkawitsch2026] H. Junkawitsch, B. Li, T. Pope, A. Bande, and T. Penfold, *XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy*, 2026.

XANESNET features
-----------------

* GPLv3-licensed open-source software.
* End-to-end workflows for data preparation, training, prediction, and analysis.
* YAML-driven runs for reproducible workflows.
* Modular components that can be added or replaced without changing the core workflow.
* Support for molecular and periodic structures, with feature-vector and graph representations.
* Neural-network models ranging from feed-forward to geometry-aware graph models.
* Forward and inverse workflows for spectra, structures, and properties.
* Single-model and ensemble training for comparison and uncertainty estimates.
* Automated analysis with metrics, comparisons, reports, and figures.
* Browser-based :doc:`configuration tool <config-editor>`.

XANESNET development team
-------------------------

XANESNET is developed by `Hendrik Junkawitsch <https://github.com/HendrikJunkawitsch>`_,
the `Penfold Group <http://penfoldgroup.co.uk>`_, and the `Research Software Engineering (RSE) team <https://rse.ncldata.dev/>`_ at `Newcastle University <https://ncl.ac.uk>`_.

| Project team:
| `Prof. Thomas Penfold <https://www.ncl.ac.uk/nes/people/profile/tompenfold.html>`_ (tom.penfold@newcastle.ac.uk)
| `Hendrik Junkawitsch <https://github.com/HendrikJunkawitsch>`_ (hendrik.junkawitsch@newcastle.ac.uk)
| `Dr. Thomas Pope <https://www.ncl.ac.uk/nes/people/profile/thomaspope2.html>`_ (thomas.pope2@newcastle.ac.uk)
| `Dr. Conor Rankine <https://www.york.ac.uk/chemistry/people/conor-rankine/>`_ (conor.rankine@york.ac.uk)

| RSE team:
| `Dr. Bowen Li <https://rse.ncldata.dev/team/bowen-li>`_ (bowen.li2@newcastle.ac.uk)

| Former RSEs:
| Dr. Nik Khadijah Nik Aznan
| Dr. Kathryn Garside
| Alex Surtee
| Dr. Lorenzo Rossi

