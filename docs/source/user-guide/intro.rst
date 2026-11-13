Introduction
============

XANESNET overview
-----------------

We present XANESNET, an open-source, PyTorch-based software package for machine learning in spectroscopy. Specifically, the framework integrates training, inference, and automated analysis for structure--spectrum learning within a plugin-based architecture. Its modular design enables users to compare, combine, and extend workflow and analysis components without modifying the core codebase, providing a flexible and reusable alternative to single-purpose implementations. XANESNET supports both forward prediction from atomic structures to spectra and inverse inference from spectra to structures, properties, or descriptors. Through PyTorch Lightning, we enable multi-device training using Distributed Data Parallel (DDP) across multiple GPUs. A shared data pipeline uniformly handles both molecular and periodic systems. Although the framework is agnostic to spectroscopic techniques, current modules primarily implement atom-centred spectra. We demonstrate its use for learning structure--spectrum relationships in X-ray absorption spectroscopy. To our knowledge, \textsc{XANESNET} is the first general-purpose machine-learning framework for spectroscopy with a central focus on extensibility and reproducibility. This facilitates systematic comparisons and evaluations of machine-learning approaches, lowering the barrier for researchers to develop and benchmark novel methods.

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
* Forward prediction and inverse inference.
* Single-model and ensemble training for comparison and uncertainty estimates.
* Automated analysis with metrics, comparisons, reports, and figures.
* Browser-based :doc:`configuration tool <config-editor>`.

XANESNET development team
-------------------------

XANESNET is developed by `Hendrik Junkawitsch <https://github.com/Junkiii>`_,
the `Penfold Group <http://penfoldgroup.co.uk>`_, and the `Research Software Engineering (RSE) team <https://rse.ncldata.dev/>`_ at `Newcastle University <https://ncl.ac.uk>`_.

| Project team:
| `Prof. Thomas Penfold <https://www.ncl.ac.uk/nes/people/profile/tompenfold.html>`_ (tom.penfold@newcastle.ac.uk)
| `Hendrik Junkawitsch <https://github.com/Junkiii>`_ (hendrik.junkawitsch@newcastle.ac.uk)
| `Dr. Thomas Pope <https://www.ncl.ac.uk/nes/people/profile/thomaspope2.html>`_ (thomas.pope2@newcastle.ac.uk)
| `Dr. Conor Rankine <https://www.york.ac.uk/chemistry/people/conor-rankine/>`_ (conor.rankine@york.ac.uk)

| RSE team:
| `Dr. Bowen Li <https://rse.ncldata.dev/team/bowen-li>`_ (bowen.li2@newcastle.ac.uk)

| Former RSEs:
| Dr. Nik Khadijah Nik Aznan
| Dr. Kathryn Garside
| Alex Surtee
| Dr. Lorenzo Rossi

