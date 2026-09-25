Introduction
============

---------------------
XANESNET overview
---------------------

XANESNET is an open-source platform for the rapid and automated analysis and
prediction of X-ray spectroscopy data. The platform provides machine learning
solutions for forward mapping problems by using an input structure to generate a spectral
observable. This links material
properties or structures with their corresponding X-ray Absorption Near Edge
Structure (XANES) spectra. 

XANESNET employs flexible deep learning models and a registry-based plugin
architecture. Users build workflows from simple YAML configuration files to set up
data, models, training strategies, and execution.

---------------------
XANESNET features
---------------------

* Open-Source: Released under the GPLv3 license.
* Easy Configuration: Uses validated YAML files with default settings.
* Modular Design: Pluggable data sources, datasets, descriptors, models, strategies, runners.
* Structural Descriptors: wACSF, RDC, MACE, SOAP, pDOS, direct.
* Model Choices: MLP, multi-head MLP/CNN, SchNet, DimeNet, GemNet, GemNet-OC, E3EE, EnvEmbed
* Training Options: Single model, k-fold, bootstrap, and deep ensembles.
* Command-Line Tools: Workflows for training, inference, and analysis.
* Monitoring: TensorBoard logging and checkpointing.
* Analysis Tools: Error-metric collectors, statistical aggregators, plots.
* User-Friendly Tools: Interactive editor.

--------------------------
XANESNET development team
--------------------------

XANESNET is developed by the
`Hendrik Junkawitsch <https://github.com/HendrikJunkawitsch>`_, the
`Penfold Group <http://penfoldgroup.co.uk>`_ and the
`Research Software Engineering (RSE) team <https://rse.ncldata.dev/>`_ at
`Newcastle University <https://ncl.ac.uk>`_.

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

