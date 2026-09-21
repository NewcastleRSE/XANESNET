Datasets
========

The ``dataset_type`` field selects a dataset class from
``DatasetRegistry``. Each type defines how structures and spectra are
preprocessed, which model families are compatible, and whether periodic
boundary conditions (``_mp`` variants) are supported.

Descriptor-based datasets
-------------------------

These datasets featurise structures with descriptors from the
``descriptors`` list in the config (see :doc:`descriptors`).

descriptor
~~~~~~~~~~

Tabular dataset for MLP and related models. Stores fixed-length feature
vectors and target spectra. Compatible with forward prediction
(structure → spectrum).

Example configs: ``configs/mlp.yaml``, ``configs/mh_mlp.yaml``.

* ``dataset_type: descriptor``
* ``descriptors`` — list of descriptor blocks
* ``split_ratios``, ``preload``, ``root``, ``skip_prepare``

descriptor_inverse
~~~~~~~~~~~~~~~~~~

Reverse mapping (spectrum → structure/descriptor space). Compatible with
inverse MLP workflows.

Example config: ``configs/mlp_inverse.yaml``.

descriptor_mp / descriptor_inverse_mp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Periodic (materials) variants of the descriptor datasets.

multihead / multihead_mp
~~~~~~~~~~~~~~~~~~~~~~~~

Multi-target datasets for models that predict several spectra or properties
from one structure. Use with ``mh_mlp`` or ``mh_cnn`` and a multi-head
datasource (``multipmgjson`` or ``multixyzspec``).

Example configs: ``configs/mh_mlp.yaml``, ``configs/mh_cnn.yaml``.

Graph and geometry datasets
---------------------------

geometrygraph / geometrygraph_mp
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph representation with wACSF-style edge features and graph-level
descriptor attributes. Used with SchNet, DimeNet, and related GNN models.

Example configs: ``configs/schnet.yaml``, ``configs/dimenet.yaml``.

gemnet / gemnet_mp
~~~~~~~~~~~~~~~~~~

Dataset pipeline for GemNet models with appropriate graph construction.

Example config: ``configs/gemnet.yaml``.

gemnet_oc / gemnet_oc_mp
~~~~~~~~~~~~~~~~~~~~~~~~

Dataset pipeline for GemNet-OC (open catalyst) models.

Example config: ``configs/gemnet_oc.yaml``.

envembed / envembed_mp
~~~~~~~~~~~~~~~~~~~~~~

Environment embedding dataset for the EnvEmbed model family.

Example config: ``configs/envembed.yaml``.

e3ee / e3ee_mp
~~~~~~~~~~~~~~

Equivariant E(3) edge embedding dataset.

Example config: ``configs/e3ee.yaml``.

e3ee_full / e3ee_full_mp
~~~~~~~~~~~~~~~~~~~~~~~~

Full E3EE variant with extended graph features.

Example config: ``configs/e3ee_full.yaml``.

Inference overlay
-----------------

The ``infer_overlay`` dataset type is used internally during inference to
align prediction inputs with a trained checkpoint signature. Users
typically do not configure this directly.

Choosing a dataset
------------------

+---------------------------+----------------------------------+
| Goal                      | Typical ``dataset_type``         |
+===========================+==================================+
| MLP on descriptors        | ``descriptor``                   |
| Inverse MLP               | ``descriptor_inverse``         |
| Multi-head outputs        | ``multihead``                    |
| SchNet / DimeNet          | ``geometrygraph``                |
| GemNet                    | ``gemnet``                       |
| GemNet-OC                 | ``gemnet_oc``                    |
| EnvEmbed                  | ``envembed``                     |
| E3EE                      | ``e3ee`` or ``e3ee_full``        |
| Periodic structures       | ``*_mp`` suffix variants         |
+---------------------------+----------------------------------+

For field-level defaults and validation rules, see the JSON Schemas under
``xanesnet/schemas/datasets/`` and the API reference for
:mod:`xanesnet.datasets`.
