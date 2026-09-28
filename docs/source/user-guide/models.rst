Models
======

The ``model_type`` field selects a registered :class:`~xanesnet.models.base.Model` implementation.
The selected model must be compatible with the dataset through a registered batch processor.
The adapter between a compatible dataset and model is described in :doc:`batch-processors`.

Available implementations
--------------------------

* ``mlp`` (:class:`~xanesnet.models.mlp.mlp.MLP`) is a feed-forward network for fixed-size descriptor vectors.
* ``mh_mlp`` (:class:`~xanesnet.models.multihead.mh_mlp.MultiHeadMLP`) is a multi-head feed-forward network for several target outputs.
* ``mh_cnn`` (:class:`~xanesnet.models.multihead.mh_cnn.MultiHeadCNN`) is a multi-head one-dimensional convolutional network for descriptor inputs.
* ``envembed`` (:class:`~xanesnet.models.envembed.envembed.EnvEmbed`) encodes an absorber environment and predicts spectral coefficients.
* ``schnet`` (:class:`~xanesnet.models.schnet.schnet.SchNet`) uses continuous-filter graph convolutions.
* ``dimenet`` (:class:`~xanesnet.models.dimenet.dimenet.DimeNet`) and ``dimenet++`` (:class:`~xanesnet.models.dimenet.dimenet_pp.DimeNetPlusPlus`) use directional message passing.
* ``gemnet`` (:class:`~xanesnet.models.gemnet.gemnet.GemNet`) uses higher-order geometric interactions.
* ``gemnet_oc`` (:class:`~xanesnet.models.gemnet_oc.gemnet_oc.GemNetOC`) uses the GemNet-OC graph representation and settings.
* ``e3ee`` (:class:`~xanesnet.models.e3ee.e3ee.E3EE`) is an absorber-centred E(3)-equivariant model.
* ``e3ee_full`` (:class:`~xanesnet.models.e3ee_full.e3ee_full.E3EEFull`) extends ``e3ee`` to use full-structure graphs and predicts for target sites in the structure.

Common configuration
--------------------

Training schemas allow selected model dimensions to use ``auto``; these values are resolved after dataset preparation.

Models may use the ``activation`` field to select a model-level activation from :doc:`activations`.

Examples
--------

MLP model:

.. code-block:: yaml

   model:
     model_type: mlp
     in_size: auto
     out_size: auto
     hidden_size: 256
     dropout: 0.1
     num_hidden_layers: 3
     shrink_rate: 0.5
     activation: prelu

SchNet model:

.. code-block:: yaml

   model:
     model_type: schnet
     hidden_channels: 128
     reduce_channels_1: 64
     reduce_channels_2: auto
     num_filters: 128
     num_interactions: 6
     num_gaussians: 50
     cutoff: 5.0

See the complete examples in ``configs/`` for the other model families.

API reference
-------------

See also :mod:`xanesnet.models` and the schemas under ``xanesnet/schemas/models/`` for complete constructor fields and defaults.
See also :mod:`xanesnet.datasets` for a dataset-model compatibility table.