Datasets
========

The ``dataset_type`` field selects how raw structures and spectra are prepared and represented for a model.
Dataset implementations write processed samples to ``root`` and provide the batches consumed by compatible models.
Generally, different models may require different dataset types.
See :doc:`models` for the model implementations and the compatibility matrix below for the registered dataset-model combinations.
The adapter between a compatible dataset and model is described in :doc:`batch-processors`.

Available implementations
--------------------------

* ``descriptor`` and ``descriptor_inverse`` (:class:`~xanesnet.datasets.torch.descriptor.DescriptorDataset`) convert structures to fixed-size descriptor features.
* ``descriptor_multihead`` and ``descriptor_multihead_inverse`` (:class:`~xanesnet.datasets.torch.descriptor_multihead.DescriptorMultiheadDataset`) prepare several target heads for multi-head models.
* ``envembed`` (:class:`~xanesnet.datasets.torch.envembed.EnvEmbedDataset`) prepares the absorber-environment representation used by the ``envembed`` model.
* ``geometrygraph`` (:class:`~xanesnet.datasets.torchgeometric.geometrygraph.GeometryGraphDataset`) builds classic geometric graphs for graph models.
* ``gemnet`` (:class:`~xanesnet.datasets.torchgeometric.gemnet.GemNetDataset`) prepares graph and higher-order interaction data for the ``gemnet`` model.
* ``gemnet_oc`` (:class:`~xanesnet.datasets.torchgeometric.gemnet.GemNetDataset`) prepares graph and higher-order interaction data for the ``gemnet_oc`` model.
* ``e3ee`` (:class:`~xanesnet.datasets.torchgeometric.e3ee.E3EEDataset`) prepares absorber-centred graphs for the ``e3ee`` model.
* ``e3ee_full`` (:class:`~xanesnet.datasets.torchgeometric.e3ee_full.E3EEFullDataset`) prepares full-structure graphs for ``e3ee_full`` predictions.

Multiprocessing
---------------
All dataset types have a ``_mp`` variant that uses multiprocessing during dataset preparation.
The ``_mp`` variants add the ``num_workers`` field to control the number of worker processes.

Common configuration
--------------------

* ``root`` is required and stores the processed samples.
* ``preload`` loads processed samples into memory instead of reading them on demand.
* ``skip_prepare`` reuses existing processed samples.
* ``split_ratios`` creates data splits and must sum to ``1.0`` when ``split_indexfile`` is ``null``.
* ``split_indexfile`` provides fixed split indices instead of ratio-based splitting.
* Descriptor datasets require a non-empty ``descriptors`` list; see :doc:`descriptors` for the available descriptor types.
* Graph datasets require ``graph_builder``; see :doc:`graphs` for the available builders.

Compatibility matrix
--------------------

The runtime selects a batch processor using both ``dataset_type`` and ``model_type``.
The following combinations are registered:

.. list-table:: Dataset and model compatibility
   :header-rows: 1
   :widths: 32 30 18

   * - Dataset type
     - Compatible model type
     - Direction
   * - ``descriptor`` / ``descriptor_mp``
     - ``mlp``
     - forward
   * - ``descriptor_inverse`` / ``descriptor_inverse_mp``
     - ``mlp``
     - inverse
   * - ``descriptor_multihead`` / ``descriptor_multihead_mp``
     - ``mh_mlp``, ``mh_cnn``
     - forward
   * - ``descriptor_multihead_inverse`` / ``descriptor_multihead_inverse_mp``
     - ``mh_mlp``, ``mh_cnn``
     - inverse
   * - ``envembed`` / ``envembed_mp``
     - ``envembed``
     - forward
   * - ``geometrygraph`` / ``geometrygraph_mp``
     - ``schnet``, ``dimenet``, ``dimenet++``
     - forward
   * - ``gemnet`` / ``gemnet_mp``
     - ``gemnet``
     - forward
   * - ``gemnet_oc`` / ``gemnet_oc_mp``
     - ``gemnet_oc``
     - forward
   * - ``e3ee`` / ``e3ee_mp``
     - ``e3ee``
     - forward
   * - ``e3ee_full`` / ``e3ee_full_mp``
     - ``e3ee_full``
     - forward

Example
-------

.. code-block:: yaml

   dataset:
     dataset_type: geometrygraph
     root: ./data/processed/toy_data_schnet/
     preload: true
     skip_prepare: false
     split_ratios: [0.8, 0.2]
     graph_builder:
       graph_builder_type: cov_radius
       cutoff: 5.0
       cov_radii_scale: 2.5
       max_num_neighbors: 50

API reference
-------------
See also :mod:`xanesnet.datasets` and the schemas under ``xanesnet/schemas/datasets/`` for complete constructor fields and defaults.
