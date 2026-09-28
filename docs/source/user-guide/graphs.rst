Graphs
======

Graph builders convert molecular or periodic structures into graph data for geometric neural networks.
They implement :class:`~xanesnet.graphs.base.GraphBuilder` and are configured under dataset graph fields such as ``graph_builder``.
See :doc:`models` for the models that consume graph datasets.

Available implementations
--------------------------

* ``radius`` (:class:`~xanesnet.graphs.radius.RadiusGraphBuilder`) connects atoms within a distance ``cutoff``.
* ``cov_radius`` (:class:`~xanesnet.graphs.radius.CovRadiusGraphBuilder`) uses a distance cutoff and a scaled sum of covalent radii.
* ``voronoi`` (:class:`~xanesnet.graphs.voronoi.VoronoiGraphBuilder`) connects atoms whose Voronoi cells share a facet.

Common configuration
--------------------

All graph builders accept:

* ``graph_builder_type``
* ``cutoff``: maximum edge distance
* ``max_num_neighbors``: maximum number of neighbors per source atom

Even though the graph builders implement different strategies, the ``cutoff`` and ``max_num_neighbors`` fields limit the number of edges in all cases.

Example
-------

.. code-block:: yaml

   dataset:
     dataset_type: geometrygraph
     root: ./data/processed/toy_data_schnet/
     graph_builder:
       graph_builder_type: cov_radius
       cutoff: 5.0
       cov_radii_scale: 2.5
       max_num_neighbors: 50

API reference
-------------

See also :mod:`xanesnet.graphs` for the base interface and registry.
