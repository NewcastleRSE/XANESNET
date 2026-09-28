Descriptors
===========

Descriptors convert atomic environments into fixed-size feature vectors.
They implement :class:`~xanesnet.descriptors.base.Descriptor`, are configured under ``dataset.descriptors``, and may be used by various :doc:`datasets`.

Available implementations
--------------------------

The descriptor registry provides six types:

* ``wacsf`` (:class:`~xanesnet.descriptors.wacsf.WACSF`) computes weighted atom-centred symmetry functions.

* ``rdc`` (:class:`~xanesnet.descriptors.rdc.RDC`) computes a radial distribution curve.
* ``mace`` (:class:`~xanesnet.descriptors.mace.MACE`) extracts per-atom features from a MACE model.
* ``direct`` (:class:`~xanesnet.descriptors.direct.DIRECT`) reads precomputed descriptor vectors from ``.txt`` files in ``source_dir``.
* ``soap`` (:class:`~xanesnet.descriptors.soap.SOAP`) computes SOAP features with dscribe.
* ``pdos`` (:class:`~xanesnet.descriptors.pdos.PDOS`) computes a projected density of states using the ``xtb`` or ``pyscf`` backend.

Examples
--------

.. code-block:: yaml

   dataset:
     dataset_type: descriptor
     descriptors:
       - descriptor_type: wacsf
         r_min: 1.0
         r_max: 6.0
         n_g2: 16
         n_g4: 32

API reference
-------------

See the API reference for :mod:`xanesnet.descriptors` for the complete constructor interfaces.
