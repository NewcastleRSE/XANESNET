Datasources
===========

A datasource reads raw structures and spectra and exposes them to dataset preparation.
XANESNET accepts both molecular and periodic structures through the common :class:`~xanesnet.datasources.base.DataSource` interface.
The loaded data are passed to :doc:`datasets` for preparation.

Available implementations
--------------------------

The ``datasource_type`` field selects one of the registered datasource types.

* ``pmgjson`` (:class:`~xanesnet.datasources.pmgjson.PMGJSONSource`) reads pymatgen JSON files from one directory.
* ``multipmgjson`` (:class:`~xanesnet.datasources.multipmgjson.MultiPMGJSONSource`) reads pymatgen JSON files from multiple subdirectories.
* ``xyzspec`` (:class:`~xanesnet.datasources.xyzspec.XYZSpecSource`) reads paired ``.xyz`` structure files and spectrum files.
* ``multixyzspec`` (:class:`~xanesnet.datasources.multixyzspec.MultiXYZSpecSource`) reads paired files from multiple subdirectories.

Important notes
--------------------

For the PMGJSON sources, ``spectrum_key`` is the pymatgen site-property key containing the spectrum.
Its default is ``spectrum``; use a different value when the source data use a key such as ``XANES``.

In general, one can use whichever datasource is convenient, as they provide a common interface.
However, when training for example the multi-head models, one must ensure that a "multi" datasource is used.


Examples
--------

PMGJSON:

.. code-block:: yaml

   datasource:
     datasource_type: pmgjson
     json_path: ./data/toy_data/
     spectrum_key: "XANES"

XYZSpec:

.. code-block:: yaml

   datasource:
     datasource_type: xyzspec
     xyz_path: ./data/toy_data/
     spectra_path: ./data/toy_data/

API reference
-------------
See also :mod:`xanesnet.datasources` and the schemas under ``xanesnet/schemas/datasources/`` for complete constructor fields and defaults.
