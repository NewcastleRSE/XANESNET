Input files
===========

XANESNET uses YAML configuration files for three workflow modes: training, inference, and analysis.
Each mode has its own top-level sections.

Every configuration is validated against the schemas in ``xanesnet/schemas/``.
The schemas define allowed fields, types, and default values.

The required top-level sections are:
* training: ``datasource``, ``dataset``, ``model``, ``trainer``, ``strategy``, and ``encodings``
* inference: ``datasource``, ``dataset``, and ``inferencer``
* analysis: ``selectors``, ``collectors``, ``aggregators``, ``reporters``, and ``plotters``

Many nested fields have defaults, so they do not need to be written when the default is suitable.
Example files are available in ``configs/``.
The command-line workflows that consume these files are described in :doc:`running`,
and schema validation is implemented by :func:`~xanesnet.serialization.schema_validation.validate_config_schema`.

Configuration sections
----------------------

The detailed configuration pages are organised by top-level section:

.. list-table::
   :header-rows: 1
   :widths: 20 45 25

   * - Section
     - Purpose
     - Details
   * - ``datasource``
     - Locate raw structures and spectra.
     - :doc:`datasources`
   * - ``dataset``
     - Prepare, represent, cache, and split the data.
     - :doc:`datasets`
   * - ``encodings``
     - Transform spectra before model input or loss calculation.
     - :doc:`encodings`
   * - ``model``
     - Select the neural network and its parameters.
     - :doc:`models`
   * - ``trainer``
     - Configure optimisation, losses, validation, and stopping.
     - :doc:`training`
   * - ``strategy``
     - Select single-model or ensemble training behaviour.
     - :doc:`strategies`
   * - ``inferencer``
     - Configure prediction from a trained checkpoint.
     - :doc:`inference`
   * - Analysis lists
     - Select, score, summarise, report, and plot predictions.
     - :doc:`analysis`
