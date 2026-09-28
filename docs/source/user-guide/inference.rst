Inferencer
==========

The inferencer evaluates a trained model or model ensemble on a prepared dataset and writes prediction records.
The checkpoint is produced by :doc:`training`.
The prepared dataset and model are described in :doc:`datasets` and :doc:`models`.

Responsibility
--------------

The inferencer evaluates a model or model ensemble on a prepared dataset.
Its public ``infer()`` method:

* moves the inference model or models to the selected device;
* creates an :class:`~xanesnet.serialization.prediction_writers.HDF5Writer` when an output path is provided;
* runs one inference pass over the data loader; and
* closes the writer and moves models back to CPU even when inference fails.

The writer stores prediction records with targets, sample identifiers, timing, and optional target-site indices.
Ensemble inference also stores the prediction standard deviation across models.

Inference does not calculate metrics, aggregate results, or create analysis plots.
It only generates and saves prediction records.
Use :doc:`analysis` to compute losses, summaries, comparisons, reports, and figures.

Available implementations
--------------------------

* ``basic`` (:class:`~xanesnet.runners.inferencers.basic.BasicInferencer`) evaluates one model and writes decoded predictions.
* ``ensemble`` (:class:`~xanesnet.runners.inferencers.ensemble.EnsembleInferencer`) evaluates all models retained by a ``kfold``, ``bootstrap``, or ``deep_ensemble`` strategy and writes the ensemble mean and standard deviation.

Checkpoint signature
--------------------

Before final validation, XANESNET loads the signature saved in the checkpoint and merges it with the user-authored inference YAML.
The signature supplies the trained dataset, model, strategy, and encoding settings.

The ``dataset`` section can be an overlay. It may change fields such as ``root``, ``preload``, ``skip_prepare``, or split settings.
Values written in signature-pinned dataset, model, strategy, or encoding sections must agree with the checkpoint signature.

For example, ``configs/mlp_infer.yaml`` provides a datasource, a dataset overlay, and a basic inferencer.
It does not repeat the model, strategy, encodings, or dataset type because those values come from the checkpoint.

Common configuration
--------------------

The basic inferencer defaults include ``batch_size: 1``, ``shuffle: false``, ``drop_last: false``, ``num_workers: 0``, and ``buffer_size: 100000``.

Ensemble inference also accepts ``model_device_policy``:

* ``all`` keeps all ensemble models on the inference device.
* ``sequential`` moves one model to the device at a time to reduce device memory use.

Strategy-inferencer compatibility
----------------------------------

The strategy determines whether inference uses one model or an ensemble:

.. list-table:: Strategy and inferencer compatibility
   :header-rows: 1
   :widths: 30 25 45

   * - Strategy
     - Inferencer
     - Behavior
   * - ``single``
     - ``basic``
     - Evaluates the single trained model.
   * - ``kfold``
     - ``ensemble``
     - Evaluates all fold models and returns the ensemble mean and standard deviation.
   * - ``bootstrap``
     - ``ensemble``
     - Evaluates all bootstrap models and returns the ensemble mean and standard deviation.
   * - ``deep_ensemble``
     - ``ensemble``
     - Evaluates all independently initialized models and returns the ensemble mean and standard deviation.
   * - ``snapshot_ensemble``
     - not available
     - The strategy is registered but not implemented.

For strategy configuration and training behavior, see :doc:`strategies`.

Examples
--------

Basic inference:

.. code-block:: yaml

   inferencer:
     inferencer_type: basic
     batch_size: 4
     shuffle: false
     drop_last: false
     num_workers: 0
     buffer_size: 1000

Ensemble inference:

.. code-block:: yaml

   inferencer:
     inferencer_type: ensemble
     batch_size: 4
     shuffle: false
     drop_last: false
     num_workers: 0
     buffer_size: 1000
     model_device_policy: sequential

API reference
-------------

See also :class:`~xanesnet.runners.inferencers.base.Inferencer`, :class:`~xanesnet.serialization.prediction_writers.HDF5Writer`, :mod:`xanesnet.runners.inferencers`,
and the schemas under ``xanesnet/schemas/runners/inferencers/`` for complete fields and defaults.
