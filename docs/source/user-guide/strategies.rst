Training strategies
===================

The ``strategy`` section controls model initialisation, checkpointing, and the number or organisation of models used by training and inference.
A strategy can retain one model or several models for ensemble inference.

The base :class:`~xanesnet.strategies.base.Strategy` coordinates the model and runner lifecycle.
Its public responsibilities are:

* ``setup_models(model_config)`` instantiates the model or models from the resolved configuration.
* ``init_model_weights()`` applies the configured weight and bias initializers.
* ``set_state_dicts()`` loads saved model weights for inference.
* ``setup_checkpointer()`` connects checkpoint storage to the model signature.
* ``setup_model_splits()`` draws model-specific dataset splits, or restores previously drawn splits.
* ``setup_encoding(config)`` resolves and fits one encoding shared by every model in the strategy.
* ``setup_trainers(device)`` creates the trainer or trainers for training.
* ``run_training()`` runs training and returns the trained models.
* ``setup_inferencers(device)`` creates the inferencer or inferencers.
* ``run_inference(path)`` runs inference and optionally writes predictions.
* ``model_signature`` and ``signature`` expose the model and strategy settings saved in checkpoints.

Training prepares model splits and the shared encoding first, resolves model configuration, then passes it to ``setup_models`` before creating the checkpointer, weights, and trainers.
Inference restores the shared encoding, then passes the checkpoint's model configuration to ``setup_models`` before loading state dictionaries.

Available implementations
--------------------------

* ``single`` (:class:`~xanesnet.strategies.single.Single`) trains one model on the configured training and validation split.
  It uses ``inferencer_type: basic`` for inference.
* ``kfold`` (:class:`~xanesnet.strategies.kfold.KFold`) splits the full dataset into folds and trains one model for each fold.
  All fold models are retained and combined with ``inferencer_type: ensemble``.
* ``bootstrap`` (:class:`~xanesnet.strategies.bootstrap.Bootstrap`) trains several models on bootstrap resamples of the training subset. \
  The ensemble inferencer combines their predictions.
* ``deep_ensemble`` (:class:`~xanesnet.strategies.deep_ensemble.DeepEnsemble`) trains several independently initialised models on the same train and validation split.
  The ensemble inferencer combines their predictions.
* ``snapshot_ensemble`` (:class:`~xanesnet.strategies.snapshot_ensemble.SnapshotEnsemble`) is not implemented yet.

Common configuration
--------------------

All strategies accept:
* ``weight_init``
* ``weight_init_params``
* ``bias_init``
The weight initializer options are ``default``, ``uniform``, ``normal``, ``xavier_uniform``, ``xavier_normal``, ``kaiming_uniform``, and ``kaiming_normal``.
Bias initializers are ``zeros`` and ``ones``.
Note that some models do use their own initialisation and ignore the strategy settings.

The ``checkpoint_interval`` specifies the number of epochs between checkpoint saves.

Model-specific data and encodings
---------------------------------

Every training strategy writes one ``split_indices.json`` file containing the train and validation indices for each model, in model order.
Strategies with shared data, such as ``single`` and ``deep_ensemble``, record the same configured split for each model.
``bootstrap`` records each resample, while ``kfold`` records each fold.
Multi-device Lightning workers reuse these exact plans instead of drawing their own.

K-fold training creates folds from the full dataset and ignores the configured dataset splits.
``n_splits`` must not exceed the number of dataset samples.
All fold models share one encoding, whose automatic fields are resolved from the full dataset before the folds are trained.
Thus each fold's validation samples contribute to the shared encoding statistics; this is an intentional transductive choice rather than a fold-isolated preprocessing estimate.
The resolved encoding is stored once in the checkpoint signature and used by every fold model during inference.

Bootstrap members are resampled from the configured training subset and share the encoding fitted on that subset.

Examples
--------

Single-model training:

.. code-block:: yaml

   strategy:
     strategy_type: single
     weight_init: xavier_uniform
     bias_init: zeros
     checkpoint_interval: 25

K-fold training:

.. code-block:: yaml

   strategy:
     strategy_type: kfold
     n_splits: 3
     n_repeats: 1

API reference
-------------

See also :mod:`xanesnet.strategies` for the base interface and registry.
See :doc:`training` for the trainer settings and :doc:`inference` for the strategy and inferencer combinations.
The strategy schemas are in ``xanesnet/schemas/strategies/``.
