Training strategies
===================

The ``strategy`` section controls model initialisation, checkpointing, and the number or organisation of models used by training and inference.
A strategy can retain one model or several models for ensemble inference.

The base :class:`~xanesnet.strategies.base.Strategy` coordinates the model and runner lifecycle.
Its public responsibilities are:

* ``setup_models()`` instantiates the model or models managed by the strategy.
* ``init_model_weights()`` applies the configured weight and bias initializers.
* ``set_state_dicts()`` loads saved model weights for inference.
* ``setup_checkpointer()`` connects checkpoint storage to the model signature.
* ``setup_trainers(device)`` creates the trainer or trainers for training.
* ``run_training()`` runs training and returns the trained models.
* ``setup_inferencers(device)`` creates the inferencer or inferencers.
* ``run_inference(path)`` runs inference and optionally writes predictions.
* ``model_signature`` and ``signature`` expose the model and strategy settings saved in checkpoints.

The normal training order is to set up models, initialize weights, create the checkpointer and trainers, and then run training.
Inference loads model state dictionaries before the strategy creates and runs its inferencers.

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

K-fold training creates folds from the full dataset and does not use the dataset ``split_ratios`` field.

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
