Training Strategies
===================

The ``strategy`` section selects how models are trained, initialised, and
checkpointed. Strategies are registered in ``StrategyRegistry`` and
configured via ``strategy_type``.

Weight initialisation (all strategies)
--------------------------------------

* ``weight_init`` — kernel initialisation (``xavier_uniform``, etc.)
* ``bias_init`` — bias initialisation (typically ``zeros``)
* ``checkpoint_interval`` — save checkpoints every *n* epochs

single
------

Train one model on the configured train/validation split. This is the
default for most example configs.

* ``strategy_type: single``
* Uses ``inferencer_type: basic`` at inference time

Example: ``configs/mlp.yaml``.

kfold
-----

K-fold cross-validation on the full dataset. Each fold trains a model;
the strategy keeps the member with the lowest validation score.

* ``strategy_type: kfold``
* ``n_splits`` — number of folds (default 3)
* ``n_repeats`` — repeat the full CV procedure (default 1)
* ``seed`` — split reproducibility

``split_ratios`` in the dataset section are ignored. Inference uses a
single best checkpoint with ``inferencer_type: basic``.

Example: ``configs/mlp_kfold.yaml``.

bootstrap
---------

Train multiple models on bootstrap resamples of the training subset.
Inference aggregates predictions (mean and standard deviation) via the
ensemble inferencer.

* ``strategy_type: bootstrap``
* ``n_models`` — number of bootstrap members
* ``sample_fraction`` — fraction of training samples per resample
  (with replacement)
* ``seeds`` — optional list of resampling seeds

Requires ``inferencer_type: ensemble`` in inference configs.

Example: ``configs/mlp_bootstrap.yaml``, ``configs/mlp_bootstrap_infer.yaml``.

deep_ensemble
-------------

Train multiple models with different weight initialisations on the same
train/validation split. Inference uses the ensemble inferencer.

* ``strategy_type: deep_ensemble``
* ``n_models`` — ensemble size
* ``seeds`` — optional per-member initialisation seeds

Example: ``configs/mlp_deep_ensemble.yaml``,
``configs/mlp_deep_ensemble_infer.yaml``.

snapshot_ensemble
-----------------

Collect snapshots from a single training run for ensembling. See
``xanesnet/schemas/strategies/snapshot_ensemble.schema.yaml`` for
parameters.

Strategy comparison
-------------------

+-------------------+-------------+------------------+----------------------------+
| ``strategy_type`` | Models kept | Inferencer       | Use case                   |
+===================+=============+==================+============================+
| ``single``        | 1           | ``basic``        | Standard training          |
| ``kfold``         | 1 (best)    | ``basic``        | Robust validation estimate |
| ``bootstrap``     | all         | ``ensemble``     | Uncertainty from sampling  |
| ``deep_ensemble`` | all         | ``ensemble``     | Uncertainty from init      |
| ``snapshot_ensemble`` | all   | ``ensemble``     | Cheap ensemble from one run|
+-------------------+-------------+------------------+----------------------------+

See :mod:`xanesnet.strategies` in the API reference for implementation
details.
