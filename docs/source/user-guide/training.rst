Trainer
========

The :class:`~xanesnet.runners.trainers.base.Trainer` runs the optimization loop for one model using the :doc:`datasets`, :doc:`models`, :doc:`encodings`, and trainer settings selected by the training configuration.

Responsibility
--------------

The trainer runs the optimization loop for one model. During setup it:

* resolves the :class:`~xanesnet.batchprocessors.base.BatchProcessor` for the selected :doc:`datasets` and :doc:`models`;
* creates the training and optional validation data loaders;
* creates the optimizer (:doc:`optimizers`), per-epoch learning-rate scheduler (:doc:`lr-schedulers`), and optional per-step warm-up scheduler;
* creates the combined loss (:doc:`losses`), regularizer (:doc:`regularizers`), and early stopper (:doc:`early-stoppers`); and
* connects the :class:`~xanesnet.checkpointing.checkpointer.Checkpointer` and :class:`~xanesnet.serialization.tensorboard.TensorBoardLogger`.

The public ``train()`` method runs each epoch, validates at the configured interval, logs metrics, advances the schedulers, checks early stopping, and saves checkpoints.
When configured, it restores the best model before returning.
Its return value is the score used for model selection.

Available implementations
--------------------------

* ``basic`` (:class:`~xanesnet.runners.trainers.basic.BasicTrainer`) provides the training and validation loop.

Trainer subcomponents are selected from these dedicated registries:

* losses: :doc:`losses`
* regularizers: :doc:`regularizers`
* early stoppers: :doc:`early-stoppers`
* optimizers: :doc:`optimizers`
* learning-rate schedulers: :doc:`lr-schedulers`

The overall model lifecycle and ensemble behavior are controlled by :doc:`strategies`.

Example
-------

A complete training configuration is available in ``configs/mlp.yaml``.
The following shows a trainer section with commonly changed options:

.. code-block:: yaml

   trainer:
     trainer_type: basic
     batch_size: 4
     shuffle: true
     drop_last: false
     num_workers: 0
     epochs: 50
     learning_rate: 0.001
     optimizer: Adam
     loss:
       - loss_type: mse
     regularizer:
       regularizer_type: none
     lr_scheduler:
       lr_scheduler_type: linear
       start_factor: 1.0
       end_factor: 0.1
       total_iters: 40
     early_stopper:
       early_stopper_type: basic
       patience: 25
       min_delta: 0.001
       restore_best: true

API reference
-------------

See also :mod:`xanesnet.runners.trainers` and the schemas under ``xanesnet/schemas/runners/trainers/`` for complete fields and defaults.
