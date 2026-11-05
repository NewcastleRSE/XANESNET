Learning-rate schedulers
=========================

A learning-rate scheduler changes the :doc:`optimizers` learning rate during training. The trainer applies the scheduler once per epoch.
The main scheduler is configured under ``trainer.lr_scheduler`` and advances once per epoch.
The trainer can also apply a separate per-step linear warm-up with ``lr_warmup`` and ``warmup_steps``.

Available implementations
--------------------------

* ``step`` (:class:`~xanesnet.components.lrscheduler.StepLRScheduler`) decays the rate every ``step_size`` epochs by ``gamma``.
* ``multistep`` (:class:`~xanesnet.components.lrscheduler.MultiStepLRScheduler`) decays the rate at the configured ``milestones``.
* ``exponential`` (:class:`~xanesnet.components.lrscheduler.ExponentialLRScheduler`) applies a multiplicative ``gamma`` each epoch.
* ``linear`` (:class:`~xanesnet.components.lrscheduler.LinearLRScheduler`) changes the rate from ``start_factor`` to ``end_factor`` over ``total_iters`` epochs.
* ``constant`` (:class:`~xanesnet.components.lrscheduler.ConstantLRScheduler`) applies a multiplicative ``factor`` for ``total_iters`` epochs and then restores the rate.
* ``none`` and ``no`` (:class:`~xanesnet.components.lrscheduler.NoOpLRScheduler`) leave the learning rate unchanged.

Example
-------

.. code-block:: yaml

   trainer:
     learning_rate: 0.001
     lr_scheduler:
       lr_scheduler_type: linear
       start_factor: 1.0
       end_factor: 0.1
       total_iters: 40

API reference
-------------

See :mod:`xanesnet.components.lrscheduler` and ``xanesnet/schemas/components/lr_schedulers/`` for complete fields and defaults.
