Early stoppers
==============

The :class:`~xanesnet.stoppers.base.EarlyStopper` implementations decide whether :doc:`training` should end before the configured number of epochs.
The trainer checks the selected stopper after each available training or validation score.

Available implementations
--------------------------

* ``none`` or ``no`` (:class:`~xanesnet.stoppers.no.NoStopper`) never stop early.
* ``basic`` (:class:`~xanesnet.stoppers.basic.BasicStopper`) stops after a configured number of epochs without meaningful improvement.
* ``time`` (:class:`~xanesnet.stoppers.time.TimeStopper`) stops after a wall clock time limit (useful for cluster computation).

Common configuration
--------------------

All implementations accept ``restore_best`` (default ``true``).
When ``restore_best`` is enabled, the trainer restores the best saved model after training stops.

Example
-------

.. code-block:: yaml

   trainer:
     early_stopper:
       early_stopper_type: basic
       patience: 25
       min_delta: 0.001
       restore_best: true

API reference
-------------

See also :mod:`xanesnet.stoppers` and ``xanesnet/schemas/stoppers/`` for complete fields and defaults.
