Regularizers
============

The :class:`~xanesnet.regularizers.base.Regularizer` implementations add a parameter penalty to the training objective.
They are configured under ``trainer.regularizer`` and applied while the :doc:`training` trainer computes the total loss.

Available implementations
--------------------------

* ``none`` or ``no`` (:class:`~xanesnet.regularizers.no.NoReg`) disable regularization.
* ``l1`` (:class:`~xanesnet.regularizers.l1.L1Reg`) applies an L1 penalty.
* ``l2`` (:class:`~xanesnet.regularizers.l2.L2Reg`) applies an L2 penalty.

Example
-------

.. code-block:: yaml

   trainer:
     regularizer:
       regularizer_type: l2
       weight: 0.0001

API reference
-------------

See also :mod:`xanesnet.regularizers` and ``xanesnet/schemas/regularizers/`` for complete fields and defaults.
