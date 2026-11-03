Optimizers
==========

The optimizer updates model parameters after each :doc:`training` batch.
It is selected by the ``trainer.optimizer`` field and is created by the trainer from the registered optimizer class.

Available implementations
--------------------------

* ``adam`` (:class:`~xanesnet.components.optim.AdamOptimizer`)
* ``sgd`` (:class:`~xanesnet.components.optim.SGDOptimizer`)
* ``rmsprop`` (:class:`~xanesnet.components.optim.RMSpropOptimizer`)
* ``adamw`` (:class:`~xanesnet.components.optim.AdamWOptimizer`)
* ``adagrad`` (:class:`~xanesnet.components.optim.AdagradOptimizer`)

Common configuration
--------------------

``learning_rate`` is configured beside ``optimizer`` in the ``trainer`` block.

Example
-------

.. code-block:: yaml

   trainer:
     optimizer: AdamW
     learning_rate: 0.001

API reference
-------------

See :mod:`xanesnet.components.optim` for the XANESNET optimizer registry.
