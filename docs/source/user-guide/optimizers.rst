Optimizers
==========

The optimizer updates model parameters after each :doc:`training` batch.
It is selected by the ``trainer.optimizer`` field and is created by the trainer from the registered optimizer class.

Available implementations
--------------------------

* ``adam`` (:class:`~torch.optim.Adam`)
* ``sgd`` (:class:`~torch.optim.SGD`)
* ``rmsprop`` (:class:`~torch.optim.RMSprop`)
* ``adamw`` (:class:`~torch.optim.AdamW`)
* ``adagrad`` (:class:`~torch.optim.Adagrad`)

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
