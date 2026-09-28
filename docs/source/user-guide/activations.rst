Activations
===========

Activation functions are model-level components used by :doc:`models`.
They might be used in various models.

Available implementations
--------------------------

The activation registry provides:

* ``relu`` (:class:`~torch.nn.ReLU`)
* ``prelu`` (:class:`~torch.nn.PReLU`)
* ``tanh`` (:class:`~torch.nn.Tanh`)
* ``sigmoid`` (:class:`~torch.nn.Sigmoid`)
* ``elu`` (:class:`~torch.nn.ELU`)
* ``leakyrelu`` (:class:`~torch.nn.LeakyReLU`)
* ``selu`` (:class:`~torch.nn.SELU`)
* ``silu`` (:class:`~torch.nn.SiLU`)
* ``gelu`` (:class:`~torch.nn.GELU`)

Example
-------

.. code-block:: yaml

   model:
     model_type: mlp
     activation: gelu

API reference
-------------

See :mod:`xanesnet.components.activation` for the XANESNET activation registry.
