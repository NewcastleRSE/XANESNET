Losses
======

The :class:`~xanesnet.losses.base.Loss` implementations measure the difference between model predictions and target spectra.
Encodings are applied before loss calculation; see :doc:`encodings`.
They are configured as a non-empty list under ``trainer.loss``.
XANESNET builds a combined loss from the selected terms before each training update.

Available implementations
--------------------------

* ``mse`` (:class:`~xanesnet.losses.mse.MSELoss`) computes mean squared error.
* ``l1`` (:class:`~xanesnet.losses.l1.L1Loss`) computes mean absolute error.
* ``bce`` (:class:`~xanesnet.losses.bcewithlogits.BCEWithLogitsLoss`) computes binary cross-entropy with logits.
* ``emd`` (:class:`~xanesnet.losses.emd.EMDLoss`) compares the cumulative distributions of prediction and target spectra.
* ``wcc`` (:class:`~xanesnet.losses.wcc.WCCLoss`) computes weighted cross-correlation.
* ``specplus`` (:class:`~xanesnet.losses.specplus.SpectralLossPlus`) combines blurred, detail, and gradient terms.
* ``msssim`` (:class:`~xanesnet.losses.msssim.MultiScale_SSIM`) computes a multi-scale structural-similarity loss for one-dimensional spectra.

Common configuration
--------------------

Each entry requires a positive ``loss_weight``.
Missing weights default to ``1.0`` and all weights are normalised to sum to one.

Example
-------

.. code-block:: yaml

   trainer:
     loss:
       - loss_type: mse
         loss_weight: 0.7
       - loss_type: wcc
         loss_weight: 0.3
         gaussian_hwhm: 10

API reference
-------------

See also :mod:`xanesnet.losses` and ``xanesnet/schemas/losses/`` for complete fields and defaults.
