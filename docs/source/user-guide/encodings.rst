Encodings
=========

Encodings (spectra encodings) transform spectra before they are passed to a model or to a :doc:`losses` function.
They are configured as an ordered list under ``encodings``.
Each encoding implements :class:`~xanesnet.encodings.base.SpectraEncoding` and provides an ``encode`` and ``decode`` operation.

Available implementations
--------------------------

* ``identity`` or ``none`` (:class:`~xanesnet.encodings.none.NoneEncoding`) leave the spectrum unchanged.
* ``scale`` (:class:`~xanesnet.encodings.scale.ScaleEncoding`) divides the spectrum by ``factor``. The factor can be global, per-point, or selected per target-site element.
* ``z_score`` (:class:`~xanesnet.encodings.zscore.ZScoreEncoding`) standardises the spectrum using ``mean`` and ``std``.
* ``min_max`` (:class:`~xanesnet.encodings.minmax.MinMaxEncoding`) normalises the spectrum using ``minimum`` and ``maximum``.
* ``subtract_average`` (:class:`~xanesnet.encodings.subtract_average.SubtractAverageEncoding`) subtracts an ``average`` spectrum.
* ``fourier`` (:class:`~xanesnet.encodings.fourier.FourierEncoding`) applies a symmetric-extension Fourier transform.
* ``gaussian`` (:class:`~xanesnet.encodings.gaussian.GaussianEncoding`) represents the spectrum with a Gaussian basis.
* ``concat`` (:class:`~xanesnet.encodings.concat.ConcatEncoding`) applies a non-empty list of independent encodings and concatenates their outputs.

Common configuration
--------------------

In forward training, the target spectrum is encoded before loss computation.
In inverse workflows, the input spectrum is encoded before it reaches the model.
Inference uses the encoding settings saved in the checkpoint signature to decode predictions consistently.

Multiple entries in the top-level ``encodings`` list are applied sequentially.
The separate ``concat`` implementation applies its sub-encodings in parallel.
A ``concat`` encoding cannot be nested inside another ``concat`` encoding.

Some input values may allow the use of ``auto``.
Check the schemas and API reference for each encoding type to see which fields support automatic resolution.

Examples
--------

Identity encoding:

.. code-block:: yaml

   encodings:
     - encoding_type: identity

Fourier encoding with the original spectrum included:

.. code-block:: yaml

   encodings:
     - encoding_type: fourier
       concat: true

Several independent encodings:

.. code-block:: yaml

   encodings:
     - encoding_type: concat
       encodings:
         - encoding_type: identity
         - encoding_type: fourier
           concat: false

API reference
-------------

See the API reference for :mod:`xanesnet.encodings` and the schemas under ``xanesnet/schemas/encodings/`` for complete fields and defaults.
