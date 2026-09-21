Input File
==========

The XANESNET input file is a YAML configuration file 
that specifies all parameters for the training, inference, and analysis.
The input file is divided into several sections, 
each corresponding to a different aspect of the configuration. 
This page details the syntax of each section, 
and provides guidance on how to customise your configuration.

Configuration file is validated against the schemas defined in ``xanesnet/schemas/``.
Default values are automatically applied during validation, 
so the input file only needs to include fields you want to override. 
The example input files can be found in the ``configs/`` directory.

Avaiable sections in training configuration files (``xanesnet train``):

  * ``seed`` — global random seed
  * ``device`` — compute device (``cpu`` or ``cuda``)
  * ``datasource`` — raw structure/spectrum source
  * ``dataset`` — how raw data are processed
  * ``encodings`` — optional spectrum encodings applied before the model
  * ``model`` — neural network architecture
  * ``trainer`` — optimisation loop
  * ``strategy`` — how models are trained and checkpointed.


Avaiable sections in inference configuration files (``xanesnet infer``):

  * ``seed`` — global random seed
  * ``device`` — compute device (``cpu`` or ``cuda``)
  * ``datasource`` — raw structure/spectrum source
  * ``dataset`` — how raw data are processed
  * ``inferencer`` — inference loop


Avaiable sections in analysis configuration files (``xanesnet analyze``):

  * ``seed`` — global random seed
  * ``selectors`` — which samples to analyse
  * ``collectors`` —  values to collect from the selected samples
  * ``aggregators`` — summary statistics over the collected values
  * ``reporters`` — reports written to disk 
  * ``plotters`` — plots written to disk


.. _datasource:

==========
datasource
==========

The datasource section defines where raw structures and spectra are read
from before dataset preparation.

* ``datasource_type`` (str): name of the datasource type, available values:

  * ``pmgjson`` (:class:`~xanesnet.datasources.pmgjson.PMGJSONSource`) —
    pymatgen JSON structure files with embedded spectra
  * ``multipmgjson`` (:class:`~xanesnet.datasources.multipmgjson.MultiPMGJSONSource`) —
    pymatgen JSON files across multiple subdirectories
  * ``xyzspec`` (:class:`~xanesnet.datasources.xyzspec.XYZSpecSource`) —
    paired ``.xyz`` structure and ``.txt`` spectrum files
  * ``multixyzspec`` (:class:`~xanesnet.datasources.multixyzspec.MultiXYZSpecSource`) —
    paired ``.xyz`` structure and ``.txt`` spectrum files across multiple subdirectories

* parameters are datasource-specific.

Example (Single directory XYZSpec dataset):
  
.. code-block:: yaml

   datasource:
     datasource_type: xyzspec 
     xyz_path: ./data/toy_data/
     spec_path: ./data/toy_data/
     spectrum_key: "XANES"

Example (Multiple directories PMGJSON dataset):

.. code-block:: yaml

  datasource:
    datasource_type: multipmgjson
    root_path: ./data/toy_data/
    spectrum_key: "XANES"

.. _dataset:

========
dataset
========

The dataset section defines how raw data are processed, cached, and split.

Common fields:

* ``dataset_type`` (str): name of the dataset type, available values:

  * ``descriptor`` (:class:`~xanesnet.datasets.torch.descriptor.DescriptorDataset`) —
    featurised structures for MLP models
  * ``multihead`` (:class:`~xanesnet.datasets.torch.multihead.MultiheadDataset`) —
    multi-head dataset for MultiHead_MLP, MultiHead_CNN models
  * ``geometrygraph`` (:class:`~xanesnet.datasets.torchgeometric.geometrygraph.GeometryGraphDataset`) —
    graph dataset for SchNet, DimeNet, and related GNNs models
  * ``gemnet`` (:class:`~xanesnet.datasets.torchgeometric.gemnet.GemNetDataset`) —
    graph dataset for GemNet and GemNet-OC models
  * ``envembed`` (:class:`~xanesnet.datasets.torch.envembed.EnvEmbedDataset`) —
    environment embedding dataset for EnvEmbed models
  * ``e3ee`` (:class:`~xanesnet.datasets.torchgeometric.e3ee.E3EEDataset`) —
    E(3)-equivariant edge embedding dataset for E3EE models
  * ``e3ee_full`` (:class:`~xanesnet.datasets.torchgeometric.e3ee_full.E3EEFullDataset`) —
    full E3EE graph dataset for E3EEFull models

* ``root`` (str): path to store the processed dataset
* ``preload`` (bool): load the full dataset into RAM when ``true``
* ``skip_prepare`` (bool): skip re-processing if cached data exist
* ``split_ratios`` (list): train/validation split fractions
* ``descriptors`` (list): name of the descriptors applied to the structure data
  (see :doc:`descriptors`)

Example (wACSF-based dataset):

.. code-block:: yaml
  
   dataset:
     dataset_type: descriptor
     root: ./data/processed/toy_data_descriptor/
     preload: True
     skip_prepare: False
     split_ratios: [0.8, 0.2]
     descriptors:
       - descriptor_type: wacsf
         r_min: 1.0
         r_max: 6.0
         n_g2: 16
         n_g4: 32

Example (SchNet dataset):

.. code-block:: yaml

  dataset:
    dataset_type:geometrygraph
    root: ./data/processed/toy_data_schnet/
    preload: True
    skip_prepare: False
    split_ratios: [0.8, 0.2]
    graph_builder:
      graph_builder_type: cov_radius
      cutoff: 5.0
      cov_radii_scale: 2.5
      max_num_neighbors: 50

.. _encodings:

=========
encodings
=========

The encodings section is a list of spectrum transforms applied to target
features before they reach the model. Each entry requires ``encoding_type``.

* ``encoding_type`` (str): name of the encoding type, available values:

  * ``identity/none`` (:class:`~xanesnet.encodings.none.NoneEncoding`) —
    leave spectra unchanged
  * ``scale`` (:class:`~xanesnet.encodings.scale.ScaleEncoding`) —
    divide spectra by a scaling factor
  * ``z_score`` (:class:`~xanesnet.encodings.zscore.ZScoreEncoding`) —
    z-score standardisation
  * ``min_max`` (:class:`~xanesnet.encodings.minmax.MinMaxEncoding`) —
    min-max normalisation to the unit interval
  * ``subtract_average`` (:class:`~xanesnet.encodings.subtract_average.SubtractAverageEncoding`) —
    average-spectrum subtraction encoding that trains on residuals about the mean spectrum
  * ``fourier`` (:class:`~xanesnet.encodings.fourier.FourierEncoding`) —
    symmetric-extension Fourier (DCT-like) transform
  * ``gaussian`` (:class:`~xanesnet.encodings.gaussian.GaussianEncoding`) —
    Gaussian-basis expansion that represents spectra as basis-expansion coefficients

* additional parameters are encoding-specific.

Example (Identity encoding):

.. code-block:: yaml

   encodings:
     - encoding_type: identity
  
  
Example (Fourier encoding with concatenation):

.. code-block:: yaml

  encodings:
    - encoding_type: fourier
    - concat: true

.. _model:

========
model
========

The model section defines the neural network architecture. See
:doc:`models` for supported ``model_type`` values and hyperparameters.

Common fields:

* ``model_type`` (str): name of the model type, supported values:

  * ``mlp`` (:class:`~xanesnet.models.mlp.mlp.MLP`) —
    feedforward network for descriptor-based forward or inverse mapping
  * ``schnet`` (:class:`~xanesnet.models.schnet.schnet.SchNet`) —
    continuous-filter graph convolution on ``geometrygraph`` data
  * ``dimenet`` (:class:`~xanesnet.models.dimenet.dimenet.DimeNet`) —
    directional message-passing GNN
  * ``dimenet++`` (:class:`~xanesnet.models.dimenet.dimenet_pp.DimeNetPlusPlus`) —
    improved DimeNet variant
  * ``gemnet`` (:class:`~xanesnet.models.gemnet.gemnet.GemNet`) —
    GemNet with triplet and quadruplet interactions
  * ``gemnet_oc`` (:class:`~xanesnet.models.gemnet_oc.gemnet_oc.GemNetOC`) —
    GemNet-OC for large catalysis-style graphs
  * ``envembed`` (:class:`~xanesnet.models.envembed.envembed.EnvEmbed`) —
    environment embedding network
  * ``e3ee`` (:class:`~xanesnet.models.e3ee.e3ee.E3EE`) —
    E(3)-equivariant edge embedding model
  * ``e3ee_full`` (:class:`~xanesnet.models.e3ee_full.e3ee_full.E3EEFull`) —
    full E3EE variant with extended graph features
  * ``mh_mlp`` (:class:`~xanesnet.models.multihead.mh_mlp.MultiHead_MLP`) —
    multi-head MLP for several target spectra or properties
  * ``mh_cnn`` (:class:`~xanesnet.models.multihead.mh_cnn.MultiHead_CNN`) —
    multi-head 1D CNN for sequential spectrum targets

* Additional keys are model-specific (hidden layers, dropout rate, graph radius,
  etc.)

.. warning::

   Model weight initialisation for training is configured under ``strategy`` rather
   than ``model`` in the current architecture.


Example (MLP model):

.. code-block:: yaml

   model:
     model_type: mlp
     in_size: auto
     out_size: auto
     hidden_size: 256
     dropout: 0.1
     num_hidden_layers: 3
     shrink_rate: 0.5
     activation: prelu  


Example (SchNet model):

.. code-block:: yaml

  model:
    model_type: schnet
    hidden_channels: 128
    reduce_channels_1: 64
    reduce_channels_2: auto
    num_filters: 128
    num_interactions: 6
    num_gaussians: 50
    cutoff: 5.0

.. _trainer:

========
trainer
========

The trainer section configures the optimisation loop.

* ``trainer_type`` (str): name of the trainer type, supported values:

  * ``basic`` (:class:`~xanesnet.runners.trainers.basic.BasicTrainer`) —
    single-process train/validate loop

* ``batch_size`` (int): number of samples per training batch
* ``shuffle`` (bool): shuffle training data each epoch
* ``drop_last`` (bool): drop the last incomplete training batch
* ``num_workers`` (int): number of data-loader worker processes
* ``epochs`` (int): total number of training epochs
* ``learning_rate`` (float): initial learning rate
* ``optimizer`` (str): optimiser name looked up via
  :class:`~xanesnet.components.OptimizerRegistry` (``Adam``, ``AdamW``,
  ``SGD``, ``RMSprop``, ``Adagrad``; matching is case-insensitive)
* ``max_norm`` (float): maximum gradient norm for clipping, or ``null`` to
  disable
* ``validation_interval`` (int): run validation every this many epochs when a
  validation subset is present
* ``lr_warmup`` (bool): apply a per-step linear learning-rate warm-up
* ``warmup_steps`` (int): number of warm-up steps when ``lr_warmup`` is
  ``true``

* ``loss`` (list): one or more loss terms. 

  * ``loss_type`` (str): supported values:
  
    * ``mse`` (:class:`~xanesnet.losses.mse.MSELoss`) — mean squared error
    * ``l1`` (:class:`~xanesnet.losses.l1.L1Loss`) — mean absolute error
    * ``bce`` (:class:`~xanesnet.losses.bcewithlogits.BCEWithLogitsLoss`) —
      binary cross-entropy with logits
    * ``emd`` (:class:`~xanesnet.losses.emd.EMDLoss`) — Earth Mover's distance
    * ``wcc`` (:class:`~xanesnet.losses.wcc.WCCLoss`) — weighted cross-correlation
    * ``specplus`` (:class:`~xanesnet.losses.specplus.SpectralLossPlus`) —
      multi-component spectral loss
    * ``msssim`` (:class:`~xanesnet.losses.msssim.MultiScale_SSIM`) —
      multi-scale structural similarity

  * additional parameters are loss-specific.

* ``regularizer``: weight regularisation. 

  * ``regularizer_type`` (str): supported values:

    * ``none`` / ``no`` (:class:`~xanesnet.regularizers.no.NoReg`) — no
      regularisation
    * ``l1`` (:class:`~xanesnet.regularizers.l1.L1Reg`) — L1 weight penalty
    * ``l2`` (:class:`~xanesnet.regularizers.l2.L2Reg`) — L2 weight penalty
  * additional parameters are regularizer-specific.

* ``lr_scheduler``: per-epoch learning-rate scheduler:

  * ``lr_scheduler_type`` (str): supported values:

    * ``step`` (:class:`~torch.optim.lr_scheduler.StepLR`) — decay at a given number of epochs
    * ``multistep`` (:class:`~torch.optim.lr_scheduler.MultiStepLR`) — decay at listed milestones
    * ``exponential`` (:class:`~torch.optim.lr_scheduler.ExponentialLR`) —

    * ``linear`` (:class:`~torch.optim.lr_scheduler.LinearLR`) — linear decay with a given start and end learning rate
    * ``constant`` (:class:`~torch.optim.lr_scheduler.ConstantLR`) — 
    keep the learning rate constant
    * ``none`` / ``no`` (:class:`~xanesnet.components.lrscheduler.NoOpLRScheduler`) —
      leave the learning rate unchanged

  * additional parameters are scheduler-specific.

* ``early_stopper``: early-stopping criterion:

  * ``early_stopper_type`` (str): supported values:

    * ``none`` / ``no`` (:class:`~xanesnet.stoppers.no.NoStopper`) — never
      stop early
    * ``basic`` (:class:`~xanesnet.stoppers.basic.BasicStopper`) — stop after
    a given number of epochs without improvement
    * ``time`` (:class:`~xanesnet.stoppers.time.TimeStopper`) — stop after 
    a given number of seconds of wall-clock time
  * additional parameters are stopper-specific.


Example:

.. code-block:: yaml

   trainer:
     trainer_type: basic
     batch_size: 4
     shuffle: true
     drop_last: false
     num_workers: 0
     epochs: 500
     learning_rate: 0.001
     optimizer: Adam
     max_norm: null
     validation_interval: 10
     lr_warmup: true
     warmup_steps: 500
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


.. _strategy:

=========
strategy
=========

The strategy section controls how models are trained, initialised, checkpointed,
and which inferencer type is required at prediction time. 

* ``strategy_type`` (str): name of the strategy type, supported values:

  * ``single`` (:class:`~xanesnet.strategies.single.Single`) —
    train one model on the configured train/validation split; use
    ``inferencer_type: basic`` at inference
  * ``kfold`` (:class:`~xanesnet.strategies.kfold.KFold`) —
    repeated k-fold cross-validation; keeps the best fold model; use ``inferencer_type: basic`` at inference
  * ``bootstrap`` (:class:`~xanesnet.strategies.bootstrap.Bootstrap`) —
    train multiple models on bootstrap resamples of the training subset; use
    ``inferencer_type: ensemble`` at inference
  * ``deep_ensemble`` (:class:`~xanesnet.strategies.deep_ensemble.DeepEnsemble`) —
    train multiple models with different initialisations on the same split; use
    ``inferencer_type: ensemble`` at inference

* ``weight_init`` (str): kernel initialisation via
  :class:`~xanesnet.components.WeightInitRegistry` (``default``, ``uniform``,
  ``normal``, ``xavier_uniform``, ``xavier_normal``, ``kaiming_uniform``,
  ``kaiming_normal``)
* ``weight_init_params`` (dict): extra keyword arguments forwarded to the
  selected initialiser
* ``bias_init`` (str): bias initialisation via
  :class:`~xanesnet.components.BiasInitRegistry` (``zeros`` or ``ones``)
* ``checkpoint_interval`` (int): save intermediate checkpoints every *n*
  epochs, or ``null`` to disable

Example:

.. code-block:: yaml

   strategy:
     strategy_type: single
     weight_init: xavier_uniform
     bias_init: zeros
     checkpoint_interval: 25

.. _inferencer:

===========
inferencer
===========

The inferencer section configures the inference loop. 

* ``inferencer_type`` (str): name of the inferencer type, supported values:

  * ``basic`` (:class:`~xanesnet.runners.inferencers.basic.BasicInferencer`) —
    single-model inference; pair with strategy types ``single`` or ``kfold``
  * ``ensemble`` (:class:`~xanesnet.runners.inferencers.ensemble.EnsembleInferencer`) —
    evaluate all ensemble members and return mean and standard deviation;
    pair with strategy types ``bootstrap`` or ``deep_ensemble``

* ``batch_size`` (int): number of samples per inference batch
* ``shuffle`` (bool): shuffle data during inference (
* ``drop_last`` (bool): drop the last incomplete batch
* ``num_workers`` (int): number of data-loader worker processes
* ``buffer_size`` (int): number of target-site rows buffered before
  predictions are flushed to disk

* ``model_device_policy`` (str, ``ensemble`` only): how ensemble models are
  placed on the inference device:

  * ``all`` — keep every model on the device 
  * ``sequential`` — move one model to the device at a time 

Example (single model):

.. code-block:: yaml

   inferencer:
     inferencer_type: basic
     batch_size: 4
     shuffle: false
     drop_last: false
     num_workers: 0
     buffer_size: 1000

Example (ensemble):

.. code-block:: yaml

   inferencer:
     inferencer_type: ensemble
     batch_size: 4
     shuffle: false
     drop_last: false
     num_workers: 0
     buffer_size: 1000
     model_device_policy: sequential

.. _analyze:

analysis 
======================

The analysis section configures the post-processing of inference predictions.

* ``seed`` (int): optional random seed for stochastic selectors

selectors
---------

Select list of samples applied to each prediction reader.

* ``selector_type`` (str): supported values:

  * ``all`` (:class:`~xanesnet.analysis.selectors.identity.IdentitySelector`) —
    keep every sample
  * ``none`` (:class:`~xanesnet.analysis.selectors.identity.IdentitySelector`) —
    keep no samples
  * ``index_list`` (:class:`~xanesnet.analysis.selectors.by_index.IndexSelector`) —
    keep explicit zero-based ``indices``
  * ``index_range`` (:class:`~xanesnet.analysis.selectors.by_range.IndexRangeSelector`) —
    keep samples from inclusive ``start`` up to exclusive ``end``
  * ``random`` (:class:`~xanesnet.analysis.selectors.bernoulli.BernoulliSelector`) —
    Bernoulli subsample with probability ``p``

collectors
----------

List of per-sample metrics computed from predictions and targets.

* ``collector_type`` (str): supported values:

  * ``error_metric`` (:class:`~xanesnet.analysis.collectors.errors.ErrorMetrics`) —
    prediction error using a registered loss

  Each ``error_metric`` entry also requires ``loss_type``: ``mse``, ``l1``,
  ``bce``, ``emd``, ``wcc``, ``specplus``, or ``msssim`` (same losses as
  :ref:`trainer`). Loss-specific parameters are forwarded to the loss
  constructor.

aggregators
-----------

List of summary statistics computed over collected values.

* ``aggregator_type`` (str): supported values:

  * ``scalar`` (:class:`~xanesnet.analysis.aggregators.scalar.ScalarAggregator`) —
    mean, standard deviation, min, max, and optional ``percentiles``

reporters
---------

List of structured outputs written to disk.

* ``reporter_type`` (str): supported values:

  * ``scalar`` (:class:`~xanesnet.analysis.reporters.scalar.ScalarReporter`) —
    per-sample scalar values as CSV files
  * ``statistics`` (:class:`~xanesnet.analysis.reporters.statistics.StatisticsReporter`) —
    aggregated statistics as YAML or JSON

plotters
--------

List of figures written to disk.

* ``plotter_type`` (str): supported values:

  * ``scalar`` (:class:`~xanesnet.analysis.plotters.scalar.ScalarPlotter`) —  
    histogram of each scalar value
  * ``spectra`` (:class:`~xanesnet.analysis.plotters.spectra.SpectraPlotter`) —
    predicted vs target spectra per sample
  * ``stat_table`` (:class:`~xanesnet.analysis.plotters.stat_table.StatTablePlotter`) —
    render comparison tables of aggregated statistics as PDF figures

Example:

.. code-block:: yaml

   seed: 42

   selectors:
     - selector_type: all
     - selector_type: random
       p: 0.1
     - selector_type: index_range
       start: 0
       end: 50

   collectors:
     - collector_type: error_metric
       loss_type: l1
     - collector_type: error_metric
       loss_type: mse

   aggregators:
     - aggregator_type: scalar
       percentiles: [25, 50, 75, 90, 95]

   reporters:
     - reporter_type: scalar
     - reporter_type: statistics
       format: yaml

   plotters:
     - plotter_type: scalar
       bins: 50
     - plotter_type: spectra
       sort_by_value: true
       sort_key: mse
       sort_ascending: true
       max_pages: 100
     - plotter_type: stat_table
       stat_keys: [mean, std, median, min, max]
       precision: 4