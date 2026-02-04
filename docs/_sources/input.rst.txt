Input File
===============

The XANESNET input file is a YAML configuration file that specifies all the necessary
parameters for the training and prediction.
The input file is divided into several sections, each corresponding to a different aspect of the configuration.
This page details the syntax of each section, and provides guidance on how to customise your model.

========
dataset
========

The dataset section defines the dataset type, file locations,
and preprocessing options used during model training.
It specifies where the raw XYZ structures and XANES spectra are stored,
where processed data should be saved, and how the data should be loaded and transformed.
See :doc:`dataset` for a detailed explanation.

* ``type`` (str): Dataset type (see :doc:`dataset`).
* ``root_path`` (str): Directory where the processed dataset will be stored or loaded from.
* ``xyz_path`` (str): Directory containing structure files.
* ``xanes_path`` (str): Directory containing XANES spectra files.
* ``params`` (dict, optional): Parameters for the chosen dataset type.

===========
descriptors
===========

The descriptor section defines the type of descriptor to be used for feature extraction.
A descriptor aims to transform atomic structures into fixed-size numeric vectors.

XANESNET provides a range of descriptor functions. Each of which requires different parameters to be specified.
See :doc:`descriptor` for a detailed explanation.

* ``type`` (str): Type of descriptor (see :doc:`descriptor`).
* ``params`` (dict, optional): Parameters for the chosen descriptor type.

========
model
========

The model section defines the architecture and specific parameters of the neural network model.
This section determines how the model is structured, including the type of neural network and the hyperparameters
that control its training and operation.

XANESNET supports a variety of widely-used deep neural network architecture.
Their architectures and parameters are further explained in the :doc:`model`.

* ``type`` (str): Type of model (see :doc:`model`).
* ``params`` (dict, optional): Model-specific parameters.
* ``weights`` (dict, optional): Configuration of the model weight initialisation.

    * ``kernel`` (str, ``default``):
      Weight initialization method for kernel parameters. Supported options:
      `uniform <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_>`__,
      `normal <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_>`__,
      `xavier_uniform <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_>`__,
      `xavier_normal <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_>`__,
      `kaiming_uniform <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_>`__,
      `kaiming_normal <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_>`__,
      *default*.
    * ``bias`` (str, ``zero``): Initialisation method for bias parameters. Supported options:
      `zeros <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.zeros_>`__,
      `ones <https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.ones_>`__.
    * ``seed`` (int): Random seed.

* ``weights_params`` (dict, optional): Additional weight parameters.

Example:
    .. code-block::

        model:
          type: mlp
          weights:
              kernel: kaiming_uniform
              bias: zeros
              seed: 2025
          weights_params:
              mode: fan_in
              nonlinearity: relu

================
hyperparameters
================

The hyperparameter section defines the settings that
govern the training process of the neural network model.

* ``batch_size`` (int, default: ``32``): Number of samples per training batch.
* ``lr`` (float, ``0.001``): Learning rate.
* ``epochs`` (int, ``100``): Maximum number of training epochs.
* ``n_earlystop`` (int, ``100``):  Number of consecutive epochs without improvement in validation loss before training is stopped.
* ``optimizer`` (str, ``adam``):
  Optimization algorithm used for training. Supported options:
  `adam <https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html>`__,
  `sgd <https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html>`__,
  `rmsprop <https://docs.pytorch.org/docs/stable/generated/torch.optim.RMSprop.html>`__,
  `adamw <https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`__,
  `adagrad <https://docs.pytorch.org/docs/stable/generated/torch.optim.Adagrad.html>`__.
* ``loss`` (str, default: ``mse``):
  Loss function. Supported options:
  `mse <https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`__,
  `bce <https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>`__,
  `l1 <https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html>`__,
  emd,
  cosine,
  wcc,
  hybrid,
  specplus,
  mw_ssim1d.
* ``loss_reg`` (str, ``none``):
  Type of regularization loss. Supported options:
  `l1 <https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html>`__,
  `l2 <https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`__,
  none.
* ``loss_lambda`` (float, ``0.0001``): Weighting factor applied to the regularization loss.
* ``loss_params`` (dict, optional): Additional parameters specific to the selected loss function.

Example:
    .. code-block::

        hyperparams:
          batch_size: 64
          lr: 0.0001
          epochs: 500
          optimizer: Adam
          seed: 2021
          loss: mse
          loss_reg: L1
          loss_lambda: 0.0001

==============
lr_scheduler
==============

The lr_scheduler section controls whether a learning rate scheduler
is used to dynamically adjust the learning rate during training.

A learning rate scheduler modifies the learning rate over time,
either according to a predefined schedule or in response to training progress.

* ``lr_scheduler`` (bool): Enables or disables the learning rate scheduler.
* ``scheduler_params`` (dict, optional):

  * ``type`` (str, ``step``): Type of learning rate scheduler to use.
    Supported options: *step* or *multistep* or *exponential* or *linear* or *constant*
  * ``step_size`` (int, ``10``):  Number of epochs between successive learning rate updates.
  * ``gamma`` (float, ``0.5``):  Multiplicative factor applied to the learning rate at each decay step.

Example:
    .. code-block::

        lr_scheduler: True
        scheduler_params:
          type: step
          step_size: 100
          gamma: 0.5


=======
kfold
=======

The kfold section controls whether K-Fold cross-validation is used during model training.
When enabled, the dataset is partitioned into k subsets (folds).
The model is trained k times, each time using a different fold
as the validation set and the remaining k-1 folds as the training set.

* ``kfold`` (bool): Enables or disables K-Fold cross-validation.
* ``kfold_params`` (dict, optional):

  * ``n_splits`` (int, ``3``): Number of folds (k).
  * ``n_repeats`` (int, ``1``): Number of times the K-Fold cross-validation is repeated.
  * ``seed`` (int): Random seed used for reproducible dataset splitting.

Example:
    .. code-block::

        kfold: True
        kfold_params:
          n_splits: 5
          n_repeats: 1
          seed: 2022

==============
bootstrap
==============

The bootstrap section controls whether bootstrap resampling is used during model training.
When enabled, the original training dataset is randomly resampled with replacement
to create new training datasets. The size of each resampled dataset is determined
by multiplying the original dataset size by a user-defined factor.
The model is trained multiple times using independently resampled datasets,
each with a different random seed.

* ``bootstrap`` (bool): Enables or disables bootstrap resampling.
* ``bootstrap_params`` (dict, optional): Parameters controlling the bootstrap procedure.

  * ``n_boot`` (int, ``3``): Number of bootstrap training runs
  * ``n_size`` (float, ``1.0``): Scaling factor applied to the original dataset size when generating each bootstrap sample.
  * ``weight_seed`` (list): List of random seeds used to generate independent bootstrap samples.

Example:
    .. code-block::

        bootstrap: True
        bootstrap_params:
          n_boot: 4
          n_size: 1.0
          weight_seed: [97, 39, 22]


==============
ensemble
==============

The ensemble section controls whether ensemble training is performed.
When enabled, multiple models are trained independently
using different random initializations of model parameters.
Each ensemble member is initialized with a distinct random seed.

* ``ensemble`` (bool): Enables or disables ensemble training.
* ``ensemble_params`` (dict, optional):

  * ``n_ens`` (int, ``3``): Number of ensemble members to train.
  * ``weight_seed`` (list): List of random seeds used to initialise model weights and biases for each ensemble member.

Example:
    .. code-block::

        ensemble: True
        ensemble_params:
          n_ens: 3
          weight_seed: [97, 39, 22]
