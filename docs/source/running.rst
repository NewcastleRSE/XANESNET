================
Running XANESNET
================





----------------
Training a model
----------------

To train a model, use the following command:

.. code-block::

    python3 -m xanesnet.cli --mode MODE --in_file <path/to/file.yaml> --save

where

* ``--mode`` specifies the training mode.
* ``--in_file`` specifies the path to the input file for training.
* ``--save`` specifies whether to save the result to disk.

The implemented training modes include:

* ``train_xyz``: uses featurised structures as input data and XANES spectra as the target.
* ``train_xanes``: uses XANES spectra as input data and the featurised structures as the target.
* ``train_all``: trains both featurised structures and XANES spectra simultaneously (only available for the AEGAN model type).


The [-\-in_file] option specifies a file path containing training and hyperparameter settings.
This file must be provided in YAML format.
More details can be found in :doc:`input`.

Below is an example command for training a model using the MLP architecture,
with featurised structures as input data:

.. code-block::

    python3 -m xanesnet.cli --mode train_xyz --in_file inputs/in_mlp.yaml --save

The resulting trained model(s) and metadata will be saved in the 'models/' directory.

-------------------------------------
Experiment Tracking & Logging
-------------------------------------

`MLFlow <https://mlflow.org>`_ is used to track hyperparameters as well as
training and validation losses for each training run.
Results are automatically logged, allowing users to compare model runs
and track experiments over time.

XANESNET provides native MLflow integration for experiment tracking,
which can be enabled using the ``--mlflow`` option. For example:

.. code-block::

    python3 -m xanesnet.cli --mode train_xyz --in_file inputs/in_mlp.yaml --save --mlflow

To open the MLflow user interface, run the following command and click on the generated hyperlink:

.. code-block::

    mlflow ui

.. warning::

   Make sure this command is executed from the parent directory of the project.

`TensorBoard <https://www.tensorflow.org/tensorboard>`_  is a tool for
visualisation and measurement tracking through the machine learning workflow.
XANESNET provides built-in support for TensorBoard logging,
which can be enabled using the ``--tensorboard`` command-line option. For example:

.. code-block::

    python3 -m xanesnet.cli --mode train_xyz --in_file inputs/in_mlp.yaml --save --tensorboard


During model training, the ``Training loss`` and ``Validation loss`` are
currently logged and accessible via TensorBoard:

.. code-block::

    tensorboard --logdir=$tensorboard/ --host 0.0.0.0

.. warning::

   Make sure this command is executed from the parent directory of the project.

Then click on the generated hyperlink and select **Custom Scalars** to visualize the logged metrics.

------------------------
Prediction
------------------------

To use a previously trained model for predictions, use the following command:

.. code-block::

    python3 -m xanesnet.cli --mode MODE --in_model <path/to/model> --in_file <path/to/file.yaml>

where

* ``--mode`` specifies the prediction mode.
* ``--in_model`` specifies a directory containing a pre-trained model and its metadata.
* ``--in_file`` specifies the path to the input file for prediction.

The implemented prediction modes include:

* ``predict_xyz`` predicts a XANES spectrum from a featurised structural input.
* ``predict_xanes`` predicts featurised structures from an input XANES spectrum.
* ``predict_all`` simultaneous prediction of both featurised structures and XANES spectra from corresponding inputs with reconstruction of inputs (only available for the AEGAN model type).

As an example, the following command predicts XANES spectra using the MLP model trained previously:

.. code-block::

    python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_std_xyz_001 --in_file inputs/in_predict.yaml

The prediction results, including raw and plot data, are automatically saved in the 'outputs/' directory.
