================
Running XANESNET
================





----------------
Training a model
----------------

To train a model, use the following command:

.. code-block::

    python3 -m xanesnet.cli --mode MODE --in_file <path/to/file.yaml>

[-\-mode] is the additional settings to specify the training mode,
with three available options:

* ``train_xanes`` Xanes spectra are used as input data, with the featurised structures are the target of the model.
* ``train_xyz`` Featurised structures are used as input data, with the xanes spectra are the target of the model.
* ``train_aegan`` Trains featurised structures and xanes spectra simultaneously.

The [-\-in_file] option specifies a file containing training and hyperparameter settings.
This file must be provided in YAML format.
More details can be found in :doc:`input`.

Below is an example command for training a model using MLP architecture
with featurised structures as input data:

.. code-block::

    python3 -m xanesnet.cli --mode train_xyz --in_file inputs/in_mlp.yaml

By default, the resulting trained model and its metadata are automatically saved in the 'models/' directory.
If you prefer not to save, use optional [-\-save] setting and toggling it to "no".

.. -------------------------------------
.. Experiment Tracking & Logging Results
.. -------------------------------------

.. `MLFlow <https://mlflow.org>`_ is used to track hyperparameters, training and validation losses for each training run. Results are automatically logged and users can compare across model runs and track experiments. To open the user interface run the following on the command line and click on the hyperlink:

.. .. code-block::

..     mlflow ui


.. Tensorboard

.. Tensorboard is a tool for visualisation and measurement tracking through the machine learning workflow. During model training the following values are currently logged and accessible through the tensor board,

.. Training & validation loss
.. To run tensorboard, run tensorboard --logdir=/tmp/tensorboard/ --host 0.0.0.0 , click on the hyperlink and choose Custom Scalar.


------------------------
Prediction
------------------------

To use a previously developed model for predictions, the following command is used:

.. code-block::

    python3 -m xanesnet.cli --mode MODE --in_model <path/to/model> --in_file <path/to/file.yaml>

The prediction mode can be specified with the [-\-mode] option:

* ``predict_xyz`` Predicts the featurised structure from an input XANES spectrum.
* ``predict_xanes`` Predicts the XANES spectrum from a featurised structural input.
* ``predict_all`` Simultaneously predicts a featurised structure and XANES spectrum from the corresponding input as well as reconstructs inputs. This mode is for AEGAN model type only.



The [-\-in_model] option specifies a directory containing pre-trained model and its metadata.
The [-\-in_file] specifies a path to input file for prediction, see :doc:`input` for more details.

As an example, the following command makes spectrum predictions using the previously trained MLP model:

.. code-block::

    python3 -m xanesnet.cli --mode predict_xanes --in_model models/model_mlp_001 --in_file inputs/in_predict.yaml

The prediction results, including raw and plot data, are automatically saved in the 'outputs/' directory.
