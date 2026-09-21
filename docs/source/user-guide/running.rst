Running XANESNET
================

XANESNET workflow are executed using three commands: ``train``, ``infer``, and ``analyze``.
Each command reads a YAML configuration file and writes output to a directory. Example configuration files 
are available in the ``configs/`` directory.

----------------
Training a model
----------------

.. code-block:: bash

   xanesnet train -i <config.yaml> [options]

Common options:

* ``-i, --in_file`` — Path to input YAML configuration file (required).
* ``-o, --out_dir`` —  Path to output directory (optional, default: ``./runs``).
* ``-n, --name`` — Name for the training run used for logging and saving (optional).
* ``-t, --tensorboard`` — Enable to write training metrics to TensorBoard logs (optional).
* ``--dry-run`` — Run one training epoch and save a model profile report (optional).
* ``-y, --yes`` — Skip interactive confirmation prompts (optional).


Training outputs are saved under ``<out_dir>/<name>/``. Typical outputs include processed
dataset, execution logs, trained model, model checkpoints, and config files.

Examples:

.. code-block:: bash

   xanesnet train -i configs/mlp.yaml 

   xanesnet train -i configs/mlp_deep_ensembles.yaml -n mlp_run -t -y


-----------------
Running inference
-----------------

.. code-block:: bash

   xanesnet infer -i <config.yaml> -m <checkpoint.pth> [options]

Common options:

* ``-i, --in_file`` — Path to the input inference YAML config (required).
* ``-m, --in_model`` — Path to a trained model .pth file (required).
* ``-o, --out_dir`` —  Path to output directory (optional, default: ``./runs``).
* ``-n, --name`` — Name for the inference run used for logging and saving (optional).
* ``-y, --yes`` — Skip interactive confirmation prompts (optional).


Inference outputs are saved under ``<out_dir>/<name>/``, including 
the predictions in ``predictions/`` (HDF5 by default), and config files.


Examples:

.. code-block:: bash

   xanesnet infer -i configs/mlp_infer.yaml -m runs/train_000/models/final.pth

   xanesnet infer -i configs/mlp_deep_ensemble_infer.yaml -m runs/train_000/models/final.pth -n mlp_infer -y

------------------
Analyzing results
------------------

.. code-block:: bash

   xanesnet infer -i <config.yaml> -p <infer_run> [options]

Common options:

* ``-i, --in_file`` — Path to the analysis YAML config (required).
* ``-p, --predictions`` — Path to the directory containing inference predictions (required).
* ``-o, --out_dir`` — Path to output directory (optional, default: ``./runs``).
* ``-n, --name`` —  Name for the analysis run used for logging and saving (optional).
* ``-y, --yes`` — Skip interactive confirmation prompts (optional).

Analysis outputs are saved under ``<out_dir>/<name>/``. Depneding on the selected analysis type, 
the outputs may include: scalar metrics in ``metrics/``, spectra plots in ``spectra/``, and statistical tables in ``statistics/``.

Examples:

.. code-block:: bash

   xanesnet infer -i configs/analyze_example.yaml -p runs/infer_000

   xanesnet infer -i configs/analyze_example.yaml -p runs/infer_000 -n analyze_000 -y

