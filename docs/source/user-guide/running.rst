Running XANESNET
================

Run XANESNET with ``train``, ``infer``, and ``analyze``.
Each command reads a YAML configuration file and writes a numbered run directory.
Example configs are available in ``configs/``.
See :doc:`input` for the configuration sections.

Training a model
----------------

.. code-block:: bash

   xanesnet train -i <config.yaml> [options]

Common options:

* ``-i, --in_file`` - Path to input YAML configuration file (required).
* ``-o, --out_dir`` - Path to output directory (optional, default: ``./runs``).
* ``-n, --name`` - Name for the training run used for logging and saving (optional).
* ``-t, --tensorboard`` - Enable training metrics in TensorBoard logs (optional).
* ``--dry-run`` - Run one real training epoch and save ``model_profile.json`` plus a readable model profile report (optional).
* ``-y, --yes`` - Skip interactive confirmation prompts (optional).


By default, training outputs are saved under ``<out_dir>/train_000/``.
With ``--name mlp_run``, the directory is ``<out_dir>/train_mlp_run_000/``.
Outputs include execution logs, model weights, checkpoints, and config files.
The processed dataset is stored under the configured ``dataset.root`` path.

Example:

.. code-block:: bash

   # MLP training configuration with default settings
   xanesnet train -i configs/mlp.yaml

Deep-ensemble example:

.. code-block:: bash

   # MLP deep ensemble training configuration with custom run name, TensorBoard logging, and no interactive confirmation
   xanesnet train -i configs/mlp_deep_ensemble.yaml -n mlp_run -t -y


Running inference
-----------------

.. code-block:: bash

   xanesnet infer -i <config.yaml> -m <checkpoint.pth> [options]

Common options:

* ``-i, --in_file`` - Path to the input inference YAML config (required).
* ``-m, --in_model`` - Path to a trained model .pth file (required).
* ``-o, --out_dir`` - Path to output directory (optional, default: ``./runs``).
* ``-n, --name`` - Name for the inference run used for logging and saving (optional).
* ``-y, --yes`` - Skip interactive confirmation prompts (optional).


By default, inference outputs are saved under ``<out_dir>/infer_000/``.
With ``--name mlp_infer``, the directory is ``<out_dir>/infer_mlp_infer_000/``.
Predictions are written to ``predictions/`` (HDF5 by default), alongside the run configuration files.

Example:

.. code-block:: bash

   # MLP inference configuration with default settings
   xanesnet infer -i configs/mlp_infer.yaml -m runs/train_000/models/final.pth

Deep-ensemble example:

.. code-block:: bash

   # MLP deep ensemble inference configuration with custom run name, and no interactive confirmation
   xanesnet infer -i configs/mlp_deep_ensemble_infer.yaml -m runs/train_000/models/final.pth -n mlp_infer -y

Analyzing results
-----------------

See :doc:`analysis` for selectors, collectors, aggregators, reporters, and plotters.

.. code-block:: bash

   xanesnet analyze -i <config.yaml> -r <infer_run> [options]

Common options:

* ``-i, --in_file`` - Path to the analysis YAML config (required).
* ``-r, --inference-runs`` - Path(s) to inference run directories (required). Repeat the option or provide multiple space-separated paths.
* ``-d, --prediction-names`` - Optional display names for the inference runs, in the same order as ``--inference-runs``.
* ``-o, --out_dir`` - Path to output directory (optional, default: ``./runs``).
* ``-n, --name`` - Name for the analysis run used for logging and saving (optional).
* ``-y, --yes`` - Skip interactive confirmation prompts (optional).

By default, analysis outputs are saved under ``<out_dir>/analyze_000/``.
With ``--name analyze_run``, the directory is ``<out_dir>/analyze_analyze_run_000/``.
Depending on the selected analysis components, outputs are written to ``plots/``, ``reports/``, and ``aux/``.

Example:

.. code-block:: bash

   # MLP analysis configuration with default settings
   xanesnet analyze -i configs/analyze_example.yaml -r runs/infer_000

Example:

.. code-block:: bash

   # MLP analysis configuration with a custom run name and no confirmation
   xanesnet analyze -i configs/analyze_example.yaml -r runs/infer_000 -n analyze_000 -y

