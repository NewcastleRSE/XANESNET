Trainer
========

The :class:`~xanesnet.runners.trainers.base.Trainer` runs the optimization loop for one model using the :doc:`datasets`, :doc:`models`, :doc:`encodings`, and trainer settings selected by the training configuration.

Responsibility
--------------

The trainer runs the optimization loop for one model. During setup it:

* resolves the :class:`~xanesnet.batchprocessors.base.BatchProcessor` for the selected :doc:`datasets` and :doc:`models`;
* creates the training and optional validation data loaders;
* creates the optimizer (:doc:`optimizers`), per-epoch learning-rate scheduler (:doc:`lr-schedulers`), and optional per-step warm-up scheduler;
* creates the combined loss (:doc:`losses`), regularizer (:doc:`regularizers`), and early stopper (:doc:`early-stoppers`); and
* connects the :class:`~xanesnet.checkpointing.checkpointer.Checkpointer` and :class:`~xanesnet.serialization.tensorboard.TensorBoardLogger`.

The public ``train()`` method runs each epoch, validates at the configured interval, logs metrics, advances the schedulers, checks early stopping, and saves checkpoints.
When configured, it restores the best model before returning.
Its return value is the score used for model selection.
When TensorBoard logging is enabled, model parameter and gradient histograms are omitted by default. Pass ``--log-weights`` to log these histograms each epoch.

Available implementations
--------------------------

* ``basic`` (:class:`~xanesnet.runners.trainers.basic.BasicTrainer`) provides the training and validation loop.
* ``lightning`` (:class:`~xanesnet.runners.trainers.lightning.LightningTrainer`) runs the same model and objective through Lightning on CPU or CUDA. Multiple CPU processes or CUDA devices use DDP.

MPS devices are currently unsupported and untested.

Trainer subcomponents are selected from these dedicated registries:

* losses: :doc:`losses`
* regularizers: :doc:`regularizers`
* early stoppers: :doc:`early-stoppers`
* optimizers: :doc:`optimizers`
* learning-rate schedulers: :doc:`lr-schedulers`

The overall model lifecycle and ensemble behavior are controlled by :doc:`strategies`.

Example
-------

A complete training configuration is available in ``configs/mlp.yaml``.
The following shows a trainer section with commonly changed options:

.. code-block:: yaml

   trainer:
     trainer_type: basic
     batch_size: 4
     shuffle: true
     drop_last: false
     num_workers: 0
     epochs: 50
     learning_rate: 0.001
     optimizer: Adam
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

.. _multi-device-lightning-training:

Multi-device Lightning training
-------------------------------

XANESNET supports multi-gpu/multi-cpu training through Lightning.
Use the special Trainer ``lightning``, set ``trainer.num_devices`` to the number of CPU training processes with ``device: cpu``, or visible GPUs with ``device: cuda``.
``device: cuda:N`` starts at visible index ``N``.
``batch_size`` and ``num_workers`` are per process.
Launch once with ``xanesnet train`` or ``python -m xanesnet.cli train``, not ``torchrun``, MPI, or multi-task ``srun``.

For two CPU training processes, change the following fields in a complete config:

.. code-block:: yaml

   device: cpu
   trainer:
     trainer_type: lightning  # <--
     num_devices: 2           # <--
     batch_size: 4
     num_workers: 0           # <--
   strategy:
     strategy_type: single

Run it through the normal CLI:

.. code-block:: bash

   python -m xanesnet.cli train -i configs/lightning_mlp.yaml --yes

``num_devices`` defaults to one in the schema. Its meaning depends on ``device``:

* ``cpu``: number of local training processes.
* ``cuda``: number of visible CUDA devices, starting at visible index zero.
* ``cuda:N``: consecutive visible CUDA indices starting at ``N``. For example,
  ``cuda:1`` with ``num_devices: 2`` selects visible indices 1 and 2.
  ``CUDA_VISIBLE_DEVICES`` can restrict and remap physical GPUs before launch.

Lightning must be launched once on one node with ``xanesnet`` or ``python -m xanesnet.cli``.
It starts its own workers; do not wrap the command in ``torchrun``, MPI, or a multi-task ``srun``.
A SLURM allocation may provide resources, but use one task on one node and request the intended GPU and CPU resources for that task. 

``batch_size`` and ``num_workers`` are per training process, not global values.

For input-dependent model branches that skip trainable parameters, set ``trainer.find_unused_parameters: true``. 
The schema default is ``false`` to avoid the extra autograd traversal for models that use every parameter.
If the used/unused parameter set and model control flow stay fixed for the entire run, ``trainer.static_graph: true`` enables DDP's static-graph mode instead.
It supports stable unused parameters without repeated unused-parameter discovery and takes precedence over ``find_unused_parameters``. 
Do not enable it when edge or path availability changes which modules run between steps.

With checkpointing enabled, Lightning writes both native XANESNET snapshots (``checkpoints/checkpoint_<n>_<epoch>.pth``)
and Lightning recovery files under ``checkpoints/model_<n>/``, including a final ``last.ckpt``. 
Native ``.pth`` snapshots use the XANESNET inference format and can be used directly. 
Lightning ``.ckpt`` files contain recovery state, including optimizer, schedulers, and stopper state, and are not native inference inputs. 

Multi-model strategies (``deep_ensemble``, ``bootstrap``, ``kfold``) train their models one after another.
Each model is trained with DDP across all configured devices.

API reference
-------------

See also :mod:`xanesnet.runners.trainers` and the schemas under ``xanesnet/schemas/runners/trainers/`` for complete fields and defaults.
