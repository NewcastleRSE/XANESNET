Project Overview
================

XANESNET is a Python toolkit for machine-learning simulation and analysis of
structure-spectra relationships. The current supported workflow is forward
prediction from molecular or periodic structures to spectra.

The project ships:

- A command-line interface (``xanesnet train``, ``xanesnet infer``,
  ``xanesnet analyze``) backed by YAML configuration files.
- A schema-validated configuration system in
  :mod:`xanesnet.serialization` whose JSON Schemas double as documentation
  and editor metadata.
- A library of model families, datasets, descriptors, and analysis
  components, all driven by a small registry pattern.
- An interactive browser-based config editor in ``tools/config-ui/``.

Quickstart
----------

The repository ``README.md`` covers installation, the example data shipped
under ``data/fe/``, and end-to-end ``train``/``infer``/``analyze`` invocations.
After installation, run::

   xanesnet train -i configs/in_mlp.yaml -n mlp_test -y
   xanesnet infer -i configs/in_mlp_infer.yaml \
       -m runs/<train_run>/models/final.pth -n mlp_infer -y
   xanesnet analyze -i configs/analyze_example.yaml \
       -r runs/<infer_run> -n mlp_analysis -y

Configuration
-------------

Every run reads one YAML config. Required top-level sections depend on the
mode but always include ``datasource``, ``dataset``, ``model``, and a runner
section (``trainer``, ``inferencer``, or analysis sub-config). Schema-backed
defaults are materialised in place during validation, so example configs only
need to spell out fields that differ from the defaults.

The schemas live under ``xanesnet/schemas/`` as JSON Schema Draft 2020-12 YAML
files. The same files back the React config editor in ``tools/config-ui/``
through a symlink.

Where to look
-------------

- :mod:`xanesnet.cli` — top-level dispatcher.
- :mod:`xanesnet.train`, :mod:`xanesnet.infer`, :mod:`xanesnet.analyze` —
  per-mode entry points used by the CLI.
- :mod:`xanesnet.core_train`, :mod:`xanesnet.core_infer`,
  :mod:`xanesnet.core_analyze` — orchestration of each pipeline.
- :mod:`xanesnet.serialization.config`,
  :mod:`xanesnet.serialization.schema_validation` — config loading and
  validation.
- :mod:`xanesnet.descriptors` — structural descriptors:
  :mod:`~xanesnet.descriptors.wacsf`,
  :mod:`~xanesnet.descriptors.rdc`,
  :mod:`~xanesnet.descriptors.mace`,
  :mod:`~xanesnet.descriptors.soap` (dscribe-based, mu2 compression),
  :mod:`~xanesnet.descriptors.pdos` (xTB/pySCF), and
  :mod:`~xanesnet.descriptors.direct` (pre-computed ``.txt`` files).
- :mod:`xanesnet.models`, :mod:`xanesnet.datasets`,
  :mod:`xanesnet.datasources`, :mod:`xanesnet.descriptors`,
  :mod:`xanesnet.runners`, :mod:`xanesnet.strategies`,
  :mod:`xanesnet.analysis` — pluggable building blocks.
