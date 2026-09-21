User Interface
==============

The XANESNET user interface is an interactive editor for ``train``,
``infer``, and ``analyze`` YAML configuration files. It is located in
``tools/config-ui/`` in the repository.

Forms are generated from JSON Schema files exposed at ``src/schemas/``, which
is a symlink to ``xanesnet/schemas/``. This keeps UI defaults and allowed
schema variants aligned with runtime Python validation.


The app supports two main tasks:

* Browse valid XANESNET configuration options.
* Generate complete YAML files for training, inference, and analysis workflows.

Features
--------

* Mode-specific forms for Train, Infer, and Analyze configs.
* YAML import with automatic mode detection.
* Infer checkpoint ``signature.yaml`` import that merges signature-provided
  ``dataset``, ``model``, and ``strategy`` fields into the Infer form.
* Signature-origin highlighting while keeping signature-loaded fields editable.
* Live YAML preview with schema defaults materialised and top-level sections
  ordered for readability.
* Light/dark theme toggle stored in local browser state.

Requirements
------------

* Node.js 20 or later (required by Vite 8)
* npm (bundled with Node.js)

Quick start
-----------

Run from ``tools/config-ui/``:

.. code-block:: bash

   npm install
   npm run dev

The development server prints a local URL, usually ``http://127.0.0.1:5173/``
or the next free Vite port.

Common commands
---------------

.. code-block:: bash

   npm run dev      # Start the Vite development server
   npm run lint     # Run ESLint
   npm run build    # Type-check and build production assets into dist/
   npm run preview  # Preview the production build locally

Project layout
--------------

``src/App.tsx``
  React UI, RJSF custom fields/templates, YAML loading, signature loading, and
  form state handling.

``src/configYaml.ts``
  Materialises schema defaults and formats generated YAML.

``src/schemaRegistry.ts``
  Loads YAML schemas through Vite, resolves local ``$ref``\ s, and decorates
  union option labels.

``src/schemas/``
  Symlink to packaged JSON Schemas for Train, Infer, Analyze, and their
  component sections.

``public/favicon.svg``
  App icon used by ``index.html``.

Schema notes
------------

Schemas are authored as YAML files and imported with Vite raw imports. Local
``$ref``\ s are resolved in ``schemaRegistry.ts``, so schema files can stay
split by domain while the app receives dereferenced schema objects.

The canonical schema files live in ``xanesnet/schemas/``; do not edit copies
under the UI tree.

Infer configs are special: at runtime XANESNET merges user Infer YAML with a
checkpoint signature before validation. The UI mirrors that by letting users
load ``signature.yaml`` in Infer mode. User-supplied Infer YAML can stay as a
checkpoint overlay without ``dataset_type``, ``model``, or ``strategy``; a
loaded signature supplies those fields when available.

When changing schemas, smoke-test at least these flows:

* Train default descriptor/MLP generation.
* Train graph dataset plus graph model branch switching.
* Infer overlay generation without a signature.
* Infer YAML loaded before a signature.
* Signature loaded before a YAML config.
* Clear after loading a signature.
* Analyze empty lists and a populated analysis example.

Maintenance
-----------

Keep generated build output out of source review. ``dist/`` and
``node_modules/`` are ignored and can be recreated with ``npm run build`` and
``npm install``.

The app intentionally does not depend on a component framework. Styling lives
in ``src/App.css`` and ``src/index.css``; form rendering is handled by
``@rjsf/core`` and ``@rjsf/validator-ajv8``.
