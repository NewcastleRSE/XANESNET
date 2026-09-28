Configuration editor
====================

The XANESNET configuration editor is a work-in-progress browser-based tool for creating and editing
YAML configuration files for ``train``, ``infer``, and ``analyze`` workflows.
Training, inference, and analysis are still run with the ``xanesnet`` command or from Python.

The editor is implemented with React and Vite. Its source is in ``tools/config-ui/``.

.. note::
   The editor is under active development.

The forms are generated from the JSON schemas in ``src/schemas/``, which is a symlink to ``xanesnet/schemas/``.
The editor therefore uses the same defaults and allowed variants as runtime validation.

Current capabilities
--------------------

* Mode-specific forms for Train, Infer, and Analyze configurations.
* YAML import.
* Inference checkpoint ``signature.yaml`` import.
* Live YAML preview.

Requirements
------------

* Node.js 20.19 or later, or 22.12 or later (required by Vite 8; Node.js 21 is not supported).
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

For additional implementation and maintenance details,
see the `configuration editor README <https://github.com/NewcastleRSE/xray-spectroscopy-ml/blob/main/tools/config-ui/README.md>`_.