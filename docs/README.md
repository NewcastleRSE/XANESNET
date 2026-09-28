# XANESNET Documentation

This directory contains the [Sphinx](https://www.sphinx-doc.org/) sources for the XANESNET API reference and user guide.

## Building

Install the documentation extras into the active environment (one-off):

```bash
pip install -e ".[docs]"
```

Then build the HTML site from this directory:

```bash
cd docs
make html
```

The rendered site is written to `build/html/index.html`. Open it in a browser to navigate the docs.

To start from a clean state:

```bash
make clean && make html
```

## Regenerating the API reference

The per-module `.rst` stubs under `source/api/` are produced by `sphinx-apidoc` from in-source Google-style docstrings.
They are checked into the repository so that builds work without first running `sphinx-apidoc`,
but they need to be refreshed whenever modules are added, removed, or renamed:

```bash
cd docs
sphinx-apidoc -o source/api ../xanesnet --separate --module-first --force
```

This rewrites `source/api/modules.rst` and every `source/api/xanesnet.*.rst` file.
Hand-maintained pages under `source/index.rst` and `source/user-guide/` are not touched.

## Layout

```
docs/
|-- Makefile              # `make html`, `make clean`, ...
|-- make.bat
|-- README.md
|-- source/
|   |-- conf.py           # Sphinx configuration
|   |-- index.rst         # landing page
|   |-- images/
|   |-- user-guide/
|   |   |-- intro.rst
|   |   |-- install.rst
|   |   |-- running.rst
|   |   |-- input.rst
|   |   |-- datasources.rst
|   |   |-- datasets.rst
|   |   |-- descriptors.rst
|   |   |-- graphs.rst
|   |   |-- encodings.rst
|   |   |-- models.rst
|   |   |-- batch-processors.rst
|   |   |-- training.rst
|   |   |-- losses.rst
|   |   |-- regularizers.rst
|   |   |-- early-stoppers.rst
|   |   |-- optimizers.rst
|   |   |-- lr-schedulers.rst
|   |   |-- activations.rst
|   |   |-- strategies.rst
|   |   |-- inference.rst
|   |   |-- analysis.rst
|   |   |-- config-editor.rst
|   |-- api/              # API reference
|       |-- modules.rst
|       |-- xanesnet.*.rst
|-- build/                # build output
```

## Conventions

- The theme is [sphinx-rtd-theme](https://sphinx-rtd-theme.readthedocs.io/).
- Docstrings are Google-style, parsed by `sphinx.ext.napoleon`.
- Type hints in signatures are rendered into the description by `sphinx-autodoc-typehints`.
- `autosummary_generate = True` is on so summary tables fill themselves in.
