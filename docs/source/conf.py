# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for the XANESNET documentation."""

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root to sys.path so autodoc can import xanesnet
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "XANESNET"
copyright = "2026, Hendrik Junkawitsch"
author = "Hendrik Junkawitsch"
release = "0.1.0-alpha"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    # Core autodoc: generates API docs from docstrings
    "sphinx.ext.autodoc",
    # Napoleon: support for Google-style and NumPy-style docstrings
    "sphinx.ext.napoleon",
    # Viewcode: adds links to highlighted source code
    "sphinx.ext.viewcode",
    # Intersphinx: cross-reference other projects (Python, NumPy, PyTorch, …)
    "sphinx.ext.intersphinx",
    # Autosummary: generate summary tables for modules/classes
    "sphinx.ext.autosummary",
    # Typehints: render PEP-484 type annotations in the docs
    "sphinx_autodoc_typehints",
    # todo: collect TODO notes across the codebase
    "sphinx.ext.todo",
    # mathjax: render LaTeX math
    "sphinx.ext.mathjax",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Silence warnings we cannot fix locally:
# - Duplicate object descriptions are emitted by sphinx-apidoc when the same
#   class is reachable through both a package's ``__init__`` re-export and the
#   submodule that defines it. Both descriptions point at the same Python
#   object, so the warning is cosmetic.
# - ``sphinx_autodoc_typehints`` emits forward-reference warnings for
#   ``pymatgen.core.structure`` types (``CompositionLike``, ``ArrayLike``)
#   that pymatgen only resolves at type-checking time.
suppress_warnings = [
    "ref.duplicate_object_description",
    "sphinx_autodoc_typehints.forward_reference",
    "sphinx_autodoc_typehints.guarded_import",
]

# -- Autosummary -------------------------------------------------------------
autosummary_generate = True  # auto-generate stub .rst files
autosummary_imported_members = False  # only document explicitly defined members

# -- Autodoc -----------------------------------------------------------------
autodoc_default_options = {
    "members": True,  # document all public members
    "undoc-members": True,  # include members without docstrings
    "private-members": False,  # skip _private members
    "special-members": "__init__",  # document __init__
    "inherited-members": False,
    "show-inheritance": True,  # show class inheritance
    "ignore-module-all": True,  # don't re-document __all__ re-exports at the package level
}
autodoc_typehints = "description"  # render type hints in the description section
autodoc_typehints_description_target = "documented"
autodoc_member_order = "bysource"  # preserve source order

# -- Napoleon (Google-style docstrings) --------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # disable NumPy style to avoid ambiguity
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True  # avoid duplicate object descriptions for dataclass attrs
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_attr_annotations = True

# -- Intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- TODO extension ----------------------------------------------------------
todo_include_todos = True

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = []

html_title = f"{project} {version}"

# -- Options for LaTeX output ------------------------------------------------
latex_elements: dict = {}
