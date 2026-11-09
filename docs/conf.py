"""Sphinx configuration of the PyPU documentation.

Build locally with ``sphinx-build -b html docs docs/_build/html``; add
``-D plot_gallery=0`` to skip running the examples.
"""

import sys
from pathlib import Path

import pypu

DOCS = Path(__file__).parent
sys.path.insert(0, str(DOCS / "sphinxext"))

# ---------------------------------------------------------------- project

project = "PyPU"
author = "Ilias Zavitsanos"
copyright = "2026, Ilias Zavitsanos"
release = pypu.__version__
version = ".".join(release.split(".")[:2])

# ---------------------------------------------------------------- general

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "sphinxext"]

# ``code`` in reStructuredText (single backticks) is Python code, as in the docstrings.
default_role = "code"

myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Copy code without the ">>> " and "... " prompts of the doctest-style blocks.
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# ---------------------------------------------------------------- API reference

autosummary_generate = True
autodoc_typehints = "none"  # the docstrings document the types

# Members are listed by the autosummary templates, so numpydoc does not list them again.
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = "all"
numpydoc_xref_aliases = {
    "ndarray": "numpy.ndarray",
    "DataFrame": "pandas.DataFrame",
    "RandomState": "numpy.random.RandomState",
    "Generator": "numpy.random.Generator",
    "estimator": ":term:`estimator <sklearn:estimator>`",
    "BasePUClassifier": "pypu.base.BasePUClassifier",
    "BasePriorEstimator": "pypu.prior.BasePriorEstimator",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "lightgbm": ("https://lightgbm.readthedocs.io/en/stable", None),
    "xgboost": ("https://xgboost.readthedocs.io/en/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

# ---------------------------------------------------------------- examples gallery

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "plot_gallery": "True",
    # Getting started first; within a section, the order of examples/README.rst.
    "subsection_order": [
        "../examples/getting_started",
        "../examples/models",
        "../examples/comparison",
    ],
    "within_subsection_order": "gallery_order.ReadmeOrder",
    # "Examples using X" on the API pages.
    "doc_module": ("pypu",),
    "backreferences_dir": "api/generated/backreferences",
    "remove_config_comments": True,
    "show_memory": False,
}

# ---------------------------------------------------------------- HTML

html_theme = "pydata_sphinx_theme"
html_title = "PyPU"
html_show_sourcelink = False
html_static_path = ["_static"]
html_favicon = "_static/logo.svg"

html_theme_options = {
    # The logo's colors read on both the light and the dark theme.
    "logo": {
        "image_light": "_static/logo.svg",
        "image_dark": "_static/logo.svg",
        "text": "PyPU",
        "alt_text": "PyPU - Home",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/izavits/PyPU",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pypu-learn/",
            "icon": "fa-brands fa-python",
        },
    ],
    # Installation, User guide, Examples, API reference, Changelog; the rest under "More".
    "header_links_before_dropdown": 5,
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "secondary_sidebar_items": ["page-toc", "sourcelink"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
}

# Pages without a left sidebar: the home page and the flat project pages.
html_sidebars = {
    "index": [],
    "install": [],
    "changelog": [],
    "contributing": [],
    "citing": [],
    "license": [],
}
