"""Sphinx configuration for tracketch documentation."""

import os
import sys
from datetime import datetime

# Add the project root so Sphinx can find the tracketch package
sys.path.insert(0, os.path.abspath(".."))

project = "tracketch"
author = "Jeppe Brage Christensen"
author_email = "jeppe.christensen@psi.ch"
institute = "Paul Scherrer Institute (PSI)"
release = "1.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "publication"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "icons/logo.png"
html_css_files = ["theme_overrides.css"]
html_context = {
    "author_name": author,
    "author_email": author_email,
    "institute": institute,
    "compilation_date": datetime.now().strftime("%Y-%m-%d"),
}
html_theme_options = {
    "logo_only": True,
}

# autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# autosummary
autosummary_generate = True

# napoleon (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# LaTeX output settings
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": (
        "\\usepackage[utf8]{inputenc}\n"
        "\\usepackage{newunicodechar}\n"
        "\\newunicodechar{\u03c3}{\\ensuremath{\\sigma}}\n"
        "\\newunicodechar{\u00b1}{\\ensuremath{\\pm}}"
    ),
}
latex_documents = [
    (
        "index",
        "tracketch.tex",
        "tracketch Documentation",
        rf"{author}\\{institute}",
        "manual",
    ),
]


def copy_pdf_to_docs(app, exception):
    """Copy compiled PDF to docs folder after build."""
    if exception is None and app.builder.name == "latex":
        import shutil

        pdf_src = os.path.join(app.outdir, "tracketch.pdf")
        pdf_dst = os.path.join(os.path.dirname(__file__), "tracketch.pdf")
        if os.path.exists(pdf_src):
            shutil.copy2(pdf_src, pdf_dst)
            print(f"Copied PDF to {pdf_dst}")


def setup(app):
    """Register Sphinx event handlers."""
    app.connect("build-finished", copy_pdf_to_docs)
