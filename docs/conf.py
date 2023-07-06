# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
from typing import Type

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

sys.path.insert(0, os.path.abspath('../python'))

# Initialize attached doc strings.
try:
    import vineyard  # noqa: E402
    version = vineyard.__version__
except ImportError:
    version = '0.0.0'

# -- Project information -----------------------------------------------------

project = 'vineyard'
copyright = '2020-2023, The Vineyard Authors'
author = 'The Vineyard Authors'

language = 'en'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'breathe',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx_copybutton",
    'sphinx_panels',
    'sphinxemoji.sphinxemoji',
    "sphinxext.opengraph",
]
suppress_warnings = []

# breathe
breathe_projects = {
    'vineyard': os.path.abspath('./_build/doxygen/xml'),
}
breathe_default_project = 'vineyard'
breathe_debug_trace_directives = True
breathe_debug_trace_doxygen_ids = True
breathe_debug_trace_qualification = True

# myst-parser
myst_enable_extensions = [
    "amsmath",
    "attrs_inline",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]
suppress_warnings.append('myst.xref_missing')

# jupyter notebooks
nbsphinx_execute = 'never'

source_suffix = ['.rst', '.md']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

master_doc = "docs"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    '.ipynb_checkpoints',
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": True,  # we use the logo
    "navigation_with_keys": True,
    "source_repository": "https://github.com/v6d-io/v6d/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/v6d-io/v6d",
            "html": "",
            "class": "fa fa-solid fa-github fa-2x",
        },
    ],
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    "images/",
    "_static/",
]

html_css_files = [
    "css/brands.min.css",    # font-awesome
    "css/v4-shims.min.css",  # font-awesome
    "css/custom.css",
    "css/panels.css",
    "css/index.css",
]

html_extra_path = [
    "artifacts/",
    "./CNAME",
    "./.nojekyll",
    "./summer.html",
]

html_additional_pages = {'index': 'index.html'}

html_title = "Vineyard"
html_logo = "images/vineyard-logo-h.png"
html_favicon = "images/vineyard.ico"

html_show_copyright= True
html_show_sphinx = False
html_last_updated = True

# add copy button to code blocks, exclude the notebook (nbsphinx) prompts
copybutton_selector = "div.notranslate:not(.prompt) div.highlight pre"

# -- patch sphinx ---------------------------------------------------------

try:
    from docutils import nodes
    from docutils.nodes import Element
    from sphinx import __display_version__
    from sphinx.builders.html import StandaloneHTMLBuilder
    from sphinx.writers.html5 import HTML5Translator

    # -- patch spinx-panels: allow link_button without URI ----------------
    class HTML5TranslatorPatched(HTML5Translator):
        def visit_reference(self, node: Element) -> None:
            atts = {'class': 'reference'}
            if node.get('internal') or 'refuri' not in node:
                atts['class'] += ' internal'
            else:
                atts['class'] += ' external'
            if 'refuri' in node:
                atts['href'] = node['refuri'] or '#'
                if self.settings.cloak_email_addresses and atts['href'].startswith('mailto:'):
                    atts['href'] = self.cloak_mailto(atts['href'])
                    self.in_mailto = True

                # patch: erase empty refuri
                if atts['href'] in ['', '""', '#']:
                    del atts['href']
            else:
                assert 'refid' in node, \
                    'References must have "refuri" or "refid" attribute.'
                atts['href'] = '#' + node['refid']
            if not isinstance(node.parent, nodes.TextElement):
                assert len(node) == 1 and isinstance(node[0], nodes.image)
                atts['class'] += ' image-reference'
            if 'reftitle' in node:
                atts['title'] = node['reftitle']
            if 'target' in node:
                atts['target'] = node['target']
            self.body.append(self.starttag(node, 'a', '', **atts))

            if node.get('secnumber'):
                self.body.append(('%s' + self.secnumber_suffix) %
                                '.'.join(map(str, node['secnumber'])))

    class StandaloneHTMLBuilderPatched(StandaloneHTMLBuilder):
        @property
        def default_translator_class(self) -> "Type[nodes.NodeVisitor]":
            return HTML5TranslatorPatched

    def setup(app):
        app.add_builder(StandaloneHTMLBuilderPatched, override=True)
        return {'version': __display_version__, 'parallel_read_safe': True}
except:  # noqa: E722, pylint: disable=bare-except
    pass
