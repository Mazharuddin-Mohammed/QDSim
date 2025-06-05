# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'QDSim'
copyright = '2024, Dr. Mazharuddin Mohammed'
author = 'Dr. Mazharuddin Mohammed'
release = '2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser',
    'nbsphinx',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
]

templates_path = ['_templates']
exclude_patterns = []

# Mock imports for modules that might not be available during documentation build
autodoc_mock_imports = [
    'qdsim_cython',
    'qdsim_cython.solvers',
    'qdsim_cython.solvers.fixed_open_system_solver',
    'qdsim_cython.memory',
    'qdsim_cython.memory.advanced_memory_manager',
    'qdsim_cython.materials',
    'qdsim_cython.visualization',
    'qdsim_cython.visualization.wavefunction_plotter',
    'cupy',
    'pycuda',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom CSS
html_css_files = [
    'custom.css',
]

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
        'processEnvironments': True,
    },
    'options': {
        'ignoreHtmlClass': 'tex2jax_ignore',
        'processHtmlClass': 'tex2jax_process'
    }
}

# MyST parser configuration
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]

# NBSphinx configuration
nbsphinx_execute = 'never'
nbsphinx_allow_errors = True

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Todo extension
todo_include_todos = True

# Source file suffixes
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# Master document
master_doc = 'index'

# Language
language = 'en'

# Pygments style
pygments_style = 'sphinx'

# HTML title
html_title = f"{project} v{release}"

# HTML short title
html_short_title = project

# HTML logo
html_logo = '_static/logo.png'

# HTML favicon
html_favicon = '_static/favicon.ico'

# Show source link
html_show_sourcelink = True

# Show copyright
html_show_copyright = True

# Show sphinx
html_show_sphinx = True

# Search language
html_search_language = 'en'

# Output file base name for HTML help builder
htmlhelp_basename = 'QDSimdoc'

# LaTeX output
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': r'''
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
\usepackage{braket}
''',
}

# LaTeX documents
latex_documents = [
    (master_doc, 'QDSim.tex', 'QDSim Documentation',
     'QDSim Development Team', 'manual'),
]

# Manual page output
man_pages = [
    (master_doc, 'qdsim', 'QDSim Documentation',
     [author], 1)
]

# Texinfo output
texinfo_documents = [
    (master_doc, 'QDSim', 'QDSim Documentation',
     author, 'QDSim', 'Advanced Quantum Dot Simulator.',
     'Miscellaneous'),
]

# Epub output
epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
