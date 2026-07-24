# tutorial notebooks

Use the `Python (pioneerml)` kernel created by the pioneerML environment setup
scripts. The framework is installed in editable mode by those scripts, and each
notebook calls `ensure_plugins_loaded()` before importing tutorial plugin code.
No notebook-specific `sys.path` edits are required.

Tutorial notebooks:
- `src/pioneerml_example_plugin/tutorial_examples/notebooks/00_quickstart.ipynb`
- `src/pioneerml_example_plugin/tutorial_examples/notebooks/01_building_zenml_pipelines.ipynb`
- `src/pioneerml_example_plugin/tutorial_examples/notebooks/04_custom_metrics_and_plots.ipynb`

Plugin repo:
- https://github.com/jaca230/pioneerML_example_plugin
