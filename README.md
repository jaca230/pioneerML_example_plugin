# pioneerML Example Plugin

Examples and tutorial components for learning the pioneerML plugin API.

This plugin contains:

- component tutorial examples
- tutorial notebooks
- small support models used by the tutorials

## Setup

Place this repository under a pioneerML checkout:

```text
plugins/example_plugin/
```

Then include it on `PYTHONPATH` with the framework:

```bash
export PYTHONPATH="<pioneerML>/src:<pioneerML>/plugins/example_plugin/src:$PYTHONPATH"
```

The plugin manifest is `plugin.json`; the Python module is
`pioneerml_example_plugin`.
