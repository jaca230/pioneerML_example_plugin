# Component Tutorial Examples

This package turns the manual's component skeletons into a small runnable
project. The theme is a synthetic sensor-health classifier:

- read the bundled `sample_data/sensor_health.csv` file of sensor readings;
- load each row as a one-node graph;
- train a small graph model;
- evaluate metrics and a plot;
- export and reload the model;
- write predictions through a custom writer and output backend;
- ask one HPO trial for suggested parameters.

The implementation is grouped by pipeline phase:

- `data_loading/`
- `modeling/`
- `evaluation/`
- `export/`
- `inference/`
- `tuning/`
- `writing/`
- `pipeline/`

Start with the notebooks in `notebooks/`. The runnable config lives in
`pipeline/config.json` and is consumed by the standard PioneerML
`training_pipeline` and `inference_pipeline`.

The inference example includes a custom batch executor. It delegates the normal
input/model/writer sequence to `BaseInferenceBatchExecutor` and uses the success
hook only to validate the sensor model's output count and finite values.

For a bottom-up view of the same config shape, open
`notebooks/04_generate_pipeline_config.ipynb`.
