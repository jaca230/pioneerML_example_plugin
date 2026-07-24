from __future__ import annotations

"""A small model-specific inference batch executor tutorial."""

from collections.abc import Mapping
from typing import Any

import numpy as np

from pioneerml.data_writer import PredictionSet
from pioneerml.inference import BaseInferenceBatchExecutor, InferenceBatchContext
from pioneerml.inference.batch_executor.factory.registry import REGISTRY


@REGISTRY.register("sensor_health_batch_executor")
class SensorHealthBatchExecutor(BaseInferenceBatchExecutor):
    """Run standard inference and validate sensor-health predictions.

    ``BaseInferenceBatchExecutor`` already performs input conversion, model
    execution, prediction conversion, and writer emission. This tutorial
    customizes only the success hook, where model-specific output invariants
    belong.
    """

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self.require_finite = bool(self.config.get("require_finite", True))

    def handle_success(
        self,
        *,
        context: InferenceBatchContext,
        prediction_set: PredictionSet,
    ) -> None:
        logits = prediction_set.model_outputs_by_name.get("main")
        if logits is None:
            raise RuntimeError("Sensor-health inference requires the 'main' model output.")

        values = np.asarray(logits).reshape(-1)
        expected = int(getattr(context.batch, "num_graphs", values.size))
        if values.size != expected:
            raise RuntimeError(
                f"Sensor-health inference produced {values.size} predictions "
                f"for a batch containing {expected} graphs."
            )
        if self.require_finite and not np.isfinite(values).all():
            raise RuntimeError("Sensor-health inference produced a non-finite prediction.")
