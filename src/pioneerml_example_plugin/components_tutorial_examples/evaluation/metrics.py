from __future__ import annotations

"""Metric plugin for binary sensor-health predictions."""

from collections.abc import Mapping
from typing import Any

import torch

from pioneerml.evaluation.metrics import BaseMetric
from pioneerml.evaluation.metrics.registry import REGISTRY as METRIC_REGISTRY


@METRIC_REGISTRY.register("sensor_health_accuracy")
class SensorHealthAccuracyMetric(BaseMetric):
    """Compute a few binary-classification metrics from evaluator tensors.

    The evaluator owns model execution; this metric only reads tensors from the
    shared context and returns scalar values to merge into the report.
    """

    def compute(self, *, context: Mapping[str, Any]) -> dict[str, Any]:
        logits = torch.as_tensor(context["logits"]).float()
        targets = torch.as_tensor(context["targets"]).float()
        threshold = float(context.get("threshold", 0.5))
        probs = torch.sigmoid(logits)
        # Keep the threshold configurable so the same metric can be reused for
        # different operating points.
        preds = (probs >= threshold).to(torch.float32)
        correct = (preds == targets).to(torch.float32)
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()
        precision = tp / max(1.0, tp + fp)
        recall = tp / max(1.0, tp + fn)
        return {
            "sensor_accuracy": float(correct.mean().item()),
            "sensor_precision": float(precision),
            "sensor_recall": float(recall),
        }
