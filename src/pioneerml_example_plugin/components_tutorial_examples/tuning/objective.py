from __future__ import annotations

"""Objective plugin that extracts a scalar score from a trained module."""

from pioneerml.integration.optuna.hpo.objective import BaseObjective
from pioneerml.integration.optuna.hpo.objective.factory.registry import REGISTRY as OBJECTIVE_REGISTRY


@OBJECTIVE_REGISTRY.register("sensor_health_val_loss")
class SensorHealthValLossObjective(BaseObjective):
    """Read the latest validation loss from a trained tutorial module.

    The smoke config is deliberately tiny, so the final train-loss fallback
    keeps one-batch experiments from reporting ``inf`` when validation history is
    unavailable.
    """

    def objective_from_module(self, module) -> float:
        values = getattr(module, "val_epoch_loss_history", None)
        if isinstance(values, list) and values:
            return float(values[-1])
        values = getattr(module, "val_loss_history", None)
        if isinstance(values, list) and values:
            return float(values[-1])
        values = getattr(module, "train_loss_history", None)
        if isinstance(values, list) and values:
            return float(values[-1])
        return float("inf")
