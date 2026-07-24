from __future__ import annotations

"""LightningModule plugin that joins model, loss, optimizer, and metrics."""

from collections.abc import Mapping
from typing import Any

from pioneerml.integration.pytorch.losses import LossFactory
from pioneerml.integration.pytorch.modules import GraphLightningModule
from pioneerml.integration.pytorch.modules.factory.registry import REGISTRY as MODULE_REGISTRY

from .losses import SensorHealthBCELoss


@MODULE_REGISTRY.register("sensor_health_module")
class SensorHealthModule(GraphLightningModule):
    """Tutorial LightningModule with sensible defaults for the sensor task.

    ``training_pipeline`` creates the model first and then injects it into this
    module through ``from_factory``. The loss can be supplied either as an
    already-built object or as a nested plugin block in config.
    """

    @classmethod
    def from_factory(cls, *, config: Mapping[str, Any] | None = None, **kwargs):
        merged = {**dict(config or {}), **dict(kwargs)}
        merged.pop("namespace", None)
        merged.pop("name", None)
        merged.pop("config", None)
        model = merged.pop("model")
        loss_fn = merged.pop("loss_fn", None)
        loss_spec = merged.pop("loss", None)
        if loss_fn is None:
            if isinstance(loss_spec, Mapping):
                # This mirrors the common config pattern: nested component
                # blocks use {"type": registry_name, "config": {...}}.
                loss_type = str(loss_spec.get("type", "sensor_health_bce"))
                loss_cfg = dict(loss_spec.get("config") or {})
                loss_fn = LossFactory(loss_name=loss_type).build(config=loss_cfg)
            else:
                loss_fn = SensorHealthBCELoss()
        return cls(model=model, loss_fn=loss_fn, **merged)
