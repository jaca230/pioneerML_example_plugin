from __future__ import annotations

"""Exporter plugin that writes a tutorial state-dict artifact."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from pioneerml.integration.pytorch.exporters import BaseExporter
from pioneerml.integration.pytorch.exporters.factory.registry import REGISTRY as EXPORTER_REGISTRY


@EXPORTER_REGISTRY.register("sensor_health_state_dict")
class SensorHealthStateDictExporter(BaseExporter):
    """Export a lightweight state-dict checkpoint for the tutorial model.

    This intentionally uses a plain state dict rather than TorchScript so the
    paired model handle can show exactly how architecture config and weights are
    restored.
    """

    @property
    def export_type(self) -> str:
        return "sensor_state_dict"

    @property
    def artifact_suffix(self) -> str:
        return "sensor_state.pt"

    def export(
        self,
        *,
        model_obj: Any,
        output_path: Path,
        prefer_cuda: bool,
        cfg: Mapping[str, Any],
        dataset: Any,
        loader_provider: Any,
    ) -> None:
        _ = prefer_cuda, dataset, loader_provider
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Store constructor settings next to weights. Without this, the model
        # handle would not know which hidden size/output dimension to recreate.
        model_config = (
            model_obj.tutorial_config()
            if hasattr(model_obj, "tutorial_config")
            else dict(cfg.get("model_config") or {})
        )
        torch.save(
            {
                "state_dict": model_obj.state_dict(),
                "model_config": model_config,
                "export_type": self.export_type,
            },
            str(output_path),
        )
