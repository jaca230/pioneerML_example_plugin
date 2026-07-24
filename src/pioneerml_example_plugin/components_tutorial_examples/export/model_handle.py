from __future__ import annotations

"""Model-handle plugin paired with the tutorial state-dict exporter."""

import torch

from pioneerml.integration.pytorch.model_handles import BaseModelHandle
from pioneerml.integration.pytorch.model_handles.registry import REGISTRY as MODEL_HANDLE_REGISTRY

from ..modeling.architecture import SensorHealthMLP


@MODEL_HANDLE_REGISTRY.register("sensor_health_state_dict")
class SensorHealthStateDictHandle(BaseModelHandle):
    """Load the tutorial state-dict artifact produced by the custom exporter.

    Model handles are inference-time adapters: pipeline config points at an
    exported artifact, and the handle returns a ready-to-call model object.
    """

    TYPE = "sensor_health_state_dict"

    def load(self, *, device: torch.device):
        """Recreate the architecture, load weights, and switch to eval mode."""

        payload = torch.load(str(self.path), map_location=device)
        model = SensorHealthMLP(**dict(payload.get("model_config") or {}))
        model.load_state_dict(payload["state_dict"])
        model.to(device)
        model.eval()
        return model
