from __future__ import annotations

"""Search-parameter plugin used inside the tutorial search space."""

import optuna

from pioneerml.integration.optuna.hpo.search_space.parameters import BaseSearchParameter
from pioneerml.integration.optuna.hpo.search_space.parameters.factory.registry import (
    REGISTRY as SEARCH_PARAMETER_REGISTRY,
)


@SEARCH_PARAMETER_REGISTRY.register("sensor_centered_float")
class SensorCenteredFloatParameter(BaseSearchParameter):
    """Suggest a float inside center +/- radius.

    This is intentionally redundant with generic float search in spirit; it
    demonstrates how teams can name domain-specific parameter distributions.
    """

    def __init__(self, *, center: float, radius: float, log: bool = False) -> None:
        self.center = float(center)
        self.radius = abs(float(radius))
        self.log = bool(log)

    def suggest(self, *, trial: optuna.Trial, name: str) -> float:
        low = max(1e-12, self.center - self.radius) if self.log else self.center - self.radius
        high = self.center + self.radius
        return float(trial.suggest_float(name, low, high, log=self.log))
