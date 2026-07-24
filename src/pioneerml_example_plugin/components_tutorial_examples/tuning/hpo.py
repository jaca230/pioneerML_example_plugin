from __future__ import annotations

"""HPO plugin with tutorial defaults for objective and search space."""

from collections.abc import Mapping
from typing import Any

from pioneerml.integration.optuna.hpo import ConfigHPO
from pioneerml.integration.optuna.hpo.factory.registry import REGISTRY as HPO_REGISTRY


@HPO_REGISTRY.register("sensor_health_hpo")
class SensorHealthHPO(ConfigHPO):
    """Config HPO with tutorial defaults for objective and search space.

    This subclass shows how a project can provide friendly defaults while still
    accepting standard ConfigHPO arguments from pipeline JSON.
    """

    def __init__(
        self,
        *,
        objective: Mapping[str, Any] | None = None,
        search_space: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            objective=objective
            or {
                "type": "sensor_health_val_loss",
                "config": {},
            },
            search_space=search_space
            or {
                "type": "sensor_health_search_space",
                "config": {"search_space": {}},
            },
            **kwargs,
        )
