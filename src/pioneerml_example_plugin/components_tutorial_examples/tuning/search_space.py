from __future__ import annotations

"""Search-space plugin that maps JSON parameter specs to Optuna suggestions."""

from collections.abc import Mapping
from typing import Any

import optuna

from pioneerml.integration.optuna.hpo.search_space import BaseSearchSpace
from pioneerml.integration.optuna.hpo.search_space.factory.registry import REGISTRY as SEARCH_SPACE_REGISTRY
from pioneerml.integration.optuna.hpo.search_space.parameters import SearchParameterFactory


@SEARCH_SPACE_REGISTRY.register("sensor_health_search_space")
class SensorHealthSearchSpace(BaseSearchSpace):
    """Reusable search space for the tutorial model/module knobs.

    Search spaces translate compact JSON specs into Optuna suggestions. The HPO
    step later partitions suggested values between model, module, and runtime
    settings.
    """

    def default_search_space(self) -> dict[str, Any]:
        return {
            "hidden": {"type": "categorical", "choices": [8, 16, 24]},
            "lr": {"type": "sensor_centered_float", "center": 0.005, "radius": 0.004, "log": True},
        }

    def suggest(self, *, trial: optuna.Trial, search_space: Mapping[str, Any] | None = None) -> dict[str, Any]:
        specs = dict(self.default_search_space())
        specs.update(dict(search_space or {}))
        out: dict[str, Any] = {}
        for name, raw_spec in specs.items():
            if not isinstance(raw_spec, Mapping):
                # Non-mapping values are treated as fixed constants.
                out[str(name)] = raw_spec
                continue
            spec = dict(raw_spec)
            param_type = str(spec.pop("type", "fixed"))
            out[str(name)] = SearchParameterFactory(search_parameter_name=param_type).build(config=spec).suggest(
                trial=trial,
                name=str(name),
            )
        return out
