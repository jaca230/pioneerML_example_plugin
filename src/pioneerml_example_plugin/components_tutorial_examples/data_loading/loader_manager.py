from __future__ import annotations

"""Loader-manager customization for package-local sample data paths."""

from collections.abc import Mapping
from typing import Any

from pioneerml.data_loader.manager.config_loader_manager import ConfigLoaderManager
from pioneerml.data_loader.manager.factory.registry import REGISTRY as LOADER_MANAGER_REGISTRY

from ..utils import resolve_tutorial_file_path


@LOADER_MANAGER_REGISTRY.register("sensor_tutorial")
@LOADER_MANAGER_REGISTRY.register("sensor_health_loader_manager")
class SensorTutorialLoaderManager(ConfigLoaderManager):
    """Config loader manager with tutorial-friendly defaults.

    Most projects can use the core ``config`` loader manager directly. This
    tutorial subclass exists so relative sample-data paths in the packaged JSON
    resolve before ``InputSourceSet`` checks that files exist.
    """

    def __init__(self, *, config: Mapping[str, Any] | None = None) -> None:
        super().__init__(config=config)
        self._resolve_tutorial_input_sources()

    @staticmethod
    def _default_chunk_workers() -> int:
        return 0

    def _resolve_tutorial_input_sources(self) -> None:
        """Make package-local file paths valid from any current directory."""

        source_spec = self.config.get("input_sources_spec")
        if not isinstance(source_spec, Mapping):
            return
        if str(source_spec.get("source_type", "file")).strip().lower() != "file":
            return

        resolved_spec = dict(source_spec)
        main_sources = resolved_spec.get("main_sources")
        if isinstance(main_sources, list):
            resolved_spec["main_sources"] = [str(resolve_tutorial_file_path(source)) for source in main_sources]

        optional_sources = resolved_spec.get("optional_sources_by_name")
        if isinstance(optional_sources, Mapping):
            resolved_optional: dict[str, list[str] | None] = {}
            for name, sources in optional_sources.items():
                if sources is None:
                    resolved_optional[str(name)] = None
                elif isinstance(sources, list):
                    resolved_optional[str(name)] = [str(resolve_tutorial_file_path(source)) for source in sources]
                else:
                    resolved_optional[str(name)] = sources
            resolved_spec["optional_sources_by_name"] = resolved_optional

        self.config["input_sources_spec"] = resolved_spec

    def resolve_loader_params(self, **kwargs) -> dict[str, Any]:
        """Fill small-run defaults after the core config manager merges blocks."""

        params = super().resolve_loader_params(**kwargs)
        params.setdefault("shuffle_batches", False)
        params.setdefault("shuffle_within_batch", False)
        params.setdefault("drop_remainders", False)
        params.setdefault("log_diagnostics", False)
        return params
