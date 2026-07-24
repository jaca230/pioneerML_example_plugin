from __future__ import annotations

"""Graph loader and loader stage for the sensor-health tutorial.

The loader shows how source columns become staged arrays, then PyG graph
batches. The graph is intentionally tiny: one row becomes one graph with one
node.
"""

from collections.abc import MutableMapping
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from pioneerml.data_loader.loaders.array_store import NDArrayColumnSpec
from pioneerml.data_loader.loaders.array_store.ndarray_store import NDArrayStore
from pioneerml.data_loader.loaders.array_store.schemas import FeatureSchema, LoaderSchema, TargetSchema
from pioneerml.data_loader.loaders.config import DataFlowConfig, GraphTensorDims, SplitSampleConfig
from pioneerml.data_loader.loaders.factory.registry import REGISTRY as LOADER_REGISTRY
from pioneerml.data_loader.loaders.input_source import InputBackend, InputSourceSet, create_input_backend
from pioneerml.data_loader.loaders.stage.stages import (
    BaseLoaderStage,
    BatchPackStage,
    ExtractFeaturesStage,
    RowFilterStage,
    RowShuffleStage,
)
from pioneerml.data_loader.loaders.structured.graph import GraphLoader
from pioneerml.staged_runtime.stage_observers import StageObserver

from ..utils import EVENT_ID_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN


class SensorHealthData(Data):
    """PyG data object that keeps source event ids stable when batching."""

    def __inc__(self, key, value, *args, **kwargs):
        if str(key) in {"graph_event_id", "source_event"}:
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class SensorGraphBuildStage(BaseLoaderStage):
    """Turn each CSV row into a one-node graph.

    A production graph loader usually has richer layout logic. This stage keeps
    the shape simple so the tutorial can focus on where custom stages plug in.
    """

    name = "build_sensor_graph"
    requires = ("features_in", "n_rows")
    provides = ("layout", "x_out", "edge_attr_out", "edge_index_out", "graph_event_id", "y_graph")

    def __init__(self, *, input_state_key: str = "features_in") -> None:
        self.input_state_key = str(input_state_key)

    def run_loader(self, *, state: MutableMapping[str, Any], owner) -> None:
        store = state.get(self.input_state_key)
        if not isinstance(store, NDArrayStore):
            raise RuntimeError(f"{self.name} expected NDArrayStore at state['{self.input_state_key}'].")

        n_rows = int(state["n_rows"])
        # One CSV row becomes one graph with one node. Therefore x_out has
        # shape [num_graphs, node_features] and edge tensors are intentionally
        # empty.
        features = [
            np.asarray(store.values(field), dtype=np.float32).reshape(n_rows)
            for field in FEATURE_COLUMNS
        ]
        x_out = np.stack(features, axis=1).astype(np.float32, copy=False)
        event_ids = np.asarray(store.values(EVENT_ID_COLUMN), dtype=np.int64).reshape(n_rows).copy()

        layout = {
            "total_graphs": int(n_rows),
            "node_ptr": np.arange(n_rows + 1, dtype=np.int64),
            "edge_ptr": np.zeros((n_rows + 1,), dtype=np.int64),
        }
        state["layout"] = layout
        state["x_out"] = x_out
        state["edge_attr_out"] = np.zeros((0, 1), dtype=np.float32)
        state["edge_index_out"] = np.zeros((2, 0), dtype=np.int64)
        state["graph_event_id"] = event_ids

        if bool(getattr(owner, "include_targets", False)) and store.has_raw(NDArrayStore.values_key(TARGET_COLUMN)):
            target = np.asarray(store.values(TARGET_COLUMN), dtype=np.float32).reshape(n_rows, 1).copy()
            state["y_graph"] = target
        else:
            state["y_graph"] = None


@LOADER_REGISTRY.register("sensor_health_loader")
class SensorHealthLoader(GraphLoader):
    """Structured graph loader for the tutorial sensor-health CSV task."""

    NODE_FEATURE_DIM = 3
    EDGE_FEATURE_DIM = 1
    TARGET_DIM = 1

    def __init__(
        self,
        input_sources: InputSourceSet,
        *,
        mode: str = GraphLoader.MODE_TRAIN,
        input_backend: InputBackend | None = None,
        input_backend_name: str = "sensor_csv",
        data_flow_config: DataFlowConfig | None = None,
        split_config: SplitSampleConfig | None = None,
        graph_dims: GraphTensorDims | None = None,
        stage_overrides: dict[str, BaseLoaderStage] | None = None,
        stage_observer: StageObserver | None = None,
        profiling: dict[str, Any] | None = None,
    ) -> None:
        self.graph_dims = graph_dims or GraphTensorDims(
            node_feature_dim=self.NODE_FEATURE_DIM,
            edge_feature_dim=self.EDGE_FEATURE_DIM,
            graph_target_dim=self.TARGET_DIM,
        )
        self.schema = self.input_schema()

        include_targets = str(mode).strip().lower() != str(self.MODE_INFERENCE).lower()
        resolved_backend = input_backend if input_backend is not None else create_input_backend(input_backend_name)
        declared_specs = self.schema.to_column_specs(include_targets=True)
        # The backend resolves declared fields against source schemas. This is
        # the handoff where logical loader fields become concrete input columns.
        self._resolved_field_specs = resolved_backend.resolve_declared_field_specs(
            input_sources=input_sources,
            field_specs=declared_specs,
            include_targets=include_targets,
        )

        super().__init__(
            input_sources=input_sources,
            input_backend=resolved_backend,
            resolved_field_specs=self._resolved_field_specs,
            mode=mode,
            data_flow_config=data_flow_config,
            split_config=split_config,
            stage_overrides=stage_overrides,
            stage_observer=stage_observer,
            profiling=profiling,
        )

    def input_schema(self) -> LoaderSchema:
        """Declare how raw source columns map into loader state arrays."""

        features = FeatureSchema(
            fields=(
                NDArrayColumnSpec(column=EVENT_ID_COLUMN, field=EVENT_ID_COLUMN, dtype=np.int64),
                NDArrayColumnSpec(column=FEATURE_COLUMNS[0], field=FEATURE_COLUMNS[0], dtype=np.float32),
                NDArrayColumnSpec(column=FEATURE_COLUMNS[1], field=FEATURE_COLUMNS[1], dtype=np.float32),
                NDArrayColumnSpec(column=FEATURE_COLUMNS[2], field=FEATURE_COLUMNS[2], dtype=np.float32),
            )
        )
        targets = TargetSchema(
            fields=(
                NDArrayColumnSpec(
                    column=TARGET_COLUMN,
                    field=TARGET_COLUMN,
                    dtype=np.float32,
                    target_only=True,
                ),
            )
        )
        return LoaderSchema(features=features, targets=targets)

    def default_stage_order(self) -> list[str]:
        """Keep the stage chain visible for tutorial readers."""

        return [
            "row_filter",
            "row_shuffle",
            "extract_features",
            "build_sensor_graph",
            "pack_batch",
        ]

    def default_stages(self) -> dict[str, BaseLoaderStage]:
        """Instantiate the stages named in ``default_stage_order``."""

        return {
            "row_filter": RowFilterStage(event_id_column=EVENT_ID_COLUMN, split_config=self.split_config),
            "row_shuffle": RowShuffleStage(),
            "extract_features": ExtractFeaturesStage(
                column_specs=self.schema.to_column_specs(include_targets=True),
                output_state_key="features_in",
            ),
            "build_sensor_graph": SensorGraphBuildStage(input_state_key="features_in"),
            "pack_batch": BatchPackStage(
                tensor_state_fields={
                    "x_node": "x_out",
                    "x_edge": "edge_attr_out",
                    "edge_index": "edge_index_out",
                    "graph_event_id": "graph_event_id",
                },
                tensor_layout_fields={"node_ptr": "node_ptr", "edge_ptr": "edge_ptr"},
                scalar_state_fields={"num_rows": "n_rows"},
                scalar_layout_fields={"num_graphs": "total_graphs"},
                optional_tensor_state_fields={"y_graph": "y_graph"},
            ),
        }

    def _slice_chunk_batch(self, chunk: dict, g0: int, g1: int) -> Data:
        """Return tutorial graph batches with non-incremented event ids."""

        data = super()._slice_chunk_batch(chunk, g0, g1)
        values = data.to_dict()
        if "graph_event_id" in values:
            values["source_event"] = values.pop("graph_event_id")
        return SensorHealthData(**values)

    def build_inference_model_input(
        self,
        *,
        batch,
        device: torch.device,
        cfg: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Adapt inference batches to this tutorial model's forward signature.

        ``GraphLoader`` defaults to returning graph tensors as separate
        positional arguments. ``SensorHealthMLP`` is deliberately simpler for
        the tutorial: it accepts the PyG batch directly, just like training.
        """

        _ = cfg
        return (batch.to(device, non_blocking=(device.type == "cuda")),), {}
