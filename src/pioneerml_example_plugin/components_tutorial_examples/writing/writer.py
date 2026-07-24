from __future__ import annotations

"""Writer plugin and writer stages for tutorial prediction outputs."""

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import torch

from pioneerml.data_writer import OutputColumnSpec, OutputSchema, PredictionSet
from pioneerml.data_writer.factory.registry import REGISTRY as WRITER_REGISTRY
from pioneerml.data_writer.stage.stages import BaseWriterStage, EmitRunOutputsStage, InitRunStateStage
from pioneerml.data_writer.structured import StructuredDataWriter, WriterPhaseOrder, WriterPhaseStages


class BuildSensorPredictionTableStage(BaseWriterStage):
    """Convert a prediction set into an Arrow prediction table."""

    name = "build_sensor_prediction_table"
    requires = ("prediction_set",)
    provides = ("table",)

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        _ = owner
        prediction_set = state["prediction_set"]
        logits = prediction_set.model_outputs_by_name.get("main")
        if logits is None:
            raise RuntimeError("SensorHealthWriter expected a 'main' model output in the prediction set.")

        probs = torch.sigmoid(torch.as_tensor(logits)).detach().cpu().numpy().reshape(-1).astype(np.float32)
        event_np = np.asarray(prediction_set.prediction_event_ids_np, dtype=np.int64).reshape(-1)
        columns = {
            # The writer emits ordinary columns; the output backend decides
            # whether those columns become parquet, JSONL, or another format.
            "event_id": pa.array(event_np, type=pa.int64()),
            "failure_probability": pa.array(probs, type=pa.float32()),
            "predicted_failure": pa.array((probs >= 0.5).astype(np.int8), type=pa.int8()),
        }
        state["table"] = pa.table(columns)


class WriteSensorPredictionTableStage(BaseWriterStage):
    """Write or buffer an Arrow prediction table."""

    name = "write_sensor_prediction_table"
    requires = ("table", "output_dir")
    provides = ("written_prediction_paths",)

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        writer = owner
        output_dir = Path(state["output_dir"]).expanduser().resolve()
        src_path = Path(state.get("src_path") or "sensor_health.csv")
        output_path = state.get("output_path")
        if output_path:
            dst_path = Path(output_path).expanduser().resolve()
        else:
            dst_path = output_dir / f"{src_path.stem}_sensor_predictions{writer.output_backend.default_extension()}"

        if not bool(state.get("streaming", False)):
            # Unified inference sends one prediction set per model batch. For a
            # single requested output_path, non-streaming mode should emit one
            # complete file instead of overwriting that path for every batch.
            buffered = state.get("buffered_prediction_tables")
            if not isinstance(buffered, list):
                buffered = []
                state["buffered_prediction_tables"] = buffered
            buffered.append(state["table"])
            state["buffered_prediction_path"] = str(dst_path)
            return

        writer.write_table(table=state["table"], dst_path=dst_path)
        written = state.get("written_prediction_paths")
        if not isinstance(written, list):
            written = []
            state["written_prediction_paths"] = written
        written.append(str(dst_path))


class FlushSensorPredictionTablesStage(BaseWriterStage):
    """Write buffered non-streaming prediction tables as one output file."""

    name = "flush_sensor_prediction_tables"

    def run_writer(self, *, state: MutableMapping[str, Any], owner) -> None:
        writer = owner
        tables = state.get("buffered_prediction_tables")
        if not isinstance(tables, list) or len(tables) == 0:
            return
        dst_path = Path(state["buffered_prediction_path"]).expanduser().resolve()
        table = pa.concat_tables(tables, promote_options="default")
        writer.write_table(table=table, dst_path=dst_path)
        written = state.get("written_prediction_paths")
        if not isinstance(written, list):
            written = []
            state["written_prediction_paths"] = written
        written.append(str(dst_path))


@WRITER_REGISTRY.register("sensor_health_writer")
class SensorHealthWriter(StructuredDataWriter):
    """Tutorial writer that emits model scores as JSONL or any output backend.

    The staged writer mirrors the staged loader: start initializes run state,
    chunk stages transform and write each batch, and finalize publishes run
    outputs.
    """

    def output_schema(self) -> OutputSchema:
        """Declare the model-output fields this writer knows how to serialize."""

        return OutputSchema(
            fields=(
                OutputColumnSpec("failure_probability", model_output_name="main", output_index=0, dtype=np.float32),
            )
        )

    def default_stage_order(self) -> WriterPhaseOrder:
        return WriterPhaseOrder(
            start=["init_run_state"],
            chunk=["build_sensor_prediction_table", "write_sensor_prediction_table"],
            finalize=["flush_sensor_prediction_tables", "emit_run_outputs"],
        )

    def default_stages(self) -> WriterPhaseStages:
        return WriterPhaseStages(
            start={"init_run_state": InitRunStateStage()},
            chunk={
                "build_sensor_prediction_table": BuildSensorPredictionTableStage(),
                "write_sensor_prediction_table": WriteSensorPredictionTableStage(),
            },
            finalize={
                "flush_sensor_prediction_tables": FlushSensorPredictionTablesStage(),
                "emit_run_outputs": EmitRunOutputsStage(),
            },
        )

    def build_prediction_set(
        self,
        *,
        batch,
        model_output: Any,
        src_path: Path,
        num_rows: int,
        cfg: dict[str, Any] | None = None,
    ) -> PredictionSet:
        _ = cfg
        logits = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
        values = torch.as_tensor(logits).detach().cpu().numpy()
        cursor = int(getattr(self, "_prediction_event_cursor", 0))
        event_np = np.arange(cursor, cursor + int(values.shape[0]), dtype=np.int64)
        self._prediction_event_cursor = int(cursor + int(values.shape[0]))
        return PredictionSet(
            src_path=Path(src_path),
            prediction_event_ids_np=event_np,
            model_outputs_by_name={"main": values},
            num_rows=int(num_rows),
        )

    def on_start(self, *, state: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        self._prediction_event_cursor = 0
        return super().on_start(state=state)

    def chunk_state(
        self,
        *,
        prediction_set: PredictionSet,
        output_dir: Path,
        output_path: str | None,
        write_timestamped: bool,
        timestamp: str,
    ) -> dict[str, Any]:
        """Build the state consumed by the tutorial writer chunk stages."""

        prediction_set.validate()
        return {
            "prediction_set": prediction_set,
            "src_path": prediction_set.src_path,
            "num_rows": int(prediction_set.num_rows),
            "output_dir": output_dir,
            "output_path": output_path,
            "write_timestamped": bool(write_timestamped),
            "timestamp": timestamp,
            "streaming": bool(self.run_config.streaming),
        }
