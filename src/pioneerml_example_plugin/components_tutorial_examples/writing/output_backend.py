from __future__ import annotations

"""Output-backend plugin that serializes prediction tables as JSONL."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa

from pioneerml.data_writer.backends import OutputBackend
from pioneerml.data_writer.backends.factory.registry import REGISTRY as OUTPUT_BACKEND_REGISTRY


@dataclass
class _JsonLinesSink:
    """Open files used by the streaming writer API."""

    dst_path: Path
    part_path: Path
    handle: Any


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


@OUTPUT_BACKEND_REGISTRY.register("sensor_jsonl")
class SensorJsonLinesOutputBackend(OutputBackend):
    """Write Arrow tables as newline-delimited JSON for easy inspection.

    Output backends isolate the storage format. The writer builds an Arrow table;
    this backend decides how that table lands on disk.
    """

    def default_extension(self) -> str:
        return ".jsonl"

    def write_table_atomic(self, *, table: pa.Table, dst_path: Path) -> None:
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        # Write to a .part file and rename at the end so readers never observe a
        # half-written JSONL file.
        with part_path.open("w", encoding="utf-8") as handle:
            for row in table.to_pylist():
                handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
        os.replace(part_path, dst_path)

    def open_sink(self, *, dst_path: Path) -> Any:
        """Open a streaming sink used when predictions are emitted chunk by chunk."""

        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        part_path = dst_path.with_suffix(dst_path.suffix + ".part")
        handle = part_path.open("w", encoding="utf-8")
        return _JsonLinesSink(dst_path=dst_path, part_path=part_path, handle=handle)

    def append_chunk(self, *, sink: Any, table: pa.Table) -> None:
        if not isinstance(sink, _JsonLinesSink):
            raise TypeError(f"Expected _JsonLinesSink, got {type(sink).__name__}.")
        for row in table.to_pylist():
            sink.handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")

    def close_sink(self, *, sink: Any) -> None:
        if not isinstance(sink, _JsonLinesSink):
            raise TypeError(f"Expected _JsonLinesSink, got {type(sink).__name__}.")
        sink.handle.close()
        os.replace(sink.part_path, sink.dst_path)
