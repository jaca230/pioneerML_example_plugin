from __future__ import annotations

"""CSV input backend used by the tutorial loader manager.

This file demonstrates the lowest-level data-loading extension point: turning
one or more configured sources into Arrow tables that loader stages can consume.
"""

from collections.abc import Iterator
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

from pioneerml.data_loader.loaders.input_source.backends import InputBackend
from pioneerml.data_loader.loaders.input_source.factory.registry import REGISTRY as INPUT_BACKEND_REGISTRY


@INPUT_BACKEND_REGISTRY.register("sensor_csv")
class SensorCSVInputBackend(InputBackend):
    """Small CSV input backend used by the component tutorial.

    CSV is intentionally not the production default in PioneerML, but it is a
    friendly format for onboarding because readers can inspect and edit it by
    hand.
    """

    def __init__(self, *, rows_per_chunk: int = 32) -> None:
        # PioneerML asks input backends for Arrow tables. This knob lets the
        # tutorial simulate chunking without requiring a real multi-row-group
        # parquet dataset.
        self.rows_per_chunk = max(1, int(rows_per_chunk))

    @staticmethod
    def _read_source(path: str) -> pa.Table:
        return pacsv.read_csv(str(Path(path).expanduser().resolve())).combine_chunks()

    def schema_fields_intersection(self, sources: tuple[str, ...]) -> set[str]:
        """Report columns common to all sources for schema validation."""

        fields: set[str] | None = None
        for source in sources:
            names = set(self._read_source(source).column_names)
            fields = names if fields is None else fields & names
        return set(fields or set())

    def iter_tables(
        self,
        *,
        sources: tuple[str, ...],
        fields: list[str],
        row_groups_per_chunk: int,
    ) -> Iterator[pa.Table]:
        chunk_rows = max(1, int(row_groups_per_chunk)) * self.rows_per_chunk
        selected_fields = [str(field) for field in fields]
        for source in sources:
            table = self._read_source(source)
            if selected_fields:
                table = table.select(selected_fields)
            # Yield small Arrow slices; downstream loader stages do not need to
            # know that this started as CSV instead of parquet.
            for start in range(0, int(table.num_rows), chunk_rows):
                yield table.slice(start, chunk_rows).combine_chunks()

    def count_rows_per_source(self, *, sources: tuple[str, ...]) -> list[int]:
        return [int(self._read_source(source).num_rows) for source in sources]
