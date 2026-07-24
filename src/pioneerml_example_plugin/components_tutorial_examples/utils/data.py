from __future__ import annotations

"""Optional synthetic data generator used for tutorial experimentation."""

import csv
from pathlib import Path

import numpy as np

from .constants import EVENT_ID_COLUMN, FEATURE_COLUMNS, TARGET_COLUMN


def make_sensor_health_rows(*, num_rows: int = 640, seed: int = 7) -> list[dict[str, float | int]]:
    """Create a small, deterministic binary-classification dataset.

    The theme is a sensor-monitoring task: each row represents one sensor
    reading, and the target marks whether that reading should trigger a
    maintenance warning.
    """

    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for event_id in range(int(num_rows)):
        # The values are synthetic but shaped like a real binary maintenance
        # task: higher vibration, unusual voltage, and high temperature make a
        # warning more likely.
        temperature = float(rng.normal(68.0, 9.0))
        vibration = float(rng.gamma(shape=1.7, scale=0.55))
        voltage = float(rng.normal(3.30, 0.18))
        temperature_score = (temperature - 68.0) / 9.0
        vibration_score = (vibration - 0.85) / 0.65
        low_voltage_score = (3.30 - voltage) / 0.18
        risk = (
            2.0 * temperature_score
            + 3.0 * vibration_score
            + 1.25 * low_voltage_score
            + float(rng.normal(0.0, 0.12))
        )
        failure = int(risk > 0.15)
        rows.append(
            {
                EVENT_ID_COLUMN: int(event_id),
                FEATURE_COLUMNS[0]: round(temperature, 6),
                FEATURE_COLUMNS[1]: round(vibration, 6),
                FEATURE_COLUMNS[2]: round(voltage, 6),
                TARGET_COLUMN: int(failure),
            }
        )
    return rows


def write_sensor_health_csv(
    path: str | Path,
    *,
    num_rows: int = 640,
    seed: int = 7,
    overwrite: bool = True,
) -> Path:
    """Write optional dummy tutorial data and return the resolved path.

    The checked-in tutorial uses the bundled CSV in ``sample_data/``. This
    helper is kept for notebook experiments where a learner wants a fresh
    deterministic toy file with a different size or seed.
    """

    out_path = Path(path).expanduser().resolve()
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = make_sensor_health_rows(num_rows=num_rows, seed=seed)
    fieldnames = [EVENT_ID_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path
