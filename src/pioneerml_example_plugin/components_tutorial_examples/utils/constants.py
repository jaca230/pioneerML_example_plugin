from __future__ import annotations

"""Shared schema names and package-local paths for the tutorial."""

from pathlib import Path

# These names are intentionally reused across the loader, sample data, and
# notebook examples so learners can follow one small schema end to end.
FEATURE_COLUMNS = ("temperature", "vibration", "voltage")
TARGET_COLUMN = "failure"
EVENT_ID_COLUMN = "event_id"

# The tutorial package root, not this utils directory. Keeping paths relative to
# the package makes the examples work from a checkout, an editable install, or a
# wheel that includes the sample CSV.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PACKAGE_DIR / "sample_data" / "sensor_health.csv"
DEFAULT_OUTPUT_DIR = "tutorial_outputs"
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "pipeline" / "config.json"


def resolve_tutorial_file_path(path: str | Path) -> Path:
    """Resolve file paths used in tutorial configs.

    Config files are easier to read when they can say
    ``sample_data/sensor_health.csv``. Core PioneerML validates file paths
    before the input backend reads them, so the tutorial loader manager uses
    this helper to turn those package-local strings into absolute paths.
    """

    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return (PACKAGE_DIR / candidate).resolve()
