from __future__ import annotations

"""Config helpers and standard pipeline exports for the tutorial package."""

from copy import deepcopy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pioneerml.pipeline.pipelines.inference import inference_pipeline
from pioneerml.pipeline.pipelines.training import training_pipeline

from ..utils import DEFAULT_CONFIG_PATH
from .config_builder import (
    build_full_config,
    build_inference_config,
    build_training_config,
    sensor_loader_manager_config,
    sensor_modeling_blocks,
)

MODEL_KEY = "components_tutorial_examples"


def load_config() -> dict[str, Any]:
    """Load the checked-in tutorial pipeline config.

    The notebooks also build a similar config in Python. Keeping this loader
    tiny makes it clear that PioneerML ultimately just needs a JSON-compatible
    dictionary keyed by pipeline step names.
    """

    return dict(json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")))


def with_output_root(config: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    """Redirect tutorial output paths for smoke tests or temporary notebook runs."""

    cfg = deepcopy(dict(config))
    root = Path(output_root).expanduser().resolve()
    training = cfg.get("training") if isinstance(cfg.get("training"), dict) else cfg
    if not isinstance(training, dict):
        return cfg

    evaluate = training.get("evaluate")
    if isinstance(evaluate, dict):
        evaluator = evaluate.get("evaluator")
        if isinstance(evaluator, dict):
            evaluator_cfg = dict(evaluator.get("config") or {})
            evaluator_cfg["plot_path"] = str(root / "sensor_score_histogram.png")
            evaluator["config"] = evaluator_cfg

    export = training.get("export")
    if isinstance(export, dict):
        exporter = export.get("exporter")
        if isinstance(exporter, dict):
            exporter_cfg = dict(exporter.get("config") or {})
            exporter_cfg["export_dir"] = str(root / "exports")
            exporter["config"] = exporter_cfg

    return cfg


def with_inference_paths(
    config: Mapping[str, Any],
    *,
    model_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Point inference config at a model artifact and local prediction output."""

    cfg = deepcopy(dict(config))
    root = Path(output_root).expanduser().resolve()
    nested = cfg.get("inference")
    if "model_handle_builder" in cfg:
        inference = cfg
    elif isinstance(nested, dict) and "model_handle_builder" in nested:
        inference = nested
    else:
        inference = cfg
    if not isinstance(inference, dict):
        return cfg

    model_handle_builder = inference.get("model_handle_builder")
    if isinstance(model_handle_builder, dict):
        model_handle = model_handle_builder.get("model_handle")
        if isinstance(model_handle, dict):
            model_handle_cfg = dict(model_handle.get("config") or {})
            model_handle_cfg["model_path"] = str(Path(model_path).expanduser().resolve())
            model_handle["config"] = model_handle_cfg

    inference_step = inference.get("inference")
    if isinstance(inference_step, dict):
        writer = inference_step.get("writer")
        if isinstance(writer, dict):
            writer_cfg = dict(writer.get("config") or {})
            predictions_dir = root / "predictions"
            writer_cfg["fallback_output_dir"] = str(predictions_dir)
            writer_cfg["output_dir"] = str(predictions_dir)
            writer_cfg["output_path"] = str(predictions_dir / "sensor_predictions.jsonl")
            writer["config"] = writer_cfg

    return cfg


__all__ = [
    "MODEL_KEY",
    "build_full_config",
    "build_inference_config",
    "build_training_config",
    "inference_pipeline",
    "load_config",
    "sensor_loader_manager_config",
    "sensor_modeling_blocks",
    "training_pipeline",
    "with_inference_paths",
    "with_output_root",
]
