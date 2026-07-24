from __future__ import annotations

"""Small builders that explain the shape of the checked-in pipeline config."""

from copy import deepcopy
from typing import Any


def sensor_loader_manager_config(*, mode: str = "train", batch_size: int = 8) -> dict[str, Any]:
    """Build the loader-manager block reused by training, evaluation, and export.

    The important idea is that data access is configured once as an
    ``input_sources_spec`` plus an ``input_backend``. Individual pipeline steps
    then choose which loader purpose they need by naming entries such as
    ``train_loader`` or ``test_loader``.
    """

    return {
        "type": "sensor_tutorial",
        "config": {
            "input_sources_spec": {
                "main_sources": ["sample_data/sensor_health.csv"],
                "optional_sources_by_name": {},
                "source_type": "file",
            },
            "input_backend": {"type": "sensor_csv", "config": {"rows_per_chunk": 64}},
            "defaults": {
                "type": "sensor_health_loader",
                "config": {
                    "batch_size": int(batch_size),
                    "chunk_row_groups": 2,
                    "chunk_workers": 0,
                    "mode": str(mode),
                    "shuffle_batches": False,
                    "shuffle_within_batch": False,
                    "train_fraction": 0.75,
                    "val_fraction": 0.25,
                    "test_fraction": 0.0,
                    "sample_fraction": 1.0,
                    "split_seed": 13,
                },
            },
            "loaders": {},
        },
    }


def sensor_modeling_blocks() -> dict[str, dict[str, Any]]:
    """Build the architecture/compiler/module/trainer blocks.

    These are the blocks consumed by the standard ``training_pipeline`` train
    step. Each ``type`` is a registry name, and each nested ``config`` dict is
    passed to that component factory.
    """

    return {
        "architecture": {
            "type": "sensor_health_mlp",
            "config": {
                "node_dim": 3,
                "edge_dim": 1,
                "hidden": 16,
                "dropout": 0.0,
                "output_dim": 1,
                "feature_mean": [68.0, 0.85, 3.30],
                "feature_std": [9.0, 0.65, 0.18],
            },
        },
        "compiler": {"type": "sensor_health_noop", "config": {"tag": "tutorial"}},
        "module": {
            "type": "sensor_health_module",
            "config": {
                "loss": {"type": "sensor_health_bce", "config": {"positive_weight": 1.0}},
                "lr": 0.01,
                "weight_decay": 0.0,
                "max_step_history": 128,
                "max_epoch_history": 16,
            },
        },
        "trainer": {
            "type": "sensor_health_trainer",
            "config": {
                "trainer_kwargs": {"max_epochs": 25, "limit_train_batches": 20, "limit_val_batches": 10},
                "early_stopping": {
                    "enabled": False,
                    "type": "relative",
                    "config": {
                        "monitor": "train_loss",
                        "mode": "min",
                        "patience": 1,
                        "min_delta": 0.0,
                        "strict": False,
                        "check_finite": True,
                        "verbose": False,
                    },
                },
            },
        },
    }


def build_training_config() -> dict[str, Any]:
    """Build the ``training`` section used by ``training_pipeline``."""

    train_loader = sensor_loader_manager_config(mode="train", batch_size=8)
    train_loader["config"]["loaders"] = {
        "train_loader": {"config": {"mode": "train", "split": "train", "shuffle_batches": True, "log_diagnostics": False}},
        "val_loader": {"config": {"mode": "train", "split": "val", "shuffle_batches": False, "log_diagnostics": False}},
    }

    hpo_loader = deepcopy(train_loader)
    eval_loader = sensor_loader_manager_config(mode="train", batch_size=8)
    eval_loader["config"]["loaders"] = {
        "test_loader": {"config": {"mode": "train", "split": "val", "shuffle_batches": False, "log_diagnostics": False}}
    }
    export_loader = sensor_loader_manager_config(mode="train", batch_size=1)
    export_loader["config"]["defaults"]["config"]["chunk_row_groups"] = 1
    export_loader["config"]["loaders"] = {
        "export_loader": {"config": {"mode": "train", "shuffle_batches": False, "log_diagnostics": False}}
    }

    hpo = {
        **sensor_modeling_blocks(),
        "loader_manager": hpo_loader,
        "hpo": {
            "type": "sensor_health_hpo",
            "config": {
                "enabled": True,
                "n_trials": 1,
                "direction": "minimize",
                "seed": 31,
                "study_name": "sensor_health_tutorial_bce",
                "storage": None,
                "fallback_dir": None,
                "allow_schema_fallback": True,
                "objective": {"type": "sensor_health_val_loss", "config": {}},
                "search_space": {
                    "type": "sensor_health_search_space",
                    "config": {
                        "search_space": {
                            "hidden": {"type": "categorical", "choices": [16, 32]},
                            "lr": {"type": "sensor_centered_float", "center": 0.01, "radius": 0.008, "log": True},
                        }
                    },
                },
            },
        },
    }
    train = {**sensor_modeling_blocks(), "loader_manager": train_loader}

    return {
        "hpo": hpo,
        "train": train,
        "evaluate": {
            "evaluator": {
                "type": "sensor_health_evaluator",
                "config": {
                    "threshold": 0.5,
                    "metrics": ["sensor_health_accuracy"],
                    "plots": ["sensor_health_score_histogram"],
                    "plot_path": "tutorial_outputs/sensor_score_histogram.png",
                },
            },
            "loader_manager": eval_loader,
        },
        "export": {
            "exporter": {
                "type": "sensor_health_state_dict",
                "config": {
                    "enabled": True,
                    "export_dir": "tutorial_outputs/exports",
                    "filename_prefix": "sensor_health",
                    "prefer_cuda": False,
                },
            },
            "loader_manager": export_loader,
        },
    }


def build_inference_config() -> dict[str, Any]:
    """Build the ``inference`` section used by ``inference_pipeline``."""

    inference_loader = sensor_loader_manager_config(mode="inference", batch_size=8)
    inference_loader["config"]["loaders"] = {
        "inference_loader": {"config": {"mode": "inference", "shuffle_batches": False, "log_diagnostics": False}}
    }
    return {
        "model_handle_builder": {
            "model_handle": {
                "type": "sensor_health_state_dict",
                "config": {"model_path": "tutorial_outputs/exports/model.pt"},
            }
        },
        "inference": {
            "runtime": {"prefer_cuda": False},
            "batch_executor": {
                "type": "sensor_health_batch_executor",
                "config": {"require_finite": True},
            },
            "writer": {
                "type": "sensor_health_writer",
                "config": {
                    "output_backend": {"type": "sensor_jsonl", "config": {}},
                    "fallback_output_dir": "tutorial_outputs/predictions",
                    "output_dir": "tutorial_outputs/predictions",
                    "output_path": "tutorial_outputs/predictions/sensor_predictions.jsonl",
                    "streaming": False,
                    "write_timestamped": False,
                    "timestamp": None,
                    "writer_params": {},
                },
            },
            "loader_manager": inference_loader,
        },
    }


def build_full_config() -> dict[str, Any]:
    """Build the full checked-in tutorial config."""

    return {"training": build_training_config(), "inference": build_inference_config()}


__all__ = [
    "build_full_config",
    "build_inference_config",
    "build_training_config",
    "sensor_loader_manager_config",
    "sensor_modeling_blocks",
]
