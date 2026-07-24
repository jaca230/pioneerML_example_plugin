from __future__ import annotations

"""Trainer plugin with small CPU-first defaults for tutorial runs."""

from typing import Any

from pioneerml.integration.pytorch.trainers import LightningModuleTrainer
from pioneerml.integration.pytorch.trainers.factory.registry import REGISTRY as TRAINER_REGISTRY


@TRAINER_REGISTRY.register("sensor_health_trainer")
class SensorHealthTrainer(LightningModuleTrainer):
    """Lightning trainer wrapper with tiny defaults for tutorial smoke runs.

    Real projects often use the stock ``lightning_module`` trainer. This wrapper
    demonstrates how a plugin can set project-specific defaults while still
    allowing config to override them.
    """

    def __init__(
        self,
        *,
        trainer_kwargs: dict[str, Any] | None = None,
        early_stopping_cfg: dict[str, Any] | None = None,
    ) -> None:
        merged = {
            # Keep tutorial runs deterministic and CPU-friendly in notebooks and
            # CI-like smoke tests.
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 2,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "num_sanity_val_steps": 0,
        }
        merged.update(dict(trainer_kwargs or {}))
        super().__init__(trainer_kwargs=merged, early_stopping_cfg=early_stopping_cfg)
