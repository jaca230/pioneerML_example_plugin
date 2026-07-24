from __future__ import annotations

"""Evaluator plugin that prepares shared metric and plot context."""

from collections.abc import Mapping

import torch

from pioneerml.evaluation.evaluators import BaseEvaluator
from pioneerml.evaluation.evaluators.factory.registry import REGISTRY as EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("sensor_health_evaluator")
class SensorHealthEvaluator(BaseEvaluator):
    """Evaluation loop for the tutorial binary graph task.

    Evaluators gather tensors once and hand a context dictionary to registered
    metric and plot plugins. That makes metrics/plots composable instead of
    embedding every calculation in the evaluator itself.
    """

    default_metric_names = ("sensor_health_accuracy",)
    default_plot_names = ("sensor_health_score_histogram",)

    def build_context(self, *, module, loader, config: Mapping[str, object]) -> dict[str, object]:
        """Run the module over a loader and prepare shared metric/plot state."""

        module.eval()
        device = next(module.parameters()).device
        logits_chunks: list[torch.Tensor] = []
        target_chunks: list[torch.Tensor] = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                raw = module(batch)
                loss, _ = module.compute_loss(raw, batch)
                preds = module.primary_predictions(raw)
                target = module.primary_target(batch, preds)
                bs = int(target.shape[0])
                total_loss += float(loss.detach().cpu().item()) * bs
                total_samples += bs
                logits_chunks.append(preds.detach().cpu())
                target_chunks.append(target.detach().cpu())

        if total_samples <= 0:
            raise RuntimeError("SensorHealthEvaluator saw no samples.")
        logits = torch.cat(logits_chunks, dim=0)
        targets = torch.cat(target_chunks, dim=0)
        threshold = float(config.get("threshold", 0.5))
        plot_path = self.resolve_plot_path(dict(config))
        return {
            # Base metrics are copied directly into the final results.
            "base_metrics": {
                "loss": float(total_loss / total_samples),
                "num_eval_samples": int(total_samples),
            },
            "metric_context": {
                "logits": logits,
                "targets": targets,
                "threshold": threshold,
            },
            "plot_kwargs_by_name": {
                # Plot kwargs are keyed by plot plugin name so different plots
                # can receive different inputs from the same evaluation pass.
                "sensor_health_score_histogram": {
                    "logits": logits,
                    "targets": targets,
                    "save_path": plot_path,
                    "show": False,
                }
            },
        }

    def finalize_results(
        self,
        *,
        results: dict[str, object],
        context: Mapping[str, object],
        config: Mapping[str, object],
    ) -> dict[str, object]:
        _ = context, config
        path = results.get("sensor_health_score_histogram_path")
        if isinstance(path, str):
            results["score_histogram_path"] = path
        return results
