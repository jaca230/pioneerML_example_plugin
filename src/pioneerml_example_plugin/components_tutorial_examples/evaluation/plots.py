from __future__ import annotations

"""Plot plugin for visualizing tutorial prediction scores."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from pioneerml.evaluation.plots import BasePlot
from pioneerml.evaluation.plots.registry import REGISTRY as PLOT_REGISTRY


@PLOT_REGISTRY.register("sensor_health_score_histogram")
class SensorHealthScoreHistogramPlot(BasePlot):
    """Render a small histogram of predicted failure probabilities.

    Plot plugins are useful when the evaluator can supply common tensors but
    each project wants different visual diagnostics.
    """

    name = "sensor_health_score_histogram"

    def render(
        self,
        *,
        logits,
        targets=None,
        save_path: str | None = None,
        show: bool = False,
        title: str = "Sensor Health Scores",
    ) -> str | None:
        probs = torch.sigmoid(torch.as_tensor(logits).float()).detach().cpu().numpy().reshape(-1)
        target_values = None if targets is None else torch.as_tensor(targets).detach().cpu().numpy().reshape(-1)
        fig, ax = plt.subplots(figsize=(5, 3))
        if target_values is None:
            ax.hist(probs, bins=12, color="#3b82f6", alpha=0.85)
        else:
            # Splitting the histogram by label makes calibration mistakes easier
            # to see than a single aggregate bar chart.
            ax.hist(probs[target_values < 0.5], bins=12, alpha=0.75, label="healthy")
            ax.hist(probs[target_values >= 0.5], bins=12, alpha=0.75, label="warning")
            ax.legend(loc="best")
        ax.set_title(title)
        ax.set_xlabel("Predicted failure probability")
        ax.set_ylabel("Count")
        fig.tight_layout()

        out: str | None = None
        if save_path:
            path = Path(save_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            out = str(path)
        if show:
            plt.show()
        plt.close(fig)
        return out
