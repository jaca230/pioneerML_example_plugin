"""Small graph classifier used by tutorial pipelines."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torch_geometric.nn import global_mean_pool
except ImportError:  # pragma: no cover - tutorials require torch_geometric at runtime
    global_mean_pool = None


class DummyGraphClassifier(nn.Module):
    """Minimal graph-level classifier for synthetic tutorial data."""

    def __init__(
        self,
        node_dim: int = 5,
        edge_dim: int | None = None,
        hidden: int = 64,
        heads: int | None = None,
        num_blocks: int = 1,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        del edge_dim, heads

        layers: list[nn.Module] = []
        in_dim = node_dim
        for _ in range(max(1, num_blocks)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(hidden, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, batch) -> torch.Tensor:
        x = batch.x.float()
        batch_index = getattr(batch, "batch", None)

        if batch_index is None:
            graph_features = x.mean(dim=0, keepdim=True)
        elif global_mean_pool is not None:
            graph_features = global_mean_pool(x, batch_index)
        else:
            num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() else 1
            graph_features = x.new_zeros((num_graphs, x.shape[-1]))
            graph_features.index_add_(0, batch_index, x)
            counts = torch.bincount(batch_index, minlength=num_graphs).clamp_min(1)
            graph_features = graph_features / counts.to(x.device).unsqueeze(-1)

        return self.network(graph_features)
