from __future__ import annotations

"""Compiler plugin that demonstrates the compile hook without changing models."""

from typing import Any

from pioneerml.integration.pytorch.compilers import BaseCompiler
from pioneerml.integration.pytorch.compilers.factory.registry import REGISTRY as COMPILER_REGISTRY


@COMPILER_REGISTRY.register("sensor_health_noop")
class SensorHealthNoopCompiler(BaseCompiler):
    """A transparent compiler that records tutorial metadata.

    A compiler plugin can wrap, optimize, or replace the model before training.
    This one deliberately does no optimization so readers can see the extension
    point without introducing torch.compile behavior.
    """

    def compile(self, *, model: Any, context: str = "run") -> Any:
        # Storing the context makes the no-op visible in tests/debugging.
        setattr(model, "tutorial_compile_context", str(context))
        return model
