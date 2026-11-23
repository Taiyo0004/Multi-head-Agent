"""Multi-head agent: one frozen backbone, PEFT role heads, shared KV cache.

Imports are lazy (PEP 562) so that torch-free modules -- ``data`` above
all -- can be imported without pulling in torch, transformers and peft.
That keeps dataset work and its tests runnable in a bare environment.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

_EXPORTS = {
    "RoleSpec": "mha_torch.backbone",
    "TorchBackbone": "mha_torch.backbone",
    "default_roles": "mha_torch.backbone",
    "pick_device": "mha_torch.backbone",
    "fork_cache": "mha_torch.cache",
    "deepcopy_cache": "mha_torch.cache",
    "LatentBridge": "mha_torch.latent",
    "RunMetrics": "mha_torch.metrics",
    "summarize": "mha_torch.metrics",
    "MultiHeadAgent": "mha_torch.pipeline",
    "PipelineConfig": "mha_torch.pipeline",
    "PipelineResult": "mha_torch.pipeline",
    "Task": "mha_torch.pipeline",
}

__all__ = sorted(_EXPORTS)
__version__ = "0.2.0"

if TYPE_CHECKING:  # pragma: no cover
    from .backbone import RoleSpec, TorchBackbone, default_roles, pick_device
    from .cache import deepcopy_cache, fork_cache
    from .latent import LatentBridge
    from .metrics import RunMetrics, summarize
    from .pipeline import MultiHeadAgent, PipelineConfig, PipelineResult, Task


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
