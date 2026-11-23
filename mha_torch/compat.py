"""Version shims for transformers v4 and v5.

v5 renamed the cache argument on ``forward`` (``past_key_values`` ->
``cache``), changed ``DynamicCache`` construction (it now takes ``config``)
and moved per-layer storage from parallel ``key_cache``/``value_cache``
lists onto ``DynamicLayer`` objects with ``keys``/``values`` attributes.

Rather than pin one version, everything version-specific is isolated here
and resolved by inspection at import time.
"""
from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache

__all__ = ["CACHE_KWARG", "new_cache", "cache_kwargs", "cache_from_output",
           "cache_layers", "cache_seq_length", "cache_bytes", "layer_tensors",
           "transformers_major"]


def transformers_major() -> int:
    import transformers
    return int(transformers.__version__.split(".")[0])


def _detect_cache_kwarg() -> str:
    """Name of the forward() keyword that accepts a Cache object."""
    try:
        from transformers import AutoModelForCausalLM  # noqa: F401
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
        params = inspect.signature(LlamaForCausalLM.forward).parameters
        if "cache" in params:
            return "cache"
        if "past_key_values" in params:
            return "past_key_values"
    except Exception:  # pragma: no cover - depends on installed version
        pass
    return "cache" if transformers_major() >= 5 else "past_key_values"


CACHE_KWARG = _detect_cache_kwarg()


def new_cache(config: Any = None) -> DynamicCache:
    """Construct a DynamicCache on either API."""
    if config is not None:
        try:
            return DynamicCache(config=config)
        except TypeError:
            pass
    return DynamicCache()


def cache_kwargs(cache: Optional[DynamicCache]) -> Dict[str, Any]:
    """Keyword dict for passing ``cache`` into a model forward call."""
    return {} if cache is None else {CACHE_KWARG: cache}


def cache_from_output(outputs: Any) -> Optional[DynamicCache]:
    for attr in ("cache", "past_key_values"):
        got = getattr(outputs, attr, None)
        if got is not None:
            return got
    return None


def cache_layers(cache: DynamicCache) -> List[Any]:
    """The per-layer objects, whatever the version calls them."""
    layers = getattr(cache, "layers", None)
    return list(layers) if layers is not None else []


def layer_tensors(cache: DynamicCache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    layers = cache_layers(cache)
    if layers:
        out = []
        for layer in layers:
            k, v = getattr(layer, "keys", None), getattr(layer, "values", None)
            if k is not None and v is not None:
                out.append((k, v))
        return out
    # transformers v4 layout
    keys = getattr(cache, "key_cache", []) or []
    values = getattr(cache, "value_cache", []) or []
    return [(k, v) for k, v in zip(keys, values) if k is not None]


def cache_seq_length(cache: Optional[DynamicCache]) -> int:
    if cache is None:
        return 0
    try:
        got = cache.get_seq_length()
        if got:
            return int(got)
    except Exception:
        pass
    tensors = layer_tensors(cache)
    return int(tensors[0][0].shape[-2]) if tensors else 0


def cache_bytes(cache: Optional[DynamicCache]) -> int:
    """Bytes of distinct K/V storage, counting shared tensors only once.

    Uses the storage pointer so a cache forked by reference is not
    double-counted -- the whole point of the fork is that those bytes exist
    once.
    """
    if cache is None:
        return 0
    seen: Dict[int, int] = {}
    for k, v in layer_tensors(cache):
        for t in (k, v):
            seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())
