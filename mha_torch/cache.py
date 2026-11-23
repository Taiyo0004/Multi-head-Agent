"""Forking a prefilled KV cache without copying it.

HuggingFace's documented recipe for reusing a prefilled prompt cache is
``copy.deepcopy(prompt_cache)`` once per continuation.  That is correct but
duplicates every prefix tensor.

It is also unnecessary.  ``DynamicLayer.update`` does
``self.keys = torch.cat([self.keys, key_states], dim=-2)``: it allocates a
new tensor and rebinds, never writing into the existing one.  So several
children can hold references to the *same* prefix tensors safely -- the
first append on any child rebinds that child's own attribute and leaves the
shared tensor untouched.

``fork_cache`` therefore copies only the container and the per-layer
objects, sharing the tensors.  ``verify.py`` asserts this produces logits
identical to the deepcopy route, and ``deepcopy_cache`` is kept so the two
can be benchmarked against each other.

Caveat worth stating in the write-up: because HF caches are
concatenation-based, the memory saving lasts until each child's first
append, after which that child holds a full-length tensor of its own.  The
*compute* saving -- never re-encoding the shared prefix through the
backbone -- is permanent, and it is the dominant cost.  A genuinely paged
cache (see ``multihead_agent/kv_cache.py``, the NumPy reference) is what
fixes the memory side too.
"""
from __future__ import annotations

import copy
from typing import Any

from transformers import DynamicCache

from .compat import cache_layers


def fork_cache(cache: DynamicCache) -> DynamicCache:
    """Branch a prefilled cache, sharing prefix tensors by reference."""
    child = copy.copy(cache)

    layers = cache_layers(cache)
    if layers:
        child.layers = [copy.copy(layer) for layer in layers]
        return child

    # transformers v4: parallel lists of tensors, replaced (not mutated) on
    # update, so copying the lists is enough to isolate the children.
    for attr in ("key_cache", "value_cache"):
        existing = getattr(cache, attr, None)
        if existing is not None:
            setattr(child, attr, list(existing))
    return child


def deepcopy_cache(cache: DynamicCache) -> DynamicCache:
    """The HuggingFace-documented route, kept for comparison."""
    return copy.deepcopy(cache)


def crop_to(cache: DynamicCache, length: int) -> DynamicCache:
    """Roll a cache back to ``length`` tokens, where the version supports it."""
    from .compat import cache_seq_length

    current = cache_seq_length(cache)
    if current <= length:
        return cache
    try:
        cache.crop(-(current - length))
    except Exception as exc:  # pragma: no cover - version dependent
        raise RuntimeError(
            "this transformers version cannot crop the cache; fork from a "
            "freshly prefilled prefix instead"
        ) from exc
    return cache
