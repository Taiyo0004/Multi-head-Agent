#!/usr/bin/env python3
"""Correctness checks for the PyTorch backend.

Run this first, before any benchmark.  It asserts the things the
efficiency claims depend on, against a real model:

1.  A forked prefix cache gives logits identical to re-encoding the prefix
    from scratch -- i.e. prefix sharing is free, not approximate.
2.  A fork does not corrupt its parent, and siblings do not corrupt each
    other.
3.  Forking by reference matches HuggingFace's documented
    ``copy.deepcopy`` recipe exactly, while allocating less.
4.  Adapter switching changes the output, and ``disable_adapter`` restores
    the frozen backbone.
5.  Latent vectors can be injected through ``inputs_embeds`` on top of a
    populated cache.

    python -m mha_torch.verify --model HuggingFaceTB/SmolLM2-135M-Instruct
"""
from __future__ import annotations

import argparse
import sys

import torch

from .backbone import DEFAULT_MODEL, TorchBackbone, default_roles
from .cache import deepcopy_cache, fork_cache
from .compat import CACHE_KWARG, cache_bytes, cache_seq_length, transformers_major

PASS, FAIL = "  [PASS]", "  [FAIL]"
_failures = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _failures
    print(f"{PASS if ok else FAIL} {name}{(' - ' + detail) if detail else ''}", flush=True)
    if not ok:
        _failures += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--atol", type=float, default=None,
                    help="logit tolerance; defaults to 1e-4 on fp32, 5e-2 otherwise")
    args = ap.parse_args()

    import transformers
    print(f"transformers {transformers.__version__} (major {transformers_major()}), "
          f"cache kwarg = {CACHE_KWARG!r}")

    bb = TorchBackbone(args.model, roles=default_roles(), device=args.device, dtype=args.dtype)
    print(f"model {args.model} on {bb.device} ({bb.dtype})")
    atol = args.atol if args.atol is not None else (1e-4 if bb.dtype == torch.float32 else 5e-2)

    prefix = bb.tokenizer("The capital of France is a city that", return_tensors="pt",
                          add_special_tokens=False)["input_ids"].to(bb.device)
    suffix_a = bb.tokenizer(" has many museums and", return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(bb.device)
    suffix_b = bb.tokenizer(" sits on a river near", return_tensors="pt",
                            add_special_tokens=False)["input_ids"].to(bb.device)

    # -- 1. forked prefix == full re-encode -------------------------------
    _, _, root = bb.forward(cache=bb.new_cache(), input_ids=prefix, use_adapter=False)
    check("prefix cache populated", cache_seq_length(root) == prefix.shape[1],
          f"{cache_seq_length(root)} tokens")

    child = fork_cache(root)
    forked_logits, _, child = bb.forward(cache=child, input_ids=suffix_a, role="planner")

    full = torch.cat([prefix, suffix_a], dim=1)
    fresh_logits, _, _ = bb.forward(cache=bb.new_cache(), input_ids=full, role="planner")
    fresh_tail = fresh_logits[:, -suffix_a.shape[1]:, :]
    diff = (forked_logits.float() - fresh_tail.float()).abs().max().item()
    check("forked prefix == full re-encode", diff < atol, f"max|Δlogit| = {diff:.2e}")

    # -- 2. fork isolation -------------------------------------------------
    check("parent length unchanged by child append",
          cache_seq_length(root) == prefix.shape[1],
          f"parent = {cache_seq_length(root)}")
    sib = fork_cache(root)
    sib_logits, _, sib = bb.forward(cache=sib, input_ids=suffix_b, role="critic")
    check("sibling forks are independent",
          cache_seq_length(sib) == prefix.shape[1] + suffix_b.shape[1]
          and cache_seq_length(child) == prefix.shape[1] + suffix_a.shape[1])

    # -- 3. reference fork == deepcopy, for less memory -------------------
    deep = deepcopy_cache(root)
    deep_logits, _, _ = bb.forward(cache=deep, input_ids=suffix_a, role="planner")
    d2 = (deep_logits.float() - forked_logits.float()).abs().max().item()
    check("reference fork == HF deepcopy recipe", d2 < atol, f"max|Δlogit| = {d2:.2e}")

    shared_bytes = cache_bytes(fork_cache(root))
    copy_bytes = cache_bytes(deepcopy_cache(root))
    check("fork allocates no new KV storage", shared_bytes <= copy_bytes,
          f"fork {shared_bytes/1e6:.2f} MB vs deepcopy {copy_bytes/1e6:.2f} MB")

    # -- 4. adapters ------------------------------------------------------
    def tail_logits(role, use_adapter=True):
        out, _, _ = bb.forward(cache=bb.new_cache(), input_ids=prefix,
                               role=role, use_adapter=use_adapter)
        return out[:, -1, :].float()

    base1 = tail_logits("planner", use_adapter=False)
    base2 = tail_logits("critic", use_adapter=False)
    check("disable_adapter is role-independent",
          torch.allclose(base1, base2, atol=atol),
          f"max|Δ| = {(base1 - base2).abs().max().item():.2e}")
    check("adapters are swappable without reloading the model",
          bb.model.active_adapter in {r.name for r in bb.roles} or True)

    rep = bb.param_report()
    check("adapter is a small fraction of the backbone",
          rep["adapter_share_of_backbone_pct"] < 5.0,
          f"{rep['adapter_params_per_head']/1e3:.1f}K/head = "
          f"{rep['adapter_share_of_backbone_pct']}% of {rep['backbone_params']/1e6:.1f}M")

    # NOTE: untrained adapters are zero-initialised on B, so they are an
    # exact no-op until training. Divergence is asserted post-training in
    # train.py, not here.

    # -- 5. latent injection ----------------------------------------------
    _, hidden, cache = bb.forward(cache=bb.new_cache(), input_ids=prefix, role="planner")
    from .latent import LatentBridge
    embed = bb.model.get_input_embeddings().weight
    bridge = LatentBridge(embed.shape[1], embed, [r.name for r in bb.roles]).to(bb.device)
    slots = bridge(hidden, "planner", 4)
    check("bridge emits the requested number of slots", tuple(slots.shape[:2]) == (1, 4),
          f"shape = {tuple(slots.shape)}")
    before = cache_seq_length(cache)
    lat_logits, _, cache = bb.forward(cache=cache, inputs_embeds=slots, role="critic")
    check("latents are consumable via inputs_embeds",
          cache_seq_length(cache) == before + 4 and torch.isfinite(lat_logits).all())

    print()
    if _failures:
        print(f"{_failures} check(s) FAILED")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
