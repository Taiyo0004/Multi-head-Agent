#!/usr/bin/env python3
"""Efficiency benchmark for the PyTorch backend: all four modes.

Also measures the cache fork itself against HuggingFace's documented
``copy.deepcopy`` prefix-reuse recipe, which is the thing this repo
replaces.

    python -m mha_torch.bench --tasks 8 --repeats 2
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

from .backbone import DEFAULT_MODEL, TorchBackbone, default_roles
from .cache import deepcopy_cache, fork_cache
from .compat import cache_bytes
from .metrics import summarize
from .pipeline import MultiHeadAgent, PipelineConfig, Task

PROMPTS = [
    "Compute the total cost of three items priced 4.50, 12.00 and 7.25 with 8 percent sales tax.",
    "A train leaves at 09:15 and arrives at 13:40 after two stops of 12 minutes each. Find the moving time.",
    "Order these by time complexity: binary search, bubble sort, hash lookup, merge sort.",
    "Explain why a read replica does not fix write amplification in a sharded database.",
    "Design a rate limiter for 1000 requests per minute per API key with burst tolerance.",
    "Given monthly revenue 12k, 15k, 11k, 19k, compute the quarter-over-quarter growth rate.",
    "Rewrite a nested triple loop over a 10k row table into a single aggregation pass.",
    "Estimate the memory of a KV cache for 32 layers, 32 heads, head dim 128, 4096 context.",
]
PAD = (" Context: the service runs on a two-node cluster, ships weekly, and has a "
       "p99 latency budget of 250 ms per request.")

def build_modes(stop_on_eos: bool):
    return [PipelineConfig(share_kv_cache=a, latent_handoff=b, stop_on_eos=stop_on_eos)
            for a, b in ((False, False), (True, False), (False, True), (True, True))]


def fork_vs_deepcopy(bb: TorchBackbone, n_forks: int = 3, pad: int = 24) -> Dict[str, float]:
    """Time and size the two ways of reusing a prefilled prefix."""
    prompt = (PROMPTS[0] + PAD * pad)
    ids = bb.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    _, _, root = bb.forward(cache=bb.new_cache(), input_ids=ids, use_adapter=False)

    def timed(fn):
        if bb.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        caches = [fn(root) for _ in range(n_forks)]
        if bb.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - t0, caches

    t_fork, forked = timed(fork_cache)
    t_deep, deeped = timed(deepcopy_cache)
    distinct = lambda cs: cache_bytes_union(cs)
    return {
        "prefix_tokens": int(ids.shape[1]),
        "n_forks": n_forks,
        "fork_seconds": round(t_fork, 6),
        "deepcopy_seconds": round(t_deep, 6),
        "fork_speedup_x": round(t_deep / max(t_fork, 1e-9), 1),
        "fork_bytes": distinct(forked),
        "deepcopy_bytes": distinct(deeped),
        "memory_reduction_pct": round(
            100.0 * (distinct(deeped) - distinct(forked)) / max(1, distinct(deeped)), 2),
    }


def cache_bytes_union(caches) -> int:
    """Distinct K/V bytes across several caches, counting shared tensors once."""
    from .compat import layer_tensors
    seen = {}
    for c in caches:
        for k, v in layer_tensors(c):
            for t in (k, v):
                seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
    return sum(seen.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--bridge", default=None, help="path to bridge.pt (defaults to <adapters>/bridge.pt)")
    ap.add_argument("--allow-early-stop", action="store_true",
                    help="let heads stop at EOS. Off by default for benchmarking: "
                         "mode changes shift the output distribution, so EOS timing "
                         "varies by mode and the cost comparison stops being like-for-like.")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--out", type=Path, default=Path("results/torch"))
    args = ap.parse_args()

    bb = TorchBackbone(args.model, roles=default_roles(args.rank),
                       device=args.device, dtype=args.dtype, adapter_dir=args.adapters)
    bridge_path = args.bridge
    if bridge_path is None and args.adapters:
        cand = Path(args.adapters) / "bridge.pt"
        bridge_path = str(cand) if cand.exists() else None
    agent = MultiHeadAgent(backbone=bb, bridge_path=bridge_path)
    tasks = [Task(f"task{i:02d}_tier{i % 3}", PROMPTS[i % len(PROMPTS)] + PAD * (i % 3))
             for i in range(args.tasks)]

    modes = build_modes(stop_on_eos=args.allow_early_stop)
    print(f"{args.model} on {bb.device} | {bb.base_param_count/1e6:.1f}M frozen params | "
          f"{len(tasks)} tasks x {args.repeats} repeats x {len(modes)} modes")
    print("decode budgets are " + ("EOS-terminated (costs not directly comparable)"
                                   if args.allow_early_stop else
                                   "fixed for every mode (like-for-like comparison)"))
    rows: List[Dict] = []
    for cfg in modes:
        for task in tasks:
            runs = [agent.run(task, cfg) for _ in range(args.repeats)]
            row = runs[0].metrics.to_row()
            secs = sorted(r.metrics.wall_seconds for r in runs)
            row["wall_seconds"] = secs[len(secs) // 2]
            row["repeats"] = args.repeats
            rows.append(row)
            print(f"  {cfg.mode:18s} {task.task_id:16s} gen={row['tokens_generated']:4d} "
                  f"enc={row['tokens_encoded']:5d} t={row['wall_seconds']:.3f}s", flush=True)

    report = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform(), "device": str(bb.device)},
        "model": args.model,
        "early_stop_allowed": bool(args.allow_early_stop),
        "adapters": args.adapters,
        "bridge_trained": agent.bridge_trained,
        "params": agent.param_report(),
        "summary": summarize(rows, "baseline"),
        "cache_fork_vs_deepcopy": fork_vs_deepcopy(bb),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "bench_rows.json").write_text(json.dumps(rows, indent=2))
    (args.out / "bench_summary.json").write_text(json.dumps(report, indent=2))

    print()
    report_base = next(e["tokens_encoded"] for e in report["summary"] if e["mode"] == "baseline")
    for e in report["summary"]:
        print(f"  {e['mode']:18s} gen={e['tokens_generated']:7.1f} "
              f"prod={e['producer_decode_steps']:6.1f} enc={e['tokens_encoded']:7.1f} "
              f"prod-red={e.get('producer_decode_reduction_pct', 0):5.1f}%  "
              f"enc-red={100 - 100*e['tokens_encoded']/max(1e-9, report_base):5.1f}%  "
              f"speedup={e.get('speedup_x')}x")
    f = report["cache_fork_vs_deepcopy"]
    print(f"\n  cache fork vs HF deepcopy ({f['prefix_tokens']} prefix tokens, "
          f"{f['n_forks']} forks): {f['fork_speedup_x']}x faster, "
          f"{f['memory_reduction_pct']}% less KV memory")
    print(f"\nwrote {args.out}/bench_summary.json")


if __name__ == "__main__":
    main()
