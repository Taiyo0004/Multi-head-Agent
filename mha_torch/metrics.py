"""Measured counters for the PyTorch pipeline."""
from __future__ import annotations

import statistics as st
import time
from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class RoleMetrics:
    role: str
    prefill_tokens: int = 0
    reused_prefix_tokens: int = 0
    decode_steps: int = 0
    emitted_latent_slots: int = 0
    seconds: float = 0.0


@dataclass
class RunMetrics:
    mode: str
    task_id: str
    roles: List[RoleMetrics] = field(default_factory=list)
    _final_role: str = "solver"
    wall_seconds: float = 0.0
    kv_bytes: int = 0
    peak_memory_bytes: int = 0

    @property
    def tokens_generated(self) -> int:
        return sum(r.decode_steps for r in self.roles)

    @property
    def producer_decode_steps(self) -> int:
        """Decode steps spent by heads whose output feeds another head.

        This is what latent hand-off replaces, and unlike total generated
        tokens it is governed by fixed budgets rather than by where each
        mode happens to emit EOS.
        """
        return sum(r.decode_steps for r in self.roles
                   if r.role not in ("shared_prefix", self._final_role))

    @property
    def tokens_encoded(self) -> int:
        return sum(r.prefill_tokens for r in self.roles)

    @property
    def tokens_total(self) -> int:
        return self.tokens_generated + self.tokens_encoded

    @property
    def prefix_tokens_reused(self) -> int:
        return sum(r.reused_prefix_tokens for r in self.roles)

    def to_row(self) -> Dict[str, object]:
        row = {
            "mode": self.mode,
            "task_id": self.task_id,
            "tokens_generated": self.tokens_generated,
            "producer_decode_steps": self.producer_decode_steps,
            "tokens_encoded": self.tokens_encoded,
            "tokens_total": self.tokens_total,
            "prefix_tokens_reused": self.prefix_tokens_reused,
            "wall_seconds": round(self.wall_seconds, 4),
            "kv_bytes": self.kv_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
        }
        for r in self.roles:
            row[f"{r.role}_decode_steps"] = r.decode_steps
            row[f"{r.role}_prefill_tokens"] = r.prefill_tokens
            row[f"{r.role}_latent_slots"] = r.emitted_latent_slots
        return row


class Stopwatch:
    """Times a block, synchronising CUDA first so the number is real."""

    def __init__(self, target: RoleMetrics | None = None, device: torch.device | None = None) -> None:
        self.target = target
        self.device = device
        self.seconds = 0.0

    def _sync(self) -> None:
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize()

    def __enter__(self) -> "Stopwatch":
        self._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self._sync()
        self.seconds = time.perf_counter() - self._t0
        if self.target is not None:
            self.target.seconds += self.seconds


def reduction(baseline: float, candidate: float) -> float:
    return 0.0 if baseline == 0 else 100.0 * (baseline - candidate) / baseline


def summarize(rows: List[Dict[str, object]], baseline_mode: str = "baseline") -> List[Dict[str, object]]:
    numeric = ["tokens_generated", "producer_decode_steps", "tokens_encoded", "tokens_total",
               "prefix_tokens_reused", "wall_seconds", "kv_bytes", "peak_memory_bytes"]
    by_mode: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_mode.setdefault(str(r["mode"]), []).append(r)
    means = {m: {k: st.fmean([float(r[k]) for r in rs]) for k in numeric}
             for m, rs in by_mode.items()}
    base = means.get(baseline_mode)
    out = []
    for mode, mean in means.items():
        entry: Dict[str, object] = {"mode": mode, "n_tasks": len(by_mode[mode])}
        entry.update({k: round(v, 4) for k, v in mean.items()})
        if base:
            entry["tokens_generated_reduction_pct"] = round(reduction(base["tokens_generated"], mean["tokens_generated"]), 2)
            entry["producer_decode_reduction_pct"] = round(
                reduction(base["producer_decode_steps"], mean["producer_decode_steps"]), 2)
            entry["tokens_total_reduction_pct"] = round(reduction(base["tokens_total"], mean["tokens_total"]), 2)
            entry["latency_reduction_pct"] = round(reduction(base["wall_seconds"], mean["wall_seconds"]), 2)
            entry["kv_memory_reduction_pct"] = round(reduction(base["kv_bytes"], mean["kv_bytes"]), 2)
            entry["speedup_x"] = round(base["wall_seconds"] / mean["wall_seconds"], 2) if mean["wall_seconds"] else None
        out.append(entry)
    return sorted(out, key=lambda e: e["tokens_total"], reverse=True)
