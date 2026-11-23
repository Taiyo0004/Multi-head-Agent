#!/usr/bin/env python3
"""End-to-end GSM8K evaluation: does the efficiency cost accuracy?

Efficiency numbers without a task metric are half an argument.  This
script runs four configurations on the same held-out questions and reports
accuracy *and* cost side by side:

    direct          base model, no adapters, no pipeline  (floor)
    solver_gold     solver head given the gold plan        (ceiling for the head)
    pipeline_text   planner -> critic -> solver, text hand-offs, no sharing
    pipeline_fast   the same pipeline with shared KV + latent hand-offs

The comparison that matters is pipeline_fast vs pipeline_text: same roles,
same weights, cheaper execution.  Any accuracy gap is the price of the
optimisation, and it is reported rather than assumed to be zero.

    python -m mha_torch.evaluate --adapters adapters --limit 200
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .backbone import DEFAULT_MODEL, TorchBackbone, default_roles
from .data import (VERDICT_BAD, VERDICT_OK, build_examples, clean_plan,
                   extract_answer, load_gsm8k, mask_answer)
from .pipeline import MultiHeadAgent, PipelineConfig, Task
from .stats import compare_accuracy, fisher_exact_two_sided

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def first_number(text: str) -> Optional[str]:
    m = NUM_RE.search(text.replace(",", ""))
    return m.group(0).rstrip(".") if m else None


def correct(pred: Optional[str], gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return False


@torch.no_grad()
def generate(bb: TorchBackbone, prompt: str, max_new: int, role: Optional[str]):
    """Plain greedy generation. Returns (text, prompt_tokens, new_tokens)."""
    ids = bb.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(bb.device)
    bb.set_role(role)
    ctx = bb.model.disable_adapter() if role is None else _null_context()
    with ctx:
        out = bb.model.generate(
            input_ids=ids, max_new_tokens=max_new, do_sample=False,
            pad_token_id=bb.tokenizer.pad_token_id or bb.tokenizer.eos_token_id,
        )
    n_prompt = int(ids.shape[1])
    new = out[0, n_prompt:]
    return bb.tokenizer.decode(new, skip_special_tokens=True), n_prompt, int(new.shape[0])


def generate_text(bb: TorchBackbone, prompt: str, max_new: int, role: Optional[str]) -> str:
    return generate(bb, prompt, max_new, role)[0]


class _null_context:
    def __enter__(self): return None
    def __exit__(self, *exc): return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapters", default="adapters", help="directory written by train.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=200, help="held-out questions")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--mask-answer-in-plan", action="store_true",
                    help="hide the answer inside the gold plan given to solver_gold, so "
                         "it has to compute the last step instead of copying it. Match "
                         "this to how the solver adapter was trained.")
    ap.add_argument("--bridge", default=None,
                    help="path to bridge.pt from mha_torch.train_latent; without it "
                         "the latent projection is random and latent-mode accuracy "
                         "is meaningless")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--out", type=Path, default=Path("results/torch"))
    args = ap.parse_args()

    adapter_dir = args.adapters if Path(args.adapters).exists() else None
    if adapter_dir is None:
        print(f"warning: {args.adapters} not found -- running with UNTRAINED adapters. "
              f"Accuracy numbers will be meaningless; run mha_torch.train first.")
    bb = TorchBackbone(args.model, roles=default_roles(args.rank),
                       device=args.device, dtype=args.dtype, adapter_dir=adapter_dir)
    bridge_path = args.bridge
    if bridge_path is None:
        default_bridge = Path(args.adapters) / "bridge.pt"
        bridge_path = str(default_bridge) if default_bridge.exists() else None
    agent = MultiHeadAgent(backbone=bb, bridge_path=bridge_path)
    if not agent.bridge_trained:
        print("warning: no trained bridge found -- the latent hand-off runs through a "
              "RANDOM projection, so pipeline_fast accuracy is not a real measurement. "
              "Run `python -m mha_torch.train_latent` first.")

    rows = load_gsm8k("test", limit=args.limit)
    # All four modes, so an accuracy change can be attributed to prefix
    # sharing or to latent compression rather than to "the fast path".
    configs = {
        "pipeline_text": PipelineConfig(share_kv_cache=False, latent_handoff=False),
        "pipeline_shared_kv": PipelineConfig(share_kv_cache=True, latent_handoff=False),
        "pipeline_latent": PipelineConfig(share_kv_cache=False, latent_handoff=True),
        "pipeline_fast": PipelineConfig(share_kv_cache=True, latent_handoff=True),
    }
    stats: Dict[str, Dict[str, float]] = {
        name: {"correct": 0, "n": 0, "tokens_generated": 0, "tokens_total": 0, "seconds": 0.0}
        for name in ["direct", "solver_gold", *configs]
    }
    # Diagnostic: GSM8K solutions usually state the answer in their last
    # step, so a "solver" given the gold plan may only be copying a number
    # out of its prompt. Measure how often that shortcut is available
    # instead of assuming it is or is not there.
    leak = {"plan_contains_answer": 0, "solver_plan_contains_answer": 0, "n": 0}
    critic = {"correct": 0, "n": 0, "correct_clean": 0, "n_clean": 0,
              "correct_corrupt": 0, "n_corrupt": 0, "predicted_wrong": 0}

    for i, row in enumerate(rows):
        question, gold = row["question"].strip(), extract_answer(row["answer"])
        if gold is None:
            continue

        # 1. base model, no pipeline
        t0 = time.perf_counter()
        text, n_prompt, n_new = generate(bb, f"Task: {question}\nAnswer:", 32, role=None)
        s = stats["direct"]
        s["correct"] += correct(first_number(text), gold); s["n"] += 1
        s["tokens_generated"] += n_new
        s["tokens_total"] += n_prompt + n_new
        s["seconds"] += time.perf_counter() - t0

        # 2. solver head, gold plan
        t0 = time.perf_counter()
        plan = clean_plan(row["answer"])
        solver_plan = mask_answer(plan, gold) if args.mask_answer_in_plan else plan
        leak["plan_contains_answer"] += gold in plan.replace(",", "")
        leak["solver_plan_contains_answer"] += gold in solver_plan.replace(",", "")
        leak["n"] += 1
        text, n_prompt, n_new = generate(
            bb, f"Task: {question}\nPlan:\n{solver_plan}\nSolver: give the final numeric answer.\nAnswer:",
            16, role="solver")
        s = stats["solver_gold"]
        s["correct"] += correct(first_number(text), gold); s["n"] += 1
        s["tokens_generated"] += n_new
        s["tokens_total"] += n_prompt + n_new
        s["seconds"] += time.perf_counter() - t0

        # 3/4. the full pipeline, two execution modes
        for name, cfg in configs.items():
            res = agent.run(Task(f"q{i}", question), cfg)
            s = stats[name]
            s["correct"] += correct(first_number(res.answer_text), gold)
            s["n"] += 1
            s["tokens_generated"] += res.metrics.tokens_generated
            s["tokens_total"] += res.metrics.tokens_total
            s["seconds"] += res.metrics.wall_seconds

        # 5. critic head: can it tell a sound plan from a perturbed one?
        ex = build_examples([row], roles=("critic",), seed=i)
        if ex:
            e = ex[0]
            verdict = generate_text(bb, e.prompt, 4, role="critic").strip().upper()
            said_wrong = VERDICT_BAD in verdict
            said_ok = VERDICT_OK in verdict and not said_wrong
            corrupted = bool(e.meta["corrupted"])
            hit = said_wrong if corrupted else said_ok
            critic["correct"] += hit
            critic["n"] += 1
            critic["predicted_wrong"] += said_wrong
            key = "corrupt" if corrupted else "clean"
            critic[f"correct_{key}"] += hit
            critic[f"n_{key}"] += 1

        if (i + 1) % 10 == 0:
            line = "  ".join(f"{k}={stats[k]['correct']/max(1,stats[k]['n']):.1%}" for k in stats)
            print(f"  [{i+1}/{len(rows)}] {line}  "
                  f"critic={critic['correct']/max(1,critic['n']):.1%}", flush=True)

    summary = {}
    for name, s in stats.items():
        n = max(1, s["n"])
        summary[name] = {
            "accuracy_pct": round(100.0 * s["correct"] / n, 2),
            "n_correct": int(s["correct"]),
            "n": s["n"],
            "mean_tokens_generated": round(s["tokens_generated"] / n, 1),
            "mean_tokens_total": round(s["tokens_total"] / n, 1),
            "mean_seconds": round(s["seconds"] / n, 4),
        }
    n_c = max(1, critic["n"])
    summary["critic"] = {
        "accuracy_pct": round(100.0 * critic["correct"] / n_c, 2),
        "accuracy_on_sound_plans_pct": round(
            100.0 * critic["correct_clean"] / max(1, critic["n_clean"]), 2),
        "accuracy_on_perturbed_plans_pct": round(
            100.0 * critic["correct_corrupt"] / max(1, critic["n_corrupt"]), 2),
        "predicted_wrong_rate_pct": round(100.0 * critic["predicted_wrong"] / n_c, 2),
        "majority_class_baseline_pct": round(
            100.0 * max(critic["n_clean"], critic["n_corrupt"]) / n_c, 2),
        "n": critic["n"],
    }
    summary["diagnostics"] = {
        "gold_plan_contains_answer_pct": round(
            100.0 * leak["plan_contains_answer"] / max(1, leak["n"]), 2),
        "solver_plan_contains_answer_pct": round(
            100.0 * leak["solver_plan_contains_answer"] / max(1, leak["n"]), 2),
        "answer_masked_for_solver_gold": bool(args.mask_answer_in_plan),
        "note": ("solver_plan_contains_answer_pct is the leakage in the plan "
                 "actually fed to solver_gold; while it is high, that row "
                 "measures answer extraction, not arithmetic. Pass "
                 "--mask-answer-in-plan (matching how the solver was trained) "
                 "to drive it to zero."),
    }
    # 4 correct out of 200 is not evidence about a 2-point gap; say so.
    n_eval = summary["pipeline_text"]["n"]
    worst = min(summary["pipeline_text"]["n_correct"], summary["pipeline_fast"]["n_correct"])
    if worst < 10:
        summary["diagnostics"]["statistical_power_warning"] = (
            f"both pipelines are near the accuracy floor "
            f"({summary['pipeline_text']['n_correct']} and "
            f"{summary['pipeline_fast']['n_correct']} correct of {n_eval}); the "
            f"accuracy delta is not statistically meaningful at this sample size. "
            f"Use a stronger backbone before drawing conclusions from it."
        )
    text_cfg = summary["pipeline_text"]
    deltas = {}
    for name in ("pipeline_shared_kv", "pipeline_latent", "pipeline_fast"):
        cfg_s = summary[name]
        deltas[name] = {
            "accuracy_points_vs_text": round(cfg_s["accuracy_pct"] - text_cfg["accuracy_pct"], 2),
            "tokens_total_reduction_pct": round(
                100.0 * (text_cfg["mean_tokens_total"] - cfg_s["mean_tokens_total"])
                / max(1e-9, text_cfg["mean_tokens_total"]), 2),
        }
    # Significance, because 32 correct vs 19 correct is not self-evidently
    # a difference at n=200.
    tests = [compare_accuracy("direct", summary["direct"]["n_correct"], summary["direct"]["n"],
                              "pipeline_text", text_cfg["n_correct"], text_cfg["n"])]
    for name in ("pipeline_shared_kv", "pipeline_latent", "pipeline_fast"):
        tests.append(compare_accuracy("pipeline_text", text_cfg["n_correct"], text_cfg["n"],
                                      name, summary[name]["n_correct"], summary[name]["n"]))
    c = summary["critic"]
    n_major = round(c["n"] * c["majority_class_baseline_pct"] / 100.0)
    tests.append(compare_accuracy(
        "critic_majority_baseline", int(n_major), c["n"],
        "critic", round(c["n"] * c["accuracy_pct"] / 100.0), c["n"]))

    summary["significance"] = tests
    summary["attribution"] = {
        "deltas_vs_pipeline_text": deltas,
        "note": ("pipeline_shared_kv isolates the cost of encoding the prefix "
                 "adapter-free; pipeline_latent isolates the cost of compressing "
                 "the hand-off into slots. pipeline_fast applies both, so its "
                 "delta should be read against these two, not on its own. Cost "
                 "figures here are EOS-terminated and therefore mode-dependent; "
                 "use bench.py for like-for-like cost."),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"model": args.model, "adapters": adapter_dir,
               "bridge": bridge_path, "bridge_trained": agent.bridge_trained,
               "n_questions": len(rows), "summary": summary}
    if not agent.bridge_trained:
        payload["warning"] = ("latent hand-off used an untrained random projection; "
                              "pipeline_fast and latent-mode accuracy are not valid")
    (args.out / "gsm8k_eval.json").write_text(json.dumps(payload, indent=2))
    print("\nsignificance (Fisher exact, two-sided):")
    for t in summary["significance"]:
        print(f"  {t['comparison']:42s} {t['accuracy_points']:+6.1f} pts  "
              f"{t['counts']:>16s}  p={t['fisher_p']:<10.2g} {t['verdict']}")
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}/gsm8k_eval.json")


if __name__ == "__main__":
    main()
