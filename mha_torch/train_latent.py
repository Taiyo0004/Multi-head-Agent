#!/usr/bin/env python3
"""Train the latent bridge, so latent hand-offs carry real information.

``train.py`` fits the role adapters on text.  That leaves the bridge --
the projection that turns a producer's hidden states into slots the
consumer reads -- at its random initialisation, and a random projection
injects noise.  Measured consequence on SmolLM2-135M: the latent pipeline
scored 0.5% on GSM8K against 3.0% for the text pipeline.

This script fixes that with a teacher-forced objective:

1.  Run the **planner** head over the question plus the *gold* plan and
    take the final-layer hidden states of the plan span (no grad: the
    planner is not being updated here).
2.  Pool them through the **bridge** into ``k`` slots.
3.  Have the **solver** predict the answer from
    ``[question][slots][instruction]``, with loss on the answer tokens only.
4.  Backpropagate into the bridge and the solver's LoRA weights.

Batches alternate between that latent form and the ordinary text form
(``[question][plan as text][instruction]``), so the one solver head stays
able to read either -- which is what the four-mode benchmark assumes.

Known limitation, stated rather than hidden: training pools *gold* plan
activations while inference pools the planner's *generated* ones, so there
is a teacher-forcing gap. Fixing it properly means sampling from the
planner during training, which costs far more compute.

    python -m mha_torch.train_latent --train-size 1500 --epochs 1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .backbone import DEFAULT_MODEL, TorchBackbone, default_roles
from .data import clean_plan, extract_answer, load_gsm8k, mask_answer
from .latent import LatentBridge
from .train import freeze_all_but

SOLVER_INSTRUCTION = "\nSolver: give the final numeric answer.\nAnswer:"


class LatentTrainer:
    def __init__(self, bb: TorchBackbone, bridge: LatentBridge, n_slots: int, max_len: int) -> None:
        self.bb = bb
        self.bridge = bridge
        self.n_slots = n_slots
        self.max_len = max_len
        self.tok = bb.tokenizer
        self.embed = bb.model.get_input_embeddings()

    def ids(self, text: str) -> List[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def plan_hidden(self, question: str, plan: str) -> torch.Tensor:
        """Planner activations over the gold plan span."""
        prompt = f"Task: {question}\nPlanner: decompose the task.\nPlan:\n"
        plan_ids = self.ids(plan)
        ids = (self.ids(prompt) + plan_ids)[-self.max_len:]
        n_plan = min(len(plan_ids), len(ids))
        ids_t = torch.tensor([ids], device=self.bb.device)
        self.bb.set_role("planner")
        out = self.bb.model(input_ids=ids_t, output_hidden_states=True, use_cache=False)
        return out.hidden_states[-1][:, -n_plan:, :].detach()

    def latent_example(self, question: str, plan: str, answer: str):
        """([question][slots][instruction][answer]) as embeddings + labels."""
        slots = self.bridge(self.plan_hidden(question, plan), "planner", self.n_slots)
        head = torch.tensor([self.ids(f"Task: {question}\n")], device=self.bb.device)
        tail = torch.tensor([self.ids(SOLVER_INSTRUCTION)], device=self.bb.device)
        target = self.ids(f" {answer}") + ([self.tok.eos_token_id] if self.tok.eos_token_id else [])
        target_t = torch.tensor([target], device=self.bb.device)

        embeds = torch.cat([
            self.embed(head), slots.to(self.embed.weight.dtype),
            self.embed(tail), self.embed(target_t),
        ], dim=1)
        n_ctx = embeds.shape[1] - len(target)
        labels = [-100] * n_ctx + target
        return embeds[0], labels

    def text_example(self, question: str, plan: str, answer: str):
        """The same supervision with the plan as text, to keep both modes alive."""
        ctx = self.ids(f"Task: {question}\nPlan:\n{plan}{SOLVER_INSTRUCTION}")
        target = self.ids(f" {answer}") + ([self.tok.eos_token_id] if self.tok.eos_token_id else [])
        ids = (ctx + target)[-self.max_len:]
        n_target = min(len(target), len(ids))
        ids_t = torch.tensor([ids], device=self.bb.device)
        embeds = self.embed(ids_t)[0]
        labels = [-100] * (len(ids) - n_target) + ids[len(ids) - n_target:]
        return embeds, labels

    def collate(self, items: List[Tuple[torch.Tensor, List[int]]]):
        width = max(e.shape[0] for e, _ in items)
        d = items[0][0].shape[-1]
        embeds = torch.zeros(len(items), width, d, device=self.bb.device,
                             dtype=items[0][0].dtype)
        attn = torch.zeros(len(items), width, dtype=torch.long, device=self.bb.device)
        labels = torch.full((len(items), width), -100, dtype=torch.long, device=self.bb.device)
        for i, (e, lab) in enumerate(items):
            n = e.shape[0]
            embeds[i, :n] = e
            attn[i, :n] = 1
            labels[i, :n] = torch.tensor(lab, device=self.bb.device)
        return embeds, attn, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapters", default="adapters",
                    help="directory holding the adapters trained by train.py")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--train-size", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--latent-fraction", type=float, default=0.5,
                    help="share of batches using the latent form (rest are text)")
    ap.add_argument("--mask-answer-in-plan", action="store_true",
                    help="hide the answer inside the plan so the solver must compute it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    adapter_dir = args.adapters if Path(args.adapters).exists() else None
    if adapter_dir is None:
        raise SystemExit(f"{args.adapters} not found -- run mha_torch.train first")
    bb = TorchBackbone(args.model, roles=default_roles(args.rank), device=args.device,
                       dtype=args.dtype, adapter_dir=adapter_dir)
    embed_w = bb.model.get_input_embeddings().weight
    bridge = LatentBridge(embed_w.shape[1], embed_w, [r.name for r in bb.roles]).to(bb.device)
    bridge.to(torch.float32)  # the bridge is small; keep it in fp32 for stable updates

    rows = load_gsm8k("train", limit=args.train_size)
    data = []
    for row in rows:
        q, sol = row["question"].strip(), row["answer"]
        plan, answer = clean_plan(sol), extract_answer(sol)
        if not plan or answer is None:
            continue
        if args.mask_answer_in_plan:
            plan = mask_answer(plan, answer)
        data.append((q, plan, answer))
    rng.shuffle(data)
    cut = max(1, int(0.95 * len(data)))
    train_set, val_set = data[:cut], data[cut:]

    model = bb.model
    model.set_adapter("solver")
    n_lora = freeze_all_but(model, "solver")
    for p in bridge.parameters():
        p.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad] + list(bridge.parameters())
    n_bridge = sum(p.numel() for p in bridge.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    trainer = LatentTrainer(bb, bridge, args.slots, args.max_len)
    accum = max(1, args.grad_accum)
    per_epoch = math.ceil(len(train_set) / args.batch_size)
    steps = max(1, math.ceil(per_epoch / accum) * args.epochs)
    print(f"bridge {n_bridge/1e3:.1f}K params + solver LoRA {n_lora/1e3:.1f}K params, "
          f"{len(train_set)} train / {len(val_set)} val, {steps} optimizer steps")

    def run_batch(batch, latent: bool):
        make = trainer.latent_example if latent else trainer.text_example
        items = [make(q, p, a) for q, p, a in batch]
        embeds, attn, labels = trainer.collate(items)
        return model(inputs_embeds=embeds, attention_mask=attn, labels=labels, use_cache=False)

    model.train()
    bridge.train()
    t0, seen, micro = time.perf_counter(), 0, 0
    for epoch in range(args.epochs):
        rng.shuffle(train_set)
        for i in range(0, len(train_set), args.batch_size):
            batch = train_set[i:i + args.batch_size]
            latent = rng.random() < args.latent_fraction
            out = run_batch(batch, latent)
            (out.loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            seen += 1
            if seen % args.log_every == 0:
                mem = (f", {torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak"
                       if bb.device.type == "cuda" else "")
                print(f"  epoch {epoch} step {seen}/{steps} "
                      f"loss {out.loss.item():.4f} ({'latent' if latent else 'text'}) "
                      f"({seen/(time.perf_counter()-t0):.2f} it/s{mem})", flush=True)
        opt.zero_grad(set_to_none=True)

    # -- validation: the comparison that matters is latent vs text --------
    model.eval()
    bridge.eval()
    report: Dict[str, float] = {}
    with torch.no_grad():
        for name, latent in (("latent", True), ("text", False)):
            total, n = 0.0, 0
            for i in range(0, len(val_set), args.batch_size):
                batch = val_set[i:i + args.batch_size]
                if not batch:
                    continue
                total += run_batch(batch, latent).loss.item() * len(batch)
                n += len(batch)
            report[f"val_loss_{name}"] = round(total / max(1, n), 4)
    gap = report["val_loss_latent"] - report["val_loss_text"]
    print(f"\nval loss  latent {report['val_loss_latent']:.4f}  "
          f"text {report['val_loss_text']:.4f}  gap {gap:+.4f}")
    print("  (gap is the cost of compressing the plan into "
          f"{args.slots} vectors; it is the number to report)")

    out_dir = Path(args.adapters)
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge.save(str(out_dir / "bridge.pt"))
    try:
        model.save_pretrained(str(out_dir), selected_adapters=["solver"])
    except TypeError:
        model.save_pretrained(str(out_dir / "solver"))
    report.update({"bridge_params": n_bridge, "solver_lora_params": n_lora,
                   "latent_slots": args.slots, "args": vars(args)})
    (out_dir / "latent_training_report.json").write_text(json.dumps(report, indent=2))
    print(f"saved bridge to {out_dir/'bridge.pt'} and the updated solver adapter")


if __name__ == "__main__":
    main()
