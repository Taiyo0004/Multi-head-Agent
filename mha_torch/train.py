#!/usr/bin/env python3
"""Train one LoRA adapter per role on GSM8K-derived supervision.

Each role is trained separately on the same frozen backbone: activate that
role's adapter, freeze every other parameter in the model, and run masked
next-token prediction on its own targets (loss on the target span only,
never on the prompt).

    python -m mha_torch.train --train-size 2000 --epochs 2 --out adapters

On CPU with the default 135M model this is slow but finishes; on any CUDA
GPU it is minutes.  Use --model to swap in something larger.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .backbone import DEFAULT_MODEL, RoleSpec, TorchBackbone, default_roles
from .data import Example, build_examples, load_gsm8k


def collate(batch: List[Example], tokenizer, max_len: int, device):
    """Tokenize prompt+target, masking the prompt out of the loss."""
    input_ids, labels = [], []
    for ex in batch:
        p = tokenizer(ex.prompt, add_special_tokens=False)["input_ids"]
        t = tokenizer(ex.target, add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            t = t + [tokenizer.eos_token_id]
        ids = (p + t)[-max_len:]
        n_target = min(len(t), len(ids))
        lab = [-100] * (len(ids) - n_target) + ids[len(ids) - n_target:]
        input_ids.append(ids)
        labels.append(lab)

    width = max(len(x) for x in input_ids)
    pad = tokenizer.pad_token_id or 0
    attn = [[1] * len(x) + [0] * (width - len(x)) for x in input_ids]
    input_ids = [x + [pad] * (width - len(x)) for x in input_ids]
    labels = [x + [-100] * (width - len(x)) for x in labels]
    t = lambda v: torch.tensor(v, dtype=torch.long, device=device)
    return t(input_ids), t(attn), t(labels)


def freeze_all_but(model, role: str) -> int:
    """Only this role's LoRA parameters stay trainable."""
    trainable = 0
    for name, param in model.named_parameters():
        on = f".{role}." in name and "lora" in name.lower()
        param.requires_grad_(on)
        trainable += param.numel() if on else 0
    return trainable


def train_role(bb: TorchBackbone, spec: RoleSpec, examples: List[Example], args) -> Dict[str, float]:
    model, tokenizer = bb.model, bb.tokenizer
    model.set_adapter(spec.name)
    if args.grad_checkpoint:
        # Trades ~30% speed for a large activation-memory saving; the usual
        # way to fit a longer sequence on a small GPU.
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    n_trainable = freeze_all_but(model, spec.name)
    if n_trainable == 0:
        raise RuntimeError(f"no trainable parameters matched role {spec.name!r}")

    rng = random.Random(args.seed)
    data = [e for e in examples if e.role == spec.name]
    rng.shuffle(data)
    cut = max(1, int(0.95 * len(data)))
    train_set, val_set = data[:cut], data[cut:]

    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tokenizer, args.max_len, bb.device),
    )
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    accum = max(1, args.grad_accum)
    steps = max(1, math.ceil(len(loader) / accum) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1, anneal_strategy="cos")

    print(f"\n[{spec.name}] {len(train_set)} train / {len(val_set)} val examples, "
          f"{n_trainable/1e3:.1f}K trainable params, {steps} optimizer steps "
          f"(batch {args.batch_size} x accum {accum})")
    model.train()
    t0, seen, micro = time.perf_counter(), 0, 0
    for epoch in range(args.epochs):
        for ids, attn, labels in loader:
            out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
            (out.loss / accum).backward()
            micro += 1
            if micro % accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            seen += 1
            if seen % args.log_every == 0:
                mem = (f", {torch.cuda.max_memory_allocated()/2**30:.2f} GiB peak"
                       if bb.device.type == "cuda" else "")
                print(f"  epoch {epoch} step {seen}/{steps} loss {out.loss.item():.4f} "
                      f"({seen/(time.perf_counter()-t0):.2f} it/s{mem})", flush=True)
        opt.zero_grad(set_to_none=True)  # drop any partial accumulation

    model.eval()
    val_loss = evaluate_loss(model, tokenizer, val_set, args, bb.device)
    print(f"[{spec.name}] val loss {val_loss:.4f} (ppl {math.exp(min(val_loss, 20)):.2f})")

    out_dir = Path(args.out) / spec.name
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(str(args.out), selected_adapters=[spec.name])
    except TypeError:  # older peft
        model.save_pretrained(str(out_dir))
    print(f"[{spec.name}] adapter saved to {out_dir}")
    return {"role": spec.name, "val_loss": val_loss, "n_train": len(train_set),
            "trainable_params": n_trainable}


@torch.no_grad()
def evaluate_loss(model, tokenizer, examples: List[Example], args, device) -> float:
    if not examples:
        return float("nan")
    total, n = 0.0, 0
    for i in range(0, len(examples), args.batch_size):
        batch = examples[i:i + args.batch_size]
        ids, attn, labels = collate(batch, tokenizer, args.max_len, device)
        out = model(input_ids=ids, attention_mask=attn, labels=labels, use_cache=False)
        total += out.loss.item() * len(batch)
        n += len(batch)
    return total / max(1, n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default=None)
    ap.add_argument("--train-size", type=int, default=2000, help="GSM8K rows to use")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2,
                    help="micro-batch; keep small on a small GPU and raise --grad-accum")
    ap.add_argument("--grad-accum", type=int, default=4,
                    help="optimizer steps every N micro-batches (effective batch = batch x accum)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="recompute activations to save memory (~30%% slower)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default="adapters")
    ap.add_argument("--roles", nargs="*", default=["planner", "critic", "solver"])
    ap.add_argument("--mask-answer-in-plan", action="store_true",
                    help="hide the answer inside the solver's plan so it must "
                         "compute the last step instead of copying it")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rows = load_gsm8k("train", limit=args.train_size)
    examples = build_examples(rows, roles=tuple(args.roles), seed=args.seed,
                              mask_answer_in_plan=args.mask_answer_in_plan)
    print(f"built {len(examples)} role examples from {len(rows)} GSM8K rows")

    specs = [s for s in default_roles(args.rank) if s.name in args.roles]
    bb = TorchBackbone(args.model, roles=specs, device=args.device, dtype=args.dtype)
    print(f"backbone {args.model} on {bb.device} ({bb.dtype}): "
          f"{bb.base_param_count/1e6:.1f}M frozen params")
    if bb.device.type == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"GPU: {torch.cuda.get_device_name(0)}, {total:.1f} GiB")

    report = [train_role(bb, spec, examples, args) for spec in specs]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "training_report.json").write_text(json.dumps(
        {"model": args.model, "args": vars(args), "roles": report}, indent=2))
    print(f"\nwrote {args.out}/training_report.json")


if __name__ == "__main__":
    main()
