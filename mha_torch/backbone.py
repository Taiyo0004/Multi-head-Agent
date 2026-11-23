"""A real pretrained backbone with per-role LoRA adapters.

One ``AutoModelForCausalLM`` is loaded once and frozen.  Each role is a
named PEFT LoRA adapter on that same model, so switching roles is
``set_adapter(name)`` -- no second model, no weight copy, no cache
invalidation.  ``disable_adapter()`` gives the adapter-free backbone used
to encode the shared prefix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cache import fork_cache
from .compat import cache_from_output, cache_kwargs, cache_seq_length, new_cache

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_TARGETS = ("q_proj", "v_proj")


@dataclass
class RoleSpec:
    """One role head: adapter hyper-parameters and generation budgets."""

    name: str
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    targets: Sequence[str] = DEFAULT_TARGETS
    text_budget: int = 48
    latent_slots: int = 4

    def lora_config(self) -> LoraConfig:
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=list(self.targets),
            bias="none",
            task_type="CAUSAL_LM",
        )


def default_roles(rank: int = 8) -> List[RoleSpec]:
    return [
        RoleSpec("planner", rank=rank, text_budget=48, latent_slots=4),
        RoleSpec("critic", rank=rank, text_budget=40, latent_slots=4),
        RoleSpec("solver", rank=rank, text_budget=64, latent_slots=4),
    ]


DTYPES = {"fp32": torch.float32, "float32": torch.float32,
          "fp16": torch.float16, "float16": torch.float16,
          "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}


def resolve_dtype(dtype, device: torch.device) -> torch.dtype:
    """Map a name (or None) to a dtype that is safe on this device.

    fp16 on CPU is painfully slow and often unsupported, and bf16 needs
    Ampere or newer, so "auto" picks fp32 on CPU and bf16 on a GPU that
    supports it.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype in (None, "auto"):
        if device.type == "cpu":
            return torch.float32
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}; choose from {sorted(DTYPES)} or 'auto'")
    return DTYPES[dtype]


def pick_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchBackbone:
    """Frozen pretrained model + named LoRA role adapters."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        roles: Optional[Sequence[RoleSpec]] = None,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype | str] = None,
        adapter_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = pick_device(device)
        dtype = resolve_dtype(dtype, self.device)
        self.dtype = dtype
        self.roles: List[RoleSpec] = list(roles or default_roles())

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        base.to(self.device)
        for p in base.parameters():
            p.requires_grad_(False)
        self.base_param_count = sum(p.numel() for p in base.parameters())

        # First role creates the PeftModel; the rest are added to it, so all
        # heads live on one frozen copy of the weights.
        first, rest = self.roles[0], self.roles[1:]
        if adapter_dir:
            self.model = PeftModel.from_pretrained(
                base, f"{adapter_dir}/{first.name}", adapter_name=first.name, is_trainable=True
            )
            for role in rest:
                self.model.load_adapter(f"{adapter_dir}/{role.name}", adapter_name=role.name)
        else:
            self.model = get_peft_model(base, first.lora_config(), adapter_name=first.name)
            for role in rest:
                self.model.add_adapter(role.name, role.lora_config())
        self.model.eval()
        self._active: Optional[str] = first.name

    # -- introspection ----------------------------------------------------
    def adapter_params(self, role: str) -> int:
        return sum(p.numel() for n, p in self.model.named_parameters() if f".{role}." in n)

    def param_report(self) -> Dict[str, float]:
        per_head = {r.name: self.adapter_params(r.name) for r in self.roles}
        total = sum(per_head.values())
        naive = self.base_param_count * len(self.roles)
        ours = self.base_param_count + total
        first = next(iter(per_head.values())) if per_head else 0
        return {
            "model": self.model_name,
            "backbone_params": self.base_param_count,
            "adapter_params_per_head": first,
            "adapter_params_total": total,
            "adapter_share_of_backbone_pct": round(100.0 * first / max(1, self.base_param_count), 4),
            "params_naive_one_model_per_role": naive,
            "params_shared_backbone": ours,
            "param_reduction_vs_naive_pct": round(100.0 * (naive - ours) / max(1, naive), 2),
        }

    # -- adapter control --------------------------------------------------
    def set_role(self, role: Optional[str]) -> None:
        """Activate a role's adapter.  ``None`` means the bare backbone."""
        if role is not None and role != self._active:
            self.model.set_adapter(role)
            self._active = role

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.get_input_embeddings()(input_ids)

    def new_cache(self):
        return new_cache(getattr(self.model, "config", None))

    @staticmethod
    def fork(cache):
        return fork_cache(cache)

    # -- forward ----------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        *,
        cache,
        input_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        role: Optional[str] = None,
        use_adapter: bool = True,
    ):
        """Encode a span, appending to ``cache``.

        Returns ``(logits, hidden, cache)``.  ``use_adapter=False`` runs the
        frozen backbone with every adapter disabled, which is how the shared
        prefix is encoded so all heads can reuse it.
        """
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids / inputs_embeds")
        self.set_role(role)

        kwargs = dict(
            use_cache=True,
            output_hidden_states=True,
            **cache_kwargs(cache),
        )
        if input_ids is not None:
            kwargs["input_ids"] = input_ids.to(self.device)
        else:
            kwargs["inputs_embeds"] = inputs_embeds.to(self.device, self.dtype)

        past = cache_seq_length(cache)
        n_new = (input_ids if input_ids is not None else inputs_embeds).shape[1]
        kwargs["cache_position"] = torch.arange(past, past + n_new, device=self.device)

        if use_adapter:
            out = self.model(**kwargs)
        else:
            with self.model.disable_adapter():
                out = self.model(**kwargs)
        return out.logits, out.hidden_states[-1], (cache_from_output(out) or cache)
