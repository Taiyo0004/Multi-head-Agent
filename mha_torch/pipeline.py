"""Planner -> Critic -> Solver over one frozen pretrained backbone.

Same four modes as the NumPy reference:

    baseline          per-head prefix re-encode, text hand-offs
    shared_kv         prefix encoded once and forked, text hand-offs
    latent            per-head prefix re-encode, latent hand-offs
    shared_kv+latent  both

The shared prefix is encoded with adapters disabled, so every head can
attend over identical keys; role adapters apply to each head's own suffix
and to all of its decode steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from .backbone import RoleSpec, TorchBackbone, default_roles
from .compat import cache_bytes, cache_seq_length
from .latent import LatentBridge
from .metrics import RoleMetrics, RunMetrics, Stopwatch

SYSTEM_PREAMBLE = (
    "You are a cooperative problem-solving system with three roles. "
    "The planner decomposes the task, the critic finds flaws in the plan, "
    "and the solver produces the final answer. Be precise and concise."
)

ROLE_INSTRUCTIONS = {
    "planner": "\nPlanner: decompose the task into ordered, checkable steps.\nPlan:",
    "critic": "\nCritic: name incorrect assumptions or missing steps in the plan.\nCritique:",
    "solver": "\nSolver: using the plan and critique, give the final answer.\nAnswer:",
}
GENERIC_INSTRUCTION = "\nAct in your role and pass on only what the next head needs.\n"


def role_instruction(role: str) -> str:
    return ROLE_INSTRUCTIONS.get(role, GENERIC_INSTRUCTION)


@dataclass(frozen=True)
class PipelineConfig:
    share_kv_cache: bool = True
    latent_handoff: bool = True
    latent_roles: Sequence[str] = ("planner", "critic")
    greedy: bool = True
    temperature: float = 0.7
    stop_on_eos: bool = True
    """Whether a head may stop early at EOS.

    Leave this on for accuracy evaluation -- it is how the system would
    really run.  Turn it OFF when comparing cost between modes: changing
    the execution mode changes the output distribution, so one mode's head
    may hit EOS at step 3 and another's at step 64, and a latency or
    token-count comparison then measures EOS luck rather than the
    architecture.  With it off, every head decodes its full budget and the
    modes do identical decode work.
    """

    @property
    def mode(self) -> str:
        if self.share_kv_cache and self.latent_handoff:
            return "shared_kv+latent"
        if self.share_kv_cache:
            return "shared_kv"
        if self.latent_handoff:
            return "latent"
        return "baseline"


@dataclass
class Task:
    task_id: str
    prompt: str


@dataclass
class PipelineResult:
    task_id: str
    mode: str
    answer_text: str
    answer_token_ids: List[int]
    texts: Dict[str, str] = field(default_factory=dict)
    metrics: Optional[RunMetrics] = None


class MultiHeadAgent:
    """The pipeline, with measurement built in."""

    def __init__(
        self,
        backbone: Optional[TorchBackbone] = None,
        roles: Optional[Sequence[RoleSpec]] = None,
        bridge: Optional[LatentBridge] = None,
        bridge_path: Optional[str] = None,
        **backbone_kwargs,
    ) -> None:
        self.backbone = backbone or TorchBackbone(roles=roles or default_roles(), **backbone_kwargs)
        self.roles = list(self.backbone.roles)
        self.role_order = [r.name for r in self.roles]
        self.specs = {r.name: r for r in self.roles}
        embed = self.backbone.model.get_input_embeddings().weight
        self.bridge = (bridge or LatentBridge(embed.shape[1], embed, self.role_order)).to(
            self.backbone.device
        )
        self.bridge_trained = False
        if bridge_path:
            self.bridge.load(bridge_path, map_location=self.backbone.device)
            self.bridge_trained = True
        self.bridge.eval()

    def param_report(self) -> Dict[str, float]:
        return self.backbone.param_report()

    # -- helpers ----------------------------------------------------------
    def _encode(self, text: str) -> torch.Tensor:
        ids = self.backbone.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return ids["input_ids"].to(self.backbone.device)

    def _next_token(self, logits: torch.Tensor, cfg: PipelineConfig) -> torch.Tensor:
        last = logits[:, -1, :]
        if cfg.greedy:
            return last.argmax(dim=-1, keepdim=True)
        probs = torch.softmax(last.float() / max(cfg.temperature, 1e-5), dim=-1)
        return torch.multinomial(probs, num_samples=1)

    @torch.no_grad()
    def _decode(self, role: str, cache, budget: int, cfg: PipelineConfig,
                rm: RoleMetrics, first_logits: torch.Tensor):
        """Greedy/sampled decode loop; returns (token ids, hidden states, cache)."""
        tok = self.backbone.tokenizer
        eos = tok.eos_token_id
        ids: List[int] = []
        hiddens: List[torch.Tensor] = []
        nxt = self._next_token(first_logits, cfg)

        for _ in range(budget):
            if cfg.stop_on_eos and eos is not None and int(nxt.item()) == eos:
                break
            ids.append(int(nxt.item()))
            with Stopwatch(rm, self.backbone.device):
                logits, hidden, cache = self.backbone.forward(
                    cache=cache, input_ids=nxt, role=role
                )
            rm.decode_steps += 1
            hiddens.append(hidden[:, -1:, :])
            nxt = self._next_token(logits, cfg)

        stacked = torch.cat(hiddens, dim=1) if hiddens else None
        return ids, stacked, cache

    # -- main entry point -------------------------------------------------
    @torch.no_grad()
    def run(self, task: Task, cfg: Optional[PipelineConfig] = None) -> PipelineResult:
        cfg = cfg or PipelineConfig()
        bb = self.backbone
        tok = bb.tokenizer
        metrics = RunMetrics(mode=cfg.mode, task_id=task.task_id,
                             _final_role=self.role_order[-1])
        if bb.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        prefix_text = f"{SYSTEM_PREAMBLE}\n\nTask: {task.prompt}\n"
        prefix_ids = self._encode(prefix_text)
        n_prefix = int(prefix_ids.shape[1])

        root = None
        with Stopwatch(None, bb.device) as total:
            if cfg.share_kv_cache:
                shared = RoleMetrics(role="shared_prefix")
                metrics.roles.append(shared)
                with Stopwatch(shared, bb.device):
                    # adapter-free, so every head may reuse these keys
                    _, _, root = bb.forward(
                        cache=bb.new_cache(), input_ids=prefix_ids, use_adapter=False
                    )
                shared.prefill_tokens += n_prefix

            carried_text: List[str] = []
            carried_latents: List[tuple] = []
            texts: Dict[str, str] = {}
            last_ids: List[int] = []
            caches = []

            for role in self.role_order:
                spec = self.specs[role]
                rm = RoleMetrics(role=role)
                metrics.roles.append(rm)

                if cfg.share_kv_cache:
                    cache = bb.fork(root)          # reference fork, no re-encode
                    rm.reused_prefix_tokens += n_prefix
                else:
                    with Stopwatch(rm, bb.device):
                        _, _, cache = bb.forward(
                            cache=bb.new_cache(), input_ids=prefix_ids, role=role
                        )
                    rm.prefill_tokens += n_prefix

                suffix = "".join(carried_text) + role_instruction(role)
                suffix_ids = self._encode(suffix)
                with Stopwatch(rm, bb.device):
                    logits, _, cache = bb.forward(cache=cache, input_ids=suffix_ids, role=role)
                rm.prefill_tokens += int(suffix_ids.shape[1])

                for producer, slots in carried_latents:
                    with Stopwatch(rm, bb.device):
                        logits, _, cache = bb.forward(cache=cache, inputs_embeds=slots, role=role)
                    rm.prefill_tokens += int(slots.shape[1])

                is_producer = role != self.role_order[-1]
                use_latent = cfg.latent_handoff and is_producer and role in cfg.latent_roles
                budget = spec.latent_slots if use_latent else spec.text_budget

                ids, hidden, cache = self._decode(role, cache, budget, cfg, rm, logits)
                texts[role] = tok.decode(ids, skip_special_tokens=True)
                last_ids = ids
                caches.append(cache)

                if is_producer:
                    if use_latent and hidden is not None:
                        slots = self.bridge(hidden, role, spec.latent_slots)
                        carried_latents.append((role, slots))
                        rm.emitted_latent_slots += int(slots.shape[1])
                    else:
                        carried_text.append(f"\n{role.capitalize()}: {texts[role]}")

            metrics.kv_bytes = sum(cache_bytes(c) for c in caches) if caches else 0

        metrics.wall_seconds = total.seconds
        if bb.device.type == "cuda":
            metrics.peak_memory_bytes = int(torch.cuda.max_memory_allocated())

        return PipelineResult(
            task_id=task.task_id,
            mode=cfg.mode,
            answer_text=texts[self.role_order[-1]],
            answer_token_ids=last_ids,
            texts=texts,
            metrics=metrics,
        )
