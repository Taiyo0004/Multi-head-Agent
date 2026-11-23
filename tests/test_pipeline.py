"""End-to-end invariants of the Planner -> Critic -> Solver pipeline."""
import pytest

torch = pytest.importorskip("torch")

from mha_torch.pipeline import PipelineConfig, Task

MODES = [PipelineConfig(False, False), PipelineConfig(True, False),
         PipelineConfig(False, True), PipelineConfig(True, True)]
TASK = Task("t0", "Add 2 and 3, then explain the carry rule.")


@pytest.mark.parametrize("cfg", MODES, ids=lambda c: c.mode)
def test_every_mode_runs(agent, cfg):
    res = agent.run(TASK, cfg)
    assert set(res.texts) == {"planner", "critic", "solver"}
    assert res.metrics.tokens_generated > 0
    assert isinstance(res.answer_text, str)


def test_shared_cache_cuts_encoded_tokens(agent):
    base = agent.run(TASK, PipelineConfig(False, False)).metrics
    shared = agent.run(TASK, PipelineConfig(True, False)).metrics
    assert shared.tokens_encoded < base.tokens_encoded
    assert shared.prefix_tokens_reused > 0 and base.prefix_tokens_reused == 0


def test_latent_handoff_cuts_generated_tokens(agent):
    base = agent.run(TASK, PipelineConfig(False, False)).metrics
    latent = agent.run(TASK, PipelineConfig(False, True)).metrics
    assert latent.tokens_generated <= base.tokens_generated
    assert sum(r.emitted_latent_slots for r in latent.roles) > 0


def test_combined_mode_is_cheapest(agent):
    totals = {c.mode: agent.run(TASK, c).metrics.tokens_total for c in MODES}
    assert totals["shared_kv+latent"] == min(totals.values()), totals


def test_runs_are_deterministic_under_greedy_decoding(agent):
    cfg = PipelineConfig(True, True, greedy=True)
    assert agent.run(TASK, cfg).answer_token_ids == agent.run(TASK, cfg).answer_token_ids


def test_latent_slots_are_injectable(agent, backbone):
    """The bridge output must be consumable through inputs_embeds."""
    from mha_torch.compat import cache_seq_length
    prompt = backbone.tokenizer("hello there", return_tensors="pt",
                                add_special_tokens=False)["input_ids"]
    _, hidden, cache = backbone.forward(cache=backbone.new_cache(),
                                        input_ids=prompt, role="planner")
    slots = agent.bridge(hidden, "planner", 2)
    before = cache_seq_length(cache)
    logits, _, cache = backbone.forward(cache=cache, inputs_embeds=slots, role="critic")
    assert cache_seq_length(cache) == before + 2
    assert torch.isfinite(logits).all()


def test_param_report_shows_a_small_adapter(agent):
    rep = agent.param_report()
    assert rep["adapter_share_of_backbone_pct"] < 25.0  # tiny test model, loose bound
    assert rep["params_shared_backbone"] < rep["params_naive_one_model_per_role"]


def test_disabling_early_stop_forces_the_full_budget(agent):
    """Cost comparisons need every mode to decode the same number of steps."""
    cfg = PipelineConfig(True, False, stop_on_eos=False)
    m = agent.run(TASK, cfg).metrics
    budgets = {r.name: r.text_budget for r in agent.roles}
    for role in agent.role_order:
        steps = next(r.decode_steps for r in m.roles if r.role == role)
        assert steps == budgets[role], (role, steps)


def test_producer_decode_steps_excludes_the_final_head(agent):
    m = agent.run(TASK, PipelineConfig(False, False, stop_on_eos=False)).metrics
    final = next(r.decode_steps for r in m.roles if r.role == agent.role_order[-1])
    assert m.producer_decode_steps == m.tokens_generated - final


def test_latent_handoff_cuts_producer_decoding(agent):
    """The unconfounded claim: producers decode slots, not prose."""
    text = agent.run(TASK, PipelineConfig(False, False, stop_on_eos=False)).metrics
    latent = agent.run(TASK, PipelineConfig(False, True, stop_on_eos=False)).metrics
    assert latent.producer_decode_steps < text.producer_decode_steps
