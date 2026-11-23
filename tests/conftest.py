"""Shared fixtures.

Torch tests run against a tiny randomly-initialised model so the suite is
fast and needs no large download.  Override with:

    set MHA_TEST_MODEL=HuggingFaceTB/SmolLM2-135M-Instruct     (Windows)
    export MHA_TEST_MODEL=...                                   (bash)
"""
from __future__ import annotations

import os

import pytest

TEST_MODEL = os.environ.get("MHA_TEST_MODEL", "hf-internal-testing/tiny-random-LlamaForCausalLM")


@pytest.fixture(scope="session")
def backbone():
    """A tiny backbone with three role adapters, loaded once per session."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from mha_torch.backbone import RoleSpec, TorchBackbone

    roles = [RoleSpec("planner", rank=4, text_budget=4, latent_slots=2),
             RoleSpec("critic", rank=4, text_budget=3, latent_slots=2),
             RoleSpec("solver", rank=4, text_budget=5, latent_slots=2)]
    try:
        return TorchBackbone(TEST_MODEL, roles=roles, device="cpu", dtype="fp32")
    except Exception as exc:  # offline, or the test model moved
        pytest.skip(f"could not load {TEST_MODEL}: {exc}")


@pytest.fixture(scope="session")
def agent(backbone):
    from mha_torch.pipeline import MultiHeadAgent
    return MultiHeadAgent(backbone=backbone)
