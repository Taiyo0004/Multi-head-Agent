"""The efficiency claim rests on forking a prefix cache being exact."""
import pytest

torch = pytest.importorskip("torch")

from mha_torch.cache import deepcopy_cache, fork_cache
from mha_torch.compat import cache_bytes, cache_seq_length

ATOL = 1e-4


def ids(backbone, text):
    return backbone.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


@pytest.fixture(scope="module")
def prefilled(backbone):
    prefix = ids(backbone, "The quick brown fox jumps over the lazy dog and then")
    _, _, cache = backbone.forward(cache=backbone.new_cache(), input_ids=prefix, use_adapter=False)
    return prefix, cache


def test_prefix_is_cached(backbone, prefilled):
    prefix, cache = prefilled
    assert cache_seq_length(cache) == prefix.shape[1]


def test_forked_prefix_matches_full_reencode(backbone, prefilled):
    """Sharing the prefix must be free, not approximate."""
    prefix, root = prefilled
    suffix = ids(backbone, " runs away quickly")

    forked, _, _ = backbone.forward(cache=fork_cache(root), input_ids=suffix, role="planner")
    full = torch.cat([prefix, suffix], dim=1)
    fresh, _, _ = backbone.forward(cache=backbone.new_cache(), input_ids=full, role="planner")

    assert torch.allclose(forked.float(), fresh[:, -suffix.shape[1]:, :].float(), atol=ATOL)


def test_fork_matches_huggingface_deepcopy_recipe(backbone, prefilled):
    _, root = prefilled
    suffix = ids(backbone, " sleeps")
    a, _, _ = backbone.forward(cache=fork_cache(root), input_ids=suffix, role="planner")
    b, _, _ = backbone.forward(cache=deepcopy_cache(root), input_ids=suffix, role="planner")
    assert torch.allclose(a.float(), b.float(), atol=ATOL)


def test_fork_does_not_mutate_the_parent(backbone, prefilled):
    prefix, root = prefilled
    before = cache_seq_length(root)
    backbone.forward(cache=fork_cache(root), input_ids=ids(backbone, " and more text"), role="critic")
    assert cache_seq_length(root) == before == prefix.shape[1]


def test_siblings_are_independent(backbone, prefilled):
    prefix, root = prefilled
    a, b = fork_cache(root), fork_cache(root)
    backbone.forward(cache=a, input_ids=ids(backbone, " one two three"), role="planner")
    backbone.forward(cache=b, input_ids=ids(backbone, " four"), role="critic")
    assert cache_seq_length(a) != cache_seq_length(b)
    assert cache_seq_length(root) == prefix.shape[1]


def test_fork_allocates_no_new_kv_storage(backbone, prefilled):
    """The whole point: a fork shares tensors, a deepcopy duplicates them."""
    _, root = prefilled
    assert cache_bytes(fork_cache(root)) <= cache_bytes(deepcopy_cache(root))
