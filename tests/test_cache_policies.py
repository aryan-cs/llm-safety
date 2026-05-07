import importlib.util

import pytest

from cache_safety_erasure.config import CachePolicyConfig

torch_spec = importlib.util.find_spec("torch")


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_sliding_window_preserves_tensor_shapes() -> None:
    import torch

    from cache_safety_erasure.cache_policies.registry import build_cache_policy

    cache = tuple((torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4)) for _ in range(3))
    policy = build_cache_policy(CachePolicyConfig(name="sliding_window", budget=6), seed=0)
    compressed, decision = policy.apply(cache, step=0)
    assert len(compressed) == 3
    assert compressed[0][0].shape == (1, 2, 6, 4)
    assert decision.retained_indices == tuple(range(4, 10))


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_random_policy_is_deterministic() -> None:
    import torch

    from cache_safety_erasure.cache_policies.registry import build_cache_policy

    cache = ((torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4)),)
    config = CachePolicyConfig(name="random_matched", budget=5, seed=123)
    policy_a = build_cache_policy(config, seed=0)
    policy_b = build_cache_policy(config, seed=999)
    _, decision_a = policy_a.apply(cache, step=3)
    _, decision_b = policy_b.apply(cache, step=3)
    assert decision_a.retained_indices == decision_b.retained_indices


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_quantize_dequantize_cache_roundtrip_shape() -> None:
    import torch

    from cache_safety_erasure.cache_policies.registry import build_cache_policy

    cache = ((torch.randn(1, 2, 7, 4), torch.randn(1, 2, 7, 4)),)
    policy = build_cache_policy(CachePolicyConfig(name="kv_int8_sim"), seed=0)
    quantized, decision = policy.apply(cache, step=0)
    assert quantized[0][0].shape == cache[0][0].shape
    assert decision.metadata["quantization_bits"] == 8


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_policy_pinned_enforces_budget_when_protected_span_is_long() -> None:
    import torch

    from cache_safety_erasure.cache_policies.registry import build_cache_policy

    cache = ((torch.randn(1, 2, 12, 4), torch.randn(1, 2, 12, 4)),)
    policy = build_cache_policy(CachePolicyConfig(name="policy_pinned", budget=4), seed=0)
    compressed, decision = policy.apply(cache, step=0, token_roles=["system"] * 10 + ["user"] * 2)
    assert compressed[0][0].shape[-2] == 4
    assert len(decision.retained_indices) == 4
    assert decision.metadata["protected_candidate_count"] == 10
    assert decision.metadata["protected_dropped_count"] == 6


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_user_pinned_uses_user_tokens_as_matched_control() -> None:
    import torch

    from cache_safety_erasure.cache_policies.registry import build_cache_policy

    cache = ((torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4)),)
    policy = build_cache_policy(CachePolicyConfig(name="user_pinned", budget=4), seed=0)
    _, decision = policy.apply(
        cache,
        step=0,
        token_roles=["system", "system", "template", "user", "user", "user", "user", "user", "user", "user"],
    )
    assert decision.metadata["protected_spans"] == "user"
    assert decision.metadata["protected_candidate_count"] == 7
    assert decision.metadata["protected_retained_count"] == 4
    assert set(decision.retained_indices).issubset(set(range(3, 10)))


@pytest.mark.skipif(torch_spec is None, reason="torch is not installed in the base interpreter")
def test_native_cache_limit_trims_before_sliding_window_attention() -> None:
    import torch

    from cache_safety_erasure.generation.hf_generate import _enforce_native_cache_limit

    cache = ((torch.randn(1, 2, 10, 4), torch.randn(1, 2, 10, 4)),)
    trimmed, decision = _enforce_native_cache_limit(
        cache,
        max_cache_len=6,
        step=3,
        token_roles=["system"] * 3 + ["user"] * 7,
    )

    assert decision is not None
    assert trimmed[0][0].shape[-2] == 6
    assert decision.policy_name == "native_sliding_window"
    assert decision.retained_indices == tuple(range(4, 10))
    assert decision.metadata["native_cache_limit"] == 6
    assert decision.metadata["evicted_system_tokens"] == 3
    assert decision.metadata["retained_user_tokens"] == 6
