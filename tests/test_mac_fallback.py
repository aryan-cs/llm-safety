from pathlib import Path

from cache_safety_erasure.config import parse_experiment_config


def test_mac_fallback_config_is_qwen3b_bounded_diagnostic() -> None:
    config, _raw = parse_experiment_config("configs/experiments/mac_qwen3b_causal_fallback.yaml")

    assert config.model.model_id == "Qwen/Qwen2.5-3B-Instruct"
    assert config.model.revision == "aa8e72537993ba99e69dfaafa59ed015b17504d1"
    assert config.model.dtype == "float16"
    assert config.model.allow_cpu_offload is False
    assert config.generation.max_new_tokens <= 96
    assert config.limit_per_suite == 5
    policy_names = [policy.name for policy in config.cache_policies]
    assert "none" in policy_names
    assert "kv_int4_sim" in policy_names
    assert "policy_pinned" in policy_names


def test_mac_fallback_script_is_bounded_and_cleans_project_local_model_cache() -> None:
    script = Path("scripts/run_mac_fallback.sh").read_text(encoding="utf-8")

    assert "configs/experiments/mac_qwen3b_causal_fallback.yaml" in script
    assert 'if [[ "$run_id" == h200_* ]]' in script
    assert "MAC_FALLBACK_MIN_UNIFIED_MEMORY_GB:-22" in script
    assert "torch.backends.mps.is_available()" in script
    assert "MAC_FALLBACK_CACHE_ROOT:-$(pwd)/.cache/mac_fallback" in script
    assert 'export HF_HOME="$cache_root/huggingface"' in script
    assert 'export TORCH_HOME="$cache_root/torch"' in script
    assert "scripts/cleanup_local_model_caches.sh --yes" in script
    assert "trap cleanup_model_cache EXIT" in script
    assert "--resume" in script
    assert "h200_qwen_full_sweep" not in script
    assert "qwen32b" not in script.lower()


def test_cleanup_local_model_cache_script_dry_runs_and_protects_evidence_paths() -> None:
    script = Path("scripts/cleanup_local_model_caches.sh").read_text(encoding="utf-8")

    assert "dry_run=1" in script
    assert "--allow-global" in script
    assert ".cache/mac_fallback/huggingface" in script
    assert ".cache/mac_fallback/torch" in script
    assert "HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE TORCH_HOME" in script
    assert 'Skipping non-repo cache path without --allow-global' in script
    assert '"$repo_dir/.cache/uv"' in script
    assert '"$repo_dir/results"' in script
    assert '"$repo_dir/snapshots"' in script


def test_readme_mac_fallback_is_separate_from_h200_evidence() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "bash scripts/run_mac_fallback.sh" in readme
    assert "fallback diagnostic, not a replacement" in readme
    assert "Do not resume `h200_*`" in readme
    assert "separate `mac_*` run ids" in readme
    assert "cleanup_local_model_caches.sh --yes" in readme
