from pathlib import Path

import pytest

from cache_safety_erasure.config import CachePolicyConfig, parse_experiment_config


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_parse_smoke_config() -> None:
    config, raw = parse_experiment_config(Path("configs/experiments/smoke_mock.yaml"))
    assert raw["run"]["name"] == "smoke_mock"
    assert config.model.provider == "mock"
    assert config.model.allow_cpu_offload is False
    assert config.cache_policies[0].name == "none"
    assert "system_leakage" in config.prompt_suites


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_tiny_hf_smoke_explicitly_allows_offload() -> None:
    config, _raw = parse_experiment_config(Path("configs/experiments/tiny_hf_smoke.yaml"))
    assert config.model.allow_cpu_offload is True


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_h200_public_sweep_uses_prompt_clusters_not_repeated_deterministic_seeds() -> None:
    config, _raw = parse_experiment_config(Path("configs/experiments/h200_public_qwen14b.yaml"))
    assert config.generation.do_sample is False
    assert config.seeds == (0,)
    assert config.limit_per_suite is None
    assert "public_system_leakage" in config.prompt_suites


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("yaml") is None,
    reason="PyYAML is not installed in the base interpreter",
)
def test_h200_ci_extension_focuses_policy_set_for_prompt_count() -> None:
    config, _raw = parse_experiment_config(
        Path("configs/experiments/h200_qwen14b_ci_extension.yaml")
    )
    assert config.seeds == (0,)
    assert {policy.name for policy in config.cache_policies} == {
        "none",
        "sliding_window",
        "kv_int4_sim",
        "policy_pinned",
    }


def test_h200_sweep_run_ids_match_paper_figure_paths() -> None:
    script = Path("scripts/run_h200_sweep.sh").read_text(encoding="utf-8")
    tex = Path("paper/latex/main.tex").read_text(encoding="utf-8")

    assert 'full_run_id="${FULL_RUN_ID:-h200_qwen_full_sweep}"' in script
    assert 'causal_run_id="${CAUSAL_RUN_ID:-h200_causal_patch_qwen7b}"' in script
    assert "scripts/assess_claims.py" in script
    assert "../../results/h200_qwen_full_sweep/figures/" in tex
    assert "../../results/h200_causal_patch_qwen7b/figures/" in tex


def test_h200_launcher_revalidates_after_gpu_gate() -> None:
    script = Path("scripts/wait_and_run_h200_sweep.sh").read_text(encoding="utf-8")

    assert 'sync_and_validate "Pre-gate"' in script
    assert 'sync_and_validate "Post-gate"' in script
    assert "scripts/wait_for_h200_gpu.sh" in script
    assert script.index('sync_and_validate "Post-gate"') > script.index(
        "bash scripts/wait_for_h200_gpu.sh"
    )
    assert "bash -n" in script


def test_h200_wait_script_logs_visible_gpu_users() -> None:
    script = Path("scripts/wait_for_h200_gpu.sh").read_text(encoding="utf-8")

    assert "--query-compute-apps=pid,process_name,used_memory" in script
    assert "nvidia-smi pmon -c 1" in script
    assert "log_visible_gpu_users" in script


def test_h200_readiness_uses_paper_grade_prompt_thresholds() -> None:
    primary = Path("scripts/run_h200_sweep.sh").read_text(encoding="utf-8")
    extension = Path("scripts/run_h200_ci_extension.sh").read_text(encoding="utf-8")
    qwen32 = Path("scripts/run_qwen32b_followup.sh").read_text(encoding="utf-8")
    publication = Path("scripts/build_publication_artifacts.sh").read_text(encoding="utf-8")

    for script in [primary, extension, qwen32]:
        assert "scripts/check_prepared_suites.py" in script
        assert "--require-public-provenance" in script
    for script in [primary, extension, qwen32, publication]:
        assert "--min-prompts-per-suite 600" in script
        assert "--suite-min-prompts system_leakage=2" in script
    for script in [primary, extension, qwen32]:
        assert "--suite-min-prompts public_xstest_safe=200" in script


def test_h200_scripts_export_multi_annotator_audit_templates() -> None:
    for script_path in [
        Path("scripts/run_h200_sweep.sh"),
        Path("scripts/run_h200_ci_extension.sh"),
        Path("scripts/run_qwen32b_followup.sh"),
    ]:
        script = script_path.read_text(encoding="utf-8")
        assert 'audit_annotator_template_count="${AUDIT_ANNOTATOR_TEMPLATE_COUNT:-2}"' in script
        assert '--annotator-template-count "$audit_annotator_template_count"' in script


def test_publication_artifact_builder_fails_without_real_results() -> None:
    script = Path("scripts/build_publication_artifacts.sh").read_text(encoding="utf-8")

    assert "require_result_artifacts" in script
    assert "Missing required result artifact" in script
    assert "require_human_audit_artifacts" in script
    assert "Missing required human-audit artifact" in script
    assert "paper/cache_mediated_safety_erasure.pdf" in script
    assert "scripts/package_arxiv_submission.py" in script
    assert "scripts/assess_claims.py" in script
    assert "scripts/report_publication_status.py" in script
    assert "REQUIRE_COMPLETE_PAPER=1 bash scripts/build_paper_pdf.sh" in script
    assert "publication_status.md" in script
    assert "--require-cache-mediated-claim" in script
    assert "--require-human-audit-support" in script
    assert "--fail-if-not-ready" in script
    assert "rm -f paper/cache_mediated_safety_erasure.pdf" in script
    assert script.index("write_publication_status --fail-if-not-ready") < script.index(
        "uv run python scripts/package_arxiv_submission.py"
    )
    assert "REQUIRE_HUMAN_AUDIT" not in script
    assert "REQUIRE_CACHE_MEDIATED_CLAIM" not in script
    assert "--required-figure prompt_effect_constellation" in script
    assert "--required-figure safety_state_atlas" in script
    assert "--required-figure causal_restoration_fraction" in script


def test_complete_paper_build_checks_publication_status_before_latex() -> None:
    script = Path("scripts/build_paper_pdf.sh").read_text(encoding="utf-8")

    assert "scripts/check_latex_placeholders.py" in script
    assert "scripts/check_paper_asset_freshness.py" in script
    assert "scripts/report_publication_status.py" in script
    assert "--allow-missing-paper-pdf" in script
    assert "--fail-if-not-ready" in script


def test_h200_scripts_use_composite_public_refusal_suite() -> None:
    for script_path in [
        Path("scripts/run_h200_sweep.sh"),
        Path("scripts/run_h200_ci_extension.sh"),
        Path("scripts/run_qwen32b_followup.sh"),
        Path("scripts/bootstrap_h200.sh"),
    ]:
        script = script_path.read_text(encoding="utf-8")
        assert "--suite public_refusal_combo" in script
        assert "--suite cyberec_prompt_injection_leakage" in script
        assert "--suite advbench" not in script


def test_patch_policy_label_includes_components() -> None:
    from cache_safety_erasure.cache_policies.registry import cache_policy_label

    label = cache_policy_label(
        CachePolicyConfig(
            name="kv_int4_sim",
            patch_from_baseline={"components": ["key"], "token_indices": [0, 1, 2]},
        )
    )
    assert label == "kv_int4_sim__patchkey__tok0to2"


def test_patch_policy_label_includes_role_controls() -> None:
    from cache_safety_erasure.cache_policies.registry import cache_policy_label

    label = cache_policy_label(
        CachePolicyConfig(
            name="kv_int4_sim",
            patch_from_baseline={
                "components": ["key", "value"],
                "token_roles": ["user"],
                "match_token_count_to_roles": ["system"],
                "max_tokens": 16,
                "selection": "first",
            },
        )
    )
    assert label == "kv_int4_sim__patchkey-value__roleuser__matchsystem__max16__selfirst"
