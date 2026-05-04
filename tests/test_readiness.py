import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

import pytest
from check_publication_readiness import (
    _check_active_compression,
    _check_causal_patch_config,
    _check_causal_restoration_metric_readiness,
    _check_generation_matrix,
    _check_paper_assets,
)


def test_generation_matrix_detects_missing_policy_seed_rows() -> None:
    manifest = {
        "cache_policy_labels": ["none", "kv_int4_sim"],
        "seeds": [0, 1],
        "expected_generation_count": 4,
    }
    prompts = [{"suite": "public_refusal_safety", "prompt_id": "p1"}]
    generations = [
        {"suite": "public_refusal_safety", "prompt_id": "p1", "policy": "none", "seed": 0},
        {"suite": "public_refusal_safety", "prompt_id": "p1", "policy": "none", "seed": 1},
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p1",
            "policy": "kv_int4_sim",
            "seed": 0,
        },
    ]
    failures: list[str] = []

    _check_generation_matrix(manifest, prompts, generations, failures)

    assert any("generation row count is 3; expected 4" in failure for failure in failures)
    assert any("missing 1 rows" in failure for failure in failures)


def test_generation_matrix_accepts_complete_grid() -> None:
    manifest = {
        "cache_policy_labels": ["none", "kv_int4_sim"],
        "seeds": [0],
        "expected_generation_count": 2,
    }
    prompts = [{"suite": "public_refusal_safety", "prompt_id": "p1"}]
    generations = [
        {"suite": "public_refusal_safety", "prompt_id": "p1", "policy": "none", "seed": 0},
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p1",
            "policy": "kv_int4_sim",
            "seed": 0,
        },
    ]
    failures: list[str] = []

    _check_generation_matrix(manifest, prompts, generations, failures)

    assert failures == []


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_active_compression_detects_noop_budget(tmp_path: Path) -> None:
    import pandas as pd

    cache_stats = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "sliding_window__budget128",
                "decode_step": 0,
                "original_seq_len": 20,
                "evicted_count": 0,
                "retained_system_tokens": 5,
                "evicted_system_tokens": 0,
                "quantization_bits": None,
                "cache_l2_before": 2.0,
                "cache_l2_after": 2.0,
            }
        ]
    ).to_parquet(cache_stats, index=False)
    failures: list[str] = []

    _check_active_compression(
        cache_stats,
        {"cache_policy_labels": ["none", "sliding_window__budget128"]},
        failures,
    )

    assert any("appears inactive" in failure for failure in failures)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_active_compression_accepts_quantization(tmp_path: Path) -> None:
    import pandas as pd

    cache_stats = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "kv_int4_sim",
                "decode_step": 0,
                "original_seq_len": 20,
                "evicted_count": 0,
                "retained_system_tokens": 5,
                "evicted_system_tokens": 0,
                "quantization_bits": 4,
                "cache_l2_before": 2.0,
                "cache_l2_after": 1.9,
            }
        ]
    ).to_parquet(cache_stats, index=False)
    failures: list[str] = []

    _check_active_compression(cache_stats, {"cache_policy_labels": ["none", "kv_int4_sim"]}, failures)

    assert failures == []


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_active_compression_accepts_policy_pinned_protected_system_tokens(
    tmp_path: Path,
) -> None:
    import pandas as pd

    cache_stats = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "policy_pinned__budget128",
                "decode_step": 0,
                "original_seq_len": 200,
                "evicted_count": 72,
                "retained_system_tokens": 5,
                "evicted_system_tokens": 0,
                "protected_candidate_count": 5,
                "protected_retained_count": 5,
                "protected_dropped_count": 0,
                "quantization_bits": None,
                "cache_l2_before": 2.0,
                "cache_l2_after": 1.4,
            }
        ]
    ).to_parquet(cache_stats, index=False)
    failures: list[str] = []

    _check_active_compression(
        cache_stats,
        {"cache_policy_labels": ["none", "policy_pinned__budget128"]},
        failures,
    )

    assert failures == []


def test_causal_patch_config_requires_system_and_matched_user_control() -> None:
    failures: list[str] = []

    _check_causal_patch_config(
        [
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["system"],
                    "components": ["key", "value"],
                },
            },
            {
                "name": "kv_int4_sim",
                "patch_from_baseline": {
                    "token_roles": ["user"],
                    "match_token_count_to_roles": ["system"],
                    "components": ["key", "value"],
                },
            },
        ],
        failures,
    )

    assert failures == []


def test_causal_patch_config_rejects_fixed_token_only_patch() -> None:
    failures: list[str] = []

    _check_causal_patch_config(
        [{"name": "kv_int4_sim", "patch_from_baseline": {"token_indices": [0, 1]}}],
        failures,
    )

    assert any("role-derived" in failure for failure in failures)


def test_causal_restoration_readiness_requires_same_endpoint_intervals() -> None:
    failures: list[str] = []

    _check_causal_restoration_metric_readiness(
        {
            "causal_restoration": {
                "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.6,
                    "safety_restoration_fraction_ci": {"ci_low": 0.4, "ci_high": 0.7},
                },
                "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.2,
                    "safety_restoration_fraction_ci": {"ci_low": 0.1, "ci_high": 0.3},
                },
            }
        },
        failures,
    )

    assert failures == []


def test_causal_restoration_readiness_rejects_missing_intervals() -> None:
    failures: list[str] = []

    _check_causal_restoration_metric_readiness(
        {
            "causal_restoration": {
                "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.6,
                },
                "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem": {
                    "compressed_policy": "kv_int4_sim",
                    "refusal_restoration_fraction": 0.2,
                    "refusal_restoration_fraction_ci": {"ci_low": 0.1, "ci_high": 0.3},
                },
            }
        },
        failures,
    )

    assert any("missing `safety_restoration_fraction_ci`" in failure for failure in failures)
    assert any("same-endpoint" in failure for failure in failures)


def test_paper_artifact_manifest_checks_tables_and_sources(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    paper_dir = tmp_path / "paper"
    (results_dir / "figures").mkdir(parents=True)
    paper_dir.mkdir()
    for name in ["manifest.json", "metrics.json", "figures/manifest.json"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    table_path = paper_dir / "main_results_table.tex"
    table_path.write_text("table", encoding="utf-8")
    write_json(
        paper_dir / "artifact_manifest.json",
        {
            "tables": {
                "main_results_table.tex": {
                    "path": str(table_path),
                    "sha256": file_sha256(table_path),
                }
            },
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "metrics.json", "figures/manifest.json"]
            },
        },
    )
    failures: list[str] = []

    _check_paper_assets(paper_dir, results_dir, failures)

    assert failures == []
