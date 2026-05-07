import json
from pathlib import Path

import pytest

from cache_safety_erasure.utils.io import (
    append_jsonl,
    file_sha256,
    read_jsonl,
    read_jsonl_tolerant,
    write_json,
)


def test_json_artifacts_roundtrip(tmp_path: Path) -> None:
    write_json(tmp_path / "environment.json", {"ok": True})
    append_jsonl(tmp_path / "generations.jsonl", [{"prompt_id": "p1", "text": "hello"}])
    assert read_jsonl(tmp_path / "generations.jsonl")[0]["prompt_id"] == "p1"


def test_tolerant_jsonl_reader_quarantines_corrupt_tail(tmp_path: Path) -> None:
    path = tmp_path / "generations.jsonl"
    path.write_text('{"prompt_id": "p1"}\n{"prompt_id": ', encoding="utf-8")

    rows, quarantine_path = read_jsonl_tolerant(path)

    assert rows == [{"prompt_id": "p1"}]
    assert read_jsonl(path) == [{"prompt_id": "p1"}]
    assert quarantine_path is not None
    assert quarantine_path.exists()
    assert quarantine_path.read_text(encoding="utf-8") == '{"prompt_id": '


def test_cache_stats_sink_preserves_existing_rows_on_resume(tmp_path: Path) -> None:
    import sys

    import pandas as pd

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _CacheStatsSink

    path = tmp_path / "cache_stats.parquet"
    first = _CacheStatsSink(path, resume=False)
    first.write([{"prompt_id": "p1", "seed": 0, "policy": "none", "decode_step": 0}])
    first.close()

    second = _CacheStatsSink(path, resume=True)
    second.write([{"prompt_id": "p2", "seed": 0, "policy": "kv_int4_sim", "decode_step": 0}])
    second.close()

    df = pd.read_parquet(path)
    assert list(df["prompt_id"]) == ["p1", "p2"]


def test_resume_reconciliation_quarantines_generations_without_cache_stats(
    tmp_path: Path,
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _reconcile_resume_generations

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"prompt_id": "p1", "suite": "suite", "policy": "none", "seed": 0},
        {"prompt_id": "p2", "suite": "suite", "policy": "kv_int4_sim", "seed": 0},
    ]
    append_jsonl(run_dir / "generations.jsonl", rows)

    kept = _reconcile_resume_generations(run_dir, rows)

    assert kept == []
    assert read_jsonl(run_dir / "generations.jsonl") == []
    orphaned = list(run_dir.glob("generations.orphaned_without_cache_stats.*.jsonl"))
    assert len(orphaned) == 1
    assert read_jsonl(orphaned[0]) == rows


def test_resume_reconciliation_keeps_only_generations_with_cache_stats(
    tmp_path: Path,
) -> None:
    import sys

    import pandas as pd

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _reconcile_resume_generations

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [
        {"prompt_id": "p1", "suite": "suite", "policy": "none", "seed": 0},
        {"prompt_id": "p2", "suite": "suite", "policy": "kv_int4_sim", "seed": 0},
    ]
    append_jsonl(run_dir / "generations.jsonl", rows)
    pd.DataFrame(
        [{"prompt_id": "p1", "policy": "none", "seed": 0, "decode_step": 0}]
    ).to_parquet(run_dir / "cache_stats.parquet", index=False)

    kept = _reconcile_resume_generations(run_dir, rows)

    assert kept == [rows[0]]
    assert read_jsonl(run_dir / "generations.jsonl") == [rows[0]]
    orphaned = list(run_dir.glob("generations.orphaned_without_cache_stats.*.jsonl"))
    assert len(orphaned) == 1
    assert read_jsonl(orphaned[0]) == rows


def test_resume_reconciliation_refuses_corrupt_cache_stats_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _reconcile_resume_generations

    monkeypatch.delenv("ALLOW_CORRUPT_CACHE_STATS_RESET", raising=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [{"prompt_id": "p1", "suite": "suite", "policy": "none", "seed": 0}]
    append_jsonl(run_dir / "generations.jsonl", rows)
    (run_dir / "cache_stats.parquet").write_bytes(b"not a parquet file")

    with pytest.raises(RuntimeError, match="cache_stats.parquet is unreadable"):
        _reconcile_resume_generations(run_dir, rows)

    assert read_jsonl(run_dir / "generations.jsonl") == rows
    assert (run_dir / "cache_stats.parquet").exists()
    assert not list(run_dir.glob("generations.orphaned_without_cache_stats.*.jsonl"))
    assert not list(run_dir.glob("generations.corrupt_cache_stats_reset.*.jsonl"))


def test_resume_reconciliation_can_explicitly_archive_corrupt_cache_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _reconcile_resume_generations

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows = [{"prompt_id": "p1", "suite": "suite", "policy": "none", "seed": 0}]
    append_jsonl(run_dir / "generations.jsonl", rows)
    (run_dir / "cache_stats.parquet").write_bytes(b"not a parquet file")
    monkeypatch.setenv("ALLOW_CORRUPT_CACHE_STATS_RESET", "1")

    kept = _reconcile_resume_generations(run_dir, rows)

    assert kept == []
    assert read_jsonl(run_dir / "generations.jsonl") == []
    generation_archives = list(run_dir.glob("generations.corrupt_cache_stats_reset.*.jsonl"))
    cache_archives = list(run_dir.glob("cache_stats.parquet.corrupt.*"))
    assert len(generation_archives) == 1
    assert len(cache_archives) == 1
    assert read_jsonl(generation_archives[0]) == rows
    assert cache_archives[0].read_bytes() == b"not a parquet file"
    assert not (run_dir / "cache_stats.parquet").exists()


def test_cache_stats_sink_uses_stable_schema_for_sparse_batches(tmp_path: Path) -> None:
    import sys

    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _CacheStatsSink

    path = tmp_path / "cache_stats.parquet"
    sink = _CacheStatsSink(path, resume=False)
    sink.write(
        [
            {
                "prompt_id": "p1",
                "seed": 0,
                "policy": "policy_pinned__budget128__sink8",
                "decode_step": 0,
                "original_seq_len": 32,
                "retained_count": 32,
                "evicted_count": 0,
                "retained_indices": "0,1",
                "evicted_indices": "",
                "cache_l2_before": 1.25,
                "cache_l2_after": 1.25,
                "retained_generated_tokens": 0,
                "evicted_generated_tokens": 0,
                "sink_tokens": 8,
                "protected_spans": "system,policy",
                "protected_candidate_count": 4,
                "protected_retained_count": 4,
                "protected_dropped_count": 0,
            }
        ]
    )
    sink.write(
        [
            {
                "prompt_id": "p2",
                "seed": 0,
                "policy": "sliding_window__budget64",
                "decode_step": 1,
                "original_seq_len": 96,
                "retained_count": 64,
                "evicted_count": 32,
                "retained_indices": "32,33",
                "evicted_indices": "0,1",
                "cache_l2_before": 2.5,
                "cache_l2_after": 2.0,
                "retained_template_tokens": 5,
                "retained_system_tokens": 3,
                "retained_user_tokens": 56,
                "evicted_template_tokens": 1,
                "evicted_system_tokens": 2,
                "evicted_user_tokens": 29,
            }
        ]
    )
    sink.close()

    parquet_file = pq.ParquetFile(path)
    assert parquet_file.metadata.num_rows == 2
    schema = parquet_file.schema_arrow
    assert str(schema.field("protected_spans").type) == "large_string"
    assert str(schema.field("sink_tokens").type) == "int64"
    assert str(schema.field("retained_generated_tokens").type) == "int64"


def test_cache_stats_parquet_rebuild_uses_durable_jsonl_checkpoint(tmp_path: Path) -> None:
    import sys

    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _CacheStatsSink, _rebuild_cache_stats_parquet_from_jsonl

    parquet_path = tmp_path / "cache_stats.parquet"
    jsonl_path = tmp_path / "cache_stats.jsonl"
    sink = _CacheStatsSink(parquet_path, resume=False)
    sink.write([{"prompt_id": "p1", "seed": 0, "policy": "none", "decode_step": 0}])
    sink.close()
    append_jsonl(
        jsonl_path,
        [
            {"prompt_id": "p1", "seed": 0, "policy": "none", "decode_step": 0},
            {"prompt_id": "p2", "seed": 0, "policy": "sliding_window", "decode_step": 0},
        ],
    )

    _rebuild_cache_stats_parquet_from_jsonl(jsonl_path, parquet_path)

    table = pq.read_table(parquet_path, columns=["prompt_id"])
    assert table.column("prompt_id").to_pylist() == ["p1", "p2"]


def test_empty_cache_stats_sink_writes_readable_schema(tmp_path: Path) -> None:
    import sys

    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import CACHE_STATS_COLUMNS, _CacheStatsSink

    path = tmp_path / "cache_stats.parquet"
    sink = _CacheStatsSink(path, resume=False)
    sink.close()

    parquet_file = pq.ParquetFile(path)
    assert parquet_file.metadata.num_rows == 0
    assert parquet_file.schema_arrow.names == CACHE_STATS_COLUMNS


def test_cache_stats_sink_migrates_legacy_sparse_schema_on_resume(tmp_path: Path) -> None:
    import sys

    import pandas as pd
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _CacheStatsSink

    path = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "prompt_id": "legacy",
                "seed": 0,
                "policy": "policy_pinned__budget128__sink8",
                "decode_step": 0,
                "original_seq_len": 64,
                "retained_count": 64,
                "evicted_count": 0,
                "retained_indices": "0,1",
                "evicted_indices": "",
                "cache_l2_before": 1.0,
                "cache_l2_after": 1.0,
                "retained_generated_tokens": 0.0,
                "sink_tokens": 8.0,
                "protected_spans": "system,policy",
            }
        ]
    ).to_parquet(path, index=False)

    sink = _CacheStatsSink(path, resume=True)
    sink.write(
        [
            {
                "prompt_id": "new",
                "seed": 1,
                "policy": "attention_h2o__budget128",
                "decode_step": 2,
                "original_seq_len": 128,
                "retained_count": 128,
                "evicted_count": 0,
                "retained_indices": "0,1",
                "evicted_indices": "",
                "cache_l2_before": 2.0,
                "cache_l2_after": 1.75,
                "quantization_bits": 4,
                "attention_scores_used": True,
                "patched_from_baseline": False,
            }
        ]
    )
    sink.close()

    table = pq.read_table(path)
    assert table.column("prompt_id").to_pylist() == ["legacy", "new"]
    schema = table.schema
    assert str(schema.field("sink_tokens").type) == "int64"
    assert str(schema.field("retained_generated_tokens").type) == "int64"
    assert str(schema.field("attention_scores_used").type) == "bool"
    assert str(schema.field("protected_spans").type) == "large_string"


def test_cache_stats_resume_recovers_valid_temp_checkpoint(tmp_path: Path) -> None:
    import sys

    import pandas as pd

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _cache_stats_generation_keys

    path = tmp_path / "cache_stats.parquet"
    temp_path = path.with_suffix(".parquet.tmp")
    pd.DataFrame([{"prompt_id": "p1", "policy": "none", "seed": 0}]).to_parquet(
        path, index=False
    )
    pd.DataFrame(
        [
            {"prompt_id": "p1", "policy": "none", "seed": 0},
            {"prompt_id": "p2", "policy": "kv_int4_sim", "seed": 0},
        ]
    ).to_parquet(temp_path, index=False)

    keys = _cache_stats_generation_keys(path)

    assert ("p1", "none", 0) in keys
    assert ("p2", "kv_int4_sim", 0) in keys
    assert not temp_path.exists()
    assert list(tmp_path.glob("cache_stats.parquet.pre_temp_recovery.*"))


def test_resume_manifest_validation_rejects_matrix_drift(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _validate_resume_manifest

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    base_manifest = _minimal_resume_manifest()
    write_json(run_dir / "manifest.json", {**base_manifest, "model_id": "old/model"})

    try:
        _validate_resume_manifest(
            run_dir,
            {**base_manifest, "model_id": "new/model"},
            {"git_commit": base_manifest["git_commit"]},
        )
    except RuntimeError as exc:
        assert "resume_manifest_mismatch:model_id" in str(exc)
    else:
        raise AssertionError("expected resume manifest drift to fail")


def test_resume_manifest_validation_requires_explicit_commit_mismatch(
    tmp_path: Path, monkeypatch
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _validate_resume_manifest

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest = _minimal_resume_manifest()
    write_json(run_dir / "manifest.json", manifest)

    try:
        _validate_resume_manifest(run_dir, manifest, {"git_commit": "different"})
    except RuntimeError as exc:
        assert "resume_manifest_mismatch:git_commit" in str(exc)
    else:
        raise AssertionError("expected git commit mismatch to fail")

    monkeypatch.setenv("ALLOW_RESUME_GIT_MISMATCH", "1")
    _validate_resume_manifest(run_dir, manifest, {"git_commit": "different"})


def test_policy_manifest_uses_json_stable_defaults() -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from run_experiment import _policy_manifest

    from cache_safety_erasure.config import CachePolicyConfig

    manifest = _policy_manifest(CachePolicyConfig(name="none"))

    assert manifest["protected_spans"] == ["system", "policy"]
    assert isinstance(manifest["protected_spans"], list)


def _minimal_resume_manifest() -> dict:
    return {
        "run_name": "run",
        "model_id": "model",
        "model_provider": "mock",
        "model_config": {"provider": "mock", "model_id": "model"},
        "git_commit": "commit",
        "resume_compatible_config_sha256": "abc",
        "prompt_suites": ["suite"],
        "prompt_counts": {"suite": 1},
        "prompt_suite_manifests": {"suite": {"sha256": "def"}},
        "cache_policy_configs": [{"name": "none"}],
        "cache_policy_labels": ["none"],
        "seeds": [0],
        "limit_per_suite": 1,
        "expected_generation_count": 1,
    }


def test_latex_table_export_escapes_policy_names(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from export_paper_assets import write_latex_table

    table_path = tmp_path / "table.tex"
    write_latex_table(
        table_path,
        ["policy", "policy_level_ssei"],
        [{"policy": "sliding_window__budget64", "policy_level_ssei": 0.125}],
        caption="Caption with SSEI.",
        label="tab:test",
    )

    text = table_path.read_text(encoding="utf-8")
    assert "sliding\\_window\\_\\_budget64" in text
    assert "0.125" in text
    assert "\\label{tab:test}" in text


def test_latex_macro_export_writes_headline_result_macros(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from export_paper_assets import write_latex_macros

    macro_path = tmp_path / "result_macros.tex"
    write_latex_macros(
        macro_path,
        {
            "publication_summary": {"policies": {"none": {}, "kv_int4_sim": {}}},
            "policy_level_contrasts": {
                "kv_int4_sim": {
                    "selective_safety_erasure_index": 0.25,
                    "selective_safety_erasure_index_ci": {
                        "ci_low": 0.1,
                        "ci_high": 0.4,
                        "n_safety": 12,
                        "n_capability": 8,
                    },
                }
            },
        },
        tmp_path / "results" / "run_001",
        "Primary",
    )

    text = macro_path.read_text(encoding="utf-8")
    assert "\\renewcommand{\\PrimaryRunId}{primary public sweep}" in text
    assert "\\renewcommand{\\PrimaryTopSSEIPolicy}{kv\\_int4\\_sim}" in text
    assert "\\renewcommand{\\PrimaryTopSSEI}{0.250}" in text
    assert "\\renewcommand{\\PrimaryTopSSEICILow}{0.100}" in text


def test_export_paper_assets_writes_ci_bearing_tables(tmp_path: Path) -> None:
    import subprocess
    import sys

    results_dir = tmp_path / "results" / "run_001"
    paper_dir = tmp_path / "paper" / "generated" / "run_001"
    (results_dir / "figures").mkdir(parents=True)
    write_json(
        results_dir / "manifest.json",
        {"run_name": "run_001", "git_commit": "run-commit", "git_dirty": False},
    )
    write_json(results_dir / "figures" / "manifest.json", {"figures": {}})
    write_json(
        results_dir / "metrics.json",
        {
            "publication_summary": {"policies": {"kv_int4_sim": {}}},
            "policy_level_contrasts": {
                "kv_int4_sim": {
                    "selective_safety_erasure_index": 0.25,
                    "selective_safety_erasure_index_ci": {
                        "ci_low": 0.10,
                        "ci_high": 0.40,
                        "n_safety": 12,
                        "n_capability": 8,
                    },
                }
            },
            "selective_safety_erasure": {
                "public_refusal_safety::kv_int4_sim": {
                    "safety_degradation": 0.12,
                    "capability_degradation": 0.02,
                    "selective_safety_erasure_index": 0.10,
                    "paired_safety_degradation_ci": {
                        "paired_n": 100,
                        "cluster_n": 95,
                        "ci_low": 0.05,
                        "ci_high": 0.18,
                    },
                }
            },
            "causal_restoration": {
                "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.62,
                    "safety_restoration_fraction_ci": {"ci_low": 0.50, "ci_high": 0.72},
                    "refusal_restoration_fraction": 0.55,
                    "refusal_restoration_fraction_ci": {"ci_low": 0.44, "ci_high": 0.66},
                    "leakage_avoidance_restoration_fraction": 0.20,
                    "leakage_avoidance_restoration_fraction_ci": {
                        "ci_low": 0.10,
                        "ci_high": 0.30,
                    },
                },
                "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.20,
                    "safety_restoration_fraction_ci": {"ci_low": 0.12, "ci_high": 0.30},
                    "refusal_restoration_fraction": 0.18,
                    "refusal_restoration_fraction_ci": {"ci_low": 0.10, "ci_high": 0.26},
                },
                "public_refusal_safety::policy_pinned__budget128__sink8": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.45,
                    "safety_restoration_fraction_ci": {"ci_low": 0.34, "ci_high": 0.56},
                    "refusal_restoration_fraction": 0.41,
                    "refusal_restoration_fraction_ci": {"ci_low": 0.30, "ci_high": 0.52},
                },
            },
        },
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/export_paper_assets.py",
            "--results-dir",
            str(results_dir),
            "--paper-dir",
            str(paper_dir),
            "--macro-prefix",
            "Primary",
        ],
        check=True,
    )

    suite_tex = (paper_dir / "suite_level_effects_table.tex").read_text(encoding="utf-8")
    causal_tex = (paper_dir / "causal_restoration_table.tex").read_text(encoding="utf-8")

    assert "safety ci low" in suite_tex
    assert "safety ci high" in suite_tex
    assert "0.050" in suite_tex
    assert "safety ci low" in causal_tex
    assert "refusal ci low" in causal_tex
    assert "leakage avoidance ci high" in causal_tex
    assert "rolesystem" in causal_tex
    assert "roleuser" in causal_tex
    assert "policy\\_pinned" in causal_tex
    artifact_manifest = json.loads(
        (paper_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert artifact_manifest["source_run_git_commit"] == "run-commit"
    assert artifact_manifest["macro_prefix"] == "Primary"
    assert artifact_manifest["analysis_git_commit"]


def test_paper_asset_freshness_recomputes_generated_outputs(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness
    from export_paper_assets import TABLE_FILES, export_paper_assets

    results_dir = tmp_path / "results" / "h200_qwen_full_sweep"
    paper_dir = tmp_path / "paper" / "generated" / "h200_qwen_full_sweep"
    _write_paper_asset_export_fixture(results_dir)
    export_paper_assets(results_dir, paper_dir, "Primary")
    _mark_clean_export_manifest(paper_dir)

    assert (
        check_paper_asset_freshness(
            paper_dir,
            results_dir,
            required_tables=TABLE_FILES,
            require_recomputed_output=True,
        )
        == []
    )

    table = paper_dir / "main_results_table.tex"
    table.write_text(
        table.read_text(encoding="utf-8").replace("0.250", "0.251", 1),
        encoding="utf-8",
    )
    _refresh_manifest_table_hash(paper_dir, "main_results_table.tex")

    failures = check_paper_asset_freshness(
        paper_dir,
        results_dir,
        required_tables=TABLE_FILES,
        require_recomputed_output=True,
    )

    assert "paper artifact table `main_results_table.tex` hash is stale" not in failures
    assert "paper artifact generated output `main_results_table.tex` differs from metrics export" in failures


def test_paper_asset_freshness_can_require_current_analysis_commit(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness
    from export_paper_assets import export_paper_assets

    results_dir = tmp_path / "results" / "h200_qwen_full_sweep"
    paper_dir = tmp_path / "paper" / "generated" / "h200_qwen_full_sweep"
    _write_paper_asset_export_fixture(results_dir)
    export_paper_assets(results_dir, paper_dir, "Primary")
    _mark_clean_export_manifest(paper_dir)
    manifest_path = paper_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_git_commit"] = "f" * 40
    write_json(manifest_path, manifest)

    assert check_paper_asset_freshness(paper_dir, results_dir) == []
    failures = check_paper_asset_freshness(
        paper_dir,
        results_dir,
        require_current_analysis_commit=True,
    )

    assert f"paper artifact manifest analysis git commit is stale: {results_dir}" in failures


def test_paper_asset_freshness_recompute_infers_causal_macro_prefix(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness
    from export_paper_assets import TABLE_FILES, export_paper_assets

    results_dir = tmp_path / "results" / "h200_causal_patch_qwen7b"
    paper_dir = tmp_path / "paper" / "generated" / "h200_causal_patch_qwen7b"
    _write_paper_asset_export_fixture(results_dir)
    export_paper_assets(results_dir, paper_dir, "Causal")
    _mark_clean_export_manifest(paper_dir, drop_macro_prefix=True)

    assert (
        check_paper_asset_freshness(
            paper_dir,
            results_dir,
            required_tables=TABLE_FILES,
            require_recomputed_output=True,
        )
        == []
    )


def test_paper_asset_freshness_recompute_fails_without_parseable_macro_prefix(
    tmp_path: Path,
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness
    from export_paper_assets import TABLE_FILES, export_paper_assets

    results_dir = tmp_path / "results" / "legacy_generated_run"
    paper_dir = tmp_path / "paper" / "generated" / "legacy_generated_run"
    _write_paper_asset_export_fixture(results_dir)
    export_paper_assets(results_dir, paper_dir, "Primary")
    _mark_clean_export_manifest(paper_dir, drop_macro_prefix=True)
    (paper_dir / "result_macros.tex").write_text(
        "% legacy malformed macros without a RunId declaration\n",
        encoding="utf-8",
    )
    _refresh_manifest_table_hash(paper_dir, "result_macros.tex")

    failures = check_paper_asset_freshness(
        paper_dir,
        results_dir,
        required_tables=TABLE_FILES,
        require_recomputed_output=True,
    )

    assert any("cannot infer macro prefix from result_macros.tex" in failure for failure in failures)


def test_paper_asset_freshness_recompute_catches_metrics_after_source_refresh(
    tmp_path: Path,
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness
    from export_paper_assets import TABLE_FILES, export_paper_assets

    results_dir = tmp_path / "results" / "h200_qwen_full_sweep"
    paper_dir = tmp_path / "paper" / "generated" / "h200_qwen_full_sweep"
    _write_paper_asset_export_fixture(results_dir)
    export_paper_assets(results_dir, paper_dir, "Primary")
    _mark_clean_export_manifest(paper_dir)

    metrics_path = results_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["policy_level_contrasts"]["kv_int4_sim"]["selective_safety_erasure_index"] = 0.33
    write_json(metrics_path, metrics)
    _refresh_manifest_source_hash(paper_dir, results_dir, "metrics.json")

    failures = check_paper_asset_freshness(
        paper_dir,
        results_dir,
        required_tables=TABLE_FILES,
        require_recomputed_output=True,
    )

    assert "paper artifact source `metrics.json` hash is stale" not in failures
    assert any("differs from metrics export" in failure for failure in failures)


def test_paper_asset_freshness_detects_stale_tables_and_sources(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness

    paper_dir = tmp_path / "paper" / "generated" / "run"
    results_dir = tmp_path / "results" / "run"
    paper_dir.mkdir(parents=True)
    (results_dir / "figures").mkdir(parents=True)
    table = paper_dir / "result_macros.tex"
    table.write_text("fresh\n", encoding="utf-8")
    for name in ["manifest.json", "metrics.json", "figures/manifest.json"]:
        path = results_dir / name
        if name == "manifest.json":
            write_json(path, {"git_commit": "run-commit", "git_dirty": False})
        else:
            path.write_text(f"{name}\n", encoding="utf-8")
    write_json(
        paper_dir / "artifact_manifest.json",
        {
            "tables": {
                "result_macros.tex": {
                    "path": str(table),
                    "sha256": file_sha256(table),
                }
            },
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "metrics.json", "figures/manifest.json"]
            },
            "source_run_git_commit": "run-commit",
            "analysis_git_commit": "analysis-commit",
        },
    )

    assert check_paper_asset_freshness(paper_dir, results_dir) == []

    table.write_text("stale\n", encoding="utf-8")
    (results_dir / "metrics.json").write_text("changed\n", encoding="utf-8")

    failures = check_paper_asset_freshness(paper_dir, results_dir)

    assert "paper artifact table `result_macros.tex` hash is stale" in failures
    assert "paper artifact source `metrics.json` hash is stale" in failures

    write_json(
        paper_dir / "artifact_manifest.json",
        {
            "tables": {
                "result_macros.tex": {
                    "path": str(table),
                    "sha256": file_sha256(table),
                }
            },
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "metrics.json", "figures/manifest.json"]
            },
            "source_run_git_commit": "run-commit",
            "source_run_git_dirty": True,
            "analysis_git_commit": "analysis-commit",
            "analysis_git_dirty": True,
        },
    )

    failures = check_paper_asset_freshness(paper_dir, results_dir)

    assert any("dirty analysis tree" in failure for failure in failures)
    assert any("source run was dirty" in failure for failure in failures)


def test_paper_asset_freshness_requires_selected_tables_under_paper_dir(
    tmp_path: Path,
) -> None:
    import sys

    sys.path.insert(0, str(Path("scripts").resolve()))
    from check_paper_asset_freshness import check_paper_asset_freshness

    paper_dir = tmp_path / "paper" / "generated" / "run"
    results_dir = tmp_path / "results" / "run"
    outside_dir = tmp_path / "outside"
    paper_dir.mkdir(parents=True)
    outside_dir.mkdir()
    (results_dir / "figures").mkdir(parents=True)
    for name in ["manifest.json", "metrics.json", "figures/manifest.json"]:
        path = results_dir / name
        if name == "manifest.json":
            write_json(path, {"git_commit": "run-commit", "git_dirty": False})
        else:
            path.write_text(f"{name}\n", encoding="utf-8")
    table = outside_dir / "main_results_table.tex"
    table.write_text("fresh but wrong location\n", encoding="utf-8")
    wrong_in_tree = paper_dir / "wrong_result_macros.tex"
    wrong_in_tree.write_text("fresh but wrong filename\n", encoding="utf-8")
    write_json(
        paper_dir / "artifact_manifest.json",
        {
            "tables": {
                "main_results_table.tex": {
                    "path": str(table),
                    "sha256": file_sha256(table),
                    "bytes": table.stat().st_size,
                },
                "result_macros.tex": {
                    "path": str(wrong_in_tree),
                    "sha256": file_sha256(wrong_in_tree),
                    "bytes": wrong_in_tree.stat().st_size + 1,
                },
            },
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "metrics.json", "figures/manifest.json"]
            },
            "source_run_git_commit": "run-commit",
            "source_run_git_dirty": False,
            "analysis_git_commit": "analysis-commit",
            "analysis_git_dirty": False,
        },
    )

    failures = check_paper_asset_freshness(
        paper_dir,
        results_dir,
        required_tables=[
            "main_results_table.tex",
            "result_macros.tex",
            "suite_level_effects_table.tex",
        ],
    )

    assert "paper artifact manifest lacks required table `suite_level_effects_table.tex`" in failures
    assert "paper artifact required table `main_results_table.tex` path is unexpected" in failures
    assert "paper artifact table `main_results_table.tex` path is outside paper dir" in failures
    assert "paper artifact required table `result_macros.tex` path is unexpected" in failures
    assert "paper artifact table `result_macros.tex` byte count is stale" in failures


def _write_paper_asset_export_fixture(results_dir: Path) -> None:
    (results_dir / "figures").mkdir(parents=True)
    write_json(
        results_dir / "manifest.json",
        {"run_name": results_dir.name, "git_commit": "run-commit", "git_dirty": False},
    )
    write_json(results_dir / "figures" / "manifest.json", {"figures": {}})
    write_json(
        results_dir / "metrics.json",
        {
            "publication_summary": {
                "policies": {
                    "kv_int4_sim": {
                        "mean_safety_score": 0.500,
                        "mean_capability_score": 0.900,
                    }
                }
            },
            "policy_level_contrasts": {
                "kv_int4_sim": {
                    "selective_safety_erasure_index": 0.25,
                    "selective_safety_erasure_index_ci": {
                        "ci_low": 0.10,
                        "ci_high": 0.40,
                        "n_safety": 12,
                        "n_capability": 8,
                    },
                }
            },
            "selective_safety_erasure": {
                "public_refusal_safety::kv_int4_sim": {
                    "safety_degradation": 0.12,
                    "capability_degradation": 0.02,
                    "selective_safety_erasure_index": 0.10,
                    "paired_safety_degradation_ci": {
                        "paired_n": 100,
                        "cluster_n": 95,
                        "ci_low": 0.05,
                        "ci_high": 0.18,
                    },
                }
            },
            "causal_restoration": {
                "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                    "compressed_policy": "kv_int4_sim",
                    "safety_restoration_fraction": 0.62,
                    "safety_restoration_fraction_ci": {"ci_low": 0.50, "ci_high": 0.72},
                    "refusal_restoration_fraction": 0.55,
                    "refusal_restoration_fraction_ci": {"ci_low": 0.44, "ci_high": 0.66},
                }
            },
        },
    )


def _mark_clean_export_manifest(paper_dir: Path, *, drop_macro_prefix: bool = False) -> None:
    manifest_path = paper_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["analysis_git_dirty"] = False
    manifest["source_run_git_dirty"] = False
    if drop_macro_prefix:
        manifest.pop("macro_prefix", None)
    write_json(manifest_path, manifest)


def _refresh_manifest_table_hash(paper_dir: Path, table_name: str) -> None:
    manifest_path = paper_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    table = paper_dir / table_name
    manifest["tables"][table_name]["sha256"] = file_sha256(table)
    manifest["tables"][table_name]["bytes"] = table.stat().st_size
    write_json(manifest_path, manifest)


def _refresh_manifest_source_hash(paper_dir: Path, results_dir: Path, source_name: str) -> None:
    manifest_path = paper_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = results_dir / source_name
    manifest["source_artifacts"][source_name]["sha256"] = file_sha256(source)
    manifest["source_artifacts"][source_name]["bytes"] = source.stat().st_size
    write_json(manifest_path, manifest)
