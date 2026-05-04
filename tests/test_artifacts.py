from pathlib import Path

from cache_safety_erasure.utils.io import append_jsonl, file_sha256, read_jsonl, write_json


def test_json_artifacts_roundtrip(tmp_path: Path) -> None:
    write_json(tmp_path / "environment.json", {"ok": True})
    append_jsonl(tmp_path / "generations.jsonl", [{"prompt_id": "p1", "text": "hello"}])
    assert read_jsonl(tmp_path / "generations.jsonl")[0]["prompt_id"] == "p1"


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
    write_json(results_dir / "manifest.json", {"run_name": "run_001"})
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
        },
    )

    assert check_paper_asset_freshness(paper_dir, results_dir) == []

    table.write_text("stale\n", encoding="utf-8")
    (results_dir / "metrics.json").write_text("changed\n", encoding="utf-8")

    failures = check_paper_asset_freshness(paper_dir, results_dir)

    assert "paper artifact table `result_macros.tex` hash is stale" in failures
    assert "paper artifact source `metrics.json` hash is stale" in failures
