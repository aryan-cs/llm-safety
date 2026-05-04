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
    assert "\\renewcommand{\\PrimaryRunId}{run\\_001}" in text
    assert "\\renewcommand{\\PrimaryTopSSEIPolicy}{kv\\_int4\\_sim}" in text
    assert "\\renewcommand{\\PrimaryTopSSEI}{0.250}" in text


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
