import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path("scripts").resolve()))

from check_publication_readiness import _check_figure_manifest, _figure_artifact_failure
from make_figures import (
    _phase_portrait_rows,
    _prompt_effect_constellation_rows,
    _safety_state_atlas_rows,
    _stream_cache_fingerprint,
    _stream_cache_summaries,
)


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_stream_cache_summaries_aggregates_without_full_read(tmp_path: Path) -> None:
    import pandas as pd

    cache_path = tmp_path / "cache_stats.parquet"
    pd.DataFrame(
        [
            {
                "policy": "sliding_window__budget64",
                "decode_step": 0,
                "cache_l2_before": 4.0,
                "cache_l2_after": 2.0,
                "retained_system_tokens": 2,
                "evicted_system_tokens": 2,
            },
            {
                "policy": "sliding_window__budget64",
                "decode_step": 0,
                "cache_l2_before": 2.0,
                "cache_l2_after": 1.0,
                "retained_system_tokens": 1,
                "evicted_system_tokens": 3,
            },
        ]
    ).to_parquet(cache_path, index=False)

    summaries = _stream_cache_summaries(cache_path)

    assert summaries["l2_rows"] == [
        {
            "policy": "sliding_window__budget64",
            "decode_step": 0,
            "l2_retained_fraction": 0.5,
        }
    ]
    assert summaries["role_rows"] == [
        {
            "policy": "sliding_window__budget64",
            "role": "system",
            "retention_fraction": 0.375,
            "retained_count": 3.0,
            "evicted_count": 5.0,
        }
    ]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyarrow") is None,
    reason="pyarrow is not installed in the base interpreter",
)
def test_stream_cache_fingerprint_uses_prompt_roles_and_position_bins(tmp_path: Path) -> None:
    import pandas as pd

    cache_path = tmp_path / "cache_stats.parquet"
    prompts_path = tmp_path / "prompts.jsonl"
    prompts_path.write_text(
        '{"prompt_id":"p1","rendered_prompt":{"token_roles":["system","system","user","user"]}}\n',
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "prompt_id": "p1",
                "policy": "sliding_window__budget2",
                "decode_step": 0,
                "original_seq_len": 4,
                "retained_indices": "2,3",
                "evicted_indices": "0,1",
            },
            {
                "prompt_id": "p1",
                "policy": "sliding_window__budget2",
                "decode_step": 1,
                "original_seq_len": 4,
                "retained_indices": "0,1",
                "evicted_indices": "2,3",
            },
        ]
    ).to_parquet(cache_path, index=False)

    rows = _stream_cache_fingerprint(cache_path, prompts_path, bin_count=4)

    assert {
        "policy": "sliding_window__budget2",
        "role": "system",
        "token_bin": 0,
        "retained_count": 0.0,
        "evicted_count": 1.0,
        "retention_fraction": 0.0,
    } in rows
    assert {
        "policy": "sliding_window__budget2",
        "role": "user",
        "token_bin": 2,
        "retained_count": 1.0,
        "evicted_count": 0.0,
        "retention_fraction": 1.0,
    } in rows


def test_phase_portrait_rows_parse_policy_budgets() -> None:
    import pandas as pd

    rows = _phase_portrait_rows(
        pd.DataFrame(
            [
                {
                    "suite": "public_refusal_safety",
                    "policy": "sliding_window__budget64",
                    "index": 0.2,
                    "safety_degradation": 0.3,
                    "capability_degradation": 0.1,
                }
            ]
        )
    )

    assert rows.to_dict(orient="records") == [
        {
            "suite": "public_refusal_safety",
            "policy": "sliding_window__budget64",
            "policy_family": "sliding_window",
            "budget_sort": 64.0,
            "budget_label": "b=64",
            "safety_degradation": 0.3,
            "capability_degradation": 0.1,
            "selective_safety_erasure_index": 0.2,
        }
    ]


def test_prompt_effect_constellation_rows_pair_against_baseline() -> None:
    import pandas as pd

    rows = _prompt_effect_constellation_rows(
        pd.DataFrame(
            [
                {
                    "suite": "public_refusal_safety",
                    "prompt_id": "p1",
                    "seed": 0,
                    "policy": "none",
                    "safety_score": 1.0,
                    "capability_score": None,
                    "refusal_expected_accuracy": 1.0,
                    "leakage_avoidance_score": None,
                    "generated_word_count": 10.0,
                },
                {
                    "suite": "public_refusal_safety",
                    "prompt_id": "p1",
                    "seed": 0,
                    "policy": "kv_int4_sim",
                    "safety_score": 0.25,
                    "capability_score": None,
                    "refusal_expected_accuracy": 0.0,
                    "leakage_avoidance_score": None,
                    "generated_word_count": 20.0,
                },
            ]
        )
    )

    assert len(rows) == 1
    assert rows[0]["safety_score_delta"] == 0.75
    assert rows[0]["refusal_expected_accuracy_delta"] == 1.0
    assert rows[0]["effect_magnitude"] == 1.0


def test_safety_state_atlas_combines_ssei_and_role_retention() -> None:
    rows = _safety_state_atlas_rows(
        [
            {
                "suite": "public_refusal_safety",
                "policy": "sliding_window__budget64",
                "index": 0.4,
                "safety_degradation": 0.5,
                "capability_degradation": 0.1,
            }
        ],
        [
            {
                "policy": "sliding_window__budget64",
                "role": "system",
                "retention_fraction": 0.25,
            },
            {
                "policy": "sliding_window__budget64",
                "role": "user",
                "retention_fraction": 0.75,
            },
        ],
    )

    assert rows == [
        {
            "suite": "public_refusal_safety",
            "policy": "sliding_window__budget64",
            "selective_safety_erasure_index": 0.4,
            "safety_degradation": 0.5,
            "capability_degradation": 0.1,
            "system_retention_fraction": 0.25,
            "user_retention_fraction": 0.75,
            "template_retention_fraction": None,
            "generated_retention_fraction": None,
        }
    ]


def test_figure_manifest_rejects_stale_hash(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    for suffix in ["png", "svg", "pdf", "csv"]:
        (figures_dir / f"figure.{suffix}").write_text(suffix, encoding="utf-8")
    write_json(
        figures_dir / "manifest.json",
        {
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
            },
            "figures": [
                {
                    "name": "figure",
                    "png": str(figures_dir / "figure.png"),
                    "png_sha256": "stale",
                    "svg": str(figures_dir / "figure.svg"),
                    "svg_sha256": file_sha256(figures_dir / "figure.svg"),
                    "pdf": str(figures_dir / "figure.pdf"),
                    "pdf_sha256": file_sha256(figures_dir / "figure.pdf"),
                    "data_csv": str(figures_dir / "figure.csv"),
                    "data_csv_sha256": file_sha256(figures_dir / "figure.csv"),
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(figures_dir, results_dir, failures, require_causal_patch=False)

    assert any("stale png hash" in failure for failure in failures)


def test_figure_manifest_rejects_malformed_visual_artifacts(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    for suffix in ["png", "svg", "pdf", "csv"]:
        (figures_dir / f"figure.{suffix}").write_text("not a valid figure\n", encoding="utf-8")
    write_json(
        figures_dir / "manifest.json",
        {
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
            },
            "figures": [
                {
                    "name": "figure",
                    "png": str(figures_dir / "figure.png"),
                    "png_sha256": file_sha256(figures_dir / "figure.png"),
                    "svg": str(figures_dir / "figure.svg"),
                    "svg_sha256": file_sha256(figures_dir / "figure.svg"),
                    "pdf": str(figures_dir / "figure.pdf"),
                    "pdf_sha256": file_sha256(figures_dir / "figure.pdf"),
                    "data_csv": str(figures_dir / "figure.csv"),
                    "data_csv_sha256": file_sha256(figures_dir / "figure.csv"),
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(figures_dir, results_dir, failures, require_causal_patch=False)

    assert "figure `figure` has invalid png: missing PNG signature" in failures
    assert "figure `figure` has invalid svg: missing SVG root" in failures
    assert "figure `figure` has invalid pdf: missing PDF signature" in failures


def test_figure_artifact_signature_validator_accepts_real_headers(tmp_path: Path) -> None:
    png = tmp_path / "figure.png"
    pdf = tmp_path / "figure.pdf"
    svg = tmp_path / "figure.svg"
    csv = tmp_path / "figure.csv"
    png.write_bytes(b"\x89PNG\r\n\x1a\npayload")
    pdf.write_bytes(b"%PDF-1.7\npayload")
    svg.write_text('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    csv.write_text("column\nvalue\n", encoding="utf-8")

    assert _figure_artifact_failure("png", png) == ""
    assert _figure_artifact_failure("pdf", pdf) == ""
    assert _figure_artifact_failure("svg", svg) == ""
    assert _figure_artifact_failure("data_csv", csv) == ""


def test_figure_manifest_requires_named_figures(tmp_path: Path) -> None:
    from cache_safety_erasure.utils.io import file_sha256, write_json

    results_dir = tmp_path / "results"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True)
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        (results_dir / name).write_text(name, encoding="utf-8")
    for suffix in ["png", "svg", "pdf", "csv"]:
        (figures_dir / f"present.{suffix}").write_text(suffix, encoding="utf-8")
    write_json(
        figures_dir / "manifest.json",
        {
            "source_artifacts": {
                name: {"sha256": file_sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]
            },
            "figures": [
                {
                    "name": "present",
                    "png": str(figures_dir / "present.png"),
                    "png_sha256": file_sha256(figures_dir / "present.png"),
                    "svg": str(figures_dir / "present.svg"),
                    "svg_sha256": file_sha256(figures_dir / "present.svg"),
                    "pdf": str(figures_dir / "present.pdf"),
                    "pdf_sha256": file_sha256(figures_dir / "present.pdf"),
                    "data_csv": str(figures_dir / "present.csv"),
                    "data_csv_sha256": file_sha256(figures_dir / "present.csv"),
                }
            ],
        },
    )
    failures: list[str] = []

    _check_figure_manifest(
        figures_dir,
        results_dir,
        failures,
        require_causal_patch=False,
        required_figures=["missing_creative_figure"],
    )

    assert "missing required figure `missing_creative_figure`" in failures
