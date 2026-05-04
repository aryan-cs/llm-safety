import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from report_publication_status import publication_status, render_markdown


def test_publication_status_reports_missing_artifacts_as_blockers(tmp_path: Path) -> None:
    status = publication_status(
        primary_results_dir=tmp_path / "primary",
        causal_results_dir=tmp_path / "causal",
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=tmp_path / "paper.pdf",
    )

    assert status["publication_ready"] is False
    assert "primary_results_complete" in status["blockers"]
    assert "claim_assessment_passed" in status["blockers"]
    assert "paper_pdf_exists" in status["blockers"]


def test_publication_status_accepts_complete_real_artifacts(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment()),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is True
    assert status["blockers"] == []
    assert "Publication ready: `true`" in render_markdown(status)


def test_publication_status_rejects_stale_audit_source_hashes(tmp_path: Path) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    (primary / "metrics.json").write_text(json.dumps({"changed": True}), encoding="utf-8")
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(_passing_claim_assessment()),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "primary_human_audit_complete" in status["blockers"]
    assert "stale_result_source:metrics.json" in status["primary_human_audit"]["failures"]


def test_publication_status_rejects_preliminary_claim_assessment_without_audit_gate(
    tmp_path: Path,
) -> None:
    primary = tmp_path / "primary"
    causal = tmp_path / "causal"
    primary_audit = tmp_path / "primary_audit"
    causal_audit = tmp_path / "causal_audit"
    _write_run(primary)
    _write_run(causal)
    _write_audit(primary_audit, primary)
    _write_audit(causal_audit, causal)
    claim_path = tmp_path / "claim_assessment.json"
    claim_path.write_text(
        json.dumps(
            {
                "publication_gate": {"passed": True},
                "passed_claim_count": 3,
                "human_audit_support": {
                    "required": False,
                    "passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n")

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=primary_audit,
        causal_audit_dir=causal_audit,
        claim_assessment_path=claim_path,
        paper_pdf=pdf_path,
    )

    assert status["publication_ready"] is False
    assert "claim_assessment_passed" in status["blockers"]
    assert "human_audit_support_not_required" in status["claim_assessment"]["failures"]


def test_publication_status_rejects_smoke_or_mock_runs(tmp_path: Path) -> None:
    primary = tmp_path / "primary_smoke"
    causal = tmp_path / "causal"
    _write_run(primary, manifest_overrides={"model_provider": "mock", "run_name": "smoke"})
    _write_run(causal)

    status = publication_status(
        primary_results_dir=primary,
        causal_results_dir=causal,
        primary_audit_dir=tmp_path / "primary_audit",
        causal_audit_dir=tmp_path / "causal_audit",
        claim_assessment_path=tmp_path / "claim_assessment.json",
        paper_pdf=tmp_path / "paper.pdf",
    )

    assert status["publication_ready"] is False
    assert "mock_model" in status["primary_results"]["disqualifiers"]
    assert "smoke_run" in status["primary_results"]["disqualifiers"]


def _write_run(path: Path, manifest_overrides: dict | None = None) -> None:
    (path / "figures").mkdir(parents=True)
    manifest = {
        "model_provider": "hf",
        "model_id": "Qwen/Qwen2.5-14B-Instruct",
        "run_name": "h200_qwen_full_sweep",
        "git_dirty": False,
        "git_commit": "abc123",
        "expected_generation_count": 10,
        "cache_policy_labels": ["none", "kv_int4_sim"],
        "prompt_counts": {"public_refusal_safety": 650},
    }
    manifest.update(manifest_overrides or {})
    for name in [
        "config.resolved.yaml",
        "environment.json",
        "prompts.jsonl",
        "generations.jsonl",
        "cache_stats.parquet",
        "figures/manifest.json",
    ]:
        (path / name).write_text("artifact\n", encoding="utf-8")
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps({"ok": True}), encoding="utf-8")


def _write_audit(path: Path, results_dir: Path) -> None:
    path.mkdir(parents=True)
    summary = {
        "expected_audit_count": 10,
        "completed_audit_count": 10,
        "completion_rate": 1.0,
        "unknown_audit_ids": [],
        "baseline_policy_deltas": {"public_refusal_safety::kv_int4_sim": {}},
    }
    for name in ["human_audit_summary.md", "human_audit_summary_table.tex", "human_audit_deltas_table.tex"]:
        (path / name).write_text("artifact\n", encoding="utf-8")
    manifest = {
        "source_artifacts": {
            "results": {
                name: {"sha256": _sha256(results_dir / name)}
                for name in ["manifest.json", "generations.jsonl", "metrics.json"]
            }
        }
    }
    (path / "audit_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "human_audit_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_claim_assessment() -> dict:
    return {
        "publication_gate": {"passed": True},
        "passed_claim_count": 3,
        "human_audit_support": {
            "required": True,
            "passed": True,
        },
    }
