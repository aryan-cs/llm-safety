from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json

REQUIRED_RUN_ARTIFACTS = [
    "config.resolved.yaml",
    "environment.json",
    "manifest.json",
    "prompts.jsonl",
    "generations.jsonl",
    "metrics.json",
    "cache_stats.parquet",
    "figures/manifest.json",
]
REQUIRED_AUDIT_ARTIFACTS = [
    "audit_manifest.json",
    "human_audit_summary.json",
    "human_audit_summary.md",
    "human_audit_summary_table.tex",
    "human_audit_deltas_table.tex",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report publication-blocking artifact and claim-gate status."
    )
    parser.add_argument(
        "--primary-results-dir",
        type=Path,
        default=Path("results/h200_qwen_full_sweep"),
    )
    parser.add_argument(
        "--causal-results-dir",
        type=Path,
        default=Path("results/h200_causal_patch_qwen7b"),
    )
    parser.add_argument(
        "--primary-audit-dir",
        type=Path,
        default=Path("paper/audit/h200_qwen_full_sweep_summary"),
    )
    parser.add_argument(
        "--causal-audit-dir",
        type=Path,
        default=Path("paper/audit/h200_causal_patch_qwen7b_summary"),
    )
    parser.add_argument(
        "--claim-assessment",
        type=Path,
        default=Path("paper/generated/claim_assessment/claim_assessment.json"),
    )
    parser.add_argument(
        "--paper-pdf",
        type=Path,
        default=Path("paper/cache_mediated_safety_erasure.pdf"),
    )
    parser.add_argument(
        "--allow-missing-paper-pdf",
        action="store_true",
        help="Permit a missing PDF when checking readiness before rebuilding the final PDF.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    args = parser.parse_args()

    status = publication_status(
        primary_results_dir=args.primary_results_dir,
        causal_results_dir=args.causal_results_dir,
        primary_audit_dir=args.primary_audit_dir,
        causal_audit_dir=args.causal_audit_dir,
        claim_assessment_path=args.claim_assessment,
        paper_pdf=args.paper_pdf,
        require_paper_pdf=not args.allow_missing_paper_pdf,
    )
    if args.output_json is not None:
        write_json(args.output_json, status)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(status), encoding="utf-8")
    print(render_markdown(status))
    if args.fail_if_not_ready and not status["publication_ready"]:
        raise SystemExit(1)


def publication_status(
    *,
    primary_results_dir: Path,
    causal_results_dir: Path,
    primary_audit_dir: Path,
    causal_audit_dir: Path,
    claim_assessment_path: Path,
    paper_pdf: Path,
    require_paper_pdf: bool = True,
) -> dict[str, Any]:
    primary = _run_status(primary_results_dir)
    causal = _run_status(causal_results_dir)
    primary_audit = _audit_status(primary_audit_dir, primary_results_dir)
    causal_audit = _audit_status(causal_audit_dir, causal_results_dir)
    claim_assessment = _claim_status(
        claim_assessment_path,
        primary_results_dir=primary_results_dir,
        causal_results_dir=causal_results_dir,
        primary_audit_dir=primary_audit_dir,
        causal_audit_dir=causal_audit_dir,
    )
    pdf = _pdf_status(paper_pdf)

    gates = {
        "primary_results_complete": primary["complete"],
        "causal_results_complete": causal["complete"],
        "primary_human_audit_complete": primary_audit["complete"],
        "causal_human_audit_complete": causal_audit["complete"],
        "claim_assessment_passed": claim_assessment["passed"],
        "paper_pdf_exists": pdf["exists"] or not require_paper_pdf,
    }
    blockers = [gate for gate, passed in gates.items() if not passed]
    return {
        "schema_version": 1,
        "publication_ready": not blockers,
        "blockers": blockers,
        "gates": gates,
        "primary_results": primary,
        "causal_results": causal,
        "primary_human_audit": primary_audit,
        "causal_human_audit": causal_audit,
        "claim_assessment": claim_assessment,
        "paper_pdf": pdf,
        "paper_pdf_required": require_paper_pdf,
    }


def render_markdown(status: dict[str, Any]) -> str:
    lines = [
        "# Publication Status",
        "",
        f"Publication ready: `{str(status['publication_ready']).lower()}`",
        "",
        "| Gate | Status |",
        "| --- | --- |",
    ]
    for gate, passed in status["gates"].items():
        lines.append(f"| `{gate}` | {'pass' if passed else 'fail'} |")
    lines.extend(["", "## Blockers", ""])
    if status["blockers"]:
        lines.extend(f"- `{blocker}`" for blocker in status["blockers"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            _artifact_line("primary results", status["primary_results"]),
            _artifact_line("causal results", status["causal_results"]),
            _artifact_line("primary human audit", status["primary_human_audit"]),
            _artifact_line("causal human audit", status["causal_human_audit"]),
            _claim_line(status["claim_assessment"]),
            _pdf_line(status["paper_pdf"]),
            "",
        ]
    )
    return "\n".join(lines)


def _run_status(results_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_RUN_ARTIFACTS if not (results_dir / name).exists()]
    manifest = _read_json(results_dir / "manifest.json")
    metrics = _read_json(results_dir / "metrics.json")
    disqualifiers: list[str] = []
    if manifest:
        model_provider = str(manifest.get("model_provider", ""))
        model_id = str(manifest.get("model_id", ""))
        run_name = str(manifest.get("run_name", ""))
        if manifest.get("git_dirty"):
            disqualifiers.append("dirty_git_tree")
        if model_provider == "mock":
            disqualifiers.append("mock_model")
        if "tiny" in model_id.lower():
            disqualifiers.append("tiny_model")
        if "smoke" in run_name.lower() or "smoke" in results_dir.name.lower():
            disqualifiers.append("smoke_run")
    return {
        "path": str(results_dir),
        "complete": not missing and not disqualifiers,
        "missing": missing,
        "disqualifiers": disqualifiers,
        "manifest_present": bool(manifest),
        "metrics_present": bool(metrics),
        "model_id": manifest.get("model_id") if manifest else None,
        "git_commit": manifest.get("git_commit") if manifest else None,
        "expected_generation_count": manifest.get("expected_generation_count") if manifest else None,
        "policy_count": len(manifest.get("cache_policy_labels", [])) if manifest else None,
        "prompt_counts": manifest.get("prompt_counts") if manifest else None,
    }


def _audit_status(audit_dir: Path, results_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_AUDIT_ARTIFACTS if not (audit_dir / name).exists()]
    summary = _read_json(audit_dir / "human_audit_summary.json")
    manifest = _read_json(audit_dir / "audit_manifest.json")
    failures = []
    if summary:
        if summary.get("completion_rate") != 1.0:
            failures.append("incomplete_annotations")
        if summary.get("unknown_audit_ids"):
            failures.append("unknown_audit_ids")
        if not summary.get("baseline_policy_deltas"):
            failures.append("missing_baseline_policy_deltas")
    failures.extend(_audit_result_source_failures(manifest, results_dir))
    return {
        "path": str(audit_dir),
        "complete": not missing and not failures,
        "missing": missing,
        "failures": failures,
        "manifest_present": bool(manifest),
        "expected_audit_count": summary.get("expected_audit_count") if summary else None,
        "completed_audit_count": summary.get("completed_audit_count") if summary else None,
        "completion_rate": summary.get("completion_rate") if summary else None,
    }


def _claim_status(
    path: Path,
    *,
    primary_results_dir: Path,
    causal_results_dir: Path,
    primary_audit_dir: Path,
    causal_audit_dir: Path,
) -> dict[str, Any]:
    assessment = _read_json(path)
    failures = _claim_failures(
        assessment,
        {
            "primary_metrics": primary_results_dir / "metrics.json",
            "primary_manifest": primary_results_dir / "manifest.json",
            "causal_metrics": causal_results_dir / "metrics.json",
            "causal_manifest": causal_results_dir / "manifest.json",
            "primary_audit_summary": primary_audit_dir / "human_audit_summary.json",
            "primary_audit_manifest": primary_audit_dir / "audit_manifest.json",
            "causal_audit_summary": causal_audit_dir / "human_audit_summary.json",
            "causal_audit_manifest": causal_audit_dir / "audit_manifest.json",
        },
    )
    return {
        "path": str(path),
        "exists": path.exists(),
        "passed": bool(assessment) and not failures,
        "failures": failures,
        "passed_claim_count": assessment.get("passed_claim_count") if assessment else None,
        "recommended_framing": assessment.get("recommended_framing") if assessment else None,
        "human_audit_required": (assessment.get("human_audit_support") or {}).get("required")
        if assessment
        else None,
        "human_audit_passed": (assessment.get("human_audit_support") or {}).get("passed")
        if assessment
        else None,
    }


def _pdf_status(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": file_sha256(path),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _audit_result_source_failures(manifest: dict[str, Any], results_dir: Path) -> list[str]:
    if not manifest:
        return []
    result_sources = (manifest.get("source_artifacts") or {}).get("results")
    if not isinstance(result_sources, dict):
        return ["missing_result_source_manifest"]
    failures = []
    for name in ["manifest.json", "generations.jsonl", "metrics.json"]:
        source = result_sources.get(name)
        if not isinstance(source, dict):
            failures.append(f"missing_result_source:{name}")
            continue
        path = results_dir / name
        if not path.exists():
            failures.append(f"missing_result_artifact:{name}")
            continue
        if source.get("sha256") != file_sha256(path):
            failures.append(f"stale_result_source:{name}")
    return failures


def _claim_failures(assessment: dict[str, Any], source_paths: dict[str, Path]) -> list[str]:
    if not assessment:
        return []
    failures = []
    if not assessment.get("publication_gate", {}).get("passed"):
        failures.append("publication_gate_failed")
    audit_support = assessment.get("human_audit_support")
    if not isinstance(audit_support, dict):
        failures.append("missing_human_audit_support")
    else:
        if audit_support.get("required") is not True:
            failures.append("human_audit_support_not_required")
        if audit_support.get("passed") is not True:
            failures.append("human_audit_support_failed")
    failures.extend(_claim_source_failures(assessment, source_paths))
    return failures


def _claim_source_failures(assessment: dict[str, Any], source_paths: dict[str, Path]) -> list[str]:
    source_artifacts = assessment.get("source_artifacts")
    if not isinstance(source_artifacts, dict):
        return ["missing_claim_source_artifacts"]
    failures = []
    for name, path in source_paths.items():
        source = source_artifacts.get(name)
        if not isinstance(source, dict):
            failures.append(f"missing_claim_source:{name}")
            continue
        if not path.exists():
            failures.append(f"missing_claim_source_file:{name}")
            continue
        if source.get("sha256") != file_sha256(path):
            failures.append(f"stale_claim_source:{name}")
    return failures


def _artifact_line(label: str, status: dict[str, Any]) -> str:
    state = "complete" if status["complete"] else "blocked"
    details = []
    if status.get("missing"):
        details.append(f"missing {len(status['missing'])}")
    if status.get("disqualifiers"):
        details.append("disqualified: " + ", ".join(status["disqualifiers"]))
    if status.get("failures"):
        details.append("failed: " + ", ".join(status["failures"]))
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"- {label}: `{state}` at `{status['path']}`{suffix}"


def _claim_line(status: dict[str, Any]) -> str:
    state = "pass" if status["passed"] else "blocked"
    suffix = ""
    if status.get("failures"):
        suffix = " (failed: " + ", ".join(status["failures"]) + ")"
    return f"- claim assessment: `{state}` at `{status['path']}`{suffix}"


def _pdf_line(status: dict[str, Any]) -> str:
    state = "exists" if status["exists"] else "missing"
    return f"- paper PDF: `{state}` at `{status['path']}`"


if __name__ == "__main__":
    main()
