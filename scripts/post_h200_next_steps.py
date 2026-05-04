from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from report_publication_status import publication_status

from cache_safety_erasure.utils.io import write_json

_DERIVED_RESULT_ARTIFACTS = {"metrics.json", "figures/manifest.json"}
_DERIVED_READINESS_EXACT_FAILURES = {
    "missing generated PNG figures",
    "missing figures/manifest.json",
    "figures manifest has no figure entries",
    "figures manifest contains non-object entry",
    "causal patch runs require causal_restoration_fraction figure",
}
_DERIVED_READINESS_PREFIXES = (
    "figure `",
    "figures manifest ",
    "figures_manifest_",
    "invalid figures/manifest.json:",
    "missing required figure `",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report the next fail-closed step after H200 result generation."
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
        "--arxiv-source-dir",
        type=Path,
        default=Path("paper/build/arxiv_source"),
    )
    parser.add_argument(
        "--arxiv-archive",
        type=Path,
        default=Path("paper/build/arxiv_source.tar.gz"),
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
        arxiv_source_dir=args.arxiv_source_dir,
        arxiv_archive=args.arxiv_archive,
        require_arxiv_bundle=True,
    )
    report = post_h200_next_steps(status)
    if args.output_json is not None:
        write_json(args.output_json, report)
    markdown = render_markdown(report)
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.fail_if_not_ready and not report["publication_ready"]:
        raise SystemExit(1)


def post_h200_next_steps(status: dict[str, Any]) -> dict[str, Any]:
    gates = status.get("gates") or {}
    primary_raw_complete = bool(gates.get("primary_results_complete")) or _raw_result_available(
        status.get("primary_results")
    )
    causal_raw_complete = bool(gates.get("causal_results_complete")) or _raw_result_available(
        status.get("causal_results")
    )
    fetched_evidence_prepared = bool(gates.get("primary_results_complete")) and bool(
        gates.get("causal_results_complete")
    )
    steps = [
        _step(
            "complete_h200_results",
            complete=primary_raw_complete and causal_raw_complete,
            ready=True,
            command="bash scripts/wait_and_run_h200_sweep.sh",
            detail="Run or wait for the registered H200 launcher until primary and causal raw result directories are ready to fetch.",
        ),
        _step(
            "prepare_after_h200_fetch",
            complete=fetched_evidence_prepared,
            ready=primary_raw_complete and causal_raw_complete,
            command=(
                "bash scripts/fetch_h200_results.sh "
                "results/h200_qwen_full_sweep "
                "results/h200_causal_patch_qwen7b && "
                "bash scripts/prepare_after_h200_fetch.sh"
            ),
            detail="Fetch raw H200 evidence, then reaggregate metrics, regenerate figures and paper tables, run readiness checks, and export audit templates from the current clean local checkout.",
        ),
        _step(
            "complete_human_audits",
            complete=bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete")),
            ready=fetched_evidence_prepared,
            command="bash scripts/aggregate_publication_human_audits.sh",
            detail=(
                "Complete the leakage-capable blinded annotator CSVs, or run the documented "
                "open local judge workflow before aggregation. Require result-source, "
                "export-protocol, judge-model, and prompt-template provenance to match the "
                "exact run artifacts."
            ),
        ),
        _step(
            "assess_claims",
            complete=bool(gates.get("claim_assessment_passed")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete"))
            and bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete")),
            command="uv run python scripts/assess_claims.py --primary-results-dir results/h200_qwen_full_sweep --causal-results-dir results/h200_causal_patch_qwen7b --primary-audit-summary paper/audit/h200_qwen_full_sweep_summary/human_audit_summary.json --causal-audit-summary paper/audit/h200_causal_patch_qwen7b_summary/human_audit_summary.json --output-dir paper/generated/claim_assessment --require-human-audit-support --require-cache-mediated-claim",
            detail="Gate the manuscript claim on H1, H2, H3, and declared audit support; do not rewrite thresholds after seeing results.",
        ),
        _step(
            "build_publication_bundle",
            complete=bool(status.get("publication_ready")),
            ready=bool(gates.get("primary_results_complete"))
            and bool(gates.get("causal_results_complete"))
            and bool(gates.get("primary_human_audit_complete"))
            and bool(gates.get("causal_human_audit_complete"))
            and bool(gates.get("claim_assessment_passed")),
            command="bash scripts/build_publication_artifacts.sh",
            detail="Regenerate metrics, figures, tables, final PDF, and arXiv source bundle from recorded evidence.",
        ),
    ]
    next_step = next((step for step in steps if not step["complete"]), None)
    return {
        "schema_version": 1,
        "publication_ready": bool(status.get("publication_ready")),
        "blockers": status.get("blockers", []),
        "blocker_details": _blocker_details(status),
        "next_step": next_step["name"] if next_step else "done",
        "steps": steps,
    }


def _raw_result_available(artifact: object) -> bool:
    if not isinstance(artifact, dict):
        return False
    missing = set(str(item) for item in artifact.get("missing", []))
    if missing - _DERIVED_RESULT_ARTIFACTS:
        return False
    disqualifiers = artifact.get("disqualifiers", [])
    if disqualifiers:
        return False
    readiness_failures = [str(item) for item in artifact.get("readiness_failures", [])]
    return all(_raw_readiness_failure_is_derived(failure) for failure in readiness_failures)


def _raw_readiness_failure_is_derived(failure: str) -> bool:
    if failure in _DERIVED_READINESS_EXACT_FAILURES:
        return True
    return failure.startswith(_DERIVED_READINESS_PREFIXES)


def _blocker_details(status: dict[str, Any]) -> dict[str, list[str]]:
    mapping = {
        "primary_results_complete": ("primary_results", ["missing", "disqualifiers", "readiness_failures"]),
        "causal_results_complete": ("causal_results", ["missing", "disqualifiers", "readiness_failures"]),
        "primary_human_audit_complete": ("primary_human_audit", ["missing", "failures"]),
        "causal_human_audit_complete": ("causal_human_audit", ["missing", "failures"]),
        "claim_assessment_passed": ("claim_assessment", ["failures"]),
        "paper_pdf_exists": ("paper_pdf", ["failure"]),
        "paper_pdf_valid": ("paper_pdf", ["failure"]),
        "arxiv_bundle_ready": ("arxiv_bundle", ["missing", "failures"]),
    }
    details: dict[str, list[str]] = {}
    for blocker in status.get("blockers", []):
        artifact_name, keys = mapping.get(str(blocker), ("", []))
        artifact = status.get(artifact_name) if artifact_name else None
        values: list[str] = []
        if isinstance(artifact, dict):
            for key in keys:
                raw_value = artifact.get(key)
                if isinstance(raw_value, list):
                    values.extend(str(item) for item in raw_value)
                elif raw_value:
                    values.append(str(raw_value))
        if values:
            details[str(blocker)] = sorted(set(values))
    return details


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Post-H200 Publication Next Steps",
        "",
        f"Publication ready: `{str(report['publication_ready']).lower()}`",
        f"Next step: `{report['next_step']}`",
        "",
        "## Blockers",
        "",
    ]
    if report["blockers"]:
        for blocker in report["blockers"]:
            lines.append(f"- `{blocker}`")
            for detail in report.get("blocker_details", {}).get(blocker, [])[:8]:
                lines.append(f"  - {detail}")
    else:
        lines.append("- none")
    lines.extend(["", "## Ordered Steps", ""])
    for step in report["steps"]:
        lines.extend(
            [
                f"### `{step['name']}`",
                f"- state: `{step['state']}`",
                f"- detail: {step['detail']}",
                f"- command: `{step['command']}`",
                "",
            ]
        )
    return "\n".join(lines)


def _step(
    name: str,
    *,
    complete: bool,
    ready: bool,
    command: str,
    detail: str,
) -> dict[str, str]:
    if complete:
        state = "complete"
    elif ready:
        state = "ready"
    else:
        state = "blocked"
    return {
        "name": name,
        "state": state,
        "complete": complete,
        "command": command,
        "detail": detail,
    }


if __name__ == "__main__":
    main()
