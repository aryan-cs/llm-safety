import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from post_h200_next_steps import post_h200_next_steps, render_markdown


def test_post_h200_next_steps_starts_with_h200_results_when_missing() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
        }
    )

    assert report["publication_ready"] is False
    assert report["next_step"] == "complete_h200_results"
    assert report["steps"][0]["state"] == "ready"
    assert report["steps"][1]["state"] == "blocked"
    assert report["steps"][2]["state"] == "blocked"
    assert "wait_and_run_h200_sweep.sh" in render_markdown(report)


def test_post_h200_next_steps_prepares_fetched_raw_results_before_audits() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [],
            },
            "causal_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [],
            },
        }
    )
    rendered = render_markdown(report)

    assert report["next_step"] == "prepare_after_h200_fetch"
    assert report["steps"][0]["state"] == "complete"
    assert report["steps"][1]["state"] == "ready"
    assert report["steps"][2]["state"] == "blocked"
    assert (
        "fetch_h200_results.sh results/h200_qwen_full_sweep "
        "results/h200_causal_patch_qwen7b"
    ) in report["steps"][1]["command"]
    assert "prepare_after_h200_fetch.sh" in rendered


def test_post_h200_next_steps_prepares_raw_results_with_only_derived_figure_failures() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": ["missing figures/manifest.json"],
            },
            "causal_results": {
                "missing": ["metrics.json"],
                "disqualifiers": [],
                "readiness_failures": [
                    "figures manifest source hash mismatch for `metrics.json`",
                    "missing required figure `causal_restoration_flow`",
                ],
            },
        }
    )

    assert report["next_step"] == "prepare_after_h200_fetch"
    assert report["steps"][0]["state"] == "complete"
    assert report["steps"][1]["state"] == "ready"


def test_post_h200_next_steps_keeps_provenance_failures_on_h200_results() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": ["public_prompts_lack_dataset_provenance:4"],
            },
            "causal_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [],
            },
        }
    )

    assert report["next_step"] == "complete_h200_results"
    assert report["steps"][0]["state"] == "ready"
    assert report["steps"][1]["state"] == "blocked"


def test_post_h200_next_steps_keeps_row_count_failures_on_h200_results() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": ["generation_row_count=12; expected=99"],
            },
            "causal_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [],
            },
        }
    )

    assert report["next_step"] == "complete_h200_results"
    assert report["steps"][0]["state"] == "ready"
    assert report["steps"][1]["state"] == "blocked"


def test_post_h200_next_steps_keeps_generation_matrix_failures_on_h200_results() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [
                    "generation_matrix_missing_rows:12; "
                    "first=suite=public_refusal_safety,prompt_id=advbench_000001,"
                    "policy=policy_pinned,seed=17"
                ],
            },
            "causal_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [],
            },
        }
    )

    assert report["next_step"] == "complete_h200_results"
    assert report["steps"][0]["state"] == "ready"
    assert report["steps"][1]["state"] == "blocked"


def test_post_h200_next_steps_keeps_profile_contract_failures_on_h200_results() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_results_complete", "causal_results_complete"],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": False,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [
                    "missing_required_policy:policy_pinned",
                    "suite_prompt_count:public_refusal_safety=120; required=600",
                    "model_id='Qwen/Qwen2.5-7B-Instruct'; "
                    "expected='Qwen/Qwen2.5-14B-Instruct'",
                ],
            },
            "causal_results": {
                "missing": ["metrics.json", "figures/manifest.json"],
                "disqualifiers": [],
                "readiness_failures": [],
            },
        }
    )

    assert report["next_step"] == "complete_h200_results"
    assert report["steps"][0]["state"] == "ready"
    assert report["steps"][1]["state"] == "blocked"


def test_post_h200_next_steps_surfaces_artifact_specific_blockers() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": [
                "primary_results_complete",
                "primary_human_audit_complete",
                "claim_assessment_passed",
                "arxiv_bundle_ready",
            ],
            "gates": {
                "primary_results_complete": False,
                "causal_results_complete": True,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": True,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
            "primary_results": {
                "missing": ["metrics.json"],
                "readiness_failures": ["public_prompts_lack_dataset_provenance:4"],
            },
            "primary_human_audit": {
                "failures": ["audit export manifest was not leakage-reference capable"]
            },
            "claim_assessment": {"failures": ["human_audit_lacks_causal_restoration_delta"]},
            "arxiv_bundle": {"failures": ["missing_required_bundle_file:figures/a.pdf"]},
        }
    )
    rendered = render_markdown(report)

    assert report["blocker_details"]["primary_results_complete"] == [
        "metrics.json",
        "public_prompts_lack_dataset_provenance:4",
    ]
    assert "audit export manifest was not leakage-reference capable" in rendered
    assert "human_audit_lacks_causal_restoration_delta" in rendered
    assert "missing_required_bundle_file:figures/a.pdf" in rendered


def test_post_h200_next_steps_requires_audits_before_claims() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["primary_human_audit_complete", "causal_human_audit_complete"],
            "gates": {
                "primary_results_complete": True,
                "causal_results_complete": True,
                "primary_human_audit_complete": False,
                "causal_human_audit_complete": False,
                "claim_assessment_passed": False,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
        }
    )

    assert report["next_step"] == "complete_human_audits"
    assert report["steps"][0]["state"] == "complete"
    assert report["steps"][1]["state"] == "complete"
    assert report["steps"][2]["state"] == "ready"
    assert report["steps"][3]["state"] == "blocked"
    assert report["steps"][2]["command"] == "bash scripts/aggregate_publication_human_audits.sh"
    assert "open local judge workflow" in report["steps"][2]["detail"]


def test_post_h200_next_steps_marks_publication_bundle_ready_after_claims() -> None:
    report = post_h200_next_steps(
        {
            "publication_ready": False,
            "blockers": ["arxiv_bundle_ready"],
            "gates": {
                "primary_results_complete": True,
                "causal_results_complete": True,
                "primary_human_audit_complete": True,
                "causal_human_audit_complete": True,
                "claim_assessment_passed": True,
                "paper_pdf_exists": True,
                "arxiv_bundle_ready": False,
            },
        }
    )

    assert report["next_step"] == "build_publication_bundle"
    assert report["steps"][-1]["state"] == "ready"
    assert "build_publication_artifacts.sh" in render_markdown(report)
