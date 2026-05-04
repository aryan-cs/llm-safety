import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from assess_claims import (
    assess_claims,
    render_abstract_status_latex,
    render_interpretation_latex,
    render_interpretation_markdown,
    render_latex_table,
)


def test_claim_assessment_passes_only_with_causal_system_control_gap() -> None:
    assessment = assess_claims(_primary_positive_metrics(), _causal_positive_metrics())

    assert assessment["publication_gate"]["passed"] is True
    assert assessment["claims"]["H1_behavioral_cache_sensitivity"]["passed"] is True
    assert assessment["claims"]["H2_selective_safety_degradation"]["passed"] is True
    assert assessment["claims"]["H3_causal_safety_state_erasure"]["passed"] is True
    assert "cache-mediated safety erasure" in assessment["recommended_framing"]


def test_claim_assessment_requires_human_audit_when_configured() -> None:
    assessment = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        require_human_audit_support=True,
    )

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["human_audit_support"]["passed"] is False
    assert "human-audit gate has not cleared" in assessment["recommended_framing"]


def test_claim_assessment_passes_with_aligned_human_audit_support() -> None:
    assessment = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        primary_audit_metrics=_audit_positive_metrics(),
        causal_audit_metrics=_causal_audit_positive_metrics(),
        require_human_audit_support=True,
    )

    assert assessment["publication_gate"]["passed"] is True
    assert assessment["human_audit_support"]["passed"] is True
    assert assessment["human_audit_support"]["best_primary_delta"]["support"] > 0
    assert assessment["human_audit_support"]["best_causal_restoration_delta"]["support"] > 0


def test_claim_assessment_rejects_human_audit_that_contradicts_direction() -> None:
    audit = _audit_positive_metrics()
    audit["baseline_policy_deltas"] = {
        "public_refusal_safety::kv_int4_sim::human_refusal_correct": {
            "treatment_minus_baseline": 0.20,
            "n": 4,
        }
    }
    assessment = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        primary_audit_metrics=audit,
        causal_audit_metrics=_causal_audit_positive_metrics(),
        require_human_audit_support=True,
    )

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["human_audit_support"]["passed"] is False


def test_claim_assessment_rejects_human_audit_for_unclaimed_primary_slice() -> None:
    audit = _audit_positive_metrics()
    audit["baseline_policy_deltas"] = {
        "system_leakage::kv_int8_sim::human_system_leakage": {
            "treatment_minus_baseline": 0.30,
            "n": 4,
        }
    }

    assessment = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        primary_audit_metrics=audit,
        causal_audit_metrics=_causal_audit_positive_metrics(),
        require_human_audit_support=True,
    )

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["human_audit_support"]["best_primary_delta"] is None
    assert any("selected automated safety evidence" in failure for failure in assessment["human_audit_support"]["failures"])


def test_claim_assessment_rejects_human_audit_without_causal_control_gap() -> None:
    causal_audit = _causal_audit_positive_metrics()
    causal_audit["baseline_policy_deltas"][
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem::human_unsafe_compliance"
    ]["treatment_minus_baseline"] = 0.05
    causal_audit["baseline_policy_deltas"][
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem::human_refusal_correct"
    ]["treatment_minus_baseline"] = -0.05

    assessment = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        primary_audit_metrics=_audit_positive_metrics(),
        causal_audit_metrics=causal_audit,
        require_human_audit_support=True,
    )

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["human_audit_support"]["passed"] is False
    assert any("causal audit best system-vs-control" in failure for failure in assessment["human_audit_support"]["failures"])


def test_claim_assessment_rejects_selective_effect_without_causal_control_gap() -> None:
    causal = _causal_positive_metrics()
    user_control = causal["causal_restoration"][
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem"
    ]
    user_control["safety_restoration_fraction"] = 0.60
    user_control["safety_restoration_fraction_ci"] = {"mean": 0.60, "ci_low": 0.55, "ci_high": 0.70}
    user_control["refusal_restoration_fraction"] = 0.60
    user_control["refusal_restoration_fraction_ci"] = {"mean": 0.60, "ci_low": 0.55, "ci_high": 0.70}

    assessment = assess_claims(_primary_positive_metrics(), causal)

    assert assessment["claims"]["H1_behavioral_cache_sensitivity"]["passed"] is True
    assert assessment["claims"]["H2_selective_safety_degradation"]["passed"] is True
    assert assessment["claims"]["H3_causal_safety_state_erasure"]["passed"] is False
    assert assessment["publication_gate"]["passed"] is False
    assert "but not the causal" in assessment["recommended_framing"]


def test_claim_assessment_rejects_unmatched_patch_variant_as_causal_evidence() -> None:
    causal = _causal_positive_metrics()
    causal["causal_restoration"][
        "public_refusal_safety::kv_int4_sim__patchkey__rolesystem__max16__selfirst"
    ] = {
        "compressed_policy": "kv_int4_sim",
        "safety_restoration_fraction": 0.95,
        "safety_restoration_fraction_ci": {
            "mean": 0.95,
            "ci_low": 0.90,
            "ci_high": 0.98,
            "cluster_n": 100,
        },
        "refusal_restoration_fraction": 0.95,
        "refusal_restoration_fraction_ci": {
            "mean": 0.95,
            "ci_low": 0.90,
            "ci_high": 0.98,
            "cluster_n": 100,
        },
    }
    system_control = causal["causal_restoration"][
        "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem"
    ]
    system_control["safety_restoration_fraction"] = 0.24
    system_control["safety_restoration_fraction_ci"] = {
        "mean": 0.24,
        "ci_low": 0.20,
        "ci_high": 0.28,
    }
    system_control["refusal_restoration_fraction"] = 0.23
    system_control["refusal_restoration_fraction_ci"] = {
        "mean": 0.23,
        "ci_low": 0.19,
        "ci_high": 0.27,
    }

    assessment = assess_claims(_primary_positive_metrics(), causal)

    assert assessment["claims"]["H3_causal_safety_state_erasure"]["passed"] is False
    best = assessment["claims"]["H3_causal_safety_state_erasure"]["best_comparison"]
    assert best["system_patch"]["key"].endswith("__patchkey-value__rolesystem")
    assert "patchkey__rolesystem" not in best["system_patch"]["key"]


def test_claim_assessment_rejects_missing_intervals() -> None:
    assessment = assess_claims({"selective_safety_erasure": {}}, {"causal_restoration": {}})

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["claims"]["H1_behavioral_cache_sensitivity"]["passed"] is False
    assert "no eligible interval" in assessment["claims"]["H1_behavioral_cache_sensitivity"]["summary"]


def test_claim_assessment_rejects_causal_point_estimates_without_intervals() -> None:
    causal = _causal_positive_metrics()
    for values in causal["causal_restoration"].values():
        values.pop("safety_restoration_fraction_ci", None)
        values.pop("refusal_restoration_fraction_ci", None)

    assessment = assess_claims(_primary_positive_metrics(), causal)

    assert assessment["claims"]["H3_causal_safety_state_erasure"]["passed"] is False
    assert "No matched system-patch" in assessment["claims"]["H3_causal_safety_state_erasure"]["summary"]


def test_claim_assessment_latex_table_is_formal_and_escaped() -> None:
    assessment = assess_claims(_primary_positive_metrics(), _causal_positive_metrics())

    table = render_latex_table(assessment)

    assert r"\label{tab:claim-assessment}" in table
    assert r"95\% CI" in table
    assert "kv\\_int4\\_sim" in table


def test_claim_interpretation_allows_full_claim_only_when_all_gates_pass() -> None:
    assessment = assess_claims(_primary_positive_metrics(), _causal_positive_metrics())

    latex = render_interpretation_latex(assessment)
    markdown = render_interpretation_markdown(assessment)

    assert "The manuscript may describe the observed effect as cache-mediated safety erasure" in latex
    assert "All registered claim gates passed" in markdown


def test_claim_assessment_writes_publication_abstract_status_only_after_gate() -> None:
    passed = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        primary_audit_metrics=_audit_positive_metrics(),
        causal_audit_metrics=_causal_audit_positive_metrics(),
        require_human_audit_support=True,
    )
    failed = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        require_human_audit_support=True,
    )

    passed_latex = render_abstract_status_latex(passed)
    failed_latex = render_abstract_status_latex(failed)

    assert r"\renewcommand{\EmpiricalStatusSentence}" in passed_latex
    assert "support the cache-mediated safety erasure claim" in passed_latex
    assert "human-audit gate has not cleared" in failed_latex
    assert "publication-ready positive safety claims" in failed_latex


def test_claim_assessment_human_audit_uses_matched_patch_signature() -> None:
    causal_audit = _causal_audit_positive_metrics()
    causal_audit["baseline_policy_deltas"] = {
        "public_refusal_safety::kv_int4_sim__patchkey__rolesystem__max16__selfirst::human_unsafe_compliance": {
            "treatment_minus_baseline": 0.01,
            "n": 4,
        },
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem::human_unsafe_compliance": {
            "treatment_minus_baseline": 0.30,
            "n": 4,
        },
    }

    assessment = assess_claims(
        _primary_positive_metrics(),
        _causal_positive_metrics(),
        primary_audit_metrics=_audit_positive_metrics(),
        causal_audit_metrics=causal_audit,
        require_human_audit_support=True,
    )

    assert assessment["publication_gate"]["passed"] is False
    assert assessment["human_audit_support"]["best_causal_restoration_delta"] is None


def test_claim_interpretation_blocks_causal_claim_for_selective_only_result() -> None:
    causal = _causal_positive_metrics()
    user_control = causal["causal_restoration"][
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem"
    ]
    user_control["safety_restoration_fraction"] = 0.60
    user_control["safety_restoration_fraction_ci"] = {"mean": 0.60, "ci_low": 0.55, "ci_high": 0.70}
    user_control["refusal_restoration_fraction"] = 0.60
    user_control["refusal_restoration_fraction_ci"] = {"mean": 0.60, "ci_low": 0.55, "ci_high": 0.70}
    assessment = assess_claims(_primary_positive_metrics(), causal)

    latex = render_interpretation_latex(assessment)

    assert "must not claim cache-mediated safety erasure" in latex
    assert "selective cache-induced safety degradation" in latex


def test_claim_assessment_require_flag_failure(tmp_path: Path) -> None:
    import json
    import subprocess

    primary_dir = tmp_path / "primary"
    causal_dir = tmp_path / "causal"
    primary_dir.mkdir()
    causal_dir.mkdir()
    (primary_dir / "metrics.json").write_text(
        json.dumps({"selective_safety_erasure": {}}), encoding="utf-8"
    )
    (causal_dir / "metrics.json").write_text(
        json.dumps({"causal_restoration": {}}), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/assess_claims.py",
            "--primary-results-dir",
            str(primary_dir),
            "--causal-results-dir",
            str(causal_dir),
            "--output-dir",
            str(tmp_path / "out"),
            "--require-cache-mediated-claim",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "do not yet justify" in result.stderr


def _primary_positive_metrics() -> dict:
    return {
        "selective_safety_erasure": {
            "public_refusal_safety::kv_int4_sim": {
                "safety_degradation": 0.12,
                "paired_safety_degradation_ci": {
                    "ci_low": 0.05,
                    "ci_high": 0.18,
                    "paired_n": 100,
                    "cluster_n": 100,
                },
            }
        },
        "policy_level_contrasts": {
            "kv_int4_sim": {
                "selective_safety_erasure_index": 0.08,
                "selective_safety_erasure_index_ci": {
                    "mean": 0.08,
                    "ci_low": 0.03,
                    "ci_high": 0.14,
                    "n_safety": 100,
                    "n_capability": 100,
                },
            }
        },
    }


def _causal_positive_metrics() -> dict:
    return {
        "causal_restoration": {
            "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem": {
                "compressed_policy": "kv_int4_sim",
                "safety_restoration_fraction": 0.62,
                "safety_restoration_fraction_ci": {
                    "mean": 0.62,
                    "ci_low": 0.50,
                    "ci_high": 0.72,
                    "cluster_n": 100,
                },
                "refusal_restoration_fraction": 0.55,
                "refusal_restoration_fraction_ci": {
                    "mean": 0.55,
                    "ci_low": 0.44,
                    "ci_high": 0.66,
                    "cluster_n": 100,
                },
            },
            "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem": {
                "compressed_policy": "kv_int4_sim",
                "safety_restoration_fraction": 0.20,
                "safety_restoration_fraction_ci": {
                    "mean": 0.20,
                    "ci_low": 0.12,
                    "ci_high": 0.30,
                    "cluster_n": 100,
                },
                "refusal_restoration_fraction": 0.18,
                "refusal_restoration_fraction_ci": {
                    "mean": 0.18,
                    "ci_low": 0.10,
                    "ci_high": 0.25,
                    "cluster_n": 100,
                },
            },
        }
    }


def _audit_positive_metrics() -> dict:
    return {
        "completion_rate": 1.0,
        "unknown_audit_ids": [],
        "label_rates": {
            "human_refusal_correct": {"n": 8},
            "human_unsafe_compliance": {"n": 8},
            "human_system_leakage": {"n": 8},
        },
        "baseline_policy_deltas": {
            "public_refusal_safety::kv_int4_sim::human_refusal_correct": {
                "treatment_minus_baseline": -0.25,
                "n": 4,
            },
            "public_refusal_safety::kv_int4_sim::human_unsafe_compliance": {
                "treatment_minus_baseline": 0.25,
                "n": 4,
            },
        },
    }


def _causal_audit_positive_metrics() -> dict:
    metrics = _audit_positive_metrics()
    metrics["baseline_policy_deltas"] = {
        "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem::human_unsafe_compliance": {
            "treatment_minus_baseline": 0.05,
            "n": 4,
        },
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem::human_unsafe_compliance": {
            "treatment_minus_baseline": 0.25,
            "n": 4,
        },
        "public_refusal_safety::kv_int4_sim__patchkey-value__rolesystem::human_refusal_correct": {
            "treatment_minus_baseline": -0.05,
            "n": 4,
        },
        "public_refusal_safety::kv_int4_sim__patchkey-value__roleuser__matchsystem::human_refusal_correct": {
            "treatment_minus_baseline": -0.25,
            "n": 4,
        },
    }
    return metrics
