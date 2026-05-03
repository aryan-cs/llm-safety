import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from plan_ci_power import build_ci_power_plan, required_cluster_count


def test_required_cluster_count_uses_full_width_target() -> None:
    assert required_cluster_count(sample_sd=0.5, target_ci_width=0.08) == 601


def test_ci_power_plan_uses_paired_prompt_cluster_deltas() -> None:
    rows = [
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p1",
            "seed": 0,
            "policy": "none",
            "safety_score": 1.0,
        },
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p1",
            "seed": 0,
            "policy": "kv_int4_sim",
            "safety_score": 0.5,
        },
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p2",
            "seed": 0,
            "policy": "none",
            "safety_score": 0.8,
        },
        {
            "suite": "public_refusal_safety",
            "prompt_id": "p2",
            "seed": 0,
            "policy": "kv_int4_sim",
            "safety_score": 0.7,
        },
    ]

    plan = build_ci_power_plan(rows, target_ci_width=0.08)

    assert plan["conservative_bernoulli_required_cluster_n"] == 601
    assert plan["pilot_estimates"][0]["suite"] == "public_refusal_safety"
    assert plan["pilot_estimates"][0]["policy"] == "kv_int4_sim"
    assert plan["pilot_estimates"][0]["metric"] == "safety_score"
    assert plan["pilot_estimates"][0]["current_cluster_n"] == 2
