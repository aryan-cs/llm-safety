import sys
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))

from export_human_audit_sample import _audit_pair, _stratified_sample


def test_blinded_audit_row_hides_policy_and_model_identifiers() -> None:
    blinded, key = _audit_pair(
        {
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "suite": "public_refusal_safety",
            "policy": "kv_int4_sim",
            "prompt_id": "p1",
            "seed": 0,
            "category": "safety",
            "system": "Follow the policy.",
            "user": "Can you help?",
            "generated_text": "I cannot help with that request.",
            "should_refuse": True,
        },
        "run_a",
        0,
    )

    assert "model_id" not in blinded
    assert "policy" not in blinded
    assert blinded["audit_id"] == key["audit_id"]
    assert key["model_id"] == "Qwen/Qwen2.5-7B-Instruct"
    assert key["policy"] == "kv_int4_sim"


def test_stratified_audit_sample_includes_matched_baseline_rows() -> None:
    rows = []
    for prompt_id in ["p1", "p2", "p3"]:
        rows.append(
            {
                "suite": "public_refusal_safety",
                "policy": "none",
                "prompt_id": prompt_id,
                "seed": 0,
            }
        )
        rows.append(
            {
                "suite": "public_refusal_safety",
                "policy": "kv_int4_sim",
                "prompt_id": prompt_id,
                "seed": 0,
            }
        )
    rows.append(
        {
            "suite": "public_refusal_safety",
            "policy": "sliding_window__budget64",
            "prompt_id": "p1",
            "seed": 0,
        }
    )

    sample = _stratified_sample(rows, per_suite_policy=1, seed=0)
    sampled_keys = {
        (row["suite"], row["policy"], row["prompt_id"], row["seed"]) for row in sample
    }

    treatment_keys = [key for key in sampled_keys if key[1] != "none"]
    assert treatment_keys
    for suite, _policy, prompt_id, seed in treatment_keys:
        assert (suite, "none", prompt_id, seed) in sampled_keys


def test_effect_audit_sample_prioritizes_largest_safety_shift() -> None:
    rows = []
    for prompt_id, treatment_safety in [("low_effect", 0.8), ("high_effect", 0.1)]:
        rows.append(
            {
                "suite": "public_refusal_safety",
                "policy": "none",
                "prompt_id": prompt_id,
                "seed": 0,
                "safety_score": 1.0,
                "capability_score": None,
            }
        )
        rows.append(
            {
                "suite": "public_refusal_safety",
                "policy": "kv_int4_sim",
                "prompt_id": prompt_id,
                "seed": 0,
                "safety_score": treatment_safety,
                "capability_score": None,
            }
        )

    sample = _stratified_sample(rows, per_suite_policy=1, seed=0, strategy="effect")
    sampled_prompt_ids = {row["prompt_id"] for row in sample}

    assert sampled_prompt_ids == {"high_effect"}


def test_random_audit_sample_is_available_for_unbiased_spot_checks() -> None:
    rows = []
    for prompt_id in ["p1", "p2", "p3"]:
        rows.append(
            {
                "suite": "public_refusal_safety",
                "policy": "none",
                "prompt_id": prompt_id,
                "seed": 0,
            }
        )
        rows.append(
            {
                "suite": "public_refusal_safety",
                "policy": "kv_int4_sim",
                "prompt_id": prompt_id,
                "seed": 0,
            }
        )

    sample = _stratified_sample(rows, per_suite_policy=2, seed=1, strategy="random")

    assert len(sample) == 4
