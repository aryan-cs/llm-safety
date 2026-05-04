from __future__ import annotations

import argparse
import csv
import hashlib
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a stratified human-audit sheet from generations.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/audit"))
    parser.add_argument("--per-suite-policy", type=int, default=3)
    parser.add_argument(
        "--strategy",
        choices=["effect", "random"],
        default="effect",
        help="Select highest automated baseline-vs-treatment shifts or random matched pairs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.results_dir / "generations.jsonl")
    if not rows:
        raise SystemExit(f"No generations found in {args.results_dir}")
    run_id = args.results_dir.name
    sample = _stratified_sample(rows, args.per_suite_policy, args.seed, strategy=args.strategy)
    audit_pairs = [_audit_pair(row, run_id, idx) for idx, row in enumerate(sample)]
    blinded_rows = [pair[0] for pair in audit_pairs]
    key_rows = [pair[1] for pair in audit_pairs]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    blinded_csv_path = args.output_dir / f"{run_id}_audit_blinded.csv"
    key_jsonl_path = args.output_dir / f"{run_id}_audit_key.jsonl"
    _write_csv(blinded_csv_path, blinded_rows)
    write_jsonl(key_jsonl_path, key_rows)
    print(f"Wrote {len(blinded_rows)} blinded audit rows to {blinded_csv_path}")
    print(f"Wrote audit key to {key_jsonl_path}")


def _stratified_sample(
    rows: list[dict[str, Any]],
    per_suite_policy: int,
    seed: int,
    *,
    strategy: str = "effect",
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for row in rows:
        groups[
            (
                str(row.get("suite")),
                str(row.get("policy")),
                str(row.get("prompt_id")),
                int(row.get("seed") or 0),
            )
        ] = row

    sampled: OrderedDict[tuple[str, str, str, int], dict[str, Any]] = OrderedDict()
    suites = sorted({suite for suite, _, _, _ in groups})
    for suite in suites:
        baseline = {
            (prompt_id, seed_value): row
            for (row_suite, policy, prompt_id, seed_value), row in groups.items()
            if row_suite == suite and policy == "none"
        }
        treatment_policies = sorted(
            {
                policy
                for row_suite, policy, _, _ in groups
                if row_suite == suite and policy != "none"
            }
        )
        for policy in treatment_policies:
            treatment = {
                (prompt_id, seed_value): row
                for (row_suite, row_policy, prompt_id, seed_value), row in groups.items()
                if row_suite == suite and row_policy == policy
            }
            paired_keys = sorted(set(baseline).intersection(treatment))
            selected_keys = _select_paired_keys(
                paired_keys,
                baseline,
                treatment,
                per_suite_policy=per_suite_policy,
                rng=rng,
                strategy=strategy,
            )
            for prompt_id, seed_value in selected_keys:
                sampled[(suite, "none", prompt_id, seed_value)] = baseline[(prompt_id, seed_value)]
                sampled[(suite, policy, prompt_id, seed_value)] = treatment[(prompt_id, seed_value)]
    return list(sampled.values())


def _select_paired_keys(
    paired_keys: list[tuple[str, int]],
    baseline: dict[tuple[str, int], dict[str, Any]],
    treatment: dict[tuple[str, int], dict[str, Any]],
    *,
    per_suite_policy: int,
    rng: random.Random,
    strategy: str,
) -> list[tuple[str, int]]:
    if strategy == "random":
        shuffled = list(paired_keys)
        rng.shuffle(shuffled)
        return shuffled[:per_suite_policy]
    if strategy != "effect":
        raise ValueError(f"Unsupported audit sampling strategy: {strategy}")
    scored = [
        (
            _pair_priority_score(baseline[(prompt_id, seed_value)], treatment[(prompt_id, seed_value)]),
            prompt_id,
            seed_value,
        )
        for prompt_id, seed_value in paired_keys
    ]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(prompt_id, seed_value) for _score, prompt_id, seed_value in scored[:per_suite_policy]]


def _pair_priority_score(baseline: dict[str, Any], treatment: dict[str, Any]) -> float:
    safety_drop = _positive_delta(baseline.get("safety_score"), treatment.get("safety_score"))
    refusal_drop = _positive_delta(
        baseline.get("refusal_expected_accuracy"), treatment.get("refusal_expected_accuracy")
    )
    leakage_increase = max(
        _positive_delta(treatment.get("exact_leakage"), baseline.get("exact_leakage")),
        _positive_delta(
            treatment.get("rouge_l_leakage_recall"), baseline.get("rouge_l_leakage_recall")
        ),
    )
    capability_drop = _positive_delta(baseline.get("capability_score"), treatment.get("capability_score"))
    generated_shift = min(
        0.1,
        abs(
            _float_or_zero(treatment.get("generated_word_count"))
            - _float_or_zero(baseline.get("generated_word_count"))
        )
        / 200.0,
    )
    return max(safety_drop, refusal_drop, leakage_increase) + generated_shift - 0.25 * capability_drop


def _positive_delta(left: Any, right: Any) -> float:
    left_value = _maybe_float(left)
    right_value = _maybe_float(right)
    if left_value is None or right_value is None:
        return 0.0
    return max(0.0, left_value - right_value)


def _float_or_zero(value: Any) -> float:
    parsed = _maybe_float(value)
    return parsed if parsed is not None else 0.0


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _audit_pair(row: dict[str, Any], run_id: str, idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_id = _audit_id(row, run_id, idx)
    blinded = {
        "audit_id": audit_id,
        "category": row.get("category"),
        "system_or_policy_text": row.get("system") or "",
        "hidden_system_reference": row.get("hidden_system") or "",
        "user_prompt": row.get("user"),
        "model_response": row.get("generated_text"),
        "human_should_refuse": "",
        "human_refusal_correct": "",
        "human_unsafe_compliance": "",
        "human_system_leakage": "",
        "human_capability_correct": "",
        "human_notes": "",
    }
    key = {
        "audit_id": audit_id,
        "run_id": run_id,
        "model_id": row.get("model_id"),
        "suite": row.get("suite"),
        "policy": row.get("policy"),
        "prompt_id": row.get("prompt_id"),
        "seed": row.get("seed"),
        "category": row.get("category"),
        "should_refuse": row.get("should_refuse"),
        "user": row.get("user"),
        "generated_text": row.get("generated_text"),
        "refusal_expected_accuracy": row.get("refusal_expected_accuracy"),
        "exact_leakage": row.get("exact_leakage"),
        "rouge_l_leakage_recall": row.get("rouge_l_leakage_recall"),
        "capability_score": row.get("capability_score"),
        "generated_word_count": row.get("generated_word_count"),
    }
    return blinded, key


def _audit_id(row: dict[str, Any], run_id: str, idx: int) -> str:
    raw = "|".join(
        [
            run_id,
            str(row.get("suite")),
            str(row.get("policy")),
            str(row.get("prompt_id")),
            str(row.get("seed")),
            str(idx),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
