from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import read_jsonl, write_json

Z_95 = 1.96


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate prompt-cluster counts needed for target CI widths."
    )
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--target-ci-width", type=float, default=0.08)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args()

    rows = []
    if args.results_dir is not None and (args.results_dir / "generations.jsonl").exists():
        rows = read_jsonl(args.results_dir / "generations.jsonl")
    plan = build_ci_power_plan(rows, target_ci_width=args.target_ci_width)
    if args.output_json:
        write_json(args.output_json, plan)
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    if not args.output_json and not args.output_md:
        print(json.dumps(plan, indent=2, sort_keys=True))


def build_ci_power_plan(rows: list[dict[str, Any]], *, target_ci_width: float) -> dict[str, Any]:
    conservative_n = required_cluster_count(
        sample_sd=0.5,
        target_ci_width=target_ci_width,
    )
    estimates = []
    for metric in ["safety_score", "capability_score"]:
        for suite, policy, deltas in _cluster_deltas(rows, metric):
            if len(deltas) < 2:
                continue
            sample_sd = float(stdev(deltas))
            estimates.append(
                {
                    "suite": suite,
                    "policy": policy,
                    "metric": metric,
                    "current_cluster_n": len(deltas),
                    "mean_delta": float(mean(deltas)),
                    "sample_sd": sample_sd,
                    "target_ci_width": target_ci_width,
                    "estimated_required_cluster_n": required_cluster_count(
                        sample_sd=sample_sd,
                        target_ci_width=target_ci_width,
                    ),
                }
            )
    return {
        "schema_version": 1,
        "target_ci_width": target_ci_width,
        "conservative_bernoulli_required_cluster_n": conservative_n,
        "pilot_estimates": estimates,
        "note": (
            "Conservative count assumes the maximum Bernoulli standard deviation of 0.5. "
            "Pilot estimates use prompt-cluster deltas from generations.jsonl and should be "
            "treated as planning guidance, not final inference."
        ),
    }


def required_cluster_count(*, sample_sd: float, target_ci_width: float) -> int:
    if target_ci_width <= 0:
        raise ValueError("target_ci_width must be positive")
    if sample_sd <= 0:
        return 2
    return max(2, int(math.ceil(((2.0 * Z_95 * sample_sd) / target_ci_width) ** 2)))


def _cluster_deltas(
    rows: list[dict[str, Any]], metric: str
) -> list[tuple[str, str, list[float]]]:
    grouped: dict[tuple[str, str, str, int], float] = {}
    for idx, row in enumerate(rows):
        value = row.get(metric)
        if value is None:
            continue
        grouped[
            (
                str(row.get("suite")),
                str(row.get("policy")),
                str(row.get("prompt_id", f"row_{idx}")),
                int(row.get("seed", 0)),
            )
        ] = float(value)
    suites = sorted({suite for suite, _, _, _ in grouped})
    policies = sorted({policy for _, policy, _, _ in grouped if policy != "none"})
    output = []
    for suite in suites:
        for policy in policies:
            by_prompt: dict[str, list[float]] = defaultdict(list)
            baseline_keys = {
                (prompt_id, seed): score
                for s, p, prompt_id, seed in grouped
                if s == suite and p == "none"
                for score in [grouped[(s, p, prompt_id, seed)]]
            }
            treatment_keys = {
                (prompt_id, seed): score
                for s, p, prompt_id, seed in grouped
                if s == suite and p == policy
                for score in [grouped[(s, p, prompt_id, seed)]]
            }
            for key in sorted(set(baseline_keys).intersection(treatment_keys)):
                prompt_id, _ = key
                by_prompt[prompt_id].append(baseline_keys[key] - treatment_keys[key])
            deltas = [mean(values) for values in by_prompt.values() if values]
            if deltas:
                output.append((suite, policy, deltas))
    return output


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# CI Width Planning",
        "",
        f"Target full CI width: `{plan['target_ci_width']:.3f}`",
        "",
        "Conservative Bernoulli prompt-cluster count: "
        f"`{plan['conservative_bernoulli_required_cluster_n']}`",
        "",
        "| suite | policy | metric | current clusters | sd | required clusters |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in plan["pilot_estimates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["suite"]),
                    str(row["policy"]),
                    str(row["metric"]),
                    str(row["current_cluster_n"]),
                    f"{row['sample_sd']:.3f}",
                    str(row["estimated_required_cluster_n"]),
                ]
            )
            + " |"
        )
    if not plan["pilot_estimates"]:
        lines.append("|  |  |  |  |  | No pilot deltas available yet. |")
    lines.extend(["", plan["note"], ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
