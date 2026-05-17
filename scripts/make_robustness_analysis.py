"""Sensitivity / robustness analysis for the cross-family SSEI claim.

Drops one model (or one family) at a time and re-checks the registered
cross-family replication rule: at least two independent instruction-tuned
families show positive SSEI whose 95% lower CI excludes zero on a registered
cache policy. The rule is robust if the claim holds for every leave-one-out
restriction.

Also reports per-model:
- the largest positive SSEI policy with CI excluding 0 (if any)
- the largest |SSEI| (effect magnitude regardless of sign)
- whether the model is a "positive" panel member under the registered threshold

Emits a JSON, Markdown, and LaTeX summary into
``docs/generated/active_primary/`` for the paper.

Run: ``uv run python scripts/make_robustness_analysis.py``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SSEI_POSITIVE_THRESHOLD = 0.01

MODEL_FAMILIES = {
    "qwen2_5_7b_base": "Qwen",
    "qwen2_5_7b_instruct": "Qwen",
    "qwen2_5_14b_instruct": "Qwen",
    "qwen2_5_14b_msm_rules": "Qwen",
    "qwen3_5_9b": "Qwen",
    "gpt_oss_20b": "OpenAI",
    "llama3_1_8b_instruct": "Llama",
    "gemma2_9b_it": "Gemma",
    "mistral_7b_instruct_v0_3": "Mistral",
    "olmo3_7b_instruct": "OLMo",
    "phi4": "Phi",
}
MODEL_LABELS = {
    "qwen2_5_7b_base": "Qwen2.5-7B base",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen2_5_14b_instruct": "Qwen2.5-14B-Instruct",
    "qwen2_5_14b_msm_rules": "Qwen2.5-14B-Instruct + MSM",
    "qwen3_5_9b": "Qwen3-9B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "gemma2_9b_it": "Gemma-2-9B-IT",
    "mistral_7b_instruct_v0_3": "Mistral-7B-Instruct-v0.3",
    "olmo3_7b_instruct": "OLMo-3-7B-Instruct",
    "phi4": "Phi-4",
}


def _safe_metrics(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _row_for_model(model_key: str, run_dir: Path) -> dict[str, Any] | None:
    metrics = _safe_metrics(run_dir)
    contrasts = metrics.get("policy_level_contrasts") or {}
    if not contrasts:
        return None

    best_positive = None
    best_abs = None
    for policy, payload in contrasts.items():
        ssei = payload.get("selective_safety_erasure_index")
        ci = payload.get("selective_safety_erasure_index_ci") or {}
        ci_low = ci.get("ci_low")
        ci_high = ci.get("ci_high")
        if ssei is None:
            continue
        if best_abs is None or abs(ssei) > abs(best_abs["ssei"]):
            best_abs = {"policy": policy, "ssei": ssei, "ci_low": ci_low, "ci_high": ci_high}
        if (
            ssei >= SSEI_POSITIVE_THRESHOLD
            and ci_low is not None
            and ci_low > 0
        ):
            if best_positive is None or ssei > best_positive["ssei"]:
                best_positive = {
                    "policy": policy,
                    "ssei": ssei,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
    return {
        "model_key": model_key,
        "family": MODEL_FAMILIES.get(model_key, model_key),
        "model_label": MODEL_LABELS.get(model_key, model_key),
        "best_positive": best_positive,
        "best_abs": best_abs,
        "is_positive_panel_member": best_positive is not None and "base" not in model_key,
    }


def leave_one_out(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    instr_rows = [r for r in rows if "_base" not in r["model_key"]]
    out: list[dict[str, Any]] = []
    for excluded_family in sorted({r["family"] for r in instr_rows}):
        remaining = [r for r in instr_rows if r["family"] != excluded_family]
        positive_families = {
            r["family"] for r in remaining if r["is_positive_panel_member"]
        }
        out.append(
            {
                "excluded_family": excluded_family,
                "remaining_models": [r["model_key"] for r in remaining],
                "positive_families_after_exclusion": sorted(positive_families),
                "claim_still_supported": len(positive_families) >= 2,
            }
        )
    return out


def render_md(rows: list[dict[str, Any]], loo: list[dict[str, Any]]) -> str:
    lines = ["## Per-model headline"]
    lines.append("")
    lines.append(
        "| Family | Model | Top positive SSEI policy | SSEI [95% CI] | Top |SSEI| policy | SSEI [95% CI] |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in rows:
        bp = r["best_positive"]
        ba = r["best_abs"]
        bp_str = (
            f"`{bp['policy']}` | {bp['ssei']:+.3f} [{bp['ci_low']:+.3f}, {bp['ci_high']:+.3f}]"
            if bp
            else "-- | --"
        )
        if ba:
            ci_str = (
                f"[{ba['ci_low']:+.3f}, {ba['ci_high']:+.3f}]"
                if ba["ci_low"] is not None
                else "[--, --]"
            )
            ba_str = f"`{ba['policy']}` | {ba['ssei']:+.3f} {ci_str}"
        else:
            ba_str = "-- | --"
        lines.append(f"| {r['family']} | {r['model_label']} | {bp_str} | {ba_str} |")
    lines.extend(["", "## Leave-one-family-out cross-family claim check", ""])
    lines.append("| Excluded family | Positive families remaining | Claim holds? |")
    lines.append("| --- | --- | --- |")
    for entry in loo:
        held = "YES" if entry["claim_still_supported"] else "NO"
        lines.append(
            f"| {entry['excluded_family']} | "
            f"{', '.join(entry['positive_families_after_exclusion']) or '(none)'} | {held} |"
        )
    return "\n".join(lines) + "\n"


def _esc(text: str) -> str:
    return text.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def render_tex(rows: list[dict[str, Any]], loo: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by scripts/make_robustness_analysis.py",
        "\\begin{table}[ht]",
        "\\centering",
        "\\small",
        "\\caption{Leave-one-family-out robustness check on the cross-family selectivity claim. Each row removes the named family and counts the remaining instruction-tuned families that retain positive SSEI with 95\\% lower CI excluding 0 on at least one registered cache policy. The registered rule requires at least two such families.}",
        "\\label{tab:robustness}",
        "\\begin{tabular}{lll}",
        "\\toprule",
        "\\textbf{Excluded family} & \\textbf{Positive families remaining} & \\textbf{Claim holds?} \\\\",
        "\\midrule",
    ]
    for entry in loo:
        held = "supported" if entry["claim_still_supported"] else "REFUTED"
        remaining = ", ".join(entry["positive_families_after_exclusion"]) or "(none)"
        lines.append(
            f"{_esc(entry['excluded_family'])} & {_esc(remaining)} & {held} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/generated/active_primary")
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(args.results_root.glob("selectivity_h200_powered_*")):
        if run_dir.name == "selectivity_h200_powered_combined":
            continue
        model_key = run_dir.name.removeprefix("selectivity_h200_powered_")
        row = _row_for_model(model_key, run_dir)
        if row is not None:
            rows.append(row)

    loo = leave_one_out(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "robustness_analysis.json"
    md_path = args.output_dir / "robustness_analysis.md"
    tex_path = args.output_dir / "robustness_analysis.tex"

    summary = {
        "ssei_positive_threshold": SSEI_POSITIVE_THRESHOLD,
        "rows": rows,
        "leave_one_family_out": loo,
        "claim_robust_to_loo": all(entry["claim_still_supported"] for entry in loo),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_md(rows, loo))
    tex_path.write_text(render_tex(rows, loo))
    print(
        f"Wrote {json_path}, {md_path}, {tex_path}. "
        f"Claim robust to leave-one-family-out: {summary['claim_robust_to_loo']}"
    )


if __name__ == "__main__":
    main()
