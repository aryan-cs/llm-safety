"""Lightweight cross-model summary table for the selectivity paper.

Aggregates per-model metrics.json (no row-level merge, no bootstrap re-run)
into a compact JSON + Markdown + LaTeX summary. Avoids the slow
``merge_selectivity_panel_results.py`` path while still giving the paper
a single artifact that quotes every model's headline numbers.

Run: ``uv run python scripts/make_cross_model_summary.py``
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MODEL_LABELS = {
    "qwen2_5_7b_base": "Qwen2.5-7B base",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen3_5_9b": "Qwen3-8B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "llama3_1_8b_instruct": "Llama-3.1-8B-Instruct",
    "gemma2_9b_it": "Gemma-2-9B-IT",
    "mistral_7b_instruct_v0_3": "Mistral-7B-Instruct-v0.3",
    "olmo3_7b_instruct": "OLMo-3-7B-Instruct",
    "phi4": "Phi-4",
    "qwen2_5_14b_msm_rules": "Qwen2.5-14B-Instruct + MSM-Rules",
    "qwen2_5_14b_msm_value_aug": "Qwen2.5-14B-Instruct + MSM-ValueAug",
    "qwen2_5_14b_instruct": "Qwen2.5-14B-Instruct",
}
MODEL_FAMILIES = {
    "qwen2_5_7b_base": "Qwen",
    "qwen2_5_7b_instruct": "Qwen",
    "qwen3_5_9b": "Qwen",
    "gpt_oss_20b": "OpenAI",
    "llama3_1_8b_instruct": "Llama",
    "gemma2_9b_it": "Gemma",
    "mistral_7b_instruct_v0_3": "Mistral",
    "olmo3_7b_instruct": "OLMo",
    "phi4": "Phi",
    "qwen2_5_14b_msm_rules": "Qwen",
    "qwen2_5_14b_msm_value_aug": "Qwen",
    "qwen2_5_14b_instruct": "Qwen",
}


def load_judgment_counts(audit_dir: Path, model_key: str, provider: str = "claude") -> dict[str, int]:
    counts = {"parsed": 0, "blocked": 0, "parse_error": 0}
    path = audit_dir / f"selectivity_h200_powered_{model_key}_judgments.{provider}.jsonl"
    if not path.exists():
        return counts
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = row.get("parser_status", "")
            if status in counts:
                counts[status] += 1
    return counts


def find_top_ssei(contrasts: dict[str, Any]) -> tuple[str | None, float | None, float | None, float | None]:
    best: tuple[str | None, float | None, float | None, float | None] = (None, None, None, None)
    best_score = float("-inf")
    for policy, payload in contrasts.items():
        ssei = payload.get("selective_safety_erasure_index")
        if ssei is None:
            continue
        if ssei > best_score:
            best_score = ssei
            ci = payload.get("selective_safety_erasure_index_ci") or {}
            best = (policy, ssei, ci.get("ci_low"), ci.get("ci_high"))
    return best


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        s = f"{float(value):.{digits}f}"
        return s.lstrip("-") if s.lstrip("-").replace("0", "").replace(".", "") == "" else s
    except (TypeError, ValueError):
        return "--"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--audit-dir", type=Path, default=Path("docs/audit"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/generated/cross_model_summary"))
    parser.add_argument("--provider", default="claude", choices=["claude", "gemini"])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(args.results_root.glob("selectivity_h200_powered_*")):
        if run_dir.name == "selectivity_h200_powered_combined":
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        model_key = run_dir.name.removeprefix("selectivity_h200_powered_")
        try:
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            continue
        contrasts = metrics.get("policy_level_contrasts") or {}
        top_policy, top_ssei, top_lo, top_hi = find_top_ssei(contrasts)
        counts = load_judgment_counts(args.audit_dir, model_key, provider=args.provider)
        total_attempts = sum(counts.values())
        rows.append(
            {
                "model_key": model_key,
                "family": MODEL_FAMILIES.get(model_key, model_key),
                "model_label": MODEL_LABELS.get(model_key, model_key),
                "judgment_parsed": counts["parsed"],
                "judgment_blocked": counts["blocked"],
                "judgment_parse_error": counts["parse_error"],
                "judgment_attempts": total_attempts,
                "policy_count": len(contrasts),
                "top_ssei_policy": top_policy,
                "top_ssei": top_ssei,
                "top_ssei_ci_low": top_lo,
                "top_ssei_ci_high": top_hi,
            }
        )

    json_out = args.output_dir / "cross_model_summary.json"
    md_out = args.output_dir / "cross_model_summary.md"
    tex_out = args.output_dir / "cross_model_summary.tex"

    summary = {
        "n_models": len(rows),
        "models_with_positive_top_ssei": sum(1 for r in rows if (r["top_ssei"] or 0) > 0),
        "rows": rows,
    }
    json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    md_lines = [
        "| Family | Model | Policies | Top SSEI policy | Top SSEI [CI] | Judge parsed / attempts |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        cells = [
            r["family"],
            r["model_label"],
            str(r["policy_count"]),
            r["top_ssei_policy"] or "--",
            f"{fmt(r['top_ssei'])} [{fmt(r['top_ssei_ci_low'])}, {fmt(r['top_ssei_ci_high'])}]",
            f"{r['judgment_parsed']} / {r['judgment_attempts']}",
        ]
        md_lines.append("| " + " | ".join(cells) + " |")
    md_out.write_text("\n".join(md_lines) + "\n")

    def _esc(text: str) -> str:
        return text.replace("_", "\\_")

    tex_lines = [
        "% Auto-generated by scripts/make_cross_model_summary.py",
        "\\begin{table}[p]",
        "\\centering",
        "\\caption{Per-model selectivity panel headline. Top SSEI is the largest positive selective-safety-erasure index across registered cache policies; 95\\% CI in brackets. Judging coverage is parsed Gemini rows / total judge attempts.}",
        "\\label{tab:cross-model-summary}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lllrrr}",
        "\\toprule",
        "\\textbf{Family} & \\textbf{Model} & \\textbf{Top policy} & \\textbf{Top SSEI} & \\textbf{95\\% CI} & \\textbf{Judge parsed / attempts} \\\\",
        "\\midrule",
    ]
    for r in rows:
        ci = f"[{fmt(r['top_ssei_ci_low'])}, {fmt(r['top_ssei_ci_high'])}]"
        tex_lines.append(
            " & ".join(
                [
                    r["family"],
                    _esc(r["model_label"]),
                    _esc(r["top_ssei_policy"] or "--"),
                    fmt(r["top_ssei"]),
                    ci,
                    f"{r['judgment_parsed']} / {r['judgment_attempts']}",
                ]
            )
            + " \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", "}", "\\end{table}", ""])
    tex_out.write_text("\n".join(tex_lines))

    print(f"Wrote {json_out}, {md_out}, {tex_out} ({len(rows)} models).")


if __name__ == "__main__":
    main()
