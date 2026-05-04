from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256, write_json

TABLE_FILES = [
    "main_results_table.md",
    "main_results_table.tex",
    "suite_level_effects_table.md",
    "suite_level_effects_table.tex",
    "causal_restoration_table.md",
    "causal_restoration_table.tex",
    "result_macros.tex",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready result tables from a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--macro-prefix", default="Primary")
    args = parser.parse_args()

    metrics_path = args.results_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    args.paper_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for policy, values in metrics.get("publication_summary", {}).get("policies", {}).items():
        contrast = metrics.get("policy_level_contrasts", {}).get(policy, {})
        ssei_ci = contrast.get("selective_safety_erasure_index_ci", {})
        summary_rows.append(
            {
                "policy": policy,
                "mean_safety_score": values.get("mean_safety_score"),
                "mean_capability_score": values.get("mean_capability_score"),
                "global_safety_degradation": values.get("global_safety_degradation"),
                "global_capability_degradation": values.get("global_capability_degradation"),
                "global_selective_safety_erasure_index": values.get(
                    "global_selective_safety_erasure_index"
                ),
                "policy_level_ssei": contrast.get("selective_safety_erasure_index"),
                "policy_level_ssei_ci_low": ssei_ci.get("ci_low"),
                "policy_level_ssei_ci_high": ssei_ci.get("ci_high"),
                "policy_level_safety_clusters": ssei_ci.get("n_safety"),
                "policy_level_capability_clusters": ssei_ci.get("n_capability"),
            }
        )
    write_markdown_table(
        args.paper_dir / "main_results_table.md",
        [
            "policy",
            "mean_safety_score",
            "mean_capability_score",
            "policy_level_ssei",
            "policy_level_ssei_ci_low",
            "policy_level_ssei_ci_high",
            "policy_level_safety_clusters",
            "policy_level_capability_clusters",
        ],
        summary_rows,
    )
    write_latex_table(
        args.paper_dir / "main_results_table.tex",
        [
            "policy",
            "mean_safety_score",
            "mean_capability_score",
            "policy_level_ssei",
            "policy_level_ssei_ci_low",
            "policy_level_ssei_ci_high",
        ],
        summary_rows,
        caption="Policy-level safety, capability, and selective safety erasure summary.",
        label="tab:main-results",
    )

    selective_rows = []
    for key, values in metrics.get("selective_safety_erasure", {}).items():
        suite, policy = key.split("::", 1)
        selective_rows.append(
            {
                "suite": suite,
                "policy": policy,
                "safety_degradation": values.get("safety_degradation"),
                "capability_degradation": values.get("capability_degradation"),
                "within_suite_ssei_if_capability_available": values.get(
                    "selective_safety_erasure_index"
                ),
                "paired_n": values.get("paired_safety_degradation_ci", {}).get("paired_n"),
                "cluster_n": values.get("paired_safety_degradation_ci", {}).get("cluster_n"),
                "safety_ci_low": values.get("paired_safety_degradation_ci", {}).get("ci_low"),
                "safety_ci_high": values.get("paired_safety_degradation_ci", {}).get("ci_high"),
            }
        )
    write_markdown_table(
        args.paper_dir / "suite_level_effects_table.md",
        [
            "suite",
            "policy",
            "safety_degradation",
            "capability_degradation",
            "within_suite_ssei_if_capability_available",
            "paired_n",
            "cluster_n",
            "safety_ci_low",
            "safety_ci_high",
        ],
        selective_rows,
    )
    write_latex_table(
        args.paper_dir / "suite_level_effects_table.tex",
        [
            "suite",
            "policy",
            "safety_degradation",
            "capability_degradation",
            "within_suite_ssei_if_capability_available",
            "paired_n",
            "cluster_n",
            "safety_ci_low",
            "safety_ci_high",
        ],
        selective_rows,
        caption="Suite-level degradation effects with paired prompt counts.",
        label="tab:suite-effects",
    )
    restoration_rows = []
    for key, values in metrics.get("causal_restoration", {}).items():
        suite, policy = key.split("::", 1)
        restoration_rows.append(
            {
                "suite": suite,
                "policy": policy,
                "compressed_policy": values.get("compressed_policy"),
                "safety_restoration_fraction": values.get("safety_restoration_fraction"),
                "safety_ci_low": values.get("safety_restoration_fraction_ci", {}).get("ci_low"),
                "safety_ci_high": values.get("safety_restoration_fraction_ci", {}).get(
                    "ci_high"
                ),
                "refusal_restoration_fraction": values.get("refusal_restoration_fraction"),
                "refusal_ci_low": values.get("refusal_restoration_fraction_ci", {}).get("ci_low"),
                "refusal_ci_high": values.get("refusal_restoration_fraction_ci", {}).get(
                    "ci_high"
                ),
                "leakage_avoidance_restoration_fraction": values.get(
                    "leakage_avoidance_restoration_fraction"
                ),
                "leakage_avoidance_ci_low": values.get(
                    "leakage_avoidance_restoration_fraction_ci", {}
                ).get("ci_low"),
                "leakage_avoidance_ci_high": values.get(
                    "leakage_avoidance_restoration_fraction_ci", {}
                ).get("ci_high"),
            }
        )
    write_markdown_table(
        args.paper_dir / "causal_restoration_table.md",
        [
            "suite",
            "policy",
            "compressed_policy",
            "safety_restoration_fraction",
            "safety_ci_low",
            "safety_ci_high",
            "refusal_restoration_fraction",
            "refusal_ci_low",
            "refusal_ci_high",
            "leakage_avoidance_restoration_fraction",
            "leakage_avoidance_ci_low",
            "leakage_avoidance_ci_high",
        ],
        restoration_rows,
    )
    write_latex_table(
        args.paper_dir / "causal_restoration_table.tex",
        [
            "suite",
            "policy",
            "compressed_policy",
            "safety_restoration_fraction",
            "safety_ci_low",
            "safety_ci_high",
            "refusal_restoration_fraction",
            "refusal_ci_low",
            "refusal_ci_high",
            "leakage_avoidance_restoration_fraction",
            "leakage_avoidance_ci_low",
            "leakage_avoidance_ci_high",
        ],
        restoration_rows,
        caption="Causal restoration effects for patched and mitigation conditions.",
        label="tab:causal-restoration",
    )
    write_latex_macros(args.paper_dir / "result_macros.tex", metrics, args.results_dir, args.macro_prefix)
    _write_artifact_manifest(args.results_dir, args.paper_dir)
    print(f"Wrote paper tables to {args.paper_dir}")


def _write_artifact_manifest(results_dir: Path, paper_dir: Path) -> None:
    tables = {
        name: {
            "path": str(paper_dir / name),
            "sha256": file_sha256(paper_dir / name),
            "bytes": (paper_dir / name).stat().st_size if (paper_dir / name).exists() else None,
        }
        for name in TABLE_FILES
    }
    source_artifacts = {
        name: {
            "path": str(results_dir / name),
            "sha256": file_sha256(results_dir / name),
            "bytes": (results_dir / name).stat().st_size if (results_dir / name).exists() else None,
        }
        for name in ["manifest.json", "metrics.json", "figures/manifest.json"]
    }
    write_json(
        paper_dir / "artifact_manifest.json",
        {
            "schema_version": 1,
            "results_dir": str(results_dir),
            "tables": tables,
            "source_artifacts": source_artifacts,
        },
    )


def write_markdown_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(
    path: Path,
    columns: list[str],
    rows: list[dict],
    *,
    caption: str,
    label: str,
) -> None:
    display_columns = [_latex_header(column) for column in columns]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\linewidth}{@{}" + "l" + "X" * (len(columns) - 1) + r"@{}}",
        r"\toprule",
        " & ".join(display_columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(_latex_escape(format_value(row.get(column))) for column in columns) + r" \\"
        )
    if not rows:
        lines.append(
            r"\multicolumn{"
            + str(len(columns))
            + r"}{c}{Results pending; no readiness-passing rows exported.} \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\caption{" + _latex_escape(caption) + r"}",
            r"\label{" + _latex_escape(label) + r"}",
            r"\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex_macros(path: Path, metrics: dict, results_dir: Path, prefix: str) -> None:
    policies = metrics.get("publication_summary", {}).get("policies", {})
    contrasts = metrics.get("policy_level_contrasts", {})
    top_policy = ""
    top_values: dict[str, object] = {}
    ranked = [
        (policy, values)
        for policy, values in contrasts.items()
        if values.get("selective_safety_erasure_index") is not None
    ]
    if ranked:
        top_policy, top_values = max(
            ranked,
            key=lambda item: float(item[1].get("selective_safety_erasure_index") or -999),
        )
    ci = top_values.get("selective_safety_erasure_index_ci", {}) if top_values else {}
    macros = {
        "RunId": _publication_run_label(prefix, results_dir.name),
        "PolicyCount": len(policies),
        "TopSSEIPolicy": top_policy,
        "TopSSEI": format_value(top_values.get("selective_safety_erasure_index"))
        if top_values
        else "",
        "TopSSEICILow": format_value(ci.get("ci_low")) if isinstance(ci, dict) else "",
        "TopSSEICIHigh": format_value(ci.get("ci_high")) if isinstance(ci, dict) else "",
        "SafetyClusterCount": format_value(ci.get("n_safety")) if isinstance(ci, dict) else "",
        "CapabilityClusterCount": format_value(ci.get("n_capability")) if isinstance(ci, dict) else "",
    }
    lines = [
        "% Auto-generated by scripts/export_paper_assets.py; do not edit by hand.",
    ]
    for suffix, value in macros.items():
        macro_name = _latex_macro_name(prefix, suffix)
        lines.append(f"\\renewcommand{{\\{macro_name}}}{{{_latex_escape(str(value))}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_macro_name(prefix: str, suffix: str) -> str:
    safe_prefix = "".join(char for char in prefix.title() if char.isalpha())
    if not safe_prefix:
        safe_prefix = "Primary"
    return f"{safe_prefix}{suffix}"


def _publication_run_label(prefix: str, fallback: str) -> str:
    normalized = "".join(char for char in prefix.lower() if char.isalpha())
    if normalized.startswith("primary"):
        return "primary public sweep"
    if normalized.startswith("causal"):
        return "causal restoration diagnostic"
    if "thirtytwo" in normalized:
        return "model-scale follow-up"
    return fallback


def _latex_header(column: str) -> str:
    return _latex_escape(column.replace("_", " "))


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
