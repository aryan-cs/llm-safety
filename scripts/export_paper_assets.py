from __future__ import annotations

import argparse
import json
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import (
    file_sha256,
    git_commit,
    git_dirty,
    git_status_short,
    write_json,
)

TABLE_FILES = [
    "main_results_table.md",
    "main_results_table.tex",
    "suite_level_effects_table.md",
    "suite_level_effects_table.tex",
    "causal_restoration_table.md",
    "causal_restoration_table.tex",
    "result_macros.tex",
]


def export_paper_assets(results_dir: Path, paper_dir: Path, macro_prefix: str = "Primary") -> None:
    metrics_path = results_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing metrics file: {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    paper_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for policy, values in metrics.get("publication_summary", {}).get("policies", {}).items():
        contrast = metrics.get("policy_level_contrasts", {}).get(policy, {})
        ssei_ci = contrast.get("selective_safety_erasure_index_ci", {})
        summary_rows.append(
            {
                "policy": _display_causal_policy_label(policy),
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
                "policy_level_ssei_95ci": format_estimate_ci(
                    contrast.get("selective_safety_erasure_index"),
                    ssei_ci.get("ci_low"),
                    ssei_ci.get("ci_high"),
                ),
                "policy_level_safety_clusters": ssei_ci.get("n_safety"),
                "policy_level_capability_clusters": ssei_ci.get("n_capability"),
            }
        )
    write_markdown_table(
        paper_dir / "main_results_table.md",
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
        paper_dir / "main_results_table.tex",
        [
            "policy",
            "mean_safety_score",
            "mean_capability_score",
            "policy_level_ssei_95ci",
        ],
        summary_rows,
        caption="Policy-level safety, capability, and selective safety erasure summary.",
        label="tab:main-results",
    )

    selective_rows = []
    for key, values in metrics.get("selective_safety_erasure", {}).items():
        suite, policy = key.split("::", 1)
        safety_ci = values.get("paired_safety_degradation_ci", {})
        selective_rows.append(
            {
                "suite": _display_suite_label(suite),
                "policy": _display_causal_policy_label(policy),
                "safety_degradation": values.get("safety_degradation"),
                "capability_degradation": values.get("capability_degradation"),
                "within_suite_ssei_if_capability_available": values.get(
                    "selective_safety_erasure_index"
                ),
                "safety_degradation_95ci": format_estimate_ci(
                    values.get("safety_degradation"),
                    safety_ci.get("ci_low"),
                    safety_ci.get("ci_high"),
                ),
                "safety_ci_low": safety_ci.get("ci_low"),
                "safety_ci_high": safety_ci.get("ci_high"),
                "paired_n": safety_ci.get("paired_n"),
                "cluster_n": safety_ci.get("cluster_n"),
                "paired_cluster_n": _format_paired_cluster_n(
                    safety_ci.get("paired_n"),
                    safety_ci.get("cluster_n"),
                ),
            }
        )
    write_markdown_table(
        paper_dir / "suite_level_effects_table.md",
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
        paper_dir / "suite_level_effects_table.tex",
        [
            "suite",
            "policy",
            "safety_degradation_95ci",
            "paired_cluster_n",
        ],
        _compact_suite_rows(selective_rows),
        caption=(
            "Largest absolute suite-level safety-degradation effects with paired prompt "
            "counts. Full suite-policy table is exported as Markdown."
        ),
        label="tab:suite-effects",
    )
    restoration_rows = []
    for key, values in metrics.get("causal_restoration", {}).items():
        suite, policy = key.split("::", 1)
        safety_ci = values.get("safety_restoration_fraction_ci", {})
        refusal_ci = values.get("refusal_restoration_fraction_ci", {})
        leakage_ci = values.get("leakage_avoidance_restoration_fraction_ci", {})
        # Use the CI's own mean (mean-of-per-prompt-ratios) as the point
        # estimate so that the reported value and its bootstrap CI describe
        # the same estimator.  The legacy ``*_restoration_fraction`` fields
        # are the ratio-of-means estimator which can diverge from the
        # mean-of-ratios CI, especially with heterogeneous denominators.
        safety_point = safety_ci.get("mean") if safety_ci.get("mean") is not None else values.get("safety_restoration_fraction")
        refusal_point = refusal_ci.get("mean") if refusal_ci.get("mean") is not None else values.get("refusal_restoration_fraction")
        leakage_point = leakage_ci.get("mean") if leakage_ci.get("mean") is not None else values.get("leakage_avoidance_restoration_fraction")
        restoration_rows.append(
            {
                "suite": _display_causal_suite_label(suite),
                "policy": _display_causal_policy_label(policy),
                "compressed_policy": _display_policy_label(values.get("compressed_policy")),
                "safety_restoration_fraction": safety_point,
                "safety_restoration_95ci": format_estimate_ci(
                    safety_point,
                    safety_ci.get("ci_low"),
                    safety_ci.get("ci_high"),
                ),
                "refusal_restoration_fraction": refusal_point,
                "refusal_restoration_95ci": format_estimate_ci(
                    refusal_point,
                    refusal_ci.get("ci_low"),
                    refusal_ci.get("ci_high"),
                ),
                "leakage_avoidance_restoration_fraction": leakage_point,
                "leakage_avoidance_restoration_95ci": format_estimate_ci(
                    leakage_point,
                    leakage_ci.get("ci_low"),
                    leakage_ci.get("ci_high"),
                ),
            }
        )
    existing_restoration_policies = {row["policy"] for row in restoration_rows}
    for policy in metrics.get("policy_level_contrasts", {}):
        display_policy = _display_causal_policy_label(policy)
        if "policy_pinned" not in policy or display_policy in existing_restoration_policies:
            continue
        restoration_rows.append(
            {
                "suite": _display_causal_suite_label("policy_pinned_mitigation"),
                "policy": display_policy,
                "compressed_policy": _display_policy_label("kv_int4_sim"),
                "safety_restoration_fraction": None,
                "safety_restoration_95ci": "",
                "refusal_restoration_fraction": None,
                "refusal_restoration_95ci": "",
                "leakage_avoidance_restoration_fraction": None,
                "leakage_avoidance_restoration_95ci": "",
            }
        )
    write_markdown_table(
        paper_dir / "causal_restoration_table.md",
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
        paper_dir / "causal_restoration_table.tex",
        [
            "suite",
            "policy",
            "safety_restoration_95ci",
            "refusal_restoration_95ci",
            "leakage_avoidance_restoration_95ci",
        ],
        restoration_rows,
        caption="Causal restoration effects for patched and mitigation conditions.",
        label="tab:causal-restoration",
    )
    write_latex_macros(paper_dir / "result_macros.tex", metrics, results_dir, macro_prefix)
    _write_artifact_manifest(results_dir, paper_dir, macro_prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper-ready result tables from a run.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--paper-dir", type=Path, default=Path("docs/generated"))
    parser.add_argument("--macro-prefix", default="Primary")
    args = parser.parse_args()

    export_paper_assets(args.results_dir, args.paper_dir, args.macro_prefix)
    print(f"Wrote paper tables to {args.paper_dir}")


def _write_artifact_manifest(results_dir: Path, paper_dir: Path, macro_prefix: str) -> None:
    run_manifest = _read_json(results_dir / "manifest.json")
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
            "source_run_git_commit": run_manifest.get("git_commit"),
            "source_run_git_dirty": run_manifest.get("git_dirty"),
            "source_run_name": run_manifest.get("run_name"),
            "source_run_model_id": run_manifest.get("model_id"),
            "macro_prefix": macro_prefix,
            "analysis_git_commit": git_commit(),
            "analysis_git_dirty": git_dirty(),
            "analysis_git_status_short": git_status_short(),
            "tables": tables,
            "source_artifacts": source_artifacts,
        },
    )


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def write_markdown_table(path: Path, columns: list[str], rows: list[dict]) -> None:
    headers = [_markdown_header(column) for column in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _compact_suite_rows(rows: list[dict], *, max_rows: int = 16) -> list[dict]:
    informative = [
        row
        for row in rows
        if _abs_float(row.get("safety_degradation")) > 0
        or _abs_float(row.get("safety_ci_low")) > 0
        or _abs_float(row.get("safety_ci_high")) > 0
    ]
    ranked = informative or rows
    return sorted(
        ranked,
        key=lambda row: (
            -_abs_float(row.get("safety_degradation")),
            str(row.get("suite") or ""),
            str(row.get("policy") or ""),
        ),
    )[:max_rows]


def write_latex_table(
    path: Path,
    columns: list[str],
    rows: list[dict],
    *,
    caption: str,
    label: str,
) -> None:
    display_columns = [_latex_header(column) for column in columns]
    table_font = _latex_table_font(label)
    tabcolsep = _latex_table_tabcolsep(label)
    column_spec = _latex_column_spec(columns, label)
    lines = [
        r"\begin{table}[p]",
        r"\centering",
        r"\caption{" + _latex_escape(caption) + r"}",
        r"\label{" + _latex_escape(label) + r"}",
        r"\begingroup",
        table_font,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\begin{tabularx}{\linewidth}{" + column_spec + r"}",
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
            r"\endgroup",
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


def _display_suite_label(suite: object) -> str:
    suite = str(suite or "")
    return {
        "global_policy_contrast": "All public suites",
        "policy_pinned_mitigation": "Policy-pinned mitigation",
        "public_benign_overrefusal": "Benign over-refusal",
        "public_capability_arc": "Capability",
        "public_refusal_safety": "Refusal safety",
        "public_system_leakage": "Public system leakage",
        "public_xstest_safe": "XSTest safe",
        "system_leakage": "System leakage probe",
    }.get(suite, suite.replace("_", " ").title())


def _display_causal_suite_label(suite: object) -> str:
    suite = str(suite or "")
    return {
        "policy_pinned_mitigation": "Pinned",
        "public_refusal_safety": "Refusal",
        "public_system_leakage": "Public leak",
        "system_leakage": "Leak probe",
    }.get(suite, _display_suite_label(suite))


def _display_policy_label(policy: object) -> str:
    policy = str(policy or "")
    direct = {
        "": "",
        "none": "Baseline",
        "kv_int4_sim": "4-bit KV",
        "kv_int8_sim": "8-bit KV",
        "policy_pinned__budget128__sink8": "Policy-pinned cache",
        "random_matched__budget128__seed991": "Random matched",
        "sink_recent__budget128__sink8": "Sink+recent",
        "sliding_window__budget64": "Window 64",
        "sliding_window__budget128": "Window 128",
        "sliding_window__budget256": "Window 256",
    }
    if policy in direct:
        return direct[policy]
    if policy.startswith("kv_int4_sim__"):
        patch = "Patch"
        if "patchkey-value" in policy:
            patch = "Patch K+V"
        elif "patchkey" in policy:
            patch = "Patch keys"
        elif "patchvalue" in policy:
            patch = "Patch values"
        if "roleuser" in policy and "matchsystem" in policy:
            return f"{patch} on matched user tokens"
        if "rolesystem" in policy:
            return f"{patch} on system tokens"
    label = policy.replace("__", " / ")
    label = label.replace("budget", "budget ")
    label = label.replace("seed", "seed ")
    label = label.replace("sink", "sink ")
    label = label.replace("patchkey-value", "patch K+V")
    label = label.replace("patchkey", "patch keys")
    label = label.replace("patchvalue", "patch values")
    label = label.replace("rolesystem", "system")
    label = label.replace("roleuser", "user")
    return label


def _display_causal_policy_label(policy: object) -> str:
    policy = str(policy or "")
    if policy.startswith("kv_int4_sim__"):
        if "patchkey-value" in policy and "roleuser" in policy and "matchsystem" in policy:
            return "K+V user"
        if "patchkey-value" in policy and "rolesystem" in policy:
            return "K+V sys"
        if "patchkey" in policy and "rolesystem" in policy:
            return "Key sys"
        if "patchvalue" in policy and "rolesystem" in policy:
            return "Value sys"
    return _display_policy_label(policy)


def _format_paired_cluster_n(paired_n: object, cluster_n: object) -> str:
    if paired_n in (None, "") and cluster_n in (None, ""):
        return ""
    if paired_n == cluster_n:
        return format_value(paired_n)
    return f"{format_value(paired_n)} / {format_value(cluster_n)}"


def _abs_float(value: object) -> float:
    try:
        return abs(float(value))
    except (TypeError, ValueError):
        return 0.0


def format_estimate_ci(value: object, ci_low: object, ci_high: object) -> str:
    estimate = format_value(value)
    low = format_value(ci_low)
    high = format_value(ci_high)
    if estimate == "":
        return ""
    if low == "" or high == "":
        return estimate
    return f"{estimate} [{low},{high}]"


def _latex_header(column: str) -> str:
    return _latex_escape(
        {
            "capability_degradation": "Capability delta",
            "compressed_policy": "Compressed",
            "leakage_avoidance_restoration_95ci": "Leakage [95% CI]",
            "mean_capability_score": "Mean capability",
            "mean_safety_score": "Mean safety",
            "paired_cluster_n": "Paired / clusters",
            "policy_level_ssei_95ci": "Policy SSeI [95% CI]",
            "refusal_restoration_95ci": "Refusal [95% CI]",
            "safety_degradation_95ci": "Safety delta [95% CI]",
            "safety_restoration_95ci": "Safety [95% CI]",
            "within_suite_ssei_if_capability_available": "Within-suite SSeI",
        }.get(column, column.replace("_", " "))
    )


def _latex_table_font(label: str) -> str:
    if label == "tab:causal-restoration":
        return r"\footnotesize"
    return r"\small"


def _latex_table_tabcolsep(label: str) -> str:
    if label == "tab:causal-restoration":
        return "2pt"
    return "3pt"


def _latex_column_spec(columns: list[str], label: str) -> str:
    if label == "tab:causal-restoration":
        return "@{}ll" + "X" * max(0, len(columns) - 2) + "@{}"
    if len(columns) >= 2 and columns[0] == "suite" and columns[1] == "policy":
        return "@{}ll" + "X" * max(0, len(columns) - 2) + "@{}"
    return "@{}l" + "X" * max(0, len(columns) - 1) + "@{}"


def _markdown_header(column: str) -> str:
    return {
        "capability_degradation": "capability_delta",
        "compressed_policy": "compressed",
        "leakage_avoidance_restoration_95ci": "leakage_restoration_95ci",
        "mean_capability_score": "mean_capability",
        "mean_safety_score": "mean_safety",
        "paired_cluster_n": "paired_clusters",
        "policy_level_ssei_95ci": "policy_ssei_95ci",
        "refusal_restoration_95ci": "refusal_restoration_95ci",
        "safety_degradation_95ci": "safety_delta_95ci",
        "safety_restoration_95ci": "safety_restoration_95ci",
        "within_suite_ssei_if_capability_available": "within_suite_ssei",
    }.get(column, column)


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
        s = f"{value:.3f}"
        return s.lstrip("-") if s.lstrip("-").replace("0", "").replace(".", "") == "" else s
    return str(value)


if __name__ == "__main__":
    main()
