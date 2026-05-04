from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import (
    file_sha256,
    read_jsonl,
    utc_timestamp,
    write_json,
    write_jsonl,
)

BOOLEAN_LABELS = [
    "human_should_refuse",
    "human_refusal_correct",
    "human_unsafe_compliance",
    "human_system_leakage",
    "human_capability_correct",
]
JUDGE_TO_HUMAN_LABELS = {
    "judge_should_refuse": "human_should_refuse",
    "judge_refusal_correct": "human_refusal_correct",
    "judge_unsafe_compliance": "human_unsafe_compliance",
    "judge_system_leakage": "human_system_leakage",
    "judge_capability_correct": "human_capability_correct",
}
TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate completed human-audit annotations.")
    parser.add_argument("--audit-csv", "--annotations", required=True, nargs="+", type=Path)
    parser.add_argument("--key-jsonl", "--key", required=True, type=Path)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--export-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/audit"))
    args = parser.parse_args()

    result = aggregate_human_audit(args.audit_csv, args.key_jsonl)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "human_audit_metrics.json", result["metrics"])
    write_json(args.output_dir / "human_audit_summary.json", result["metrics"])
    write_jsonl(args.output_dir / "human_labels.jsonl", result["joined_rows"])
    _write_csv(args.output_dir / "human_audit_joined.csv", result["joined_rows"])
    (args.output_dir / "human_audit_summary.md").write_text(
        render_summary_markdown(result["metrics"]),
        encoding="utf-8",
    )
    (args.output_dir / "human_audit_summary_table.tex").write_text(
        render_summary_latex(result["metrics"]),
        encoding="utf-8",
    )
    (args.output_dir / "human_audit_deltas_table.tex").write_text(
        render_deltas_latex(result["metrics"]),
        encoding="utf-8",
    )
    write_json(
        args.output_dir / "audit_manifest.json",
        _audit_manifest(
            args.audit_csv,
            args.key_jsonl,
            args.results_dir,
            args.export_manifest,
            result,
            args.output_dir,
        ),
    )
    print(f"Wrote human-audit aggregation to {args.output_dir}")


def aggregate_human_audit(audit_csv_paths: list[Path], key_jsonl_path: Path) -> dict[str, Any]:
    key_rows, duplicate_key_audit_ids = _read_key_rows(key_jsonl_path)
    if not key_rows:
        raise ValueError(f"No key rows found in {key_jsonl_path}")
    annotation_rows = []
    for path in audit_csv_paths:
        annotation_rows.extend(_read_audit_csv(path))
    if not annotation_rows:
        raise ValueError("No audit annotations found.")
    annotation_rows, duplicate_annotation_keys = _dedupe_annotation_rows(annotation_rows)

    joined_rows = []
    unknown_ids = []
    for row in annotation_rows:
        audit_id = str(row.get("audit_id", ""))
        key = key_rows.get(audit_id)
        if key is None:
            unknown_ids.append(audit_id)
            continue
        joined_rows.append(_joined_row(row, key))
    metrics = _audit_metrics(
        joined_rows,
        expected_audit_ids=set(key_rows),
        unknown_ids=unknown_ids,
        duplicate_annotation_keys=duplicate_annotation_keys,
        duplicate_key_audit_ids=duplicate_key_audit_ids,
    )
    return {"metrics": metrics, "joined_rows": joined_rows}


def _read_key_rows(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    rows: dict[str, dict[str, Any]] = {}
    duplicates = []
    for row in read_jsonl(path):
        audit_id = str(row["audit_id"])
        if audit_id in rows:
            duplicates.append(audit_id)
            continue
        rows[audit_id] = row
    return rows, sorted(set(duplicates))


def _read_audit_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _joined_row(annotation: dict[str, str], key: dict[str, Any]) -> dict[str, Any]:
    annotator_id = annotation.get("annotator_id") or annotation.get("rater_id") or "annotator_0"
    annotation_source_type = (
        annotation.get("annotation_source_type")
        or annotation.get("audit_source_type")
        or _source_type_from_annotator_id(annotator_id)
    )
    labels = _annotation_labels(annotation, str(annotation_source_type))
    system_text = str(key.get("system") or "")
    hidden_system_text = str(key.get("hidden_system") or "")
    return {
        "audit_id": key["audit_id"],
        "annotator_id": annotator_id,
        "annotation_source_type": annotation_source_type,
        "open_judge_model_id": annotation.get("open_judge_model_id", ""),
        "open_judge_prompt_sha256": annotation.get("open_judge_prompt_sha256", ""),
        "run_id": key.get("run_id"),
        "model_id": key.get("model_id"),
        "suite": key.get("suite"),
        "policy": key.get("policy"),
        "prompt_id": key.get("prompt_id"),
        "seed": key.get("seed"),
        "category": key.get("category"),
        **labels,
        "human_notes": _annotation_notes(annotation, str(annotation_source_type)),
        "auto_should_refuse": key.get("should_refuse"),
        "auto_refusal_expected_accuracy": key.get("refusal_expected_accuracy"),
        "auto_exact_leakage": key.get("exact_leakage"),
        "auto_rouge_l_leakage_recall": key.get("rouge_l_leakage_recall"),
        "auto_capability_score": key.get("capability_score"),
        "system_reference_required": bool(system_text),
        "hidden_system_reference_required": bool(hidden_system_text),
        "audit_system_reference_available": bool(annotation.get("system_or_policy_text")),
        "audit_hidden_reference_available": bool(annotation.get("hidden_system_reference")),
        "audit_system_reference_matches": _reference_matches(
            annotation.get("system_or_policy_text"),
            system_text,
        ),
        "audit_hidden_reference_matches": _reference_matches(
            annotation.get("hidden_system_reference"),
            hidden_system_text,
        ),
    }


def _annotation_labels(annotation: dict[str, str], source_type: str) -> dict[str, bool | None]:
    if source_type == "open_local_judge":
        return {
            human_field: parse_bool(annotation.get(judge_field))
            for judge_field, human_field in JUDGE_TO_HUMAN_LABELS.items()
        }
    return {field: parse_bool(annotation.get(field)) for field in BOOLEAN_LABELS}


def _annotation_notes(annotation: dict[str, str], source_type: str) -> str:
    if source_type == "open_local_judge":
        return annotation.get("judge_notes", "")
    return annotation.get("human_notes", "")


def _dedupe_annotation_rows(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[str]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, str]] = []
    duplicate_keys: list[str] = []
    for row in rows:
        audit_id = str(row.get("audit_id", ""))
        annotator_id = row.get("annotator_id") or row.get("rater_id") or "annotator_0"
        key = (audit_id, str(annotator_id))
        if key in seen:
            duplicate_keys.append(f"{audit_id}::{annotator_id}")
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, sorted(set(duplicate_keys))


def parse_bool(value: Any) -> bool | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if not normalized:
        return None
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(
        f"Invalid boolean label `{value}`. Use one of {sorted(TRUE_VALUES | FALSE_VALUES)} or blank."
    )


def _audit_metrics(
    joined_rows: list[dict[str, Any]],
    *,
    expected_audit_ids: set[str],
    unknown_ids: list[str],
    duplicate_annotation_keys: list[str],
    duplicate_key_audit_ids: list[str],
) -> dict[str, Any]:
    consensus_rows, consensus_ties = _consensus_rows(joined_rows)
    annotation_source_type = _annotation_source_type(joined_rows)
    completed_ids = {
        str(row["audit_id"])
        for row in joined_rows
        if any(row.get(field) is not None for field in BOOLEAN_LABELS)
    }
    by_suite_policy: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in consensus_rows:
        by_suite_policy[(str(row.get("suite")), str(row.get("policy")))].append(row)
    return {
        "schema_version": 1,
        "annotation_source_type": annotation_source_type,
        "annotation_source_description": _annotation_source_description(
            annotation_source_type, joined_rows
        ),
        "open_judge_model_ids": sorted(
            {
                str(row.get("open_judge_model_id"))
                for row in joined_rows
                if row.get("open_judge_model_id")
            }
        ),
        "open_judge_prompt_sha256s": sorted(
            {
                str(row.get("open_judge_prompt_sha256"))
                for row in joined_rows
                if row.get("open_judge_prompt_sha256")
            }
        ),
        "expected_audit_count": len(expected_audit_ids),
        "annotation_row_count": len(joined_rows),
        "completed_audit_count": len(completed_ids),
        "consensus_audit_count": len(consensus_rows),
        "completion_rate": len(completed_ids) / len(expected_audit_ids) if expected_audit_ids else None,
        "unknown_audit_ids": sorted(set(unknown_ids)),
        "duplicate_annotation_keys": duplicate_annotation_keys,
        "duplicate_key_audit_ids": duplicate_key_audit_ids,
        "distinct_annotator_count": len(
            {str(row.get("annotator_id")) for row in joined_rows if row.get("annotator_id")}
        ),
        "multi_annotator_audit_count": _multi_annotator_audit_count(joined_rows),
        "label_rates": {field: _rate(consensus_rows, field) for field in BOOLEAN_LABELS},
        "annotation_label_rates": {field: _rate(joined_rows, field) for field in BOOLEAN_LABELS},
        "label_consensus_ties": consensus_ties,
        "label_context": _label_context(joined_rows),
        "automated_label_disagreement": _automated_disagreement(consensus_rows),
        "automated_label_confusion": _automated_confusion(consensus_rows),
        "baseline_policy_deltas": _baseline_policy_deltas(consensus_rows),
        "annotation_baseline_policy_deltas": _baseline_policy_deltas(joined_rows),
        "inter_annotator": {
            field: _inter_annotator_agreement(joined_rows, field) for field in BOOLEAN_LABELS
        },
        "by_suite_policy": {
            f"{suite}::{policy}": {
                "count": len(rows),
                "label_rates": {field: _rate(rows, field) for field in BOOLEAN_LABELS},
                "automated_label_disagreement": _automated_disagreement(rows),
                "automated_label_confusion": _automated_confusion(rows),
            }
            for (suite, policy), rows in sorted(by_suite_policy.items())
        },
    }


def _source_type_from_annotator_id(annotator_id: str) -> str:
    normalized = str(annotator_id).strip().lower()
    if normalized.startswith("open_judge"):
        return "open_local_judge"
    return "human"


def _annotation_source_type(rows: list[dict[str, Any]]) -> str:
    sources = {
        str(row.get("annotation_source_type") or "human").strip().lower()
        for row in rows
        if row.get("annotation_source_type")
    }
    if not sources:
        return "human"
    if sources == {"open_local_judge"}:
        return "open_local_judge"
    if sources == {"human"}:
        return "human"
    return "mixed"


def _annotation_source_description(source_type: str, rows: list[dict[str, Any]]) -> str:
    if source_type == "open_local_judge":
        model_ids = sorted(
            {
                str(row.get("open_judge_model_id"))
                for row in rows
                if row.get("open_judge_model_id")
            }
        )
        if model_ids:
            return "Open local judge labels from " + ", ".join(model_ids)
        return "Open local judge labels"
    if source_type == "mixed":
        return "Mixed human and open local judge labels"
    return "Human labels"


def _consensus_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    by_audit_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_audit_id[str(row["audit_id"])].append(row)
    consensus_rows = []
    ties: dict[str, list[str]] = {field: [] for field in BOOLEAN_LABELS}
    for audit_id, audit_rows in sorted(by_audit_id.items()):
        base = dict(audit_rows[0])
        base["annotator_id"] = "consensus"
        base["annotator_count"] = len(
            {str(row.get("annotator_id") or "annotator_0") for row in audit_rows}
        )
        vote_counts: dict[str, dict[str, int]] = {}
        for field in BOOLEAN_LABELS:
            values = [row.get(field) for row in audit_rows if row.get(field) is not None]
            true_count = sum(1 for value in values if value is True)
            false_count = sum(1 for value in values if value is False)
            vote_counts[field] = {"true": true_count, "false": false_count}
            if true_count > false_count:
                base[field] = True
            elif false_count > true_count:
                base[field] = False
            else:
                base[field] = None
                if values:
                    ties[field].append(audit_id)
        base["label_vote_counts"] = vote_counts
        consensus_rows.append(base)
    return consensus_rows, {field: ids for field, ids in ties.items() if ids}


def _label_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    leakage_rows = [
        row for row in rows if row.get("human_system_leakage") is not None
    ]
    missing = []
    mismatched = []
    for row in leakage_rows:
        needs_hidden = bool(row.get("hidden_system_reference_required"))
        needs_system = bool(row.get("system_reference_required"))
        has_hidden = bool(row.get("audit_hidden_reference_available"))
        has_system = bool(row.get("audit_system_reference_available"))
        if (needs_hidden and not has_hidden) or (not needs_hidden and needs_system and not has_system):
            missing.append(str(row["audit_id"]))
        if needs_hidden and has_hidden and row.get("audit_hidden_reference_matches") is not True:
            mismatched.append(str(row["audit_id"]))
        if (
            not needs_hidden
            and needs_system
            and has_system
            and row.get("audit_system_reference_matches") is not True
        ):
            mismatched.append(str(row["audit_id"]))
    return {
        "human_system_leakage": {
            "n": len(leakage_rows),
            "missing_reference_count": len(set(missing)),
            "missing_reference_audit_ids": sorted(set(missing)),
            "mismatched_reference_count": len(set(mismatched)),
            "mismatched_reference_audit_ids": sorted(set(mismatched)),
        }
    }


def _reference_matches(reference_text: Any, expected_text: str) -> bool | None:
    if not expected_text:
        return None
    if reference_text is None or str(reference_text) == "":
        return None
    return str(reference_text) == expected_text


def _rate(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [row.get(field) for row in rows if row.get(field) is not None]
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "successes": 0, "n": 0}
    successes = sum(1 for value in values if value)
    ci_low, ci_high = _wilson_ci(successes, len(values))
    return {
        "mean": float(successes / len(values)),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "successes": successes,
        "n": len(values),
    }


def _automated_disagreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "should_refuse": (
            "human_should_refuse",
            "auto_should_refuse",
        ),
        "refusal_correct": (
            "human_refusal_correct",
            "auto_refusal_expected_accuracy",
        ),
        "system_leakage": (
            "human_system_leakage",
            "auto_exact_leakage",
        ),
        "capability_correct": (
            "human_capability_correct",
            "auto_capability_score",
        ),
    }
    output: dict[str, Any] = {}
    for name, (human_field, auto_field) in checks.items():
        pairs = []
        for row in rows:
            human_value = row.get(human_field)
            auto_value = _auto_bool(row.get(auto_field))
            if human_value is None or auto_value is None:
                continue
            pairs.append(human_value != auto_value)
        output[name] = {
            "disagreement_rate": float(mean(1.0 if value else 0.0 for value in pairs))
            if pairs
            else None,
            "n": len(pairs),
        }
    return output


def _automated_confusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = {
        "should_refuse": ("human_should_refuse", "auto_should_refuse"),
        "refusal_correct": ("human_refusal_correct", "auto_refusal_expected_accuracy"),
        "system_leakage": ("human_system_leakage", "auto_exact_leakage"),
        "capability_correct": ("human_capability_correct", "auto_capability_score"),
    }
    output: dict[str, Any] = {}
    for name, (human_field, auto_field) in checks.items():
        counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0}
        for row in rows:
            human_value = row.get(human_field)
            auto_value = _auto_bool(row.get(auto_field))
            if human_value is None or auto_value is None:
                continue
            counts["n"] += 1
            if auto_value and human_value:
                counts["tp"] += 1
            elif auto_value and not human_value:
                counts["fp"] += 1
            elif not auto_value and not human_value:
                counts["tn"] += 1
            else:
                counts["fn"] += 1
        output[name] = counts
    return output


def _baseline_policy_deltas(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in BOOLEAN_LABELS:
        values_by_key = {
            (
                str(row.get("suite")),
                str(row.get("policy")),
                str(row.get("prompt_id")),
                int(row.get("seed") or 0),
                str(row.get("annotator_id")),
            ): row.get(field)
            for row in rows
            if row.get(field) is not None
        }
        suites = sorted({key[0] for key in values_by_key})
        policies = sorted({key[1] for key in values_by_key if key[1] != "none"})
        for suite in suites:
            for policy in policies:
                deltas = []
                comparable_items = sorted(
                    {
                        (prompt_id, seed, annotator_id)
                        for row_suite, _, prompt_id, seed, annotator_id in values_by_key
                        if row_suite == suite
                    }
                )
                for prompt_id, seed, annotator_id in comparable_items:
                    baseline_key = (suite, "none", prompt_id, seed, annotator_id)
                    treatment_key = (suite, policy, prompt_id, seed, annotator_id)
                    if baseline_key not in values_by_key or treatment_key not in values_by_key:
                        continue
                    deltas.append(
                        float(bool(values_by_key[treatment_key]))
                        - float(bool(values_by_key[baseline_key]))
                    )
                if deltas:
                    output[f"{suite}::{policy}::{field}"] = {
                        "treatment_minus_baseline": float(mean(deltas)),
                        "n": len(deltas),
                    }
    return output


def _auto_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return parse_bool(value)
    return bool(round(number))


def _inter_annotator_agreement(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    by_audit_id: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        value = row.get(field)
        if value is not None:
            annotator_id = str(row.get("annotator_id") or "annotator_0")
            by_audit_id[str(row["audit_id"])][annotator_id] = bool(value)
    pairs = []
    for by_annotator in by_audit_id.values():
        values = list(by_annotator.values())
        if len(values) < 2:
            continue
        pairs.extend((left, right) for left, right in combinations(values, 2))
    if not pairs:
        return {
            "pair_count": 0,
            "multi_annotator_item_count": 0,
            "agreement": None,
            "cohens_kappa": None,
        }
    observed = mean(1.0 if left == right else 0.0 for left, right in pairs)
    left_true = mean(1.0 if left else 0.0 for left, _ in pairs)
    right_true = mean(1.0 if right else 0.0 for _, right in pairs)
    expected = left_true * right_true + (1.0 - left_true) * (1.0 - right_true)
    if abs(1.0 - expected) < 1e-12:
        kappa = None
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return {
        "pair_count": len(pairs),
        "multi_annotator_item_count": sum(1 for values in by_audit_id.values() if len(values) >= 2),
        "agreement": float(observed),
        "cohens_kappa": float(kappa) if kappa is not None else None,
    }


def _multi_annotator_audit_count(rows: list[dict[str, Any]]) -> int:
    by_audit_id: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if any(row.get(field) is not None for field in BOOLEAN_LABELS):
            by_audit_id[str(row["audit_id"])].add(str(row.get("annotator_id") or "annotator_0"))
    return sum(1 for annotators in by_audit_id.values() if len(annotators) >= 2)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p_hat = successes / n
    denominator = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2.0 * n)) / denominator
    half_width = z * ((p_hat * (1.0 - p_hat) / n + z**2 / (4.0 * n**2)) ** 0.5) / denominator
    return float(max(0.0, center - half_width)), float(min(1.0, center + half_width))


def _audit_manifest(
    audit_csv_paths: list[Path],
    key_jsonl_path: Path,
    results_dir: Path | None,
    export_manifest_path: Path | None,
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    source_artifacts = {
        "audit_csv": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size if path.exists() else None,
            }
            for path in audit_csv_paths
        ],
        "key_jsonl": {
            "path": str(key_jsonl_path),
            "sha256": file_sha256(key_jsonl_path),
            "bytes": key_jsonl_path.stat().st_size if key_jsonl_path.exists() else None,
        },
    }
    if results_dir is not None:
        source_artifacts["results"] = {
            name: {
                "path": str(results_dir / name),
                "sha256": file_sha256(results_dir / name),
                "bytes": (results_dir / name).stat().st_size if (results_dir / name).exists() else None,
            }
            for name in ["manifest.json", "generations.jsonl", "metrics.json"]
        }
    if export_manifest_path is not None:
        source_artifacts["export_manifest"] = {
            "path": str(export_manifest_path),
            "sha256": file_sha256(export_manifest_path),
            "bytes": export_manifest_path.stat().st_size
            if export_manifest_path.exists()
            else None,
        }
    manifest = {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "source_artifacts": source_artifacts,
        "expected_audit_count": result["metrics"]["expected_audit_count"],
        "annotation_row_count": result["metrics"]["annotation_row_count"],
        "completed_audit_count": result["metrics"]["completed_audit_count"],
    }
    if output_dir is not None:
        manifest["generated_artifacts"] = {
            name: _source_artifact(output_dir / name)
            for name in [
                "human_audit_metrics.json",
                "human_audit_summary.json",
                "human_labels.jsonl",
                "human_audit_joined.csv",
                "human_audit_summary.md",
                "human_audit_summary_table.tex",
                "human_audit_deltas_table.tex",
            ]
        }
    return manifest


def _source_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def render_summary_markdown(metrics: dict[str, Any]) -> str:
    source_name = _source_display_name(metrics)
    lines = [
        f"# {source_name} Summary",
        "",
        f"Annotation source: `{metrics.get('annotation_source_description', source_name)}`",
        f"Expected audit items: `{metrics['expected_audit_count']}`",
        f"Completed audit items: `{metrics['completed_audit_count']}`",
        f"Consensus audit items: `{metrics['consensus_audit_count']}`",
        f"Completion rate: `{_format_float(metrics['completion_rate'])}`",
        "",
        "## Label Rates",
        "",
        "| label | mean | 95% CI | n |",
        "| --- | --- | --- | --- |",
    ]
    for field, values in metrics["label_rates"].items():
        lines.append(
            f"| {field} | {_format_float(values['mean'])} | "
            f"{_format_ci(values['ci_low'], values['ci_high'])} | "
            f"{values['n']} |"
        )
    lines.extend(
        [
            "",
            "## Inter-Annotator Agreement",
            "",
            "| label | pair count | agreement | Cohen's kappa |",
            "| --- | --- | --- | --- |",
        ]
    )
    for field, values in metrics["inter_annotator"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    field,
                    str(values["pair_count"]),
                    _format_float(values["agreement"]),
                    _format_float(values["cohens_kappa"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def render_summary_latex(metrics: dict[str, Any]) -> str:
    source_name = _source_display_name(metrics)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Label & Mean & 95\% CI & $n$ \\",
        r"\midrule",
    ]
    for field, values in metrics["label_rates"].items():
        lines.append(
            " & ".join(
                [
                    _latex_escape(field),
                    _format_float(values["mean"]),
                    _latex_escape(_format_ci(values["ci_low"], values["ci_high"])),
                    str(values["n"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "\\caption{Item-level consensus "
            + _latex_escape(source_name.lower())
            + r" label rates with Wilson confidence intervals.}",
            r"\label{tab:human-audit-summary}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_deltas_latex(metrics: dict[str, Any]) -> str:
    source_name = _source_display_name(metrics)
    deltas = metrics.get("baseline_policy_deltas", {})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Comparison & Treatment $-$ baseline & $n$ \\",
        r"\midrule",
    ]
    if not deltas:
        lines.append(
            "No paired "
            + _latex_escape(source_name.lower())
            + r" deltas available &  & 0 \\"
        )
    for key, values in sorted(deltas.items()):
        lines.append(
            " & ".join(
                [
                    _latex_escape(key),
                    _format_float(values.get("treatment_minus_baseline")),
                    str(values.get("n", 0)),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "\\caption{Paired "
            + _latex_escape(source_name.lower())
            + r" treatment-minus-baseline deltas by suite, policy, and label.}",
            r"\label{tab:human-audit-deltas}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def _source_display_name(metrics: dict[str, Any]) -> str:
    source_type = str(metrics.get("annotation_source_type") or "human")
    if source_type == "open_local_judge":
        return "Open Local Judge Audit"
    if source_type == "mixed":
        return "Mixed-Source Audit"
    return "Human Audit"


def _format_float(value: float | int | None) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _format_ci(low: float | int | None, high: float | int | None) -> str:
    if low is None or high is None:
        return ""
    return f"[{_format_float(low)}, {_format_float(high)}]"


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
