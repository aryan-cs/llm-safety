from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.utils.io import file_sha256

REQUIRED_FIGURE_DATA_COLUMNS = {
    "selective_safety_erasure_heatmap": {
        "suite",
        "policy",
        "safety_degradation",
        "capability_degradation",
    },
    "safety_capability_phase_portrait": {
        "suite",
        "policy",
        "policy_family",
        "safety_degradation",
        "capability_degradation",
        "selective_safety_erasure_index",
    },
    "prompt_effect_constellation": {
        "suite",
        "prompt_id",
        "policy",
        "x",
        "y",
        "effect_magnitude",
    },
    "cache_state_fingerprint": {
        "policy",
        "role",
        "token_bin",
        "retention_fraction",
    },
    "safety_state_atlas": {
        "suite",
        "policy",
        "selective_safety_erasure_index",
        "system_retention_fraction",
        "user_retention_fraction",
    },
    "causal_restoration_fraction": {
        "suite",
        "policy",
        "compressed_policy",
        "safety_restoration_fraction",
    },
    "causal_restoration_flow": {
        "suite",
        "policy",
        "compressed_policy",
        "safety_restoration_fraction",
        "label",
    },
}
FIGURE_DATA_COLUMN_ALIASES = {
    "selective_safety_erasure_heatmap": [
        {"selective_safety_erasure_index", "index"},
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a result directory is paper-ready.")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--paper-dir", type=Path, default=None)
    parser.add_argument("--min-prompts-per-suite", type=int, default=100)
    parser.add_argument(
        "--suite-min-prompts",
        action="append",
        default=[],
        help="Optional per-suite threshold override, e.g. system_leakage=2.",
    )
    parser.add_argument("--max-ci-width", type=float, default=0.08)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-mock-model", action="store_true")
    parser.add_argument("--allow-tiny-model", action="store_true")
    parser.add_argument("--allow-smoke-run", action="store_true")
    parser.add_argument("--min-policies", type=int, default=3)
    parser.add_argument("--required-suite", action="append", default=[])
    parser.add_argument("--required-policy", action="append", default=[])
    parser.add_argument("--required-figure", action="append", default=[])
    parser.add_argument("--require-public-provenance", action="store_true")
    parser.add_argument("--require-causal-patch", action="store_true")
    parser.add_argument("--require-policy-pinned", action="store_true")
    parser.add_argument(
        "--allow-inactive-compression",
        action="store_true",
        help="Allow policies whose cache stats show no eviction or quantization activity.",
    )
    args = parser.parse_args()
    suite_min_prompts = _parse_suite_min_prompts(args.suite_min_prompts)

    failures: list[str] = []
    generations = args.results_dir / "generations.jsonl"
    metrics_path = args.results_dir / "metrics.json"
    manifest_path = args.results_dir / "manifest.json"
    prompts_path = args.results_dir / "prompts.jsonl"
    for required in [
        "config.resolved.yaml",
        "environment.json",
        "manifest.json",
        "prompts.jsonl",
        "generations.jsonl",
        "metrics.json",
        "cache_stats.parquet",
    ]:
        if not (args.results_dir / required).exists():
            failures.append(f"missing artifact: {required}")

    manifest = {}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("git_dirty") and not args.allow_dirty:
            failures.append("run was produced from a dirty git working tree")
        if manifest.get("model_provider") == "mock" and not args.allow_mock_model:
            failures.append("mock model runs are not paper evidence")
        model_id = str(manifest.get("model_id", ""))
        if "tiny" in model_id.lower() and not args.allow_tiny_model:
            failures.append(f"tiny model `{model_id}` is not paper evidence")
        run_name = str(manifest.get("run_name", ""))
        if "smoke" in run_name.lower() and not args.allow_smoke_run:
            failures.append(f"smoke run `{run_name}` is not paper evidence")
        if not manifest.get("cache_policy_configs"):
            failures.append("manifest lacks full cache policy configs")
        if not manifest.get("cache_policy_labels"):
            failures.append("manifest lacks cache policy labels")
        if manifest.get("expected_generation_count") is None:
            failures.append("manifest lacks expected generation count")
        if not manifest.get("prompt_counts"):
            failures.append("manifest lacks prompt counts")
        prompt_suite_manifests = manifest.get("prompt_suite_manifests") or {}
        if args.require_public_provenance:
            for suite in manifest.get("prompt_suites", []):
                if str(suite).startswith("public_"):
                    suite_manifest = prompt_suite_manifests.get(suite)
                    if not suite_manifest:
                        failures.append(f"missing processed suite manifest for `{suite}`")
                    elif not suite_manifest.get("sha256") or not suite_manifest.get("record_count"):
                        failures.append(f"processed suite manifest for `{suite}` lacks hash/count")
        policy_configs = manifest.get("cache_policy_configs") or []
        if len(policy_configs) < args.min_policies:
            failures.append(f"manifest has {len(policy_configs)} policies; need >= {args.min_policies}")
        policy_names = {str(policy.get("name")) for policy in policy_configs if isinstance(policy, dict)}
        for required_policy in args.required_policy:
            if required_policy not in policy_names and not any(
                str(policy.get("name", "")).startswith(required_policy)
                for policy in policy_configs
                if isinstance(policy, dict)
            ):
                failures.append(f"missing required policy `{required_policy}`")
        if args.require_policy_pinned and "policy_pinned" not in policy_names:
            failures.append("missing policy_pinned mitigation policy")
        if args.require_causal_patch:
            _check_causal_patch_config(policy_configs, failures)
        prompt_counts = manifest.get("prompt_counts") or {}
        for required_suite in args.required_suite:
            if required_suite not in prompt_counts:
                failures.append(f"missing required suite `{required_suite}`")

    prompt_rows: list[dict] = []
    if prompts_path.exists():
        token_span_failures = 0
        public_without_provenance = 0
        public_without_precise_provenance = 0
        with prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_rows.append(row)
                rendered = row.get("rendered_prompt", {})
                if manifest.get("model_provider") != "mock" and (
                    rendered.get("token_count") is None or not rendered.get("token_role_spans")
                ):
                    token_span_failures += 1
                if args.require_public_provenance and str(row.get("suite", "")).startswith("public_"):
                    metadata = row.get("metadata", {})
                    if not metadata.get("source_dataset") or not metadata.get("source_split"):
                        public_without_provenance += 1
                    if _public_prompt_precise_provenance_failure(row, metadata):
                        public_without_precise_provenance += 1
        if token_span_failures:
            failures.append(f"{token_span_failures} prompts lack tokenizer token-role spans")
        if public_without_provenance:
            failures.append(f"{public_without_provenance} public prompts lack dataset provenance")
        if public_without_precise_provenance:
            failures.append(
                f"{public_without_precise_provenance} public prompts lack precise dataset provenance"
            )

    figure_dir = args.results_dir / "figures"
    if not figure_dir.exists() or not list(figure_dir.glob("*.png")):
        failures.append("missing generated PNG figures")
    _check_figure_manifest(
        figure_dir,
        args.results_dir,
        failures,
        args.require_causal_patch,
        required_figures=args.required_figure,
    )
    if not args.allow_inactive_compression and (args.results_dir / "cache_stats.parquet").exists():
        _check_active_compression(args.results_dir / "cache_stats.parquet", manifest, failures)
    if args.require_causal_patch and (args.results_dir / "cache_stats.parquet").exists():
        _check_causal_patch_cache_stats(args.results_dir / "cache_stats.parquet", manifest, failures)

    generation_rows: list[dict] = []
    if generations.exists():
        counts: dict[str, set[str]] = {}
        with generations.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                generation_rows.append(row)
                counts.setdefault(row["suite"], set()).add(row["prompt_id"])
        for suite, prompt_ids in counts.items():
            required_count = suite_min_prompts.get(suite, args.min_prompts_per_suite)
            if len(prompt_ids) < required_count:
                failures.append(
                    f"suite `{suite}` has {len(prompt_ids)} prompts; need >= {required_count}"
                )
        if manifest:
            _check_generation_matrix(manifest, prompt_rows, generation_rows, failures)

    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        policy_summary = metrics.get("publication_summary", {}).get("policies", {})
        has_global_safety = any(
            value.get("mean_safety_score") is not None for value in policy_summary.values()
        )
        has_global_capability = any(
            value.get("mean_capability_score") is not None for value in policy_summary.values()
        )
        if has_global_safety and has_global_capability:
            contrasts = metrics.get("policy_level_contrasts", {})
            if not contrasts:
                failures.append("missing policy-level safety-vs-capability contrasts")
            for policy, contrast in contrasts.items():
                ssei_ci = contrast.get("selective_safety_erasure_index_ci", {})
                if policy != "none" and ssei_ci.get("mean") is None:
                    failures.append(f"{policy}: missing policy-level SSEI CI")
        if args.require_causal_patch and not metrics.get("causal_restoration"):
            failures.append("missing causal restoration metrics")
        if args.require_causal_patch and metrics.get("causal_restoration"):
            _check_causal_restoration_metric_readiness(metrics, failures)
        for key, value in metrics.get("selective_safety_erasure", {}).items():
            ci = value.get("paired_safety_degradation_ci", {})
            if ci.get("ci_low") is None or ci.get("ci_high") is None:
                failures.append(f"{key}: missing paired safety CI")
                continue
            width = ci["ci_high"] - ci["ci_low"]
            if width > args.max_ci_width:
                failures.append(
                    f"{key}: paired safety CI width {width:.3f}; target <= {args.max_ci_width:.3f}"
                )
    if args.paper_dir is not None:
        _check_paper_assets(args.paper_dir, args.results_dir, failures)

    if failures:
        print("NOT PAPER READY")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("PAPER READY CHECK PASSED")


def _parse_suite_min_prompts(values: list[str]) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Expected --suite-min-prompts value like suite=100, got `{value}`")
        suite, raw_count = value.split("=", 1)
        parsed[suite] = int(raw_count)
    return parsed


def _check_figure_manifest(
    figure_dir: Path,
    results_dir: Path,
    failures: list[str],
    require_causal_patch: bool,
    required_figures: list[str] | None = None,
) -> None:
    manifest_path = figure_dir / "manifest.json"
    if not manifest_path.exists():
        failures.append("missing figures/manifest.json")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"invalid figures/manifest.json: {exc}")
        return
    figures = manifest.get("figures")
    if not isinstance(figures, list) or not figures:
        failures.append("figures manifest has no figure entries")
        return
    source_artifacts = manifest.get("source_artifacts") or {}
    for name in ["manifest.json", "generations.jsonl", "metrics.json", "cache_stats.parquet"]:
        source = source_artifacts.get(name)
        if not isinstance(source, dict):
            failures.append(f"figures manifest lacks source artifact `{name}`")
            continue
        expected_sha = source.get("sha256")
        actual_sha = file_sha256(results_dir / name)
        if expected_sha != actual_sha:
            failures.append(f"figures manifest source hash mismatch for `{name}`")

    figure_names = set()
    for figure in figures:
        if not isinstance(figure, dict):
            failures.append("figures manifest contains non-object entry")
            continue
        name = figure.get("name", "<unnamed>")
        figure_names.add(str(name))
        for key in [
            "png",
            "png_sha256",
            "svg",
            "svg_sha256",
            "pdf",
            "pdf_sha256",
            "data_csv",
            "data_csv_sha256",
        ]:
            if not figure.get(key):
                failures.append(f"figure `{name}` lacks `{key}`")
        for path_key, hash_key in [
            ("png", "png_sha256"),
            ("svg", "svg_sha256"),
            ("pdf", "pdf_sha256"),
            ("data_csv", "data_csv_sha256"),
        ]:
            raw_path = figure.get(path_key)
            if not raw_path:
                continue
            path = Path(str(raw_path))
            if not path.exists():
                failures.append(f"figure `{name}` references missing {path_key}: {raw_path}")
                continue
            if figure.get(hash_key) != file_sha256(path):
                failures.append(f"figure `{name}` has stale {path_key} hash")
            artifact_failure = _figure_artifact_failure(path_key, path)
            if artifact_failure:
                failures.append(f"figure `{name}` has invalid {path_key}: {artifact_failure}")
        if figure.get("data_csv"):
            row_count_failure = _figure_data_row_count_failure(figure)
            if row_count_failure:
                failures.append(f"figure `{name}` has invalid data_csv: {row_count_failure}")
    if require_causal_patch and "causal_restoration_fraction" not in figure_names:
        failures.append("causal patch runs require causal_restoration_fraction figure")
    for required_figure in required_figures or []:
        if required_figure not in figure_names:
            failures.append(f"missing required figure `{required_figure}`")


def _figure_artifact_failure(path_key: str, path: Path) -> str:
    try:
        content = path.read_bytes()
    except OSError as exc:
        return str(exc)
    prefix = content[:4096]
    if path_key == "png":
        if not prefix.startswith(b"\x89PNG\r\n\x1a\n"):
            return "missing PNG signature"
        visual_failure = _png_visual_failure(path)
        if visual_failure:
            return visual_failure
    elif path_key == "pdf":
        if not prefix.startswith(b"%PDF-"):
            return "missing PDF signature"
        if len(content) < 32:
            return "PDF too small"
        if b"%%EOF" not in content[-2048:]:
            return "missing PDF EOF marker"
        pdf_failure = _pdf_page_failure(path)
        if pdf_failure:
            return pdf_failure
    elif path_key == "svg":
        lowered = prefix.lstrip().lower()
        if b"<svg" not in lowered:
            return "missing SVG root"
    elif path_key == "data_csv":
        first_line = prefix.splitlines()[0].strip() if prefix.splitlines() else b""
        if not first_line:
            return "missing CSV header"
    return ""


def _figure_data_row_count_failure(figure: dict) -> str:
    raw_path = figure.get("data_csv")
    if not raw_path:
        return ""
    path = Path(str(raw_path))
    if not path.exists():
        return ""
    try:
        nonempty_lines = [
            line for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line
        ]
    except OSError as exc:
        return str(exc)
    observed_rows = max(0, len(nonempty_lines) - 1)
    raw_declared = figure.get("data_row_count")
    if raw_declared is None:
        return "missing data_row_count"
    try:
        declared_rows = int(raw_declared)
    except (TypeError, ValueError):
        return f"invalid data_row_count={raw_declared!r}"
    if declared_rows <= 0:
        return "data_row_count must be positive"
    if observed_rows != declared_rows:
        return f"data_row_count={declared_rows}; observed_rows={observed_rows}"
    schema_failure = _figure_data_schema_failure(figure, path)
    if schema_failure:
        return schema_failure
    return ""


def _png_visual_failure(path: Path) -> str:
    try:
        import matplotlib.image as mpimg
        import numpy as np

        image = mpimg.imread(path)
    except Exception as exc:
        return f"PNG visual decode failed: {exc}"
    if image.ndim < 2 or image.shape[0] < 64 or image.shape[1] < 64:
        return f"PNG dimensions too small: {tuple(image.shape)}"
    array = np.asarray(image)
    if array.size == 0:
        return "PNG has no pixels"
    channels = array[..., :3] if array.ndim == 3 and array.shape[-1] >= 3 else array
    if float(np.nanstd(channels)) < 1e-4:
        return "PNG appears visually blank"
    return ""


def _pdf_page_failure(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
    except Exception as exc:
        return f"PDF parse failed: {exc}"
    if not reader.pages:
        return "PDF has no pages"
    for index, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        width = float(box.width)
        height = float(box.height)
        if width <= 32 or height <= 32:
            return f"PDF page {index} dimensions too small: {width}x{height}"
    return ""


def _figure_data_schema_failure(figure: dict, path: Path) -> str:
    required_columns = REQUIRED_FIGURE_DATA_COLUMNS.get(str(figure.get("name")))
    if not required_columns:
        return ""
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except OSError as exc:
        return str(exc)
    observed = {str(column) for column in header}
    missing = sorted(required_columns - observed)
    for aliases in FIGURE_DATA_COLUMN_ALIASES.get(str(figure.get("name")), []):
        if observed.isdisjoint(aliases):
            missing.append("/".join(sorted(aliases)))
    if missing:
        return f"missing required columns: {', '.join(missing)}"
    return ""


def _public_prompt_precise_provenance_failure(row: dict, metadata: dict) -> bool:
    required_value_fields = [
        "source_dataset",
        "source_split",
        "source_revision",
        "source_fingerprint",
        "source_version",
    ]
    if any(not metadata.get(field) for field in required_value_fields):
        return True
    required_present_fields = [
        "source_config",
        "source_config_name",
        "source_homepage",
        "source_license",
    ]
    if any(field not in metadata for field in required_present_fields):
        return True
    source_locator_fields = [
        "source_id",
        "source_row_index",
        "source_group_id",
        "xstest_task_id",
    ]
    if any(metadata.get(field) not in {None, ""} for field in source_locator_fields):
        return False
    return not re.search(r"_\d{6}$", str(row.get("prompt_id") or ""))


def _check_paper_assets(paper_dir: Path, results_dir: Path, failures: list[str]) -> None:
    manifest_path = paper_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        failures.append(f"missing paper artifact manifest: {manifest_path}")
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"invalid paper artifact manifest: {exc}")
        return
    for name, table in (manifest.get("tables") or {}).items():
        if not isinstance(table, dict):
            failures.append(f"paper artifact table entry `{name}` is malformed")
            continue
        path = Path(str(table.get("path", "")))
        if not path.exists():
            failures.append(f"paper artifact table `{name}` is missing")
            continue
        if table.get("sha256") != file_sha256(path):
            failures.append(f"paper artifact table `{name}` hash is stale")
    for name in ["manifest.json", "metrics.json", "figures/manifest.json"]:
        source = (manifest.get("source_artifacts") or {}).get(name)
        if not isinstance(source, dict):
            failures.append(f"paper artifact manifest lacks source `{name}`")
            continue
        source_path = results_dir / name
        if not source_path.exists():
            failures.append(f"paper artifact source `{name}` is missing")
            continue
        if source.get("sha256") != file_sha256(source_path):
            failures.append(f"paper artifact source `{name}` hash is stale")


def _check_causal_patch_config(policy_configs: list[dict], failures: list[str]) -> None:
    patch_specs = [
        policy.get("patch_from_baseline")
        for policy in policy_configs
        if isinstance(policy, dict) and policy.get("patch_from_baseline")
    ]
    if not patch_specs:
        failures.append("missing cache patch policy with patch_from_baseline")
        return
    role_patches = [patch for patch in patch_specs if _patch_roles(patch)]
    if not role_patches:
        failures.append("causal patch policies must use role-derived token selection")
    system_patches = [patch for patch in role_patches if "system" in _patch_roles(patch)]
    if not system_patches:
        failures.append("missing system-role cache patch")
    matched_controls = [
        patch
        for patch in role_patches
        if "user" in _patch_roles(patch)
        and "system" in _patch_match_roles(patch)
    ]
    if not matched_controls:
        failures.append("missing matched user-role cache patch control")


def _check_causal_restoration_metric_readiness(
    metrics: dict, failures: list[str]
) -> None:
    grouped: dict[tuple[str, str, str], set[str]] = {}
    for key, values in metrics.get("causal_restoration", {}).items():
        if "::" not in key:
            failures.append(f"malformed causal restoration key `{key}`")
            continue
        suite, policy = key.split("::", 1)
        compressed_policy = str(values.get("compressed_policy") or "")
        role = _patch_role_class_from_label(policy)
        if role is None or not compressed_policy:
            continue
        for metric in [
            "safety_restoration_fraction",
            "refusal_restoration_fraction",
            "leakage_avoidance_restoration_fraction",
        ]:
            if values.get(metric) is None:
                continue
            ci = values.get(f"{metric}_ci") or {}
            if ci.get("ci_low") is None or ci.get("ci_high") is None:
                failures.append(f"{key}: missing `{metric}_ci` interval")
                continue
            grouped.setdefault((suite, compressed_policy, metric), set()).add(role)
    if not any({"system", "user_control"}.issubset(roles) for roles in grouped.values()):
        failures.append(
            "causal restoration metrics lack same-endpoint system patch and matched user control intervals"
        )


def _patch_role_class_from_label(policy: str) -> str | None:
    patch_part = policy.split("__patch", 1)[1] if "__patch" in policy else policy
    normalized = "".join(char for char in patch_part.lower() if char.isalnum())
    has_user = "roleuser" in normalized or "tokenroleuser" in normalized
    has_system = "rolesystem" in normalized or "tokenrolesystem" in normalized
    matched_system = "matchsystem" in normalized or "matchedsystem" in normalized
    if has_user and matched_system:
        return "user_control"
    if has_system and not has_user:
        return "system"
    return None


def _patch_roles(patch: dict) -> set[str]:
    return set(_as_string_list(patch.get("token_roles") or patch.get("roles") or patch.get("role")))


def _patch_match_roles(patch: dict) -> set[str]:
    return set(
        _as_string_list(
            patch.get("match_token_count_to_roles")
            or patch.get("matched_token_roles")
            or patch.get("match_roles")
        )
    )


def _as_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _check_generation_matrix(
    manifest: dict,
    prompt_rows: list[dict],
    generation_rows: list[dict],
    failures: list[str],
) -> None:
    expected_count = manifest.get("expected_generation_count")
    if expected_count is not None and len(generation_rows) != int(expected_count):
        failures.append(
            f"generation row count is {len(generation_rows)}; expected {int(expected_count)}"
        )

    policy_labels = [str(label) for label in manifest.get("cache_policy_labels", [])]
    seeds = [int(seed) for seed in manifest.get("seeds", [])]
    if not prompt_rows or not policy_labels or not seeds:
        return

    expected_keys = {
        (str(prompt["suite"]), str(prompt["prompt_id"]), policy, seed)
        for prompt in prompt_rows
        for policy in policy_labels
        for seed in seeds
    }
    observed_keys = []
    malformed = 0
    for row in generation_rows:
        try:
            seed = int(row["seed"])
            observed_keys.append(
                (str(row["suite"]), str(row["prompt_id"]), str(row["policy"]), seed)
            )
        except (KeyError, TypeError, ValueError):
            malformed += 1
    if malformed:
        failures.append(f"{malformed} generation rows have malformed matrix keys")
    observed_key_set = set(observed_keys)
    duplicate_count = len(observed_keys) - len(observed_key_set)
    if duplicate_count:
        failures.append(f"generation matrix has {duplicate_count} duplicate rows")
    missing = expected_keys - observed_key_set
    extra = observed_key_set - expected_keys
    if missing:
        failures.append(
            f"generation matrix is missing {len(missing)} rows; "
            f"first missing: {_format_matrix_key(sorted(missing)[0])}"
        )
    if extra:
        failures.append(
            f"generation matrix has {len(extra)} rows outside the manifest; "
            f"first extra: {_format_matrix_key(sorted(extra)[0])}"
        )


def _format_matrix_key(key: tuple[str, str, str, int]) -> str:
    suite, prompt_id, policy, seed = key
    return f"suite={suite}, prompt_id={prompt_id}, policy={policy}, seed={seed}"


def _check_active_compression(cache_stats_path: Path, manifest: dict, failures: list[str]) -> None:
    expected_policies = [
        str(policy)
        for policy in manifest.get("cache_policy_labels", [])
        if str(policy) != "none"
    ]
    if not expected_policies:
        return
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        failures.append("pyarrow is required to validate active compression")
        return
    try:
        parquet_file = pq.ParquetFile(cache_stats_path)
    except Exception as exc:
        failures.append(f"cannot inspect cache stats for active compression: {exc}")
        return
    available_columns = set(parquet_file.schema.names)
    required_columns = {
        "policy",
        "decode_step",
        "original_seq_len",
        "evicted_count",
        "retained_system_tokens",
        "evicted_system_tokens",
        "protected_candidate_count",
        "protected_retained_count",
        "protected_dropped_count",
        "quantization_bits",
        "cache_l2_before",
        "cache_l2_after",
    }
    columns = [column for column in required_columns if column in available_columns]
    if "policy" not in columns:
        failures.append("cache stats lack policy column for active-compression check")
        return
    stats: dict[str, dict[str, float]] = {
        policy: {
            "rows": 0.0,
            "evicted": 0.0,
            "quantized": 0.0,
            "l2_delta": 0.0,
            "pre_response_evicted": 0.0,
            "pre_response_quantized": 0.0,
            "pre_response_system_touched": 0.0,
        }
        for policy in expected_policies
    }
    for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
        table = batch.to_pydict()
        policies = table.get("policy", [])
        for idx, raw_policy in enumerate(policies):
            policy = str(raw_policy)
            if policy not in stats:
                continue
            stats[policy]["rows"] += 1
            original_seq_len = _float_at(table, "original_seq_len", idx)
            decode_step = _float_at(table, "decode_step", idx)
            evicted_count = _float_at(table, "evicted_count", idx)
            retained_system = _float_at(table, "retained_system_tokens", idx)
            evicted_system = _float_at(table, "evicted_system_tokens", idx)
            protected_candidate_count = _float_at(table, "protected_candidate_count", idx)
            protected_retained_count = _float_at(table, "protected_retained_count", idx)
            protected_dropped_count = _float_at(table, "protected_dropped_count", idx)
            quantization_bits = table.get("quantization_bits", [None] * len(policies))[idx]
            before = _float_at(table, "cache_l2_before", idx)
            after = _float_at(table, "cache_l2_after", idx)
            stats[policy]["evicted"] += evicted_count
            if quantization_bits is not None and original_seq_len > 0:
                stats[policy]["quantized"] += 1
            if decode_step <= 1:
                stats[policy]["pre_response_evicted"] += evicted_count
                if quantization_bits is not None and original_seq_len > 0:
                    stats[policy]["pre_response_quantized"] += 1
                    if retained_system > 0:
                        stats[policy]["pre_response_system_touched"] += 1
                if evicted_system > 0:
                    stats[policy]["pre_response_system_touched"] += evicted_system
                if (
                    policy.startswith("policy_pinned")
                    and protected_candidate_count > 0
                    and (protected_retained_count > 0 or protected_dropped_count > 0)
                ):
                    stats[policy]["pre_response_system_touched"] += max(
                        protected_retained_count, protected_dropped_count
                    )
            stats[policy]["l2_delta"] += abs(before - after)
    for policy, policy_stats in stats.items():
        if policy_stats["rows"] == 0:
            failures.append(f"cache policy `{policy}` has no cache-stat rows")
            continue
        if policy_stats["evicted"] <= 0 and policy_stats["quantized"] <= 0:
            failures.append(
                f"cache policy `{policy}` appears inactive: no evictions or quantization rows"
            )
        if (
            policy_stats["pre_response_evicted"] <= 0
            and policy_stats["pre_response_quantized"] <= 0
        ):
            failures.append(
                f"cache policy `{policy}` appears inactive before first generated token"
            )
        if policy_stats["pre_response_system_touched"] <= 0:
            failures.append(
                f"cache policy `{policy}` never touches system-role tokens before generation"
            )


def _check_causal_patch_cache_stats(
    cache_stats_path: Path, manifest: dict, failures: list[str]
) -> None:
    expected_patch_policies = [
        str(policy)
        for policy in manifest.get("cache_policy_labels", [])
        if "__patch" in str(policy)
    ]
    if not expected_patch_policies:
        return
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        failures.append("pyarrow is required to validate causal patch metadata")
        return
    parquet_file = pq.ParquetFile(cache_stats_path)
    columns = [
        column
        for column in [
            "policy",
            "patched_from_baseline",
            "patched_token_count",
            "patched_roles",
            "patch_matched_roles",
        ]
        if column in set(parquet_file.schema.names)
    ]
    if "policy" not in columns:
        failures.append("cache stats lack policy column for causal patch check")
        return
    stats: dict[str, dict[str, object]] = {
        policy: {"rows": 0, "patched_rows": 0, "max_tokens": 0.0, "roles": set(), "match_roles": set()}
        for policy in expected_patch_policies
    }
    for batch in parquet_file.iter_batches(columns=columns, batch_size=100_000):
        table = batch.to_pydict()
        policies = table.get("policy", [])
        for idx, raw_policy in enumerate(policies):
            policy = str(raw_policy)
            if policy not in stats:
                continue
            policy_stats = stats[policy]
            policy_stats["rows"] = int(policy_stats["rows"]) + 1
            patched = _truthy_at(table, "patched_from_baseline", idx)
            if patched:
                policy_stats["patched_rows"] = int(policy_stats["patched_rows"]) + 1
            policy_stats["max_tokens"] = max(
                float(policy_stats["max_tokens"]),
                _float_at(table, "patched_token_count", idx),
            )
            _update_role_set(policy_stats["roles"], table, "patched_roles", idx)
            _update_role_set(policy_stats["match_roles"], table, "patch_matched_roles", idx)
    for policy, policy_stats in stats.items():
        if int(policy_stats["rows"]) == 0:
            failures.append(f"patch policy `{policy}` has no cache-stat rows")
            continue
        if int(policy_stats["patched_rows"]) == 0:
            failures.append(f"patch policy `{policy}` has no patched cache-stat rows")
        if float(policy_stats["max_tokens"]) <= 0:
            failures.append(f"patch policy `{policy}` patched zero tokens")
    if not any("system" in policy_stats["roles"] for policy_stats in stats.values()):
        failures.append("cache stats show no system-role patch rows")
    if not any(
        "user" in policy_stats["roles"] and "system" in policy_stats["match_roles"]
        for policy_stats in stats.values()
    ):
        failures.append("cache stats show no matched user-role patch control rows")


def _float_at(table: dict[str, list], column: str, idx: int) -> float:
    values = table.get(column)
    if values is None:
        return 0.0
    value = values[idx]
    if value is None:
        return 0.0
    return float(value)


def _truthy_at(table: dict[str, list], column: str, idx: int) -> bool:
    values = table.get(column)
    if values is None:
        return False
    value = values[idx]
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes"}


def _update_role_set(target: object, table: dict[str, list], column: str, idx: int) -> None:
    if not isinstance(target, set):
        return
    values = table.get(column)
    if values is None:
        return
    value = values[idx]
    if value is None:
        return
    for role in str(value).split(","):
        if role:
            target.add(role)


if __name__ == "__main__":
    main()
