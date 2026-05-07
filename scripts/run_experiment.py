from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from _path import add_src_to_path

add_src_to_path()

from cache_safety_erasure.cache_policies.registry import build_cache_policy, cache_policy_label
from cache_safety_erasure.config import dump_yaml, parse_experiment_config
from cache_safety_erasure.evals.io import load_prompt_suite, load_prompt_suite_manifest
from cache_safety_erasure.evals.rendering import rendered_prompt_manifest
from cache_safety_erasure.evals.spans import character_span_manifest
from cache_safety_erasure.generation.runner import generate_one
from cache_safety_erasure.metrics.aggregate import compute_example_metrics, compute_run_metrics
from cache_safety_erasure.models.loader import hf_device_map, load_model
from cache_safety_erasure.utils.io import (
    append_jsonl,
    environment_snapshot,
    make_run_dir,
    read_jsonl,
    read_jsonl_tolerant,
    utc_timestamp,
    write_json,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cache safety erasure experiments.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-id", help="Override run.run_id without editing the config file.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the target run directory and skip completed prompt/policy/seed rows.",
    )
    args = parser.parse_args()

    config, raw_config = parse_experiment_config(args.config)
    if args.run_id or args.resume:
        config = replace(
            config,
            run=replace(
                config.run,
                run_id=args.run_id or config.run.run_id,
                resume=config.run.resume or args.resume,
            ),
        )
        raw_config = _raw_config_with_run_overrides(raw_config, args.run_id, args.resume)
    run_dir = make_run_dir(
        config.run.output_dir, config.run.name, config.run.run_id, config.run.resume
    )
    env = environment_snapshot()
    generations_path = run_dir / "generations.jsonl"
    cache_stats_jsonl_path = run_dir / "cache_stats.jsonl"
    if config.run.resume:
        existing, corrupt_tail_path = read_jsonl_tolerant(generations_path)
        if corrupt_tail_path is not None:
            print(
                "Resume recovery quarantined a corrupt generations.jsonl tail at "
                f"{corrupt_tail_path}."
            )
    else:
        existing = []

    suite_prompts = {}
    prompt_counts = {}
    prompt_suite_manifests = {}
    for suite in config.prompt_suites:
        prompts = load_prompt_suite(suite)
        if config.limit_per_suite is not None:
            prompts = prompts[: config.limit_per_suite]
        suite_prompts[suite] = prompts
        prompt_counts[suite] = len(prompts)
        prompt_suite_manifests[suite] = load_prompt_suite_manifest(suite)
    policy_labels = [cache_policy_label(policy) for policy in config.cache_policies]
    expected_generation_count = sum(prompt_counts.values()) * len(config.seeds) * len(policy_labels)

    manifest_base = {
        "run_name": config.run.name,
        "model_id": config.model.model_id,
        "model_family": config.model.family,
        "model_track": config.model.track,
        "model_provider": config.model.provider,
        "backend_name": _backend_name(config.model.provider),
        "backend_version": _backend_version(config.model.provider),
        "launch_command": _launch_command(args),
        "model_config": asdict(config.model),
        "model_device_map": None,
        "model_context_length": config.model.context_length,
        "tokenizer_source": config.model.model_id,
        "chat_template_required": config.model.chat_template_required,
        "cache_policy_support": _cache_policy_support_manifest(config.cache_policies),
        "repository_url": (env.get("project") or {}).get("repository_url"),
        "repository_git_url": (env.get("project") or {}).get("repository_git_url"),
        "git_remote_origin": env.get("git_remote_origin"),
        "git_commit": env.get("git_commit"),
        "git_dirty": env.get("git_dirty"),
        "config_sha256": _stable_hash(raw_config),
        "resume_compatible_config_sha256": _stable_hash(_resume_compatible_config(raw_config)),
        "prompt_suites": list(config.prompt_suites),
        "prompt_counts": prompt_counts,
        "prompt_suite_manifests": prompt_suite_manifests,
        "cache_policy_configs": [_policy_manifest(policy) for policy in config.cache_policies],
        "cache_policy_labels": policy_labels,
        "seeds": list(config.seeds),
        "limit_per_suite": config.limit_per_suite,
        "expected_generation_count": expected_generation_count,
    }
    if config.run.resume:
        _validate_resume_manifest(run_dir, manifest_base, env)
        existing = _reconcile_resume_generations(run_dir, existing)
    done_keys = {
        (row["prompt_id"], row["suite"], row["policy"], int(row["seed"])) for row in existing
    }
    _write_progress(
        run_dir,
        completed=len(done_keys),
        expected=expected_generation_count,
        activity="loading_model",
        current=None,
    )

    model_bundle = load_model(config.model)
    device_map = (
        hf_device_map(model_bundle.model)
        if getattr(model_bundle, "model", None) is not None
        else None
    )
    manifest = {
        **manifest_base,
        "model_device_map": device_map,
        "tokenizer_source": _tokenizer_source(getattr(model_bundle, "tokenizer", None), config),
        "chat_template_source_sha256": _chat_template_sha256(
            getattr(model_bundle, "tokenizer", None)
        ),
    }
    _write_run_start_artifacts(run_dir, raw_config, env, manifest, resume=config.run.resume)

    cache_stat_rows: list[dict] = []
    cache_stats_sink = _CacheStatsSink(run_dir / "cache_stats.parquet", resume=config.run.resume)
    prompt_manifest_rows: list[dict] = []

    for _suite, prompts in suite_prompts.items():
        prompt_manifest_rows.extend(
            {
                "prompt_id": prompt.id,
                "suite": prompt.suite,
                "category": prompt.category,
                "system": prompt.system,
                "user": prompt.user,
                "should_refuse": prompt.should_refuse,
                "expected_answer": prompt.expected_answer,
                "choices": prompt.choices,
                "hidden_system": prompt.hidden_system,
                "prompt_sha256": _stable_hash(prompt.to_dict()),
                "character_spans": character_span_manifest(prompt),
                "rendered_prompt": rendered_prompt_manifest(
                    getattr(model_bundle, "tokenizer", None), prompt
                ),
                "metadata": prompt.metadata,
            }
            for prompt in prompts
        )
        for seed in config.seeds:
            for policy_config in config.cache_policies:
                policy = build_cache_policy(policy_config, seed)
                policy_name = getattr(policy, "name", policy_config.name)
                for prompt in prompts:
                    key = (prompt.id, prompt.suite, policy_name, seed)
                    if key in done_keys:
                        continue
                    result = generate_one(
                        model_bundle=model_bundle,
                        prompt=prompt,
                        policy=policy,
                        generation_config=config.generation,
                        patch_from_baseline=policy_config.patch_from_baseline,
                    )
                    metrics = compute_example_metrics(prompt, result.text)
                    row = {
                        "prompt_id": prompt.id,
                        "suite": prompt.suite,
                        "category": prompt.category,
                        "policy": policy_name,
                        "seed": seed,
                        "model_id": config.model.model_id,
                        "model_family": config.model.family,
                        "model_track": config.model.track,
                        "generated_at": utc_timestamp(),
                        "system": prompt.system,
                        "user": prompt.user,
                        "should_refuse": prompt.should_refuse,
                        "expected_answer": prompt.expected_answer,
                        "hidden_system": prompt.hidden_system,
                        "prompt_metadata": prompt.metadata,
                        "generated_text": result.text,
                        **metrics,
                    }
                    append_jsonl(generations_path, [row])
                    decision_rows = []
                    for decision in result.cache_decisions:
                        decision_rows.extend(decision.to_rows(prompt.id, seed))
                    if decision_rows:
                        append_jsonl(cache_stats_jsonl_path, decision_rows)
                        cache_stat_rows.extend(decision_rows)
                        if len(cache_stat_rows) >= 10_000:
                            cache_stats_sink.write(cache_stat_rows)
                            cache_stat_rows = []
                    done_keys.add(key)
                    _write_progress(
                        run_dir,
                        completed=len(done_keys),
                        expected=expected_generation_count,
                        activity="generating",
                        current={
                            "suite": prompt.suite,
                            "prompt_id": prompt.id,
                            "policy": policy_name,
                            "seed": seed,
                            "model_id": config.model.model_id,
                        },
                    )

    rows = read_jsonl(generations_path)
    metrics = compute_run_metrics(rows)
    write_jsonl(run_dir / "prompts.jsonl", prompt_manifest_rows)
    write_json(run_dir / "metrics.json", metrics)
    if cache_stat_rows:
        cache_stats_sink.write(cache_stat_rows)
    cache_stats_sink.close()
    _rebuild_cache_stats_parquet_from_jsonl(cache_stats_jsonl_path, cache_stats_sink.path)
    _write_progress(
        run_dir,
        completed=len(rows),
        expected=expected_generation_count,
        activity="complete",
        current={"model_id": config.model.model_id},
    )
    print(f"Completed run: {run_dir}")


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _backend_name(model_provider: str) -> str:
    if model_provider == "hf":
        return "transformers"
    if model_provider == "mock":
        return "mock"
    return model_provider


def _backend_version(model_provider: str) -> str | None:
    if model_provider == "hf":
        try:
            return importlib.metadata.version("transformers")
        except importlib.metadata.PackageNotFoundError:
            return None
    return None


def _launch_command(args: argparse.Namespace) -> list[str]:
    command = [sys.executable, "scripts/run_experiment.py", "--config", str(args.config)]
    if args.run_id:
        command.extend(["--run-id", str(args.run_id)])
    if args.resume:
        command.append("--resume")
    return command


def _tokenizer_source(tokenizer: object | None, config: object) -> str:
    source = getattr(tokenizer, "name_or_path", None)
    if source:
        return str(source)
    return str(getattr(config.model, "model_id", ""))


def _chat_template_sha256(tokenizer: object | None) -> str | None:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return _stable_hash(str(template))


def _cache_policy_support_manifest(policies: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "name": policy.name,
            "label": cache_policy_label(policy),
            "requires_role_spans": policy.name in {"policy_pinned", "user_pinned"}
            or bool(policy.patch_from_baseline),
            "requires_attention_scores": policy.name in {"attention_h2o"},
            "requires_patch_from_baseline": bool(policy.patch_from_baseline),
        }
        for policy in policies
    ]


def _raw_config_with_run_overrides(
    raw_config: dict[str, Any], run_id: str | None, resume: bool
) -> dict[str, Any]:
    updated = json.loads(json.dumps(raw_config, default=str))
    run = updated.setdefault("run", {})
    if run_id:
        run["run_id"] = run_id
    if resume:
        run["resume"] = True
    return updated


def _resume_compatible_config(raw_config: dict[str, Any]) -> dict[str, Any]:
    updated = json.loads(json.dumps(raw_config, default=str))
    run = updated.get("run")
    if isinstance(run, dict):
        run.pop("resume", None)
    return updated


RESUME_MANIFEST_STABLE_KEYS = [
    "run_name",
    "model_id",
    "model_provider",
    "model_config",
    "prompt_suites",
    "prompt_counts",
    "prompt_suite_manifests",
    "cache_policy_configs",
    "cache_policy_labels",
    "seeds",
    "limit_per_suite",
    "expected_generation_count",
]


class CorruptCacheStatsError(RuntimeError):
    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


def _validate_resume_manifest(
    run_dir: Path, planned_manifest: dict[str, Any], env: dict[str, Any]
) -> None:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for key in RESUME_MANIFEST_STABLE_KEYS:
        if existing_manifest.get(key) != planned_manifest.get(key):
            failures.append(f"resume_manifest_mismatch:{key}")
    existing_compat_hash = existing_manifest.get("resume_compatible_config_sha256")
    planned_compat_hash = planned_manifest.get("resume_compatible_config_sha256")
    if existing_compat_hash and planned_compat_hash and existing_compat_hash != planned_compat_hash:
        failures.append("resume_manifest_mismatch:resume_compatible_config_sha256")
    existing_commit = existing_manifest.get("git_commit")
    current_commit = env.get("git_commit")
    allow_commit_mismatch = os.environ.get("ALLOW_RESUME_GIT_MISMATCH") == "1"
    if existing_commit and current_commit and existing_commit != current_commit:
        if allow_commit_mismatch:
            print(
                "WARNING: resuming across git commits because "
                "ALLOW_RESUME_GIT_MISMATCH=1. "
                f"existing={existing_commit} current={current_commit}"
            )
        else:
            failures.append("resume_manifest_mismatch:git_commit")
    if not failures:
        return
    details = "\n".join(f"  - {failure}" for failure in failures)
    raise RuntimeError(
        "Refusing to resume from an incompatible run directory. "
        "Use the original git commit, a new run id, or set "
        "ALLOW_RESUME_GIT_MISMATCH=1 only after verifying the experiment matrix "
        f"is unchanged.\n{details}"
    )


def _write_run_start_artifacts(
    run_dir: Path,
    raw_config: dict[str, Any],
    env: dict[str, Any],
    manifest: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if resume and (run_dir / "manifest.json").exists():
        stamp = utc_timestamp()
        dump_yaml(raw_config, run_dir / f"config.resume.{stamp}.yaml")
        write_json(run_dir / f"environment.resume.{stamp}.json", env)
        write_json(run_dir / f"manifest.resume.{stamp}.json", manifest)
        return
    dump_yaml(raw_config, run_dir / "config.resolved.yaml")
    write_json(run_dir / "environment.json", env)
    write_json(run_dir / "manifest.json", manifest)


def _policy_manifest(policy: Any) -> dict[str, Any]:
    # Keep resume comparisons stable against the JSON form written to manifest.json.
    return json.loads(json.dumps(asdict(policy), sort_keys=True))


def _reconcile_resume_generations(run_dir: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return rows
    cache_stats_path = run_dir / "cache_stats.parquet"
    try:
        cache_keys = _cache_stats_generation_keys(cache_stats_path)
    except CorruptCacheStatsError as exc:
        if os.environ.get("ALLOW_CORRUPT_CACHE_STATS_RESET") != "1":
            raise RuntimeError(f"{cache_stats_path.name} is unreadable: {exc}") from exc
        stamp = utc_timestamp()
        generation_archive = run_dir / f"generations.corrupt_cache_stats_reset.{stamp}.jsonl"
        cache_archive = run_dir / f"{cache_stats_path.name}.corrupt.{stamp}"
        write_jsonl(generation_archive, rows)
        write_jsonl(run_dir / "generations.jsonl", [])
        if exc.path.exists():
            exc.path.replace(cache_archive)
        print(
            "Resume reconciliation reset generations because cache_stats.parquet was unreadable; "
            f"archived generations at {generation_archive} and cache stats at {cache_archive}."
        )
        return []
    kept = [
        row
        for row in rows
        if (
            str(row.get("prompt_id")),
            str(row.get("policy")),
            int(row.get("seed", 0)),
        )
        in cache_keys
    ]
    if len(kept) == len(rows):
        return rows
    orphan_path = run_dir / f"generations.orphaned_without_cache_stats.{utc_timestamp()}.jsonl"
    write_jsonl(orphan_path, rows)
    write_jsonl(run_dir / "generations.jsonl", kept)
    print(
        "Resume reconciliation removed "
        f"{len(rows) - len(kept)} generation row(s) without cache-stat evidence; "
        f"archived original rows at {orphan_path}."
    )
    return kept


def _cache_stats_generation_keys(cache_stats_path: Path) -> set[tuple[str, str, int]]:
    jsonl_path = cache_stats_path.with_suffix(".jsonl")
    if jsonl_path.exists():
        rows, corrupt_tail_path = read_jsonl_tolerant(jsonl_path)
        if corrupt_tail_path is not None:
            print(
                "Resume recovery quarantined a corrupt cache_stats.jsonl tail at "
                f"{corrupt_tail_path}."
            )
        keys = _cache_stats_generation_keys_from_rows(rows)
        if keys:
            return keys
    cache_stats_path = _promote_recoverable_cache_stats_temp(cache_stats_path)
    if not cache_stats_path.exists():
        return set()
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for cache stats resume reconciliation.") from exc
    try:
        table = pq.read_table(cache_stats_path, columns=["prompt_id", "policy", "seed"])
    except Exception as exc:
        raise CorruptCacheStatsError(cache_stats_path, str(exc)) from exc
    prompt_ids = table.column("prompt_id").to_pylist()
    policies = table.column("policy").to_pylist()
    seeds = table.column("seed").to_pylist()
    return {
        (str(prompt_id), str(policy), int(seed or 0))
        for prompt_id, policy, seed in zip(prompt_ids, policies, seeds, strict=False)
        if prompt_id is not None and policy is not None
    }


def _cache_stats_generation_keys_from_rows(rows: list[dict[str, Any]]) -> set[tuple[str, str, int]]:
    keys = set()
    for row in rows:
        prompt_id = row.get("prompt_id")
        policy = row.get("policy")
        if prompt_id is None or policy is None:
            continue
        keys.add((str(prompt_id), str(policy), int(row.get("seed") or 0)))
    return keys


def _write_progress(
    run_dir: Path,
    *,
    completed: int,
    expected: int,
    activity: str,
    current: dict[str, Any] | None,
) -> None:
    percent = round((completed / expected) * 100, 3) if expected else 100.0
    write_json(
        run_dir / "progress.json",
        {
            "updated_at": utc_timestamp(),
            "activity": activity,
            "completed": completed,
            "expected": expected,
            "progress_percent": percent,
            "current": current or {},
        },
    )


def _promote_recoverable_cache_stats_temp(cache_stats_path: Path) -> Path:
    temp_path = cache_stats_path.with_suffix(".parquet.tmp")
    if not temp_path.exists():
        return cache_stats_path
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for cache stats resume reconciliation.") from exc

    def num_rows(path: Path) -> int | None:
        if not path.exists():
            return None
        try:
            return int(pq.ParquetFile(path).metadata.num_rows)
        except Exception:
            return None

    temp_rows = num_rows(temp_path)
    if temp_rows is None:
        return cache_stats_path
    main_rows = num_rows(cache_stats_path)
    if main_rows is not None and temp_rows < main_rows:
        return cache_stats_path
    if cache_stats_path.exists():
        backup_path = cache_stats_path.with_name(
            f"{cache_stats_path.name}.pre_temp_recovery.{utc_timestamp()}"
        )
        cache_stats_path.replace(backup_path)
    temp_path.replace(cache_stats_path)
    print(
        "Recovered resume cache stats from valid temporary parquet "
        f"with {temp_rows} row(s): {cache_stats_path}"
    )
    return cache_stats_path


CACHE_STATS_COLUMNS = [
    "prompt_id",
    "seed",
    "policy",
    "decode_step",
    "original_seq_len",
    "retained_count",
    "evicted_count",
    "retained_indices",
    "evicted_indices",
    "layer",
    "layer_count",
    "cache_l2_before",
    "cache_l2_after",
    "cache_l2_measurement",
    "retained_special_tokens",
    "retained_template_tokens",
    "retained_system_tokens",
    "retained_user_tokens",
    "retained_generated_tokens",
    "retained_unknown_tokens",
    "evicted_special_tokens",
    "evicted_template_tokens",
    "evicted_system_tokens",
    "evicted_user_tokens",
    "evicted_generated_tokens",
    "evicted_unknown_tokens",
    "sink_tokens",
    "recent_tokens",
    "policy_seed",
    "attention_scores_used",
    "quantization_bits",
    "protected_spans",
    "protected_candidate_count",
    "protected_retained_count",
    "protected_dropped_count",
    "patched_from_baseline",
    "patched_token_count",
    "patched_roles",
    "patched_token_indices",
    "patch_selection",
    "patch_matched_roles",
    "patch_layers",
    "patch_heads",
    "patch_components",
    "cache_l2_after_patch",
]


def _normalize_cache_stat_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{column: row.get(column) for column in CACHE_STATS_COLUMNS} for row in rows]


def _cache_stats_schema() -> Any:
    import pyarrow as pa

    int_columns = {
        "seed",
        "decode_step",
        "original_seq_len",
        "retained_count",
        "evicted_count",
        "layer",
        "layer_count",
        "retained_special_tokens",
        "retained_template_tokens",
        "retained_system_tokens",
        "retained_user_tokens",
        "retained_generated_tokens",
        "retained_unknown_tokens",
        "evicted_special_tokens",
        "evicted_template_tokens",
        "evicted_system_tokens",
        "evicted_user_tokens",
        "evicted_generated_tokens",
        "evicted_unknown_tokens",
        "sink_tokens",
        "recent_tokens",
        "policy_seed",
        "quantization_bits",
        "protected_candidate_count",
        "protected_retained_count",
        "protected_dropped_count",
        "patched_token_count",
    }
    float_columns = {"cache_l2_before", "cache_l2_after", "cache_l2_after_patch"}
    bool_columns = {"attention_scores_used", "patched_from_baseline"}
    fields = []
    for column in CACHE_STATS_COLUMNS:
        if column in int_columns:
            fields.append(pa.field(column, pa.int64()))
        elif column in float_columns:
            fields.append(pa.field(column, pa.float64()))
        elif column in bool_columns:
            fields.append(pa.field(column, pa.bool_()))
        else:
            fields.append(pa.field(column, pa.large_string()))
    return pa.schema(fields)


class _CacheStatsSink:
    def __init__(self, path: Path, *, resume: bool) -> None:
        self.path = path
        self.resume = resume
        self.writer: Any | None = None
        self.temp_path: Path | None = None

    def write(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        schema = _cache_stats_schema()
        table = _align_table_to_schema(_cache_stats_table(rows), schema)
        if self.writer is None:
            write_path = self.path
            if self.resume and self.path.exists():
                self.temp_path = self.path.with_suffix(".parquet.tmp")
                write_path = self.temp_path
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
            self.writer = pq.ParquetWriter(write_path, schema)
            if self.temp_path is not None:
                _copy_existing_cache_stats(self.path, self.writer, schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            if self.temp_path is not None:
                self.temp_path.replace(self.path)
            return
        if not self.path.exists():
            try:
                import pyarrow.parquet as pq
            except ModuleNotFoundError as exc:
                raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
            pq.write_table(_cache_stats_table([]), self.path)


def _cache_stats_table(rows: list[dict[str, Any]]) -> Any:
    try:
        import pyarrow as pa
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
    normalized = _normalize_cache_stat_rows(rows)
    schema = _cache_stats_schema()
    arrays = [
        pa.array([row[field.name] for row in normalized], type=field.type) for field in schema
    ]
    return pa.Table.from_arrays(arrays, schema=schema)


def _rebuild_cache_stats_parquet_from_jsonl(jsonl_path: Path, parquet_path: Path) -> None:
    if not jsonl_path.exists():
        return
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for cache_stats.parquet.") from exc
    temp_path = parquet_path.with_suffix(".parquet.rebuild.tmp")
    schema = _cache_stats_schema()
    row_count = 0
    with pq.ParquetWriter(temp_path, schema) as writer:
        for batch in _iter_cache_stat_jsonl_batches(jsonl_path):
            row_count += len(batch)
            writer.write_table(_align_table_to_schema(_cache_stats_table(batch), schema))
    if row_count == 0:
        pq.write_table(_cache_stats_table([]), temp_path)
    temp_path.replace(parquet_path)


def _iter_cache_stat_jsonl_batches(
    path: Path, *, batch_size: int = 100_000
) -> Any:
    batch: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path} has malformed JSON at line {line_number}; "
                    "refusing to rebuild cache_stats.parquet from partial data"
                ) from exc
            if not isinstance(row, dict):
                raise RuntimeError(f"{path} line {line_number} is not a JSON object")
            batch.append(row)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _copy_existing_cache_stats(path: Path, writer: Any, schema: Any) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required for resume-safe cache stats.") from exc
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        writer.write_table(_align_table_to_schema(table, schema))


def _align_table_to_schema(table: Any, schema: Any) -> Any:
    import pyarrow as pa

    columns = []
    for field in schema:
        if field.name in table.column_names:
            column = table[field.name]
            if not column.type.equals(field.type):
                column = column.cast(field.type)
            columns.append(column)
        else:
            columns.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(columns, schema=schema)


if __name__ == "__main__":
    main()
