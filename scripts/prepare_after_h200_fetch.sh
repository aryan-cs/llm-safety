#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
primary_generated_dir="${PRIMARY_GENERATED_DIR:-paper/generated/h200_qwen_full_sweep}"
causal_generated_dir="${CAUSAL_GENERATED_DIR:-paper/generated/h200_causal_patch_qwen7b}"
publication_status_dir="${PUBLICATION_STATUS_DIR:-paper/build}"
arxiv_source_dir="${ARXIV_SOURCE_DIR:-paper/build/arxiv_source}"
arxiv_archive="${ARXIV_ARCHIVE:-paper/build/arxiv_source.tar.gz}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
causal_ci_width="${CAUSAL_CI_WIDTH:-0.12}"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to prepare paper evidence from a dirty git working tree." >&2
  echo "Commit or stash code/documentation changes before regenerating paper assets." >&2
  git status --short >&2
  exit 1
fi

require_result_artifacts() {
  local results_dir="$1"
  for required in manifest.json generations.jsonl metrics.json cache_stats.parquet; do
    [[ -f "$results_dir/$required" ]] && continue
    echo "Missing required completed-run artifact: $results_dir/$required" >&2
    echo "Run scripts/fetch_h200_results.sh after the guarded H200 launcher completes." >&2
    exit 1
  done
}

rebuild_primary() {
  require_result_artifacts "$primary_results"
  uv run python scripts/aggregate_results.py --results-dir "$primary_results"
  uv run python scripts/make_figures.py --results-dir "$primary_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$primary_results" \
    --paper-dir "$primary_generated_dir" \
    --macro-prefix Primary
  uv run python scripts/plan_ci_power.py \
    --results-dir "$primary_results" \
    --target-ci-width "$target_ci_width" \
    --output-json "$primary_results/ci_power.json" \
    --output-md "$primary_generated_dir/ci_power.md"
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$primary_results" \
    --paper-dir "$primary_generated_dir" \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --suite-min-prompts public_xstest_safe=200 \
    --max-ci-width "$target_ci_width" \
    --required-suite system_leakage \
    --required-suite public_system_leakage \
    --required-suite public_refusal_safety \
    --required-suite public_benign_overrefusal \
    --required-suite public_xstest_safe \
    --required-suite public_capability_arc \
    --required-policy none \
    --required-policy sliding_window \
    --required-policy sink_recent \
    --required-policy random_matched \
    --required-policy kv_int8_sim \
    --required-policy kv_int4_sim \
    --require-policy-pinned \
    --required-figure safety_capability_phase_portrait \
    --required-figure selective_safety_erasure_heatmap \
    --required-figure prompt_effect_constellation \
    --required-figure cache_state_fingerprint \
    --required-figure safety_state_atlas \
    --require-public-provenance
}

rebuild_causal() {
  require_result_artifacts "$causal_results"
  uv run python scripts/aggregate_results.py --results-dir "$causal_results"
  uv run python scripts/make_figures.py --results-dir "$causal_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$causal_results" \
    --paper-dir "$causal_generated_dir" \
    --macro-prefix Causal
  uv run python scripts/plan_ci_power.py \
    --results-dir "$causal_results" \
    --target-ci-width "$causal_ci_width" \
    --output-json "$causal_results/ci_power.json" \
    --output-md "$causal_generated_dir/ci_power.md"
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$causal_results" \
    --paper-dir "$causal_generated_dir" \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --max-ci-width "$causal_ci_width" \
    --required-suite system_leakage \
    --required-suite public_system_leakage \
    --required-suite public_refusal_safety \
    --required-policy none \
    --required-policy kv_int4_sim \
    --require-causal-patch \
    --require-policy-pinned \
    --required-figure causal_restoration_fraction \
    --required-figure causal_restoration_flow \
    --require-public-provenance
}

write_publication_status() {
  mkdir -p "$publication_status_dir"
  uv run python scripts/report_publication_status.py \
    --primary-results-dir "$primary_results" \
    --causal-results-dir "$causal_results" \
    --primary-generated-dir "$primary_generated_dir" \
    --causal-generated-dir "$causal_generated_dir" \
    --arxiv-source-dir "$arxiv_source_dir" \
    --arxiv-archive "$arxiv_archive" \
    --output-json "$publication_status_dir/publication_status.json" \
    --output-md "$publication_status_dir/publication_status.md"
}

uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q

rebuild_primary
rebuild_causal
bash scripts/export_publication_audit_samples.sh
write_publication_status
uv run python scripts/post_h200_next_steps.py \
  --output-json paper/generated/post_h200_next_steps.json \
  --output-md paper/generated/post_h200_next_steps.md

echo "Post-H200 fetched evidence prepared:"
echo "- $primary_generated_dir"
echo "- $causal_generated_dir"
echo "- paper/audit/*_audit_blinded_annotator_*.csv"
echo "- paper/generated/post_h200_next_steps.md"
