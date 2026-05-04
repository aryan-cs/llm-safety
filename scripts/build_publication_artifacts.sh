#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
qwen32_results="${QWEN32_RESULTS_DIR:-results/h200_qwen32b_public_followup_primary}"
primary_generated_dir="${PRIMARY_GENERATED_DIR:-paper/generated/h200_qwen_full_sweep}"
causal_generated_dir="${CAUSAL_GENERATED_DIR:-paper/generated/h200_causal_patch_qwen7b}"
claim_generated_dir="${CLAIM_GENERATED_DIR:-paper/generated/claim_assessment}"
qwen32_generated_dir="${QWEN32_GENERATED_DIR:-paper/generated/h200_qwen32b_public_followup}"
primary_audit_summary="${PRIMARY_AUDIT_SUMMARY_DIR:-paper/audit/h200_qwen_full_sweep_summary}"
causal_audit_summary="${CAUSAL_AUDIT_SUMMARY_DIR:-paper/audit/h200_causal_patch_qwen7b_summary}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
causal_ci_width="${CAUSAL_CI_WIDTH:-0.12}"
qwen32_ci_width="${QWEN32_CI_WIDTH:-0.10}"
require_qwen32_followup="${REQUIRE_QWEN32_FOLLOWUP:-0}"
publication_status_dir="${PUBLICATION_STATUS_DIR:-paper/build}"
arxiv_source_dir="${ARXIV_SOURCE_DIR:-paper/build/arxiv_source}"
arxiv_archive="${ARXIV_ARCHIVE:-paper/build/arxiv_source.tar.gz}"

result_artifacts_complete() {
  local results_dir="$1"
  for required in manifest.json generations.jsonl metrics.json cache_stats.parquet; do
    if [[ ! -f "$results_dir/$required" ]]; then
      return 1
    fi
  done
  return 0
}

require_result_artifacts() {
  local results_dir="$1"
  for required in manifest.json generations.jsonl metrics.json cache_stats.parquet; do
    [[ -f "$results_dir/$required" ]] && continue
    echo "Missing required result artifact: $results_dir/$required" >&2
    exit 1
  done
}

require_human_audit_artifacts() {
  local audit_dir="$1"
  local results_dir="$2"
  for required in \
    audit_manifest.json \
    human_audit_summary.json \
    human_audit_summary.md \
    human_audit_summary_table.tex \
    human_audit_deltas_table.tex; do
    if [[ ! -f "$audit_dir/$required" ]]; then
      echo "Missing required human-audit artifact: $audit_dir/$required" >&2
      echo "Aggregate completed annotations with scripts/aggregate_human_audit.py before publication." >&2
      exit 1
    fi
  done
  uv run python scripts/check_human_audit_readiness.py \
    --summary-json "$audit_dir/human_audit_summary.json" \
    --audit-manifest "$audit_dir/audit_manifest.json" \
    --results-dir "$results_dir" \
    --require-result-source-match \
    --require-baseline-deltas
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

rebuild_qwen32_if_present() {
  if [[ ! -d "$qwen32_results" ]]; then
    echo "Skipping Qwen 32B follow-up artifacts; directory not found: $qwen32_results"
    return
  fi
  if [[ "$require_qwen32_followup" != "1" ]] && ! result_artifacts_complete "$qwen32_results"; then
    echo "Skipping optional Qwen 32B follow-up artifacts; directory exists but is incomplete: $qwen32_results"
    echo "Set REQUIRE_QWEN32_FOLLOWUP=1 to make incomplete Qwen 32B artifacts fail the publication build."
    return
  fi
  require_result_artifacts "$qwen32_results"
  uv run python scripts/aggregate_results.py --results-dir "$qwen32_results"
  uv run python scripts/make_figures.py --results-dir "$qwen32_results"
  uv run python scripts/export_paper_assets.py \
    --results-dir "$qwen32_results" \
    --paper-dir "$qwen32_generated_dir" \
    --macro-prefix QwenThirtyTwo
  uv run python scripts/check_publication_readiness.py \
    --results-dir "$qwen32_results" \
    --paper-dir "$qwen32_generated_dir" \
    --min-prompts-per-suite 600 \
    --suite-min-prompts system_leakage=2 \
    --suite-min-prompts public_xstest_safe=200 \
    --max-ci-width "$qwen32_ci_width" \
    --required-suite system_leakage \
    --required-suite public_system_leakage \
    --required-suite public_refusal_safety \
    --required-suite public_benign_overrefusal \
    --required-suite public_xstest_safe \
    --required-suite public_capability_arc \
    --required-policy none \
    --required-policy sliding_window \
    --required-policy sink_recent \
    --required-policy kv_int4_sim \
    --require-policy-pinned \
    --required-figure safety_capability_phase_portrait \
    --required-figure selective_safety_erasure_heatmap \
    --required-figure prompt_effect_constellation \
    --required-figure cache_state_fingerprint \
    --required-figure safety_state_atlas \
    --require-public-provenance
}

assess_claims() {
  local claim_args=(
    --primary-results-dir "$primary_results"
    --causal-results-dir "$causal_results"
    --output-dir "$claim_generated_dir"
    --primary-audit-summary "$primary_audit_summary/human_audit_summary.json"
    --causal-audit-summary "$causal_audit_summary/human_audit_summary.json"
    --require-human-audit-support
    --require-cache-mediated-claim
  )
  uv run python scripts/assess_claims.py "${claim_args[@]}"
  uv run python scripts/plan_registered_followups.py \
    --claim-assessment "$claim_generated_dir/claim_assessment.json" \
    --primary-ci-power "$primary_results/ci_power.json" \
    --causal-ci-power "$causal_results/ci_power.json" \
    --output-dir paper/generated/registered_followup_plan
}

write_publication_status() {
  mkdir -p "$publication_status_dir"
  uv run python scripts/report_publication_status.py \
    --primary-results-dir "$primary_results" \
    --causal-results-dir "$causal_results" \
    --primary-audit-dir "$primary_audit_summary" \
    --causal-audit-dir "$causal_audit_summary" \
    --claim-assessment "$claim_generated_dir/claim_assessment.json" \
    --arxiv-source-dir "$arxiv_source_dir" \
    --arxiv-archive "$arxiv_archive" \
    --output-json "$publication_status_dir/publication_status.json" \
    --output-md "$publication_status_dir/publication_status.md" \
    "$@"
}

uv sync --frozen --extra dev
uv run ruff check .
uv run pytest -q
write_publication_status

require_human_audit_artifacts "$primary_audit_summary" "$primary_results"
require_human_audit_artifacts "$causal_audit_summary" "$causal_results"

rebuild_primary
rebuild_causal
assess_claims
rebuild_qwen32_if_present

rm -f paper/cache_mediated_safety_erasure.pdf
PRIMARY_RESULTS_DIR="$primary_results" \
CAUSAL_RESULTS_DIR="$causal_results" \
PRIMARY_PAPER_DIR="$primary_generated_dir" \
CAUSAL_PAPER_DIR="$causal_generated_dir" \
PRIMARY_AUDIT_SUMMARY_DIR="$primary_audit_summary" \
CAUSAL_AUDIT_SUMMARY_DIR="$causal_audit_summary" \
CLAIM_ASSESSMENT_PATH="$claim_generated_dir/claim_assessment.json" \
ARXIV_SOURCE_DIR="$arxiv_source_dir" \
ARXIV_ARCHIVE="$arxiv_archive" \
REQUIRE_COMPLETE_PAPER=1 \
bash scripts/build_paper_pdf.sh
cp paper/build/cache_mediated_safety_erasure.pdf paper/cache_mediated_safety_erasure.pdf
write_publication_status --fail-if-not-ready
uv run python scripts/package_arxiv_submission.py \
  --output-dir "$arxiv_source_dir" \
  --archive "$arxiv_archive" \
  --primary-results-dir "$primary_results" \
  --causal-results-dir "$causal_results" \
  --primary-generated-dir "$primary_generated_dir" \
  --causal-generated-dir "$causal_generated_dir" \
  --claim-generated-dir "$claim_generated_dir" \
  --qwen32-generated-dir "$qwen32_generated_dir" \
  --primary-audit-dir "$primary_audit_summary" \
  --causal-audit-dir "$causal_audit_summary"
write_publication_status --require-arxiv-bundle --fail-if-not-ready

echo "Publication artifacts rebuilt:"
echo "- paper/cache_mediated_safety_erasure.pdf"
echo "- ${arxiv_archive}"
echo "- ${publication_status_dir}/publication_status.md"
