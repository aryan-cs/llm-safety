#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-$(pwd)/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TORCH_HOME="${TORCH_HOME:-$(pwd)/.cache/torch}"

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to run H200 sweep from a dirty git working tree." >&2
  echo "Commit or stash local changes so generated artifacts point to an exact commit." >&2
  exit 1
fi

uv sync --frozen --extra dev

public_prompt_limit="${PUBLIC_PROMPT_LIMIT:-650}"
target_ci_width="${TARGET_CI_WIDTH:-0.08}"
audit_per_suite_policy="${AUDIT_PER_SUITE_POLICY:-10}"
audit_annotator_template_count="${AUDIT_ANNOTATOR_TEMPLATE_COUNT:-2}"

uv run python scripts/prepare_data.py --suite all
uv run python scripts/prepare_data.py --source hf --suite cyberec_prompt_injection_leakage --limit "$public_prompt_limit" --output-suite public_system_leakage
uv run python scripts/prepare_data.py --source hf --suite public_refusal_combo --limit "$public_prompt_limit" --output-suite public_refusal_safety
uv run python scripts/prepare_data.py --source hf --suite dolly_benign --limit "$public_prompt_limit" --output-suite public_benign_overrefusal
uv run python scripts/prepare_data.py --source hf --suite xstest_safe --limit "$public_prompt_limit" --output-suite public_xstest_safe
uv run python scripts/prepare_data.py --source hf --suite arc_easy --limit "$public_prompt_limit" --output-suite public_capability_arc
uv run python scripts/check_prepared_suites.py \
  --min-records 600 \
  --suite-min-records system_leakage=2 \
  --suite-min-records public_xstest_safe=200 \
  --require-public-provenance \
  --suite system_leakage \
  --suite public_system_leakage \
  --suite public_refusal_safety \
  --suite public_benign_overrefusal \
  --suite public_xstest_safe \
  --suite public_capability_arc

uv run python scripts/preflight_h200.py \
  --config configs/experiments/qwen7b_smoke.yaml \
  --config configs/experiments/h200_public_qwen14b.yaml \
  --config configs/experiments/h200_causal_patch_qwen7b.yaml \
  --config configs/experiments/h200_attention_diagnostic_qwen7b.yaml

smoke_run_id="${SMOKE_RUN_ID:-qwen7b_smoke_h200}"
full_run_id="${FULL_RUN_ID:-h200_qwen_full_sweep}"
causal_run_id="${CAUSAL_RUN_ID:-h200_causal_patch_qwen7b}"
attention_run_id="${ATTENTION_RUN_ID:-h200_attention_diagnostic_qwen7b_primary}"

echo "Running Qwen 7B smoke validation..."
uv run python scripts/run_experiment.py \
  --config configs/experiments/qwen7b_smoke.yaml \
  --run-id "$smoke_run_id" \
  --resume

latest_smoke="results/$smoke_run_id"
uv run python scripts/aggregate_results.py --results-dir "$latest_smoke"
uv run python scripts/make_figures.py --results-dir "$latest_smoke"
uv run python scripts/export_paper_assets.py --results-dir "$latest_smoke" --paper-dir paper/generated/qwen7b_smoke

echo "Running primary H200 Qwen 14B sweep..."
uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_public_qwen14b.yaml \
  --run-id "$full_run_id" \
  --resume

latest_full="results/$full_run_id"
uv run python scripts/aggregate_results.py --results-dir "$latest_full"
uv run python scripts/make_figures.py --results-dir "$latest_full"
uv run python scripts/export_paper_assets.py \
  --results-dir "$latest_full" \
  --paper-dir paper/generated/h200_qwen_full_sweep \
  --macro-prefix Primary
uv run python scripts/plan_ci_power.py \
  --results-dir "$latest_full" \
  --target-ci-width "$target_ci_width" \
  --output-json "$latest_full/ci_power.json" \
  --output-md paper/generated/h200_qwen_full_sweep/ci_power.md
uv run python scripts/check_publication_readiness.py \
  --results-dir "$latest_full" \
  --paper-dir paper/generated/h200_qwen_full_sweep \
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
uv run python scripts/export_human_audit_sample.py \
  --results-dir "$latest_full" \
  --per-suite-policy "$audit_per_suite_policy" \
  --annotator-template-count "$audit_annotator_template_count"

echo "Running causal patch diagnostic on Qwen 7B..."
uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_causal_patch_qwen7b.yaml \
  --run-id "$causal_run_id" \
  --resume

latest_causal="results/$causal_run_id"
uv run python scripts/aggregate_results.py --results-dir "$latest_causal"
uv run python scripts/make_figures.py --results-dir "$latest_causal"
uv run python scripts/export_paper_assets.py \
  --results-dir "$latest_causal" \
  --paper-dir paper/generated/h200_causal_patch_qwen7b \
  --macro-prefix Causal
uv run python scripts/plan_ci_power.py \
  --results-dir "$latest_causal" \
  --target-ci-width 0.12 \
  --output-json "$latest_causal/ci_power.json" \
  --output-md paper/generated/h200_causal_patch_qwen7b/ci_power.md
uv run python scripts/check_publication_readiness.py \
  --results-dir "$latest_causal" \
  --paper-dir paper/generated/h200_causal_patch_qwen7b \
  --min-prompts-per-suite 600 \
  --suite-min-prompts system_leakage=2 \
  --max-ci-width 0.12 \
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
uv run python scripts/assess_claims.py \
  --primary-results-dir "$latest_full" \
  --causal-results-dir "$latest_causal" \
  --output-dir paper/generated/preliminary_claim_assessment
uv run python scripts/export_human_audit_sample.py \
  --results-dir "$latest_causal" \
  --per-suite-policy "$audit_per_suite_policy" \
  --annotator-template-count "$audit_annotator_template_count"

echo "Running attention-policy diagnostic on Qwen 7B..."
uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_attention_diagnostic_qwen7b.yaml \
  --run-id "$attention_run_id" \
  --resume

latest_attention="results/$attention_run_id"
uv run python scripts/aggregate_results.py --results-dir "$latest_attention"
uv run python scripts/make_figures.py --results-dir "$latest_attention"
uv run python scripts/export_paper_assets.py --results-dir "$latest_attention" --paper-dir paper/generated/h200_attention_diagnostic_qwen7b

echo "Primary sweep complete: $latest_full"
