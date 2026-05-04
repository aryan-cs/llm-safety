#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

primary_run_id="${PRIMARY_RUN_ID:-h200_qwen_full_sweep}"
causal_run_id="${CAUSAL_RUN_ID:-h200_causal_patch_qwen7b}"
audit_input_dir="${AUDIT_INPUT_DIR:-paper/audit}"
judge_model_id="${OPEN_JUDGE_MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
judge_dtype="${OPEN_JUDGE_DTYPE:-bfloat16}"
judge_device_map="${OPEN_JUDGE_DEVICE_MAP:-auto}"
judge_max_new_tokens="${OPEN_JUDGE_MAX_NEW_TOKENS:-256}"
judge_limit="${OPEN_JUDGE_LIMIT:-}"

run_open_judge_variant() {
  local run_id="$1"
  local variant="$2"
  local input_csv="$audit_input_dir/${run_id}_audit_blinded.csv"
  local output_csv="$audit_input_dir/${run_id}_audit_blinded_annotator_open_judge_${variant}.csv"

  if [[ ! -f "$input_csv" ]]; then
    echo "Missing blinded audit CSV: $input_csv" >&2
    echo "Run scripts/export_publication_audit_samples.sh first." >&2
    exit 1
  fi

  local limit_args=()
  if [[ -n "$judge_limit" ]]; then
    limit_args+=(--limit "$judge_limit")
  fi

  uv run python scripts/run_open_judge_audit.py \
    --audit-csv "$input_csv" \
    --output-csv "$output_csv" \
    --model-id "$judge_model_id" \
    --dtype "$judge_dtype" \
    --device-map "$judge_device_map" \
    --prompt-variant "$variant" \
    --annotator-id "open_judge_${variant}" \
    --max-new-tokens "$judge_max_new_tokens" \
    "${limit_args[@]}"
}

uv sync --frozen --extra dev
run_open_judge_variant "$primary_run_id" v1
run_open_judge_variant "$primary_run_id" v2
run_open_judge_variant "$causal_run_id" v1
run_open_judge_variant "$causal_run_id" v2

echo "Open local judge audit CSVs written to $audit_input_dir"
echo "Aggregate them with scripts/aggregate_publication_human_audits.sh."
