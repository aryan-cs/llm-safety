#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <remote-run-dir-relative-to-repo> [local-results-dir]" >&2
  exit 2
fi

remote_run_dir="$1"
local_results_dir="${2:-results}"
remote_root="/home/aryang9/sandbox/llm-safety"
remote="uiuc-h200"
remote_parent="$(dirname "$remote_run_dir")"
remote_base="$(basename "$remote_run_dir")"

mkdir -p "$local_results_dir"

local_run_dir="${local_results_dir}/${remote_base}"
mkdir -p "$local_run_dir"

ssh "$remote" "cd '$remote_root/$remote_parent' && tar -czf - '$remote_base'" \
  | tar -xzf - -C "$local_results_dir"

echo "Fetched ${remote}:${remote_root}/${remote_run_dir} -> ${local_run_dir}"
