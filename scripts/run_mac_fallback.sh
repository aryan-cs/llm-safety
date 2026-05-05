#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$repo_dir"

config="${MAC_FALLBACK_CONFIG:-configs/experiments/mac_qwen3b_causal_fallback.yaml}"
run_id="${MAC_FALLBACK_RUN_ID:-mac_qwen3b_causal_fallback}"
paper_dir="${MAC_FALLBACK_PAPER_DIR:-paper/generated/mac_qwen3b_causal_fallback}"
cache_root="${MAC_FALLBACK_CACHE_ROOT:-$(pwd)/.cache/mac_fallback}"
delete_models_after="${MAC_FALLBACK_DELETE_MODELS_AFTER:-1}"
min_unified_memory_gb="${MAC_FALLBACK_MIN_UNIFIED_MEMORY_GB:-22}"

if [[ ! "$run_id" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid MAC_FALLBACK_RUN_ID=${run_id}; expected only letters, numbers, dot, underscore, or dash." >&2
  exit 2
fi
if [[ "$run_id" == h200_* ]]; then
  echo "Refusing to use an H200 run id for Mac fallback: ${run_id}" >&2
  echo "Use a separate mac_* run id so local diagnostics cannot contaminate H200 evidence." >&2
  exit 2
fi

if [[ -n "$(git status --short)" && "${ALLOW_DIRTY_MAC_FALLBACK:-0}" != "1" ]]; then
  echo "Refusing to run Mac fallback from a dirty git working tree." >&2
  echo "Commit or stash local changes, or set ALLOW_DIRTY_MAC_FALLBACK=1 for a non-evidence debug run." >&2
  exit 1
fi

cleanup_model_cache() {
  if [[ "$delete_models_after" == "1" ]]; then
    HF_HOME="$cache_root/huggingface" \
    HF_HUB_CACHE="$cache_root/huggingface/hub" \
    TRANSFORMERS_CACHE="$cache_root/huggingface/transformers" \
    TORCH_HOME="$cache_root/torch" \
      bash scripts/cleanup_local_model_caches.sh --yes
  fi
}
trap cleanup_model_cache EXIT

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv sync --frozen --extra dev

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python - <<PY
from __future__ import annotations

import platform
import sys

min_gb = float("${min_unified_memory_gb}")
if platform.system() != "Darwin":
    raise SystemExit("Mac fallback is only intended for macOS/M-series local runs.")


def read_unified_memory_bytes() -> int:
    """Read physical memory without requiring macOS sysctl access."""
    try:
        import os

        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        page_count = int(os.sysconf("SC_PHYS_PAGES"))
        if page_size > 0 and page_count > 0:
            return page_size * page_count
    except Exception:
        pass
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        if total > 0:
            return total
    except Exception:
        pass
    raise SystemExit("Could not read unified memory with os.sysconf or psutil.")


mem_bytes = read_unified_memory_bytes()
mem_gb = mem_bytes / 1024**3
if mem_gb < min_gb:
    raise SystemExit(
        f"Mac fallback requires at least {min_gb:.1f} GiB unified memory; detected {mem_gb:.1f} GiB."
    )
try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit("PyTorch is not installed; run `uv sync --extra dev` first.") from exc
if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
    raise SystemExit("Mac fallback requires an available PyTorch MPS backend.")
print(f"Mac fallback preflight passed: macOS, {mem_gb:.1f} GiB unified memory, MPS available.")
PY

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export HF_HOME="$cache_root/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$cache_root/torch"

UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/prepare_data.py --suite all
UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/run_experiment.py \
  --config "$config" \
  --run-id "$run_id" \
  --resume

latest="results/$run_id"
UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/aggregate_results.py --results-dir "$latest"
UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/make_figures.py --results-dir "$latest"
UV_CACHE_DIR="${UV_CACHE_DIR:-.cache/uv}" uv run python scripts/export_paper_assets.py \
  --results-dir "$latest" \
  --paper-dir "$paper_dir" \
  --macro-prefix MacFallback

echo "Mac fallback run complete: ${latest}"
echo "Mac fallback paper assets: ${paper_dir}"
