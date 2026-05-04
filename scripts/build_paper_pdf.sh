#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

src_dir="paper/latex"
build_dir="paper/build"
primary_results="${PRIMARY_RESULTS_DIR:-results/h200_qwen_full_sweep}"
causal_results="${CAUSAL_RESULTS_DIR:-results/h200_causal_patch_qwen7b}"
primary_paper_dir="${PRIMARY_PAPER_DIR:-paper/generated/h200_qwen_full_sweep}"
causal_paper_dir="${CAUSAL_PAPER_DIR:-paper/generated/h200_causal_patch_qwen7b}"
mkdir -p "$build_dir"

if [[ "${REQUIRE_COMPLETE_PAPER:-0}" == "1" ]]; then
  uv run python scripts/check_latex_placeholders.py --tex "$src_dir/main.tex"
  uv run python scripts/check_paper_asset_freshness.py \
    --pair "$primary_paper_dir=$primary_results" \
    --pair "$causal_paper_dir=$causal_results"
  uv run python scripts/report_publication_status.py \
    --paper-pdf "$build_dir/cache_mediated_safety_erasure.pdf" \
    --allow-missing-paper-pdf \
    --fail-if-not-ready
fi

if command -v tectonic >/dev/null 2>&1; then
  (
    cd "$src_dir"
    tectonic --outdir ../build main.tex
  )
elif command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory="$build_dir" "$src_dir/main.tex"
elif command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
  (
    cd "$src_dir"
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
    bibtex ../build/main
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
    pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build main.tex
  )
else
  echo "No supported LaTeX builder found. Install tectonic, latexmk, or pdflatex+bibtex." >&2
  exit 1
fi

if [[ -f "$build_dir/main.pdf" ]]; then
  mv "$build_dir/main.pdf" "$build_dir/cache_mediated_safety_erasure.pdf"
fi

echo "Wrote $build_dir/cache_mediated_safety_erasure.pdf"
