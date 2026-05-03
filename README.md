# Testing Cache-Mediated Safety Erasure

This repository tests a phenomenon-first alignment hypothesis:

> Inference-time KV-cache optimizations may selectively weaken safety/refusal behavior while preserving ordinary model capability, because some safety behavior may depend on fragile cache-resident routing state.

The project is intentionally built around open models, local inference, and reproducible artifacts. It does not depend on paid endpoints, closed-source judges, or private datasets.

## Why This Project Exists

Earlier candidate work focused on safety-classifier supply-chain auditing. That is useful, but the closest prior work already covers much of the attack and audit surface: Anthropic's classifier poisoning post, Rapid Poison, AI-BOM/provenance work, and guardrail robustness benchmarks. This repository instead targets a more surprising mechanism: **deployment-time inference infrastructure itself may alter alignment behavior without changing model weights or prompts**.

Closest adjacent work to cite and distinguish:

- KV-cache compression can damage multi-instruction following and system prompt privacy: <https://arxiv.org/abs/2510.00231>
- KV-cache compression can be interpreted as a routing/accessibility perturbation: <https://arxiv.org/abs/2603.01426>
- MiKV reports that exhaustive eviction can create safety breaches, hallucinations, and context loss: <https://arxiv.org/abs/2402.18096>
- KV-cache editing can defend against indirect prompt injection: <https://arxiv.org/abs/2504.21228>
- Refusal/alignment behavior may route through sparse gate and amplifier heads: <https://arxiv.org/abs/2604.04385>
- Subliminal learning and token entanglement are examples of the kind of phenomenon-first contribution this project is aiming for.

The claims ladder is deliberately strict:

1. cache policies change behavior;
2. safety degrades more than ordinary capability;
3. targeted system-role cache preservation/restoration causally recovers safety more than matched user-role controls.

Only the third result justifies the stronger "safety erasure" language.

## Hardware Assumptions

Development target:

- MacBook M4 Pro with 24 GB RAM for code, tests, and tiny smoke runs.

Full sweep target:

- Illinois Computes Research Notebooks H200 with 141 GB VRAM, 10 CPUs, and a 32 GB RAM cgroup.
- Avoid CPU offload-heavy configurations. H200 preflight rejects configs that permit CPU/disk offload, and the Hugging Face loader fails paper runs if `hf_device_map` places modules on CPU or disk.

Primary model targets:

- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`

Optional targets if locally available and licensing/gating is resolved:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it` or a current open Gemma instruct model

## Quickstart

Install dependencies:

```bash
uv sync --extra dev
```

Prepare the built-in diagnostic prompt suites:

```bash
uv run python scripts/prepare_data.py --suite all
```

Run the local artifact smoke test with a deterministic mock model:

```bash
uv run python scripts/run_experiment.py --config configs/experiments/smoke_mock.yaml
```

Run the tiny Hugging Face plumbing test:

```bash
uv run python scripts/run_experiment.py --config configs/experiments/tiny_hf_smoke.yaml
```

Run the unit tests:

```bash
uv run pytest
uv run ruff check .
```

Run a real small-model smoke test after downloading an open Hugging Face model:

```bash
uv run python scripts/run_experiment.py --config configs/experiments/qwen7b_smoke.yaml
```

Resume or pin a run id without editing YAML:

```bash
uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_public_qwen14b.yaml \
  --run-id h200_public_qwen14b_primary \
  --resume
```

Run the primary H200 workflow:

```bash
bash scripts/run_h200_sweep.sh
```

Initialize or update the H200 checkout under the authorized notebook folder:

```bash
bash scripts/setup_h200_remote.sh
```

That wrapper runs `scripts/bootstrap_h200.sh` over `ssh uiuc-h200` and refuses to operate outside `/home/aryang9/sandbox/llm-safety`.

Preflight the H200 configs without launching a sweep:

```bash
uv run python scripts/preflight_h200.py \
  --config configs/experiments/h200_public_qwen14b.yaml \
  --config configs/experiments/h200_causal_patch_qwen7b.yaml \
  --config configs/experiments/h200_attention_diagnostic_qwen7b.yaml
```

Aggregate a run:

```bash
uv run python scripts/aggregate_results.py --results-dir results/<run_id>
```

Make figures:

```bash
uv run python scripts/make_figures.py --results-dir results/<run_id>
```

Build the current LaTeX paper draft as a readable PDF:

```bash
bash scripts/build_paper_pdf.sh
```

Package arXiv-style source files:

```bash
uv run python scripts/package_arxiv_submission.py
```

Export paper tables:

```bash
uv run python scripts/export_paper_assets.py --results-dir results/<run_id>
```

Check publication readiness:

```bash
uv run python scripts/check_publication_readiness.py --results-dir results/<run_id>
```

Estimate prompt counts needed for a target confidence interval width:

```bash
uv run python scripts/plan_ci_power.py --results-dir results/<run_id> --target-ci-width 0.08
```

Export a small blinded human-audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

Run the optional Qwen 32B public-suite follow-up after the primary 14B/7B workflow passes:

```bash
bash scripts/run_qwen32b_followup.sh
```

## Artifact Contract

Every run writes:

- `config.resolved.yaml`: fully resolved config
- `environment.json`: Python/platform/package/device metadata
- `manifest.json`: run metadata, git commit, dirty-tree state, model config, model device map, prompt counts, full policy configs, policy labels, seeds, and expected generation count
- `prompts.jsonl`: raw prompt fields, rendered chat text, prompt hashes, token IDs, tokenizer offsets, and token-role spans
- `generations.jsonl`: raw prompt metadata, generated text, and per-example metrics
- `metrics.json`: aggregate suite/policy metrics, policy-level safety-vs-capability contrasts, and prompt-clustered intervals
- `paper/generated/<run>/main_results_table.md`: paper-ready summary table with policy-level SSEI confidence intervals
- `paper/generated/<run>/main_results_table.tex`: LaTeX version of the paper-ready summary table
- `paper/generated/<run>/suite_level_effects_table.md`: suite-level effect table with paired CIs
- `paper/generated/<run>/suite_level_effects_table.tex`: LaTeX version of the suite-level effect table
- `cache_stats.parquet`: retained/evicted cache-token stats by policy application, including layer count and role-level retained/evicted token counts
- `figures/*.png`, `figures/*.svg`, and `figures/*.pdf`: generated by `scripts/make_figures.py`
- `figures/*.csv` and `figures/manifest.json`: figure source data plus SHA256 hashes for every plotted artifact
- `data/processed/<suite>.manifest.json`: processed prompt-suite record counts, prompt IDs, SHA256 hashes, and HF dataset revisions when applicable

Mock-model runs are for engineering smoke tests only. They must not be used as research evidence.
Tiny-model runs are also plumbing checks only. The readiness script rejects mock, tiny, dirty, smoke, inactive-compression, incomplete generation matrices, and missing prompt provenance by default unless explicit override flags are passed.

## Experiment Axes

The core sweep varies:

- model
- prompt suite
- cache policy
- compression budget
- prompt-suite order, when configs explicitly vary it
- seed

Implemented cache policies:

- `none`: uncompressed baseline
- `sliding_window`: keep last `N` cached tokens
- `sink_recent`: keep first `S` plus last `N` cached tokens
- `random_matched`: random eviction matched to the same budget
- `attention_h2o`: keep sink/recent tokens plus high-attention historical tokens when attention scores are available
- `kv_int8_sim`: symmetric per-tensor int8 quantize/dequantize simulation
- `kv_int4_sim`: symmetric per-tensor int4 quantize/dequantize simulation
- `policy_pinned`: mitigation policy that protects configured token roles, currently system-role spans, while evicting other tokens

For causal diagnostics, `patch_from_baseline` supports role-derived token selection, for example patching `token_roles: [system]` and comparing it against `token_roles: [user]` with `match_token_count_to_roles: [system]`. Hard-coded token indices are kept only for low-level debugging.

## Paper And Visuals

The manuscript lives in `paper/latex/main.tex` and builds to `paper/build/cache_mediated_safety_erasure.pdf`. The default format is an arXiv-friendly ML preprint because the target venue is not fixed. The planned paper visuals are documented in `paper/visuals.md`: cache-state fingerprints, safety-capability phase portraits, restoration flow diagrams, prompt-level effect constellations, and a safety-state atlas. These are designed to show structured cache-state patterns rather than only scatterplots and bar charts.

## Safety And Data Policy

This repository is for safety evaluation. The built-in prompt suites intentionally avoid procedural harmful details. Publication-quality runs should use open public datasets through `scripts/prepare_data.py` or documented dataset ingestion configs, and every dataset source must be logged in the resolved config.

Do not use closed-source model judges or paid endpoints. Use local metrics and open guard/classifier models only.
