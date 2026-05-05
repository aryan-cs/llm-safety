# Testing Cache-Mediated Safety Erasure

Repository: <https://github.com/aryan-cs/kv-cache-safety>

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
  --config configs/experiments/h200_qwen_full_sweep.yaml \
  --run-id h200_qwen_full_sweep \
  --resume
```

Run the primary H200 workflow:

```bash
bash scripts/run_h200_sweep.sh
```

The primary workflow defaults to `PUBLIC_PROMPT_LIMIT=650`, one deterministic seed, `AUDIT_PER_SUITE_POLICY=10`, `AUDIT_ANNOTATOR_TEMPLATE_COUNT=2`, and `AUDIT_INCLUDE_HIDDEN_REFERENCE=1`. The public refusal suite combines AdvBench with JailbreakBench harmful behaviors, and the public system-leakage suite uses a prompt-injection benchmark, so both safety and leakage prompt counts clear the 600-cluster paper-readiness threshold. This keeps runtime lower than repeated deterministic seeds while targeting prompt-cluster counts needed for narrow confidence intervals and producing duplicate leakage-capable blinded audit templates for inter-annotator agreement. For a cheaper pilot, run `PUBLIC_PROMPT_LIMIT=200 AUDIT_PER_SUITE_POLICY=3 AUDIT_ANNOTATOR_TEMPLATE_COUNT=0 bash scripts/run_h200_sweep.sh`.

If the H200 GPU is busy, queue the sweep behind an availability gate from the H200 checkout:

```bash
setsid -f bash scripts/wait_and_run_h200_sweep.sh </dev/null > logs/h200/launcher.out 2>&1
```

The launcher refuses to run outside `/home/aryang9/sandbox/llm-safety`, pulls `master`, checks that the tree is clean, runs the CPU-only test suite, waits until `nvidia-smi` is below `MAX_USED_MIB=20000` and `MAX_UTIL_PCT=20`, then pulls and validates `master` again before starting the selected sweep. Override `SWEEP_SCRIPT=scripts/run_h200_ci_extension.sh` or `SWEEP_SCRIPT=scripts/run_qwen32b_followup.sh` only after the earlier registered stage has passed.

Summarize the H200 wait/run state without changing it:

```bash
uv run python scripts/report_h200_status.py \
  --output-json logs/h200/h200_status_latest.json \
  --output-md logs/h200/h200_status_latest.md
uv run python scripts/write_h200_admin_report.py \
  --status-json logs/h200/h200_status_latest.json \
  --output-md logs/h200/h200_admin_report.md
uv run python scripts/package_h200_support_bundle.py \
  --output logs/h200/h200_support_bundle_latest.tar.gz
```

If the status report says `Hidden GPU context likely: true`, `nvidia-smi` is showing high memory or utilization without a visible compute process inside the notebook namespace. Treat that as an infrastructure/allocation blocker, not an experiment result. Do not kill the waiting launcher, and do not run `nvidia-smi --gpu-reset` on shared infrastructure unless an administrator explicitly authorizes it. First preserve the status report, then release or restart the H200 notebook allocation from the Illinois Computes/Jupyter UI if this is your session. After reconnecting, return to `/home/aryang9/sandbox/llm-safety` and rerun `uv run python scripts/report_h200_status.py`; the existing launcher should continue waiting or start automatically once the GPU gate clears. If the launcher process is gone, restart it with the `setsid -f bash scripts/wait_and_run_h200_sweep.sh ...` command above from a clean `master` checkout.

From the local checkout, copy the latest H200 status and admin-support report into `logs/h200/`:

```bash
bash scripts/fetch_h200_reports.sh
uv run python scripts/package_h200_support_bundle.py \
  --output logs/h200/h200_support_bundle_latest.tar.gz
```

If the notebook allocation may expire before the current run finishes, snapshot
the partial run from the local checkout. The snapshot includes the run directory,
checksums, the observed generation count, the expected generation count, and the
remote git commit. To include bounded H200 launcher/finalizer log tails, set
`H200_SNAPSHOT_LOG_FILE_LIMIT=<n>`:

```bash
RUN_ID=h200_causal_patch_qwen7b bash scripts/snapshot_h200_run.sh
```

After restarting the H200 allocation, restore the latest snapshot and resume
from the H200 checkout:

```bash
SNAPSHOT_DIR=snapshots/h200/latest bash scripts/restore_h200_snapshot.sh
ssh uiuc-h200
cd /home/aryang9/sandbox/llm-safety
UV_CACHE_DIR=.cache/uv uv run python scripts/run_experiment.py \
  --config configs/experiments/h200_causal_patch_qwen7b.yaml \
  --run-id h200_causal_patch_qwen7b \
  --resume
```

Resume is fail-closed. It preserves the original `manifest.json`,
`config.resolved.yaml`, and `environment.json`; writes
`manifest.resume.<timestamp>.json`, `config.resume.<timestamp>.yaml`, and
`environment.resume.<timestamp>.json`; quarantines a truncated
`generations.jsonl` tail; recovers a valid `cache_stats.parquet.tmp`; and refuses
to append if the model, prompt suites, policy matrix, seeds, expected row count,
or git commit do not match the original run. Use
`ALLOW_RESUME_GIT_MISMATCH=1` only after confirming the experiment matrix is
unchanged and the newer code is a resume-only compatibility patch.

If the H200 allocation expires and is not restarted for a while, keep the Mac
working on a bounded Qwen 3B diagnostic instead of attempting the H200-only 14B
or 32B sweeps locally:

```bash
bash scripts/run_mac_fallback.sh
```

This fallback is intentionally conservative for a 24 GB M4 Pro: it checks macOS
and the PyTorch MPS backend, requires at least 22 GiB unified memory, uses
`configs/experiments/mac_qwen3b_causal_fallback.yaml`, writes artifacts to
`results/mac_qwen3b_causal_fallback`, and isolates model downloads under
`.cache/mac_fallback`. By default it deletes `.cache/mac_fallback/huggingface`
and `.cache/mac_fallback/torch` when the script exits, even on failure or
interruption. It is a fallback diagnostic, not a replacement for the registered
H200 Qwen 14B CI extension or any Qwen 32B follow-up. Do not resume `h200_*`
run ids on the Mac; Mac fallback runs use separate `mac_*` run ids so they
cannot contaminate H200 evidence.

To clean model caches manually, dry-run first and then delete only repo-local
model caches:

```bash
bash scripts/cleanup_local_model_caches.sh
bash scripts/cleanup_local_model_caches.sh --yes
```

Run the prompt-count extension for narrower confidence intervals after the primary pilot identifies viable effects:

```bash
SWEEP_SCRIPT=scripts/run_h200_ci_extension.sh \
bash scripts/wait_and_run_h200_sweep.sh
```

The CI extension uses `CI_PROMPT_LIMIT=650` by default and focuses on fewer policies so prompt-cluster counts, not repeated deterministic seeds, do the statistical work. Run it through the guarded launcher so the H200 checkout syncs `master`, revalidates after the GPU gate, and holds the launcher lock. Override with `CI_PROMPT_LIMIT=<n>` or `TARGET_CI_WIDTH=<width>` if needed.

Initialize or update the H200 checkout under the authorized notebook folder:

```bash
bash scripts/setup_h200_remote.sh
```

That wrapper runs `scripts/bootstrap_h200.sh` over `ssh uiuc-h200` and refuses to operate outside `/home/aryang9/sandbox/llm-safety`.

Preflight the H200 configs without launching a sweep:

```bash
uv run python scripts/preflight_h200.py \
  --config configs/experiments/h200_qwen_full_sweep.yaml \
  --config configs/experiments/h200_qwen14b_ci_extension.yaml \
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

The draft PDF is refreshed in both `paper/build/cache_mediated_safety_erasure.pdf`
and `paper/cache_mediated_safety_erasure.pdf`. It remains a pre-results draft until
the publication-readiness gates pass.

Package arXiv-style source files:

```bash
uv run python scripts/package_arxiv_submission.py
```

After the guarded H200 launcher completes, fetch raw result evidence into the
local checkout with checksum verification, then regenerate publication assets
locally from the current clean checkout:

```bash
bash scripts/fetch_h200_results.sh results/h200_qwen_full_sweep results/h200_causal_patch_qwen7b
bash scripts/prepare_after_h200_fetch.sh
```

This writes remote and local artifact manifests in `logs/h200/`, compares hashes
and byte counts, and refuses paths outside `results/`, `paper/generated/`, and
`paper/audit/`. It does not pull code or start jobs on the H200. The default
fetch also includes remote diagnostic and audit-export files when the full
launcher has finished. For the publication preparation handoff, use the explicit
primary and causal result directories above; `prepare_after_h200_fetch.sh`
requires that passing checksum manifest before regenerating publication-valid
audit sheets locally.

After the post-causal H200 finalizer has run, fetch the finalized handoff set
instead. This includes optional merged-primary CI-extension artifacts,
open-local-judge audit CSVs, audit summaries, generated claim files, and active
paper assets when they exist:

```bash
FETCH_H200_FINALIZED=1 bash scripts/fetch_h200_results.sh
```

If you need only an intermediate run or want to archive remote-generated debug
artifacts, pass explicit artifact paths, for example:

```bash
bash scripts/fetch_h200_results.sh results/h200_qwen_full_sweep paper/generated/h200_qwen_full_sweep
```

To include all remote-generated H200 paper/debug artifacts in the default fetch,
set `FETCH_H200_REMOTE_GENERATED=1`; do this only when you are intentionally
archiving those files rather than preparing final paper assets.

Regenerate publication-valid audit sheets from already fetched completed runs:

```bash
bash scripts/prepare_after_h200_fetch.sh
```

This reaggregates the fetched raw results using the current local checkout,
regenerates paper figures/tables and CI planning files, runs readiness checks,
then exports publication-valid blinded audit sheets. It regenerates
`metrics.json` from the fetched `generations.jsonl`, creates duplicate annotator
templates, and includes the hidden/system references needed for leakage labels
while leaving model and policy identities blinded.

Then rebuild all paper artifacts from recorded results:

```bash
bash scripts/build_publication_artifacts.sh
```

This command regenerates aggregate metrics, figures, paper tables, CI planning files, the evidence-gated claim assessment, readiness checks, the readable PDF, and the arXiv source bundle. It fails if the required real result artifacts, completed human-audit summaries, or cache-mediated-safety-erasure claim gates are missing.
Optional Qwen 32B follow-up artifacts are packaged only when that follow-up was
rebuilt and passed readiness in the same publication build, so stale optional
generated directories cannot enter the final arXiv bundle by accident.

To see the next fail-closed step without running the full rebuild:

```bash
uv run python scripts/post_h200_next_steps.py
```

This checklist reports whether the next legitimate action is completing H200 results, completing human audits, assessing claims, or building the final publication bundle.

If early results are weak or mixed, do not silently search for a better framing. Generate an evidence-gated follow-up plan from the completed claim assessment:

```bash
uv run python scripts/plan_registered_followups.py \
  --claim-assessment paper/generated/preliminary_claim_assessment/claim_assessment.json \
  --primary-ci-power results/h200_qwen_full_sweep/ci_power.json \
  --causal-ci-power results/h200_causal_patch_qwen7b/ci_power.json \
  --output-dir paper/generated/preliminary_followup_plan
```

The follow-up planner records whether the next legitimate step is a causal extension, a powered selectivity extension, a human-audit repair, a model-family replication, or a clearly preregistered pivot. It preserves the novelty search while preventing post-hoc threshold changes or unregistered suite/policy additions from becoming the main paper claim.

Human-audit summaries must also pass the audit-readiness gate:

```bash
uv run python scripts/check_human_audit_readiness.py \
  --summary-json paper/audit/h200_qwen_full_sweep_summary/human_audit_summary.json \
  --audit-manifest paper/audit/h200_qwen_full_sweep_summary/audit_manifest.json \
  --results-dir results/h200_qwen_full_sweep \
  --require-baseline-deltas \
  --require-result-source-match
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

Summarize publication blockers without mutating artifacts:

```bash
uv run python scripts/report_publication_status.py
```

Export a small blinded human-audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The default audit export uses prompt-matched baseline/treatment pairs and prioritizes the largest automated safety, leakage, or benign-over-refusal shifts. Use `--strategy random` for an unbiased spot-check sample.
Add `--annotator-template-count 2` to write duplicate blinded CSVs with prefilled annotator IDs for inter-annotator agreement.

Aggregate completed human-audit labels:

```bash
bash scripts/aggregate_publication_human_audits.sh
```

This aggregates the completed primary and causal annotator CSVs, validates
inter-annotator coverage, checks leakage-reference context, and verifies that
the audit summaries still match the fetched result artifacts and audit-export
manifests.
If both human annotation files and open-local-judge files are present, set
`AUDIT_SOURCE=human` or `AUDIT_SOURCE=open_judge`; the wrapper refuses to choose
between mixed sources implicitly.

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
- `paper/generated/claim_assessment/`: H1/H2/H3 claim-ladder assessment generated from primary and causal metrics
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
