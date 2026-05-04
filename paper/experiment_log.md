# Experiment Log

Use this file to summarize meaningful runs after aggregation.

Required fields for each entry:

- date
- commit hash
- config path
- run id
- machine / GPU
- model
- prompt suites
- cache policies
- main metrics
- decision: keep, rerun, discard, or extend

## 2026-05-03 Local Plumbing Runs

- date: 2026-05-03
- commit hash: local dirty tree during development; do not cite
- config path: `configs/experiments/tiny_hf_smoke.yaml`
- run id: `tiny_hf_guard_smoke_20260503`
- machine / GPU: local development environment
- model: `sshleifer/tiny-gpt2`
- prompt suites: `capability_smoke`
- cache policies: `none`, `sliding_window`, `kv_int8_sim`
- main metrics: not meaningful; artifact/readiness plumbing produced `generations.jsonl`, `metrics.json`, `cache_stats.parquet`, and figures with explicit tiny/smoke override flags
- decision: discard as evidence; keep only as plumbing validation

## 2026-05-04 H200 Launcher

- date: 2026-05-04
- commit hash: `d3670cc`
- config path: `scripts/run_h200_sweep.sh`
- run id: pending; launcher waits for `h200_qwen_full_sweep` and `h200_causal_patch_qwen7b`
- machine / GPU: UIUC H200 notebook
- model: pending; Qwen2.5 sweep has not started
- prompt suites: pending; public-suite preparation has not started in the launcher
- cache policies: pending; sweep has not started
- main metrics: no empirical metrics yet; `scripts/wait_and_run_h200_sweep.sh` passed ruff and the full test suite, then entered the GPU gate because the H200 was saturated
- decision: keep launcher waiting; do not cite as evidence

## 2026-05-04 H200 Hidden-Context Blocker

- date: 2026-05-04
- commit hash: `a9a310a`
- status command: `uv run python scripts/report_h200_status.py`
- machine / GPU: UIUC H200 notebook
- result: launcher still waiting; no experiment process running; expected result and audit artifacts missing
- blocker: H200 reported approximately 142 GB of 143 GB memory in use with high utilization, while `nvidia-smi --query-compute-apps` and `nvidia-smi pmon` showed no visible compute process
- decision: treat as infrastructure/allocation blocker, not experiment evidence; keep the launcher waiting and release or restart the notebook allocation externally if the context remains hidden

## 2026-05-04 H200 Blocker Refresh

- date: 2026-05-04
- commit hash: `e068362`
- status command: `uv run python scripts/report_h200_status.py --output-json logs/h200/h200_status_latest.json --output-md logs/h200/h200_status_latest.md`
- machine / GPU: UIUC H200 notebook
- result: launcher still waiting; no experiment process running; expected primary, causal, generated-paper, claim-assessment, and human-audit artifacts are still missing
- blocker: H200 memory has dropped to `5422/143771 MiB`, but utilization remains about `67%` with no visible compute apps, no accounted apps, no local `/proc/*/fd` NVIDIA holders, and `nvidia-smi -q -d PIDS` reporting `Processes : None`
- wait history: `43` samples over `210.1` minutes; latest gate block window is `utilization` for `15` samples over `70.0` minutes
- support artifact: `logs/h200/h200_support_bundle_latest.tar.gz` contains only infrastructure diagnostics and explicitly excludes model generations and paper evidence
- decision: keep the launcher waiting; escalate the generated admin report/support bundle or release/restart the notebook allocation externally; do not cite this as evidence

## 2026-05-04 H200 Primary Sweep Active

- date: 2026-05-04
- commit hash: `2e22cba` on the H200 checkout; local publication-gate hardening has advanced to `7653ede` and must be applied only after the active run completes
- config path: `configs/experiments/h200_qwen_full_sweep.yaml`
- run id: `h200_qwen_full_sweep`
- machine / GPU: UIUC H200 notebook
- model: `Qwen/Qwen2.5-14B-Instruct`
- prompt suites: prepared public suites plus built-in smoke suites; primary generation matrix expected count is `25299`
- cache policies: registered primary sweep policies from the resolved H200 config
- main metrics: no primary `metrics.json` yet; generation is in progress with `18454 / 25299` rows observed, and no causal diagnostic metrics have been produced
- decision: keep the active launcher running; do not pull code or start a duplicate process on H200; after completion, fetch results locally and reaggregate with the latest clean `master` gates before making paper claims
