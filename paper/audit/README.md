# Human Audit Samples

Use `scripts/export_human_audit_sample.py` after a real run to create a stratified audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The script writes a blinded CSV for annotation and a private key JSONL that maps `audit_id` back to model, suite, policy, prompt id, seed, and automated metrics. Do not treat automated refusal-string scores as final unsafe-compliance labels without a small blinded human audit or a documented open local judge.

Use [labeling_guide.md](labeling_guide.md) when completing the blinded CSV. The export samples prompt-matched baseline/treatment pairs so the aggregation can compute paired human-audit deltas. By default, the exporter prioritizes pairs with the largest automated safety, leakage, or over-refusal shifts so human effort concentrates on claim-relevant examples. Add `--strategy random` for unbiased spot checks.

After annotation, aggregate the completed sheet:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/<run_id>_audit_blinded.csv \
  --key-jsonl paper/audit/<run_id>_audit_key.jsonl \
  --output-dir paper/audit/<run_id>_summary
```

For the publication build, the expected summary directories are:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/h200_qwen_full_sweep_audit_blinded.csv \
  --key-jsonl paper/audit/h200_qwen_full_sweep_audit_key.jsonl \
  --results-dir results/h200_qwen_full_sweep \
  --output-dir paper/audit/h200_qwen_full_sweep_summary

uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/h200_causal_patch_qwen7b_audit_blinded.csv \
  --key-jsonl paper/audit/h200_causal_patch_qwen7b_audit_key.jsonl \
  --results-dir results/h200_causal_patch_qwen7b \
  --output-dir paper/audit/h200_causal_patch_qwen7b_summary
```

The completed CSV may include an optional `annotator_id` column. Multiple rows with the same `audit_id` are treated as multiple annotations and are used to compute pairwise agreement and Cohen's kappa for each boolean label.

Accepted label values are `yes`, `no`, `true`, `false`, `1`, `0`, or blank. The aggregation writes:

- `human_audit_metrics.json`
- `human_audit_summary.json`
- `human_labels.jsonl`
- `human_audit_joined.csv`
- `human_audit_summary.md`
- `human_audit_summary_table.tex`
- `human_audit_deltas_table.tex`
- `audit_manifest.json`

The JSON summary includes Wilson confidence intervals for label rates, automated-vs-human confusion matrices, pairwise inter-annotator agreement, and paired baseline-vs-policy deltas when the same `prompt_id`, `seed`, and annotator appear under `none` and a treatment policy.

Before using the audit in the paper, run:

```bash
uv run python scripts/check_human_audit_readiness.py \
  --summary-json paper/audit/<run_id>_summary/human_audit_summary.json \
  --require-baseline-deltas
```

By default this requires complete annotations, no unknown audit IDs, non-empty core safety labels, paired treatment-minus-baseline deltas, and at least one inter-annotator pair for each core label. Use `--allow-single-annotator` only for a clearly documented draft or ablation.
