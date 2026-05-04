# Human Audit Samples

Use `scripts/export_human_audit_sample.py` after a real run to create a stratified audit sheet:

```bash
uv run python scripts/export_human_audit_sample.py --results-dir results/<run_id>
```

The script writes a blinded CSV for annotation and a private key JSONL that maps `audit_id` back to model, suite, policy, prompt id, seed, hidden/system text, and automated metrics. By default, a direct script call redacts raw system and hidden-system text and includes only short digests. The H200 launcher passes `--include-hidden-reference` by default because `human_system_leakage` labels require annotators to compare the response against the hidden reference while still hiding model and policy identity. Do not treat automated refusal-string scores as final unsafe-compliance labels without a blinded human audit or a documented open local judge.

Use [labeling_guide.md](labeling_guide.md) when completing the blinded CSV. The export samples prompt-matched baseline/treatment pairs so the aggregation can compute paired human-audit deltas. By default, the exporter prioritizes pairs with the largest automated safety, leakage, or over-refusal shifts so human effort concentrates on claim-relevant examples. Add `--strategy random` for unbiased spot checks.

To create duplicate blinded sheets for inter-annotator agreement, add:

```bash
uv run python scripts/export_human_audit_sample.py \
  --results-dir results/<run_id> \
  --annotator-template-count 2
```

This writes the standard blinded CSV plus `*_annotator_01.csv`, `*_annotator_02.csv`, and so on with prefilled `annotator_id` values. Aggregate the completed annotator files together with one `--audit-csv` argument per file.

After annotation, aggregate the completed sheet:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/<run_id>_audit_blinded.csv \
  --key-jsonl paper/audit/<run_id>_audit_key.jsonl \
  --output-dir paper/audit/<run_id>_summary
```

For the publication build, the expected summary directories are:

```bash
bash scripts/export_publication_audit_samples.sh
```

This regenerates leakage-capable publication audit templates from the fetched
primary and causal completed-run artifacts before annotation. Then aggregate and
validate both completed annotator file sets:

```bash
bash scripts/aggregate_publication_human_audits.sh
```

For one-off debugging, the equivalent per-run commands are:

```bash
uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/h200_qwen_full_sweep_audit_blinded_annotator_*.csv \
  --key-jsonl paper/audit/h200_qwen_full_sweep_audit_key.jsonl \
  --results-dir results/h200_qwen_full_sweep \
  --export-manifest paper/audit/h200_qwen_full_sweep_audit_export_manifest.json \
  --output-dir paper/audit/h200_qwen_full_sweep_summary

uv run python scripts/aggregate_human_audit.py \
  --audit-csv paper/audit/h200_causal_patch_qwen7b_audit_blinded_annotator_*.csv \
  --key-jsonl paper/audit/h200_causal_patch_qwen7b_audit_key.jsonl \
  --results-dir results/h200_causal_patch_qwen7b \
  --export-manifest paper/audit/h200_causal_patch_qwen7b_audit_export_manifest.json \
  --output-dir paper/audit/h200_causal_patch_qwen7b_summary
```

The completed CSV may include an optional `annotator_id` column. Multiple rows with the same `audit_id` are treated as multiple annotations only when they come from distinct annotator IDs. Duplicate `(audit_id, annotator_id)` rows are deduplicated, reported in the summary, and block publication readiness.

Accepted label values are `yes`, `no`, `true`, `false`, `1`, `0`, or blank. The aggregation writes:

- `human_audit_metrics.json`
- `human_audit_summary.json`
- `human_labels.jsonl`
- `human_audit_joined.csv`
- `human_audit_summary.md`
- `human_audit_summary_table.tex`
- `human_audit_deltas_table.tex`
- `audit_manifest.json`

The exporter also writes `<run_id>_audit_export_manifest.json`, and the aggregation manifest records its hash so publication checks can verify the sampling strategy, seed, hidden-reference mode, and annotator-template count. The JSON summary reports publication-facing label rates at the item level after majority consensus across annotators; unresolved ties are listed and block readiness. It also keeps annotation-level label rates as diagnostics, includes Wilson confidence intervals, automated-vs-human confusion matrices, pairwise inter-annotator agreement across distinct annotators, duplicate-annotation diagnostics, and paired baseline-vs-policy deltas when the same `prompt_id` and `seed` appear under `none` and a treatment policy.

Before using the audit in the paper, run:

```bash
uv run python scripts/check_human_audit_readiness.py \
  --summary-json paper/audit/<run_id>_summary/human_audit_summary.json \
  --audit-manifest paper/audit/<run_id>_summary/audit_manifest.json \
  --results-dir results/<run_id> \
  --require-baseline-deltas \
  --require-result-source-match
```

By default this requires complete annotations, no unknown audit IDs, no duplicate `(audit_id, annotator_id)` rows, at least two distinct annotators, full multi-annotator coverage, non-empty core safety labels, no unresolved consensus ties, hidden/system reference context for leakage labels, paired treatment-minus-baseline deltas, and at least one inter-annotator pair for each core label. Use `--allow-single-annotator` only for a clearly documented draft or ablation.

## Open Local Judge Alternative

If human annotators are not available, use a documented open local judge rather
than filling the human-label columns by hand. The wrapper below runs two
deterministic prompt variants and marks every row with
`annotation_source_type=open_local_judge`, the judge model id, and the judge
prompt hash:

```bash
bash scripts/run_publication_open_judge_audits.sh
bash scripts/aggregate_publication_human_audits.sh
```

These labels are audit-support labels, not human labels. The paper and claim
assessment must report the source as an open local judge, and any final claim
must remain limited by that weaker validation source.
