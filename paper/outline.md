# Paper Outline: Testing Cache-Mediated Safety Erasure

## Working Thesis

Safety/refusal behavior in aligned LLMs may depend on fragile cache-resident routing state. The paper earns the name "Cache-Mediated Safety Erasure" only if targeted cache preservation or restoration recovers refusal behavior more than matched non-policy controls.

## Target Contributions

1. Demonstrate or falsify selective safety degradation under cache eviction/quantization at compression levels where benign capability remains comparatively stable.
2. Localize degradation by token role and, where feasible, by layer/head.
3. Show causal restoration for length-preserving perturbations and policy-pinned mitigation for eviction perturbations.
4. Define matched negative controls so the effect is not just "more cache helps."

## Key Figures

1. Safety degradation vs capability degradation across cache policies and budgets.
2. Selective Safety Erasure Index by model and policy.
3. Token-span retention heatmaps for system/policy/user spans.
4. Causal patching restoration curves by layer/head/span.
5. Mitigation comparison: naive compression vs policy-pinned compression.

## Acceptance Criteria For A Paper Claim

- Effects reproduce on at least two open model families, or the claim is limited to the tested model family.
- Safety/refusal degradation is meaningfully larger than ordinary capability degradation.
- At least one causal patching experiment restores a safety metric without globally restoring the full cache and beats matched non-policy patch controls.
- Policy-pinned mitigation preserves safety at similar cache budgets more than naive sink/recent retention.
- All runs are reproducible from committed configs and scripts.
- Paper runs are produced from committed configs and a complete prompt x policy x seed generation matrix.
- A stratified human audit or documented open local judge checks the most important safety-compliance labels.
