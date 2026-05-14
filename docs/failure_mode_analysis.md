# Why Some Models Don't Show the Pattern

The 9-model panel shows positive Selective Safety Erasure Index (SSEI) in 4
families (Llama, OLMo, Phi, Qwen) but ~zero or negative SSEI in three models:

| Model | Architecture | Top SSEI | Why we think it doesn't fit |
| --- | --- | --- | --- |
| Gemma-2-9B-IT | Interleaved local (4K SWA) + global attention | ≈ 0 | Half its layers are pre-trained on local sliding-window attention, so the safety reasoning circuit does not depend on a long global KV cache. Further cache eviction is "in distribution" for the local layers. |
| Mistral-7B-Instruct-v0.3 | All-layer sliding-window attention (4K SWA) | +0.009 (sub-threshold) | Even more aggressively local than Gemma; the model is *natively* trained against a 4K-token sliding window. Eviction is the same regime it always operated in. Mistral 7B v0.3 also has no built-in moderation per the original model card, so the absolute safety scores are low to begin with. |
| GPT-OSS-20B | MoE transformer (~3.6B active params); harmony chain-of-thought format | -0.068 (negative) | Trained on the harmony response format and "instruction hierarchy"; degrades into a single uniform output mode when its cache is compressed (safety_score = 0.554 on every treatment policy vs 0.129 baseline). This is a degenerate "stuck" state from out-of-distribution input formatting, not robust safety. The fixed 0.554 score across all eviction policies is the smoking gun for stuck behavior. |

The five models that DO show selective safety erasure (Qwen3-9B, Phi-4,
OLMo-3, Llama-3.1, Qwen2.5-Instruct) are all dense transformers with vanilla
full attention. Phi-4 and Qwen3-9B in particular were never trained on
truncated-attention regimes and have safety reasoning that appears to live in
the cache state of system-role tokens.

## A unifying hypothesis

**Selective safety erasure scales with how much the model relies on a long
global KV cache for its safety reasoning.** Models that:

1. Use full global attention at every layer (Qwen, Llama, Phi, OLMo) → strong SSEI
2. Mix global with local attention (Gemma 2) → near-zero SSEI
3. Use only local sliding-window attention (Mistral) → effect at the edge of detectability

Models in a different regime entirely:

- GPT-OSS-20B's MoE + harmony stack does not "gracefully degrade" under cache
  pressure — it collapses to a single output mode. Whether that mode is
  conventionally safer than the baseline depends on the suite, but the
  uniform safety score (identical on every treatment) is incompatible with
  the hypothesis that cache state encodes a *modulated* safety signal.

This is consistent with the registered causal-localization protocol from
`RESEARCH.md` Phase 4: if safety degradation tracks the eviction of
system-role KV state, then models whose layers do not attend to those tokens
at long range — because they are constrained to local windows — should not
show the effect. They do not. The Phase 4 causal-patching runs in progress
(Qwen3-9B, then Phi-4) will test the converse: that *patching back* the
system-role cache state into a compressed run restores safety in the models
that *do* show the pattern.

## Caveats

1. n = 3 "non-conforming" models is too few to test the architectural
   hypothesis statistically. Treat this as exploratory.
2. Some architectural features bundle together with training-data choices we
   cannot factor cleanly: Gemma 2 was distilled from a much larger teacher;
   GPT-OSS uses harmony plus deliberative alignment; Mistral 7B v0.3 has no
   safety post-training at all. Any of these could also explain the failure
   mode.
3. The GPT-OSS-20B "stuck mode" deserves a deeper diagnostic — it may break
   the panel's blinded-judge protocol because the judge sees nearly identical
   outputs across every treatment.

## Sources

- [Gemma 2 technical report — Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118)
- [Gemma 2 architecture deep dive with PyTorch](https://amaarora.github.io/posts/2024-07-07%20Gemma.html)
- [gpt-oss-120b & gpt-oss-20b Model Card (OpenAI)](https://openai.com/index/gpt-oss-model-card/)
- [gpt-oss-120b & gpt-oss-20b model card (arXiv)](https://arxiv.org/abs/2508.10925)
- [Mistral 7B paper](https://arxiv.org/abs/2310.06825)
- [Mistral 7B HuggingFace docs](https://huggingface.co/docs/transformers/en/model_doc/mistral)
