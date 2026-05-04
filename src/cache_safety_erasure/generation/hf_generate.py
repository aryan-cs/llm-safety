from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cache_safety_erasure.cache_policies.base import CachePolicyDecision
from cache_safety_erasure.cache_policies.cache_utils import (
    cache_l2_norm,
    cache_layer_count,
    cache_seq_len,
    maybe_from_legacy_cache,
)
from cache_safety_erasure.config import GenerationConfig
from cache_safety_erasure.evals.prompt_record import PromptRecord
from cache_safety_erasure.evals.rendering import build_chat_text, token_roles_for_prompt
from cache_safety_erasure.generation.cache_patching import (
    patch_cache_from_baseline,
    resolve_patch_from_baseline_spec,
)


@dataclass
class GenerationResult:
    text: str
    cache_decisions: list[CachePolicyDecision]


def hf_generate(
    *,
    model: Any,
    tokenizer: Any,
    prompt: PromptRecord,
    policy: Any,
    generation_config: GenerationConfig,
    patch_from_baseline: dict[str, Any] | None = None,
) -> GenerationResult:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Hugging Face generation.") from exc

    text = build_chat_text(tokenizer, prompt)
    encoded = tokenizer(text, return_tensors="pt")
    device = _model_device(model)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    token_roles = token_roles_for_prompt(tokenizer, prompt, input_ids)
    cache_decisions: list[CachePolicyDecision] = []
    generated_ids: list[int] = []

    with torch.inference_mode():
        baseline_full_prompt_past = None
        if patch_from_baseline:
            baseline_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=False,
                return_dict=True,
            )
            baseline_full_prompt_past = baseline_outputs.past_key_values
        if int(input_ids.shape[-1]) > 1:
            prefill_ids = input_ids[:, :-1]
            last_prompt_token = input_ids[:, -1:]
            prefill_mask = attention_mask[:, :-1] if attention_mask is not None else None
            outputs = model(
                input_ids=prefill_ids,
                attention_mask=prefill_mask,
                use_cache=True,
                output_attentions=generation_config.capture_attentions,
                return_dict=True,
            )
            baseline_prefill_past = outputs.past_key_values
            past = outputs.past_key_values
            past, decision = policy.apply(
                past,
                step=0,
                token_roles=token_roles[:-1],
                attention_scores=getattr(outputs, "attentions", None),
            )
            if patch_from_baseline:
                past, patch_metadata = patch_from_baseline_cache(
                    past,
                    baseline_prefill_past,
                    patch_from_baseline,
                    token_roles=token_roles[:-1],
                )
                decision.metadata.update(patch_metadata)
            cache_decisions.append(decision)
            outputs = _forward_one_token(
                model=model,
                token_id=last_prompt_token,
                past=past,
                absolute_position=int(input_ids.shape[-1]) - 1,
                output_attentions=generation_config.capture_attentions,
            )
            past = outputs.past_key_values
            past, decision = policy.apply(
                past,
                step=1,
                token_roles=token_roles,
                attention_scores=getattr(outputs, "attentions", None),
            )
            if patch_from_baseline:
                past, patch_metadata = patch_from_baseline_cache(
                    past,
                    baseline_full_prompt_past
                    if baseline_full_prompt_past is not None
                    else baseline_prefill_past,
                    patch_from_baseline,
                    token_roles=token_roles,
                )
                decision.metadata.update(patch_metadata)
            cache_decisions.append(decision)
            absolute_position = int(input_ids.shape[-1])
            decode_step_start = 2
        else:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_attentions=generation_config.capture_attentions,
                return_dict=True,
            )
            baseline_prefill_past = outputs.past_key_values
            past = outputs.past_key_values
            past, decision = policy.apply(
                past,
                step=0,
                token_roles=token_roles,
                attention_scores=getattr(outputs, "attentions", None),
            )
            if patch_from_baseline:
                past, patch_metadata = patch_from_baseline_cache(
                    past,
                    baseline_full_prompt_past
                    if baseline_full_prompt_past is not None
                    else baseline_prefill_past,
                    patch_from_baseline,
                    token_roles=token_roles,
                )
                decision.metadata.update(patch_metadata)
            cache_decisions.append(decision)
            absolute_position = int(input_ids.shape[-1])
            decode_step_start = 1
        next_token = _sample_next_token(outputs.logits[:, -1, :], generation_config)

        for step in range(decode_step_start, decode_step_start + generation_config.max_new_tokens):
            token_id = int(next_token.item())
            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)

            outputs = _forward_one_token(
                model=model,
                token_id=next_token.reshape(1, 1),
                past=past,
                absolute_position=absolute_position,
                output_attentions=generation_config.capture_attentions,
            )
            absolute_position += 1

            past = outputs.past_key_values
            extended_roles = token_roles + (["generated"] * len(generated_ids))
            past, decision = policy.apply(
                past,
                step=step,
                token_roles=extended_roles,
                attention_scores=getattr(outputs, "attentions", None),
            )
            cache_decisions.append(decision)
            if patch_from_baseline and baseline_full_prompt_past is not None:
                past, patch_metadata = patch_from_baseline_cache(
                    past,
                    baseline_full_prompt_past,
                    patch_from_baseline,
                    token_roles=token_roles,
                )
                decision.metadata.update(patch_metadata)
            next_token = _sample_next_token(outputs.logits[:, -1, :], generation_config)

            if _has_stop_string(tokenizer, generated_ids, generation_config.stop_strings):
                break

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    _ = cache_layer_count  # imported for downstream callers and import validation.
    return GenerationResult(text=decoded, cache_decisions=cache_decisions)


def patch_from_baseline_cache(
    past: Any,
    baseline_prefill_past: Any,
    patch_from_baseline: dict[str, Any],
    *,
    token_roles: list[str] | None,
) -> tuple[Any, dict[str, Any]]:
    resolved_patch, metadata = resolve_patch_from_baseline_spec(
        patch_from_baseline,
        token_roles=token_roles,
        target_cache=past,
        baseline_cache=baseline_prefill_past,
    )
    patched = patch_cache_from_baseline(
        past,
        baseline_prefill_past,
        layers=resolved_patch.get("layers"),
        heads=resolved_patch.get("heads"),
        token_indices=resolved_patch.get("token_indices"),
        components=resolved_patch.get("components"),
    )
    metadata["cache_l2_after_patch"] = cache_l2_norm(patched)
    return maybe_from_legacy_cache(patched, baseline_prefill_past), metadata


def _forward_one_token(
    *,
    model: Any,
    token_id: Any,
    past: Any,
    absolute_position: int,
    output_attentions: bool,
) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch is required for Hugging Face generation.") from exc

    device = token_id.device
    cache_len = cache_seq_len(past)
    kwargs = {
        "input_ids": token_id,
        "attention_mask": torch.ones((1, cache_len + 1), dtype=torch.long, device=device),
        "past_key_values": past,
        "use_cache": True,
        "output_attentions": output_attentions,
        "return_dict": True,
        "position_ids": torch.tensor([[absolute_position]], dtype=torch.long, device=device),
    }
    try:
        kwargs["cache_position"] = torch.tensor([absolute_position], dtype=torch.long, device=device)
        return model(**kwargs)
    except TypeError:
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def _model_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return "cpu"


def _sample_next_token(logits: Any, generation_config: GenerationConfig) -> Any:
    import torch

    if generation_config.do_sample and generation_config.temperature > 0:
        scaled = logits / generation_config.temperature
        if generation_config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative = probs.cumsum(dim=-1)
            mask = cumulative > generation_config.top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            probs = torch.softmax(sorted_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            return sorted_indices.gather(-1, sampled).squeeze(-1)
        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    return torch.argmax(logits, dim=-1)


def _has_stop_string(tokenizer: Any, generated_ids: list[int], stop_strings: tuple[str, ...]) -> bool:
    if not stop_strings:
        return False
    partial_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return any(stop in partial_text for stop in stop_strings)
