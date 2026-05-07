from __future__ import annotations

import hashlib
from typing import Any

from cache_safety_erasure.evals.prompt_record import PromptRecord


def build_chat_text(tokenizer: Any, prompt: PromptRecord) -> str:
    messages = []
    if prompt.system:
        messages.append({"role": "system", "content": prompt.system})
    messages.append({"role": "user", "content": prompt.user})
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            if not (prompt.system and _is_unsupported_system_role_error(exc)):
                raise
            return tokenizer.apply_chat_template(
                _fold_system_into_user_messages(prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
    if prompt.system:
        return f"System: {prompt.system}\nUser: {prompt.user}\nAssistant:"
    return f"User: {prompt.user}\nAssistant:"


def rendered_prompt_manifest(tokenizer: Any | None, prompt: PromptRecord) -> dict[str, Any]:
    if tokenizer is None:
        rendered_text = _mock_rendered_text(prompt)
        return {
            "rendered_text": rendered_text,
            "rendered_text_sha256": hashlib.sha256(rendered_text.encode()).hexdigest(),
            "component_char_spans": raw_component_char_spans(prompt),
            "token_count": None,
            "token_ids": [],
            "token_offsets": [],
            "token_roles": [],
            "token_role_spans": [],
            "offset_mapping_available": False,
        }
    rendered_text = build_chat_text(tokenizer, prompt)
    try:
        tokenized = tokenizer(rendered_text, return_offsets_mapping=True)
        offsets = tokenized.get("offset_mapping")
    except (NotImplementedError, TypeError, ValueError):
        tokenized = tokenizer(rendered_text)
        offsets = None
    token_ids = [int(x) for x in tokenized["input_ids"]]
    component_spans = rendered_component_char_spans(rendered_text, prompt)
    if offsets is None:
        token_roles = _fallback_token_roles(tokenizer, prompt, len(token_ids))
        token_offsets: list[dict[str, int | str | None]] = [
            {"index": idx, "start": None, "end": None, "role": role}
            for idx, role in enumerate(token_roles)
        ]
        offset_available = False
    else:
        normalized_offsets = [(int(start), int(end)) for start, end in offsets]
        token_roles = [
            _role_for_token_span(start, end, component_spans) for start, end in normalized_offsets
        ]
        token_offsets = [
            {"index": idx, "start": start, "end": end, "role": token_roles[idx]}
            for idx, (start, end) in enumerate(normalized_offsets)
        ]
        offset_available = True
    return {
        "rendered_text": rendered_text,
        "rendered_text_sha256": hashlib.sha256(rendered_text.encode()).hexdigest(),
        "component_char_spans": component_spans,
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "token_offsets": token_offsets,
        "token_roles": token_roles,
        "token_role_spans": _merge_token_role_spans(token_roles),
        "offset_mapping_available": offset_available,
    }


def token_roles_for_prompt(tokenizer: Any, prompt: PromptRecord, input_ids: Any) -> list[str]:
    manifest = rendered_prompt_manifest(tokenizer, prompt)
    roles = list(manifest.get("token_roles", []))
    total = int(input_ids.shape[-1])
    if len(roles) == total:
        return roles
    if not roles:
        return _fallback_token_roles(tokenizer, prompt, total)
    if len(roles) > total:
        return roles[:total]
    return roles + (["unknown"] * (total - len(roles)))


def raw_component_char_spans(prompt: PromptRecord) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    cursor = 0
    if prompt.system:
        spans.append({"role": "system", "start": cursor, "end": cursor + len(prompt.system)})
        if prompt.hidden_system:
            hidden_start = prompt.system.find(prompt.hidden_system)
            if hidden_start >= 0:
                spans.append(
                    {
                        "role": "hidden_system",
                        "start": cursor + hidden_start,
                        "end": cursor + hidden_start + len(prompt.hidden_system),
                    }
                )
        cursor += len(prompt.system)
    if prompt.user:
        spans.append({"role": "user", "start": cursor, "end": cursor + len(prompt.user)})
    return spans


def rendered_component_char_spans(rendered_text: str, prompt: PromptRecord) -> list[dict[str, int | str]]:
    spans: list[dict[str, int | str]] = []
    if prompt.system:
        system_span = _find_text_span(rendered_text, prompt.system, "system", 0)
        if system_span:
            spans.append(system_span)
        if prompt.hidden_system:
            hidden_span = _find_text_span(rendered_text, prompt.hidden_system, "hidden_system", 0)
            if hidden_span:
                spans.append(hidden_span)
    if prompt.user:
        search_start = int(spans[0]["end"]) if spans else 0
        user_span = _find_text_span(rendered_text, prompt.user, "user", search_start)
        if user_span:
            spans.append(user_span)
    return sorted(spans, key=lambda span: (int(span["start"]), int(span["end"])))


def _find_text_span(
    haystack: str, needle: str, role: str, start_at: int
) -> dict[str, int | str] | None:
    start = haystack.find(needle, start_at)
    if start < 0:
        start = haystack.find(needle)
    if start < 0:
        return None
    return {"role": role, "start": start, "end": start + len(needle)}


def _role_for_token_span(
    token_start: int, token_end: int, component_spans: list[dict[str, int | str]]
) -> str:
    if token_end <= token_start:
        return "special"
    best_role = "template"
    best_overlap = 0
    for span in component_spans:
        role = str(span["role"])
        if role == "hidden_system":
            role = "system"
        overlap = max(
            0,
            min(token_end, int(span["end"])) - max(token_start, int(span["start"])),
        )
        if overlap > best_overlap:
            best_role = role
            best_overlap = overlap
    return best_role


def _merge_token_role_spans(token_roles: list[str]) -> list[dict[str, int | str]]:
    if not token_roles:
        return []
    spans: list[dict[str, int | str]] = []
    start = 0
    current = token_roles[0]
    for idx, role in enumerate(token_roles[1:], start=1):
        if role != current:
            spans.append({"role": current, "start": start, "end": idx})
            start = idx
            current = role
    spans.append({"role": current, "start": start, "end": len(token_roles)})
    return spans


def _fallback_token_roles(tokenizer: Any, prompt: PromptRecord, total: int) -> list[str]:
    if not prompt.system:
        return ["user"] * total
    system_ids = tokenizer(prompt.system, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    system_len = min(total, max(1, int(system_ids.shape[-1]) + 8))
    return ["system" if idx < system_len else "user" for idx in range(total)]


def _mock_rendered_text(prompt: PromptRecord) -> str:
    if prompt.system:
        return f"System: {prompt.system}\nUser: {prompt.user}\nAssistant:"
    return f"User: {prompt.user}\nAssistant:"


def _fold_system_into_user_messages(prompt: PromptRecord) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "System policy:\n"
                f"{prompt.system}\n\n"
                "User request:\n"
                f"{prompt.user}"
            ),
        }
    ]


def _is_unsupported_system_role_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "system role not supported" in message or "role system" in message
