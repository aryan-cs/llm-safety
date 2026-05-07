from cache_safety_erasure.evals.prompt_record import PromptRecord
from cache_safety_erasure.evals.rendering import build_chat_text, rendered_prompt_manifest


class OffsetTokenizer:
    chat_template = None

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        tokens = []
        offsets = []
        cursor = 0
        for part in text.split(" "):
            start = text.find(part, cursor)
            end = start + len(part)
            tokens.append(len(tokens) + 1)
            offsets.append((start, end))
            cursor = end
        result = {"input_ids": tokens}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class NoSystemRoleTokenizer(OffsetTokenizer):
    chat_template = "template"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if any(message["role"] == "system" for message in messages):
            raise ValueError("System role not supported")
        content = messages[0]["content"]
        suffix = "\nAssistant:" if add_generation_prompt else ""
        return f"<start_of_turn>user\n{content}<end_of_turn>{suffix}"


def test_rendered_manifest_labels_exact_system_and_user_token_roles() -> None:
    prompt = PromptRecord(
        id="p1",
        suite="system_leakage",
        system="policy words SECRET",
        hidden_system="SECRET",
        user="user words",
    )
    manifest = rendered_prompt_manifest(OffsetTokenizer(), prompt)
    assert manifest["offset_mapping_available"] is True
    assert any(span["role"] == "system" for span in manifest["token_role_spans"])
    assert any(span["role"] == "user" for span in manifest["token_role_spans"])
    assert any(span["role"] == "hidden_system" for span in manifest["component_char_spans"])


def test_system_role_unsupported_template_folds_policy_into_user_turn() -> None:
    prompt = PromptRecord(
        id="p1",
        suite="system_leakage",
        system="policy words SECRET",
        hidden_system="SECRET",
        user="user words",
    )
    tokenizer = NoSystemRoleTokenizer()

    rendered = build_chat_text(tokenizer, prompt)
    manifest = rendered_prompt_manifest(tokenizer, prompt)

    assert "System policy:" in rendered
    assert "policy words SECRET" in rendered
    assert "User request:" in rendered
    assert "user words" in rendered
    assert any(span["role"] == "system" for span in manifest["token_role_spans"])
    assert any(span["role"] == "user" for span in manifest["token_role_spans"])
