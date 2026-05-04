from cache_safety_erasure.generation.hf_generate import _has_stop_string


class DecodeCountingTokenizer:
    def __init__(self, text: str = "") -> None:
        self.text = text
        self.decode_calls = 0

    def decode(self, _ids: list[int], *, skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        self.decode_calls += 1
        return self.text


def test_stop_string_check_skips_decode_when_no_stop_strings() -> None:
    tokenizer = DecodeCountingTokenizer("stop")

    assert _has_stop_string(tokenizer, [1, 2, 3], ()) is False
    assert tokenizer.decode_calls == 0


def test_stop_string_check_decodes_when_stop_strings_configured() -> None:
    tokenizer = DecodeCountingTokenizer("answer<stop>")

    assert _has_stop_string(tokenizer, [1, 2, 3], ("<stop>",)) is True
    assert tokenizer.decode_calls == 1
