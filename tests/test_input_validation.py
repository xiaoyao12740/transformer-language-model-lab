import pytest

from app.input_validation import PromptValidationError, validate_prompt
from src.tokenizer import CharacterTokenizer


def test_empty_prompt_rejected_cleanly():
    tokenizer = CharacterTokenizer().fit("abc")
    with pytest.raises(PromptValidationError, match="cannot be empty"):
        validate_prompt("", tokenizer)


def test_oov_prompt_rejected_cleanly_with_dynamic_vocab_size():
    tokenizer = CharacterTokenizer().fit("abc")
    with pytest.raises(PromptValidationError, match="3-character vocabulary"):
        validate_prompt("ab🙂", tokenizer)


def test_context_and_encoding_share_same_truncation():
    tokenizer = CharacterTokenizer().fit("abcdef")
    context, encoded = validate_prompt("abcdef", tokenizer, 3)
    assert context == "def"
    assert tokenizer.decode(encoded) == context
