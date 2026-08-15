class PromptValidationError(ValueError):
    """A user-facing prompt validation error."""


def validate_prompt(prompt, tokenizer, block_size=None):
    if not prompt:
        raise PromptValidationError("Prompt cannot be empty / 输入不能为空。")
    context = prompt[-block_size:] if block_size is not None else prompt
    try:
        encoded = tokenizer.encode(context)
    except ValueError as error:
        raise PromptValidationError(
            f"Character outside the {tokenizer.vocab_size}-character vocabulary / "
            f"包含 {tokenizer.vocab_size} 字符词表之外的字符。"
        ) from error
    return context, encoded
