def supports_input_role(tokenizer):
    """
    Determines whether the given tokenizer's chat template supports the "input" role.

    Many Hugging Face chat models use structured message templates with roles like "system", "user", "assistant", and optionally "input". This function inspects the tokenizer's `chat_template` to check if it includes logic for handling messages with the "input" role, which is commonly used to inject retrieved documents in RAG setups.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer associated with a chat model.

    Returns:
        bool: True if the chat template references an "input" role, False otherwise.
    """
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        return False
    return "input" in template or "inputs" in template
