BOX_INSTRUCTION = "Put your final answer inside \\boxed{}."


def build_user_message(prompt: str) -> str:
    return f"{prompt}\n{BOX_INSTRUCTION}"


def build_assistant_message(answer: str, reasoning: str | None = None) -> str:
    answer = str(answer).strip()
    if reasoning:
        return f"{reasoning.strip()}\n\nTherefore the final answer is \\boxed{{{answer}}}."
    return f"\\boxed{{{answer}}}"


def build_training_text(tokenizer, prompt: str, answer: str, use_chat_template: bool = True, reasoning: str | None = None) -> str:
    messages = [
        {"role": "user", "content": build_user_message(prompt)},
        {"role": "assistant", "content": build_assistant_message(answer, reasoning)},
    ]
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        except Exception:
            pass
    return f"User: {messages[0]['content']}\nAssistant: {messages[1]['content']}"
