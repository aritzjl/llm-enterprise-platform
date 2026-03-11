"""Prompt builders for the chat completion LangGraph agent."""

import json

ROUTER_SYSTEM_PROMPT = (
    "You are a routing model. Classify whether the user request is SIMPLE or COMPLEX. "
    "SIMPLE: short factual requests, greetings, quick formatting, single-step asks. "
    "COMPLEX: multi-step reasoning, planning, deep comparisons, coding/debugging, "
    "or requests requiring detailed structured output. "
    "Return ONLY strict JSON in one line with this exact schema: "
    '{"route":"simple"} or {"route":"complex"}. '
    "No extra keys, no markdown, no prose."
)


def build_router_classification_messages(
    messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Build messages used by the router model for simple/complex classification."""
    conversation_json = json.dumps(messages, ensure_ascii=True)
    user_prompt = (
        "Classify this conversation as simple or complex. "
        f"Conversation JSON: {conversation_json}"
    )
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
