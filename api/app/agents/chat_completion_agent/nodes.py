"""LangGraph nodes for chat completion with router-first execution."""

from __future__ import annotations

import json
import re
from time import perf_counter
from typing import Any

from app.clients.openai_client import get_openai_client
from app.observability.langfuse import (
    create_llm_generation,
    end_llm_generation_error,
    end_llm_generation_success,
)

from .prompts import build_router_classification_messages
from .state import ChatAgentRoute, ChatAgentState

ROUTER_CLASSIFICATION_TEMPERATURE = 0.0
ROUTER_CLASSIFICATION_MAX_TOKENS = 32


def _extract_usage_details(completion: dict[str, Any]) -> dict[str, int] | None:
    usage = completion.get("usage")
    if not isinstance(usage, dict):
        return None

    details = {
        key: value
        for key, value in usage.items()
        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
        and isinstance(value, int)
    }
    return details or None


def _completion_content(completion: dict[str, Any]) -> str:
    choices = completion.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _parse_route_from_content(content: str) -> ChatAgentRoute:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    candidate_payloads = [cleaned]
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match is not None:
        candidate_payloads.append(json_match.group(0))

    for payload in candidate_payloads:
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            route = parsed.get("route")
            if route == "simple":
                return "simple"
            if route == "complex":
                return "complex"

    return "complex"


def route_selector(state: ChatAgentState) -> ChatAgentRoute:
    """Select branch after classification, defaulting to complex for safety."""
    return "simple" if state.get("selected_route") == "simple" else "complex"


def classify_route_node(state: ChatAgentState) -> ChatAgentState:
    """Classify conversation complexity with the router model."""
    start = perf_counter()
    router_messages = build_router_classification_messages(state["messages"])
    generation = create_llm_generation(
        trace=state.get("trace"),
        provider=state["provider"],
        base_url=state["base_url"],
        model=state["router_model"],
        messages=router_messages,
        temperature=ROUTER_CLASSIFICATION_TEMPERATURE,
        max_tokens=ROUTER_CLASSIFICATION_MAX_TOKENS,
        generation_name="llm.router.classification",
        metadata={
            "stage": "classification",
            "router_model": state["router_model"],
            "primary_model": state["model"],
        },
    )

    payload: dict[str, Any] = {
        "model": state["router_model"],
        "messages": router_messages,
        "stream": False,
        "temperature": ROUTER_CLASSIFICATION_TEMPERATURE,
        "max_tokens": ROUTER_CLASSIFICATION_MAX_TOKENS,
    }

    try:
        completion = get_openai_client(base_url=state["base_url"]).chat.completions.create(
            **payload
        )
    except Exception as exc:
        end_llm_generation_error(
            generation=generation,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            extra_metadata={"stage": "classification"},
        )
        raise

    completion_json = completion.model_dump(mode="json")
    route = _parse_route_from_content(_completion_content(completion_json))

    end_llm_generation_success(
        generation=generation,
        output=completion_json,
        usage_details=_extract_usage_details(completion_json),
        latency_ms=int((perf_counter() - start) * 1000),
        extra_metadata={
            "stage": "classification",
            "selected_route": route,
        },
    )
    return {"selected_route": route}


def _call_response_model(
    state: ChatAgentState,
    *,
    target_model: str,
    generation_name: str,
    stage: str,
) -> ChatAgentState:
    start = perf_counter()
    generation = create_llm_generation(
        trace=state.get("trace"),
        provider=state["provider"],
        base_url=state["base_url"],
        model=target_model,
        messages=state["messages"],
        temperature=state.get("temperature"),
        max_tokens=state.get("max_tokens"),
        generation_name=generation_name,
        metadata={
            "stage": stage,
            "selected_route": state.get("selected_route", "complex"),
            "responder_model": target_model,
            "router_model": state["router_model"],
            "primary_model": state["model"],
        },
    )

    payload: dict[str, Any] = {
        "model": target_model,
        "messages": state["messages"],
        "stream": False,
    }
    if state.get("max_tokens") is not None:
        payload["max_tokens"] = state["max_tokens"]
    if state.get("temperature") is not None:
        payload["temperature"] = state["temperature"]

    try:
        completion = get_openai_client(base_url=state["base_url"]).chat.completions.create(
            **payload
        )
    except Exception as exc:
        end_llm_generation_error(
            generation=generation,
            error_message=str(exc),
            latency_ms=int((perf_counter() - start) * 1000),
            extra_metadata={
                "stage": stage,
                "selected_route": state.get("selected_route", "complex"),
                "responder_model": target_model,
            },
        )
        raise

    completion_json = completion.model_dump(mode="json")
    end_llm_generation_success(
        generation=generation,
        output=completion_json,
        usage_details=_extract_usage_details(completion_json),
        latency_ms=int((perf_counter() - start) * 1000),
        extra_metadata={
            "stage": stage,
            "selected_route": state.get("selected_route", "complex"),
            "responder_model": target_model,
        },
    )
    return {
        "completion": completion_json,
        "responder_model": target_model,
    }


def call_router_model_node(state: ChatAgentState) -> ChatAgentState:
    """Generate final response with router model for simple requests."""
    return _call_response_model(
        state,
        target_model=state["router_model"],
        generation_name="llm.router.response",
        stage="router_response",
    )


def call_primary_model_node(state: ChatAgentState) -> ChatAgentState:
    """Generate final response with primary model for complex requests."""
    return _call_response_model(
        state,
        target_model=state["model"],
        generation_name="llm.generation",
        stage="primary_response",
    )
