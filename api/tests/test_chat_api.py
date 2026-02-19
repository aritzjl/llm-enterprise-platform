import json

from fastapi.testclient import TestClient

from app.core.config import clear_settings_cache
from app.main import app
from app.services import providers
from app.api.routes import chat as chat_route


client = TestClient(app)


def teardown_function() -> None:
    clear_settings_cache()


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_provider_returns_500(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "invalid")
    clear_settings_cache()

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )

    assert response.status_code == 500
    assert "Invalid LLM_PROVIDER" in response.json()["detail"]


def test_vllm_non_stream_uses_default_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "vllm")
    monkeypatch.setenv("VLLM_MODEL", "Qwen/Test-Model")
    clear_settings_cache()

    captured_payload = {}

    async def fake_call_vllm(payload, settings):
        captured_payload.update(payload)
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }

    monkeypatch.setattr(chat_route.providers, "call_vllm_non_stream", fake_call_vllm)

    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hola"}]},
    )

    assert response.status_code == 200
    assert captured_payload["model"] == "Qwen/Test-Model"
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_streaming_with_ollama_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    clear_settings_cache()

    async def fake_stream_ollama(payload, settings, model, observation):
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "delta": {"content": "hola"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    monkeypatch.setattr(chat_route.providers, "stream_ollama", fake_stream_ollama)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "qwen2:1.5b",
            "messages": [{"role": "user", "content": "hola"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert "data: [DONE]" in response.text


def test_build_ollama_payload_maps_generation_options() -> None:
    payload = {
        "messages": [{"role": "user", "content": "hola"}],
        "temperature": 0.2,
        "max_tokens": 42,
        "top_p": 0.95,
        "stop": ["FIN"],
        "format": "json",
    }

    ollama_payload = providers.build_ollama_payload(payload, model="qwen2:1.5b", stream=False)

    assert ollama_payload["model"] == "qwen2:1.5b"
    assert ollama_payload["stream"] is False
    assert ollama_payload["options"]["num_predict"] == 42
    assert ollama_payload["options"]["temperature"] == 0.2
    assert ollama_payload["options"]["top_p"] == 0.95
    assert ollama_payload["options"]["stop"] == ["FIN"]
    assert ollama_payload["format"] == "json"
