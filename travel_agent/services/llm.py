from __future__ import annotations

from openai import OpenAI

from travel_agent.config import get_config
from travel_agent.errors import ConfigError, ExternalServiceError


def get_llm_client(api_key: str | None = None, base_url: str | None = None) -> OpenAI:
    config = get_config()
    resolved_api_key = api_key if api_key is not None else config.llm_api_key
    resolved_base_url = base_url if base_url is not None else config.llm_base_url

    if not resolved_api_key:
        raise ConfigError(
            "LLM API key is not configured. Please set it in Settings.",
            "llm_api_key",
        )
    client_kwargs = {"api_key": resolved_api_key}
    if resolved_base_url:
        client_kwargs["base_url"] = resolved_base_url
    return OpenAI(**client_kwargs)


# Common model fallbacks for providers that don't support models.list()
_PROVIDER_FALLBACK_MODELS: dict[str, list[str]] = {
    "api.moonshot.cn": [
        "kimi-k2-0711-preview",
        "kimi-k2-0711",
        "kimi-k1.5-preview",
        "kimi-k1.5",
        "kimi-latest",
    ],
}


def _is_moonshot_api(base_url: str) -> bool:
    return "moonshot" in base_url.lower() or "api.moonshot.cn" in base_url.lower()


def list_models(api_key: str | None = None, base_url: str | None = None) -> list[str]:
    config = get_config()
    resolved_api_key = api_key if api_key is not None else config.llm_api_key
    resolved_base_url = base_url if base_url is not None else config.llm_base_url

    # Special handling for Moonshot/Kimi API which has limited models.list support
    if _is_moonshot_api(resolved_base_url):
        return _PROVIDER_FALLBACK_MODELS["api.moonshot.cn"]

    client = get_llm_client(api_key=resolved_api_key, base_url=resolved_base_url)
    try:
        response = client.models.list()
        model_ids = sorted(model.id for model in response.data)
        return model_ids
    except Exception as exc:
        # If models.list() fails, try to return fallback models based on base_url
        for provider_domain, models in _PROVIDER_FALLBACK_MODELS.items():
            if provider_domain in resolved_base_url:
                return models
        # Re-raise if no fallback available
        raise ExternalServiceError(f"Failed to fetch models from the LLM provider: {exc}") from exc


def _is_kimi_model(model: str) -> bool:
    return model.lower().startswith("kimi-")


def _get_temperature(config) -> float | None:
    """Get temperature for LLM call. Kimi models don't use temperature when thinking is disabled."""
    if _is_kimi_model(config.llm_model):
        return None  # Kimi with disabled thinking doesn't use temperature
    return config.llm_temperature


def _get_extra_body(config) -> dict | None:
    """Get extra body parameters for specific providers."""
    if _is_kimi_model(config.llm_model):
        # Disable thinking for Kimi models via extra_body
        return {"thinking": {"type": "disabled"}}
    return None

def invoke_text(prompt: str) -> str:
    config = get_config()
    client = get_llm_client()

    temperature = _get_temperature(config)
    extra_body = _get_extra_body(config)

    try:
        request_params = {
            "model": config.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise slow-travel planning assistant. Return only the requested format.",
                },
                {"role": "user", "content": prompt},
            ],
        }

        if temperature is not None:
            request_params["temperature"] = temperature
        if extra_body is not None:
            request_params["extra_body"] = extra_body

        response = client.chat.completions.create(**request_params)
    except Exception as exc:
        raise ExternalServiceError(f"Failed to call the selected LLM model: {exc}") from exc

    content = response.choices[0].message.content
    if not content:
        raise ExternalServiceError("The selected LLM returned an empty response.")
    return str(content).strip()
