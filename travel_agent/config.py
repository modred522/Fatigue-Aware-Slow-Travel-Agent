from functools import lru_cache
import os

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel


load_dotenv(find_dotenv(), override=False)


class AppConfig(BaseModel):
    amap_api_key: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_model: str = "qwen-turbo"
    llm_temperature: float = 0.3


def _parse_temperature(value: str | None) -> float:
    if not value:
        return 0.3
    try:
        temp = float(value)
        return max(0.0, min(2.0, temp))
    except ValueError:
        return 0.3


@lru_cache(maxsize=1)
def get_config() -> AppConfig:
    return AppConfig(
        amap_api_key=os.getenv("AMAP_API_KEY", ""),
        llm_api_key=os.getenv("LLM_API_KEY", "") or os.getenv("DASHSCOPE_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL", "") or _default_base_url(),
        llm_model=os.getenv("LLM_MODEL", "qwen-turbo"),
        llm_temperature=_parse_temperature(os.getenv("LLM_TEMPERATURE")),
    )


def refresh_config() -> AppConfig:
    get_config.cache_clear()
    return get_config()


def _default_base_url() -> str:
    if os.getenv("DASHSCOPE_API_KEY"):
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return ""
