from typing import Literal

from pydantic import BaseModel, SecretStr


class Config(BaseModel):
    llm_backend: Literal["auto", "openai", "vllm"] = "auto"
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    api_key: SecretStr = SecretStr("EMPTY")
