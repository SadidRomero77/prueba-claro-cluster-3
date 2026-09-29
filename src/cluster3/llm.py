"""Fábrica de modelos de lenguaje. El proveedor se elige por variables de entorno.

LLM_PROVIDER = anthropic | openai | openrouter | databricks
LLM_MODEL    = nombre del modelo (opcional; si está vacío se usa el default del proveedor)
"""
from __future__ import annotations

import os

from cluster3 import config

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DEFAULTS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o-mini",
    "openrouter": "anthropic/claude-sonnet-5",
    "databricks": "databricks-meta-llama-3-3-70b-instruct",
}
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "databricks": "DATABRICKS_HOST",
}


def llm_available() -> bool:
    return bool(os.getenv(KEYS.get(config.LLM_PROVIDER, ""), ""))


def model_name() -> str:
    return config.LLM_MODEL or DEFAULTS[config.LLM_PROVIDER]


def get_chat_model(temperature: float = 0.0, max_tokens: int = 2048):
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model_name(), temperature=temperature, max_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model_name(), temperature=temperature, max_tokens=max_tokens)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        class ChatOpenRouter(ChatOpenAI):
            """OpenRouter no soporta json_schema estricto en todos los modelos: la salida estructurada va por
            function calling, que sí soportan todos los modelos con herramientas."""

            def with_structured_output(self, schema, *, method="function_calling", **kw):
                return super().with_structured_output(schema, method=method, **kw)

        return ChatOpenRouter(
            model=model_name(),
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=OPENROUTER_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={"X-Title": "Claro Cluster 3"},
        )
    if provider == "databricks":
        from databricks_langchain import ChatDatabricks  # pip install databricks-langchain

        return ChatDatabricks(endpoint=model_name(), temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"LLM_PROVIDER no soportado: {provider}")
