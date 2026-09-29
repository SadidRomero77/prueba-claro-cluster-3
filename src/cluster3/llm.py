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
# Modelo rápido para decisiones cortas (enrutar, juzgar): baja la latencia sin tocar la calidad de las respuestas.
DEFAULTS_RAPIDO = {
    "anthropic": "claude-haiku-4-5-20251001",
    "openai": "gpt-4o-mini",
    "openrouter": "anthropic/claude-haiku-4.5",
    "databricks": "databricks-meta-llama-3-3-70b-instruct",
}
# Modelos que se pueden elegir en la app (OpenRouter). Todos soportan tool calling, que los agentes necesitan.
MODELOS_OPENROUTER = {
    "anthropic/claude-sonnet-5": "Claude Sonnet 5 · predeterminado",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "anthropic/claude-haiku-4.5": "Claude Haiku 4.5 · rápido",
    "anthropic/claude-opus-5.5": "Claude Opus 5.5 · más capaz",
    "openai/gpt-6-sol": "OpenAI GPT-6 Sol",
    "openai/gpt-6-luna": "OpenAI GPT-6 Luna · económico",
    "google/gemini-3.8-flash": "Google Gemini 3.8 Flash",
    "deepseek/deepseek-v4-pro": "DeepSeek V4 Pro",
}
KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "databricks": "DATABRICKS_HOST",
}


def llm_available() -> bool:
    return bool(os.getenv(KEYS.get(config.LLM_PROVIDER, ""), ""))


def model_name(rapido: bool = False) -> str:
    if rapido:
        return os.getenv("LLM_MODEL_RAPIDO") or DEFAULTS_RAPIDO[config.LLM_PROVIDER]
    return config.LLM_MODEL or DEFAULTS[config.LLM_PROVIDER]


def get_chat_model(temperature: float = 0.0, max_tokens: int = 2048, rapido: bool = False,
                   modelo: str | None = None):
    """rapido=True usa el modelo liviano (LLM_MODEL_RAPIDO) para enrutar y juzgar.
    modelo: nombre elegido en la app; reemplaza a LLM_MODEL (no aplica a las llamadas rápidas)."""
    provider = config.LLM_PROVIDER
    nombre = model_name(True) if rapido else (modelo or model_name())
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=nombre, temperature=temperature, max_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=nombre, temperature=temperature, max_tokens=max_tokens)
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        class ChatOpenRouter(ChatOpenAI):
            """OpenRouter no soporta json_schema estricto en todos los modelos: la salida estructurada va por
            function calling, que sí soportan todos los modelos con herramientas."""

            def with_structured_output(self, schema, *, method="function_calling", **kw):
                return super().with_structured_output(schema, method=method, **kw)

        return ChatOpenRouter(
            model=nombre,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=OPENROUTER_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={"X-Title": "Claro Cluster 3"},
        )
    if provider == "databricks":
        from databricks_langchain import ChatDatabricks  # pip install databricks-langchain

        return ChatDatabricks(endpoint=nombre, temperature=temperature, max_tokens=max_tokens)
    raise ValueError(f"LLM_PROVIDER no soportado: {provider}")
