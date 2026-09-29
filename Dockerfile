# App de sustentación (Streamlit + sistema multiagente). Los artefactos del pipeline
# (outputs/, models/, data/processed/) se montan como volúmenes: la imagen no lleva datos.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CLUSTER3_ROOT=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# libgomp1: requerido por LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY .streamlit ./.streamlit
RUN uv pip install --system --no-cache ".[agents,llm]"

RUN useradd -m app && mkdir -p outputs models data/processed data/labels docs && chown -R app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "src/cluster3/app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
