# Guardiã AI — container da aplicação (interface web + grafo + modelo + RAG).
# A LLM é remota (LLM_BASE_URL / JUDGE_BASE_URL no .env); o modelo de ML e o índice RAG ficam na imagem.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 \
    HF_HOME=/app/data/generated/hf TOKENIZERS_PARALLELISM=false
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl ffmpeg && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.12.6 /uv /usr/local/bin/uv

# dependências primeiro (cache de camadas)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --group audio --no-install-project --no-cache

# código, modelo treinado, protocolos
COPY src ./src
COPY models ./models
COPY data/protocols ./data/protocols
COPY docs/figures ./docs/figures
RUN uv sync --frozen --no-dev --group audio --no-cache

# pré-baixa o modelo de embeddings e indexa os protocolos (sem rede em runtime); limpa caches de download
RUN uv run --no-sync python -c "from guardia.rag.retriever import vectorstore; print('chunks', vectorstore()._collection.count())" \
    && rm -rf /root/.cache /app/data/generated/hf/hub/.locks

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD curl -fs http://localhost:8000/api/health || exit 1
CMD ["uv", "run", "--no-sync", "uvicorn", "guardia.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
