.DEFAULT_GOAL := help
PORT ?= 8000

help: ## Lista os alvos
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Cria .venv e instala dependências (uv)
	uv sync

env: ## Cria .env a partir do exemplo
	@test -f .env || cp .env.example .env

data: ## SINAN bruto → tabela de treino (data/processed)
	uv run guardia build-data

train: ## Treina LR, RF e HGB; calibra; SHAP; figuras (use GA=1 para o algoritmo genético)
	uv run python -m guardia.ml.train $(if $(GA),--ga,)
	uv run python scripts/update_readme_metrics.py

serve: ## Sobe a interface web em http://localhost:$(PORT)
	uv run uvicorn guardia.web.app:app --host 0.0.0.0 --port $(PORT) --reload

status: ## Checa LLM, juiz, modelo e RAG
	uv run guardia status

demo: ## Roda o grafo num caso de exemplo (precisa da LLM)
	uv run guardia demo

test: ## Testes automatizados
	uv run pytest -q

lint: ## Ruff
	uv run ruff check src tests scripts

pod-create: ## Cria o pod vLLM no RunPod (geradora + juiz)
	uv run python scripts/runpod_vllm.py create

pod-status: ## Status do pod e dos dois servidores
	uv run python scripts/runpod_vllm.py status

pod-stop: ## Para o pod (custo zero)
	uv run python scripts/runpod_vllm.py stop

pod-start: ## Religa o pod
	uv run python scripts/runpod_vllm.py start

synth: ## Gera relatos sintéticos com a LLM (data/synthetic)
	uv run python scripts/gerar_relatos_sinteticos.py

eval: ## Avalia o pipeline nos relatos sintéticos (juiz, latência, citação)
	uv run python scripts/avaliar_pipeline.py

eval-rag: ## Avalia a recuperação do RAG sem LLM (perguntas do chat + relatos do F1), grava docs/eval_rag.json
	uv run python scripts/avaliar_rag.py --json docs/eval_rag.json

relatorio: ## Gera docs/relatorio_tecnico.html e .pdf a partir do .md (Chrome headless)
	uv run --with markdown scripts/relatorio_pdf.py

publish-hf: ## Publica modelo e métricas na org do Hugging Face
	uv run python scripts/publish_hf.py

app-create: ## Sobe o app num pod de CPU do RunPod (clona o repo; --image usa o GHCR; LLM via OpenRouter)
	uv run --env-file .env python scripts/runpod_app.py create

app-status: ## Status e URL pública do app no RunPod
	uv run --env-file .env python scripts/runpod_app.py status

app-stop: ## Para o app no RunPod (custo zero)
	uv run --env-file .env python scripts/runpod_app.py stop

app-start: ## Religa o app no RunPod
	uv run --env-file .env python scripts/runpod_app.py start

app-redeploy: ## Publica a main no app público: para, religa (o pod refaz o clone) e espera o /api/health
	uv run --env-file .env python scripts/runpod_app.py stop && sleep 5 && uv run --env-file .env python scripts/runpod_app.py start && uv run --env-file .env python scripts/runpod_app.py wait

docker-push: ## Publica a imagem no GHCR (precisa de `gh auth refresh -s write:packages`)
	gh auth token | docker login ghcr.io -u $$(gh api user --jq .login) --password-stdin
	docker tag guardia-ai ghcr.io/fiap-ssp-2025/guardia-ai:latest
	docker push ghcr.io/fiap-ssp-2025/guardia-ai:latest

docker-build: ## Build da imagem
	docker build -t guardia-ai .

docker-run: ## Roda o container com o .env
	docker run --rm -p $(PORT):8000 --env-file .env -v $(PWD)/data/generated:/app/data/generated guardia-ai

notebooks: ## Exporta os scripts de dados/treino como notebooks (jupytext)
	uv run jupytext --to ipynb src/guardia/data/features.py -o notebooks/02_preprocessing.ipynb
	uv run jupytext --to ipynb src/guardia/ml/train.py -o notebooks/03_model_comparison.ipynb

.PHONY: relatorio help install env data train serve status demo test lint pod-create pod-status pod-stop pod-start app-create app-status app-stop app-start app-redeploy docker-push synth eval eval-rag publish-hf docker-build docker-run notebooks
