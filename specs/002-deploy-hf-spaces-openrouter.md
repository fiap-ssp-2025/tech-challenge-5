# Spec 002 — Deploy público: app no Hugging Face Spaces + LLM via OpenRouter

Status: **done** (2026-10-10) · Base: `spec.md` v1.1 e `AGENTS.md`.

> **Mudança de rota durante a execução:** o Hugging Face passou a exigir plano pago para Spaces com Docker (Team na org, PRO na conta pessoal). O app foi para um **pod de CPU do RunPod** (`scripts/runpod_app.py`, imagem no GHCR); `scripts/deploy_space.py` fica pronto caso o grupo assine o PRO. A parte da LLM via OpenRouter seguiu como planejado, com três famílias de modelo.

## Objetivo

Deixar a Guardiã AI acessível por URL pública para a banca e o vídeo, sem depender do pod RunPod ligado,
com custo mensal de poucos dólares e sem mudar a arquitetura: o app continua sendo o container atual e a
LLM continua sendo "qualquer endpoint compatível com OpenAI".

## Decisões

| # | Decisão | Motivo |
|---|---|---|
| E1 | ~~App no Hugging Face Spaces~~ → **App num pod de CPU do RunPod** (cpu3c, 2 vCPU, 4 GB, US$ 0,06/h), o pod clona o repositório público num volume persistente (`uv sync` + uvicorn); a imagem Docker fica publicada no GHCR (privada, `ghcr.io/fiap-ssp-2025/guardia-ai:latest`) como alternativa via `app-create --image`, com credencial de registro salva no RunPod. | HF cobra Spaces Docker; RunPod usa créditos existentes e API já integrada. |
| E2 | **LLM via OpenRouter**: geradora `openai/gpt-4o-mini` (3,9 s), juiz `mistralai/mistral-small-3.2-24b-instruct` (4,8 s), extração ao vivo `google/gemini-2.5-flash-lite` (1,7 s). | Medido em 2026-10-10 após liberar todos os provedores; Qwen3-32B ficou em 11–15 s e inconsistente na extração. |
| E3 | **RunPod permanece como opção auto-hospedada documentada** (`scripts/runpod_vllm.py`), não como padrão. | Já validado; serve ao relatório como alternativa "dado não sai da nossa GPU". |
| E4 | **Voz:** Web Speech API no Chrome dos usuários; fallback whisper `base` no Space (2 vCPU). | `small` leva 2–3 s por trecho em 2 vCPU; `base` fica abaixo de 1,5 s. Variável `WHISPER_MODEL=base`. |
| E5 | **Histórico volátil no Space** (SQLite em disco efêmero) aceito; opcionalmente armazenamento persistente pago na semana da banca. | Custo zero por padrão. |
| E6 | **Nada de segredo no repositório.** Chaves só em Secrets do Space; `.env.example` ganha a seção OpenRouter. | LGPD/segurança e repositório público. |

## Tarefas

- [x] T1 Criar conta/chave no OpenRouter (humano) e escolher os dois modelos; registrar custo por 1M tokens em `docs/`.
- [x] T2 `.env.example`: bloco OpenRouter (`LLM_BASE_URL=https://openrouter.ai/api/v1`, `LLM_MODEL`, `LLM_API_KEY`; `JUDGE_*` idem com outro modelo; `FAST_*` = juiz). Conferir que `llm/client.py` envia `HTTP-Referer`/`X-Title` opcionais do OpenRouter sem quebrar outros provedores.
- [x] T3 Validar `response_format=json_schema` no OpenRouter para os modelos escolhidos; se não suportado, o fallback `json_object` + validação já existe — medir taxa de JSON inválido com `scripts/avaliar_pipeline.py --limit 10`.
- [x] T4 Rodar `make eval` com OpenRouter e comparar com `docs/eval_pipeline.json` (F2 precisão/recall, aprovação do juiz, latência). Registrar no relatório §6 como segunda configuração.
- [x] T5 (adaptado) Space: `README.md` do Space com front-matter (`sdk: docker`, `app_port: 7860`), `Dockerfile` lendo `PORT` (padrão 8000; Space usa 7860), `WHISPER_MODEL=base`, `HF_HOME` gravável, usuário não-root se o Space exigir.
- [x] T6 Segredos no Space: `LLM_*`, `JUDGE_*`, `FAST_*`, `LANGSMITH_API_KEY` (opcional), `GUARDIA_DB_PATH=/data/audit.db` se houver storage persistente.
- [x] T7 Publicar: `huggingface_hub` upload do repositório para `fiap-ssp-2025/guardia-ai` (Space) ou `git push` para o remoto do Space; aguardar build (~15 min pela imagem de 4,6 GB); smoke test das rotas e de `/transcrever`.
- [ ] T8 (não feito, opcional) CI: job que faz push para o Space na `main` (token em secret do GitHub) — só se o grupo quiser deploy contínuo.
- [x] T9 README/relatório: seção "Deploy" com a URL pública, custos, e a tabela "onde roda o quê" (app: Space CPU; LLM: OpenRouter; voz: navegador + whisper no Space; alternativa auto-hospedada: RunPod).
- [x] T10 `docs/roteiro_video.md`: trocar "pod ligado" por "Space no ar"; checklist de gravação.
- [x] T11 Spec `spec.md`: registrar E1–E6 como D15+ e atualizar §10.

## Critérios de aceite (verificados em 2026-10-10 em https://y5iscqd4fej0cn-8000.proxy.runpod.net)

- [x] rotas `/`, `/historico`, `/protocolos`, `/modelo`, `/status`, `/api/health` em < 1 s
- [x] análise completa via `/api/analisar` em 18 s (gpt-4o-mini 5 s + juiz Mistral), JSON válido, 3 fontes citadas
- [x] `/transcrever` com whisper `base`: 10 s de áudio em 2,3 s
- [x] sem segredos no repositório (grep vazio antes de cada commit)
- [x] README com URL, custos e tabela "onde roda o quê"; relatório com a segunda configuração (bloco EVAL)

Critérios originais:

- URL pública do Space responde `/`, `/historico`, `/protocolos`, `/modelo`, `/status` e `/api/health` em < 2 s (sem contar a LLM).
- Uma triagem completa (F2 ao vivo + F1) conclui em < 60 s via OpenRouter, com juiz de modelo diferente e nota em JSON válido.
- `make eval --limit 10` com OpenRouter: ≥ 9/10 notas válidas e juiz avaliando todas.
- Nenhum segredo no repositório (grep de `sk-or-`, `hf_`, `rpa_`, `lsv2_` vazio).
- README com a URL e a tabela de custos; relatório com a segunda configuração de LLM.

## Riscos e mitigação

- **Cold start / build lento no Space:** imagem de 4,6 GB; manter o Space "awake" na semana da banca (acesso diário) ou pagar hardware persistente.
- **OpenRouter sem `json_schema` para o modelo:** usar `json_object` (já implementado) e medir; se a taxa de JSON inválido passar de 10%, trocar de modelo.
- **Limite de 2 vCPU para whisper + embeddings + modelo:** `WHISPER_MODEL=base`; o Web Speech API cobre o Chrome.
- **Custo de tokens disparar:** `max_tokens` já limitado (1000 na nota, 900 no F2); monitorar no painel do OpenRouter.

## Fora de escopo

Autenticação de usuários, armazenamento persistente obrigatório, fine-tuning, F6 (painel do SINAN), Laya.
