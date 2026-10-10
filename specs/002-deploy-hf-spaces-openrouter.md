# Spec 002 — Deploy público: app em pod de CPU do RunPod + LLM via OpenRouter

Status: **fechada** (2026-10-10) · Base: `spec.md` v1.1 (D15–D18) e `AGENTS.md`.

> **Mudança de rota durante a execução:** o Hugging Face passou a exigir plano pago para Spaces com Docker (Team na org, PRO
> na conta pessoal). O app foi para um **pod de CPU do RunPod** (`scripts/runpod_app.py`); `scripts/deploy_space.py` fica
> pronto caso o grupo assine o PRO. A parte da LLM via OpenRouter seguiu como planejado, com três famílias de modelo, e a
> voz passou a usar um modelo de áudio do mesmo provedor. O nome do arquivo foi mantido porque é referenciado em `AGENTS.md`
> e no `spec.md`.

## Objetivo

Deixar a Guardiã AI acessível por URL pública para a banca e o vídeo, sem depender de um pod de GPU ligado,
com custo de poucos dólares e sem mudar a arquitetura: o app continua sendo o mesmo código e a LLM continua
sendo "qualquer endpoint compatível com OpenAI".

## Decisões

| # | Decisão | Motivo |
|---|---|---|
| E1 | ~~App no Hugging Face Spaces~~ → **App num pod de CPU do RunPod** (cpu3c, 2 vCPU, 4 GB, US$ 0,066/h com o volume). O pod clona o repositório público em `/workspace/app` (volume de 20 GB), faz `uv sync` e sobe o uvicorn; `make app-redeploy` publica a `main`. A imagem Docker (5,5 GB) está no GHCR (`ghcr.io/fiap-ssp-2025/guardia-ai:latest`, privada) como alternativa via `app-create --image`. | HF cobra Spaces Docker; RunPod usa créditos existentes e API já integrada. O clone dispensa registro de imagem e segredo de pull. |
| E2 | **LLM via OpenRouter**: geradora `openai/gpt-4o-mini` (3,9 s), juiz `mistralai/mistral-small-3.2-24b-instruct` (4,8 s), extração ao vivo `google/gemini-2.5-flash-lite` (1,7 s). Preços por milhão de tokens no README. | Medido em 2026-10-10 após liberar todos os provedores; Qwen3-32B ficou em 11–15 s e inconsistente na extração. |
| E3 | **RunPod permanece como opção auto-hospedada documentada** (`scripts/runpod_vllm.py`), não como padrão. | Já validado; serve ao relatório como alternativa "dado não sai da nossa GPU". |
| E4 | **Voz:** Web Speech API no Chrome dos usuários; fallback no servidor com `google/gemini-2.5-flash` via OpenRouter (áudio inline em MP3, prompt literal), whisper `base` só de reserva. | Whisper `base` em 2 vCPU transcrevia mal; o Gemini faz 19 s de áudio em 1,9 s sem erro por menos de um centavo (spec.md D18). |
| E5 | **Histórico persistente no volume do pod.** O SQLite fica em `data/generated/audit.db` dentro do clone, no volume de 20 GB. | A API do RunPod ignorou `volumeInGb` na criação do pod de CPU (ficou 0 e o histórico zerava a cada restart); um `PATCH` posterior anexou o volume e o script agora confere isso após criar. Verificado em 2026-10-10: análise gravada antes de um stop/start continuou no `/historico` depois. |
| E6 | **Nada de segredo no repositório.** As chaves vão do `.env` local para o ambiente do pod pelo script (`LLM_*`, `JUDGE_*`, `FAST_*`, `TRANSCRIBE_*`, `OPENROUTER_API_KEY`, LangSmith); `.env.example` documenta OpenRouter e voz. | LGPD/segurança e repositório público. |

## Tarefas

- [x] T1 Chave no OpenRouter (humano), modelos escolhidos; preços por milhão de tokens registrados no README (seção Deploy).
- [x] T2 `.env.example` com os blocos OpenRouter (`LLM_*`, `JUDGE_*`, `FAST_*`) e voz (`TRANSCRIBE_*`); `llm/client.py` envia `HTTP-Referer`/`X-Title` (ASCII) sem quebrar outros provedores e resolve `${VAR}` nas chaves.
- [x] T3 `response_format=json_schema` validado no OpenRouter para os quatro modelos; fallback `json_object` + validação continua. Zero JSON inválido em 30 relatos.
- [x] T4 `make eval` com OpenRouter: 30 relatos em `docs/eval_pipeline.json`; comparação com a configuração auto-hospedada (`docs/eval_pipeline_runpod.json`) no relatório §6.
- [x] T5 (adaptada) Container: `Dockerfile` lê `PORT`, `WHISPER_MODEL=base`, grupo `audio`; imagem publicada no GHCR. O pod público usa o bootstrap por clone (`scripts/runpod_app.py`).
- [x] T6 (adaptada) Segredos injetados como variáveis de ambiente do pod pelo script, a partir do `.env` local.
- [x] T7 (adaptada) Publicação: `make app-create` cria o pod e imprime a URL; `make app-redeploy` republica a `main`; smoke test das rotas, de `/api/analisar` e de `/transcrever` pela URL pública.
- [x] T8 (encerrada sem CI) Deploy contínuo substituído por `make app-redeploy` (stop + start + espera do `/api/health`, ~2,5 min). Um job de CI faria o mesmo via API; não vale o segredo extra no GitHub para o prazo do TC.
- [x] T9 README: seção "Deploy público" com URL, custos, preços de tokens e a tabela "onde roda o quê".
- [x] T10 `docs/roteiro_video.md` sem referência a pod de GPU ligado nem às páginas retiradas.
- [x] T11 `spec.md`: E1–E6 registrados como D15–D18 e §10 atualizado.

## Critérios de aceite (verificados em 2026-10-10 em https://y5iscqd4fej0cn-8000.proxy.runpod.net)

- [x] rotas `/`, `/historico`, `/protocolos`, `/status/pill`, `/api/health` em < 1 s (as páginas `/modelo` e `/status` foram retiradas do site)
- [x] análise completa via `/api/analisar` em 14 s com o app aquecido (gpt-4o-mini 4 s + juiz Mistral), JSON válido, fontes citadas; 34 s na primeira análise após subir
- [x] `/transcrever` com Gemini 2.5 Flash: 6 s de áudio em 1,6–2,3 s e 19 s de áudio em 1,9 s, texto idêntico ao falado
- [x] histórico sobrevive a stop/start do pod (E5)
- [x] `make app-redeploy` leva a `main` ao ar em ~2,5 min
- [x] sem segredos no repositório (grep de `sk-or-`, `hf_`, `rpa_`, `lsv2_` vazio antes de cada commit)
- [x] README com URL, custos e tabela "onde roda o quê"; relatório com a segunda configuração (bloco EVAL)

Critérios originais (Space): URL pública respondendo em < 2 s; triagem completa em < 60 s via OpenRouter com juiz de modelo
diferente; `make eval --limit 10` com ≥ 9/10 notas válidas; nenhum segredo no repositório; README e relatório atualizados.
Todos atendidos, com folga, na rota final.

## Saldo em 2026-10-10

RunPod US$ 10,53 (pod ligado gasta US$ 0,066/h: ~160 h); OpenRouter US$ 13,57 de 20 (a avaliação de 30 relatos custou menos de
US$ 0,50). Parar o pod com `make app-stop` fora da semana da banca.

## Riscos e mitigação

- **URL muda se o pod for recriado** (`make app-create` gera outro id): usar `make app-stop`/`app-start`/`app-redeploy`, que mantêm o pod; se recriar, atualizar README, apresentação e vídeo.
- **Primeira análise após subir é lenta** (modelo, RAG e embeddings carregando): abrir a URL alguns minutos antes da demonstração.
- **OpenRouter sem `json_schema` para um modelo:** `json_object` + validação já implementado; trocar de modelo se a taxa de JSON inválido passar de 10%.
- **Custo de tokens disparar:** `max_tokens` limitado (1000 na nota, 900 no F2, proporcional à duração no áudio); acompanhar no painel do OpenRouter.
- **Volume ignorado na criação do pod:** o script confere `volumeInGb` após criar e faz o `PATCH`; conferir `make app-status`.

## Fora de escopo

Autenticação de usuários, fine-tuning, F6 (painel do SINAN), Laya, deploy contínuo por CI.
