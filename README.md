# Guardiã AI — apoio ao atendimento de mulheres em situação de violência

Tech Challenge 5 · FIAP Pós-Tech IA para Devs · grupo **fiap-ssp-2025**

A profissional cola o relato do atendimento. A Guardiã AI organiza as informações na ficha do SINAN, estima com um
modelo de machine learning se o caso tem perfil de **encaminhamento à rede de proteção**, consulta leis e protocolos
cadastrados (RAG) e devolve uma **nota de apoio com fontes citadas**, auditada por um segundo modelo de linguagem.
A decisão é sempre da profissional: o sistema não decide encaminhamentos, não diagnostica e não prescreve.

> Especificação e decisões: [`spec.md`](spec.md) · por que esta base: [`analise-base-tc5.md`](analise-base-tc5.md) ·
> relatório técnico: [`docs/relatorio_tecnico.md`](docs/relatorio_tecnico.md) · convenções: [`AGENTS.md`](AGENTS.md)

## O problema e o cenário escolhido

O roteiro pede um cenário principal em saúde ou segurança da mulher com uma jornada completa
(dados → ML → consulta de informações → LLM → resultado para a profissional). Escolhemos **segurança**: violência
doméstica e familiar. A dimensão de saúde entra pela jornada (encaminhamentos de saúde, norma técnica do MS sobre
violência sexual), não como um segundo escopo.

**Alvo do modelo.** Nas fases anteriores o grupo usou `OUT_VEZES` ("ocorreu outras vezes?") como alvo e chamou de
predição de recorrência. É um campo da mesma ficha, preenchido junto com as features: um estudo transversal, não uma
predição, e no produto a atendente pode simplesmente perguntar. O SINAN não tem desfecho futuro, mas registra em ~94%
das fichas para onde o caso foi encaminhado. O alvo passou a ser `ENCAMINHA_PROTECAO` (delegacia da mulher, centro de
atendimento à mulher, Ministério Público ou Defensoria). `OUT_VEZES` virou feature. Detalhes em `spec.md` §3.

## Arquitetura

```
relato + ficha ──▶ LangGraph (F1) ──▶ nota de apoio
                   ├─ assess_risk          regras determinísticas (arma, ameaça, repetição, gestação…)
                   ├─ run_model            HistGB/RF calibrado + SHAP do caso          ← models/encaminhamento.joblib
                   ├─ retrieve_protocols   Chroma + MiniLM multilíngue, 11 documentos   ← data/protocols
                   ├─ define_actions       ações e encaminhamentos por risco
                   ├─ build_explanation / prepare_audit
                   ├─ generate_note        LLM geradora, JSON por schema (vLLM no RunPod · fallback OpenAI)
                   ├─ judge                blocklist + LLM-as-judge em outro servidor/modelo
                   └─ persist              SQLite: hash do relato, risco, probabilidade, fontes, veredito
```

| Fluxo | O que faz | Onde |
|---|---|---|
| **F1 Triagem** | caixa de relato no estilo chat (escrever ou **ditar por voz**) → nota de apoio, com progresso real por etapa do grafo | `/` |
| **F2 Estruturar relato** | enquanto a profissional escreve ou fala, o modelo rápido preenche a ficha (JSON por schema); o que foi detectado aparece como chips e os campos ficam em grupos recolhíveis | ao vivo em `/` |
| **F3 Consultar protocolos** | chat com memória no servidor sobre a base RAG, sempre citando a fonte | `/protocolos` |
| **F4 Histórico** | atendimentos anteriores com linha do tempo digerida: o que cada etapa fez, quanto levou, modelo, fontes e validador | `/historico` |
| Status | saúde dos componentes (LLM, juiz, modelo, RAG, voz) no indicador do cabeçalho; métricas e figuras do modelo ficam no relatório e em `docs/figures` | cabeçalho |

Stack: Python 3.12 · scikit-learn · SHAP · LangGraph · Chroma · sentence-transformers · vLLM (RunPod) ·
FastAPI + Jinja2 · **htmx 4.0.0 + Tailwind CSS 4 + Basecoat** (HTML renderizado no servidor, sem SPA) · SQLite · Docker.

**Interface (decisões de UX).** Revelação progressiva: a ficha mostra primeiro o que o relato preencheu e esconde o
resto em grupos com contagem. Espera honesta: a análise roda como job no servidor e a página consulta o progresso real
dos 9 nós a cada 2 s (etapa atual, tempo por etapa, estimativa), em vez de um spinner. Voz: botão de ditado com Web
Speech API (texto provisório em cinza, nada é enviado sozinho) e fallback de gravação em blocos transcritos no servidor
por faster-whisper. Extração ao vivo usa o modelo pequeno do pod (7 s); a nota final usa o de 27B.

## Dados

SINAN/DATASUS 2024, violência interpessoal e autoprovocada (público, anonimizado). Filtro: mulheres, lesão não
autoprovocada, campo de encaminhamento respondido → **280.757 notificações**, prevalência do alvo **28,3%**.
42 features: 10 tipos de violência, 9 meios de agressão, 5 grupos de relação com o autor, 13 campos de contexto
(repetição, álcool, gestação, local, motivação, raça/cor, ciclo de vida, UF…), escolaridade ordinal e idade.
Ausentes viram categoria `IGNORADO`; nada é descartado por campo vazio. Notebooks: `notebooks/02_preprocessing.ipynb`.

## Modelos e métricas

Três modelos no mesmo `Pipeline` (OneHot nominais + imputação mediana), 70/30 estratificado, validação de 15% dentro do
treino para limiar (máximo F2), calibração isotônica e fitness do algoritmo genético; o teste é usado uma vez.

<!-- METRICS:START -->
Conjunto: 280.757 linhas · treino 167.049 · validação 29.480 · teste 84.228 · prevalência 0.283 · acurácia da classe majoritária 0.717.

| Modelo | PR-AUC | ROC-AUC | Recall@thr | Precisão@thr | F1@thr | Especificidade | Limiar |
|---|---:|---:|---:|---:|---:|---:|---:|
| logistic_regression | 0.612 | 0.777 | 0.906 | 0.375 | 0.531 | 0.406 | 0.308 |
| random_forest | 0.652 | 0.803 | 0.930 | 0.378 | 0.537 | 0.396 | 0.313 |
| hist_gradient_boosting **(implantado)** | 0.674 | 0.816 | 0.923 | 0.392 | 0.551 | 0.437 | 0.287 |
| random_forest_ga | 0.680 | 0.819 | 0.918 | 0.401 | 0.558 | 0.460 | 0.280 |

Modelo implantado: **hist_gradient_boosting** calibrado (isotônica): Brier 0.1739 → 0.1462; no limiar calibrado 0.150: recall 0.923, precisão 0.392, F1 0.551. Gap de recall por raça/cor: 0.180. Artefato: 0.5 MB. Figuras em `docs/figures/` (PR, calibração, SHAP).
Algoritmo genético (RF): melhores hiperparâmetros {'n_estimators': 150, 'max_depth': 20, 'min_samples_leaf': 10, 'max_features': 0.3}.
Top features (média |SHAP|): IDADE_ANOS, REL_PARCEIRO_INTIMO, VIOL_MOTIV_SEXISMO, AG_AMEACA, AUTOR_SEXO_MASCULINO, OUT_VEZES_SIM.
<!-- METRICS:END -->

Por que **recall da classe encaminhada** e não acurácia: a prevalência é 28%, "nunca sugerir" acerta 72% e não ajuda
ninguém. Um falso negativo deixa de levar à equipe um caso com perfil de risco; um falso positivo custa uma conversa a
mais. Por isso o limiar privilegia recall e a probabilidade exibida é calibrada. Explicabilidade: SHAP global e por caso
(nota de apoio), coeficientes da regressão logística no relatório. Equidade: recall por raça/cor.
Limite a declarar sempre: o modelo aprende a prática registrada em 2024, inclusive sub-encaminhamento.

### Avaliação da aplicação (30 relatos sintéticos, `make eval`)

F2 extrai os campos da ficha com precisão 0,92 e recall 0,93 por campo e acerta a repetição (`OUT_VEZES`) em 100%; latência mediana 22 s.
F1 gera 30 de 30 notas; o juiz avaliou todas e aprovou 29 (1 bloqueada); latência mediana da LLM 25 s, tempo total mediano 52 s;
2,8 fontes recuperadas e 2,1 citadas por nota; a sugestão do modelo concorda com o encaminhamento real da ficha em 77% dos casos.
Com a configuração padrão via OpenRouter (gpt-4o-mini, Mistral Small, Gemini Flash Lite), os mesmos 30 relatos deram: F2 precisão 0,90 e recall 0,95,
juiz aprovou 30 de 30, LLM 5 s de mediana e 20 s no total por análise, concordância do modelo com a ficha 0,80
(`docs/eval_pipeline.json`; a configuração auto-hospedada ficou em `docs/eval_pipeline_runpod.json`).

## LLM, RAG e segurança

- **Geradora, juiz e extração ao vivo** em qualquer endpoint compatível com OpenAI (`LLM_*`, `JUDGE_*`, `FAST_*`).
  Configuração padrão: OpenRouter com `openai/gpt-4o-mini` (nota), `mistralai/mistral-small-3.2-24b-instruct` (juiz) e
  `google/gemini-2.5-flash-lite` (F2). Configuração auto-hospedada validada: `QuantTrio/Qwen3.6-27B-AWQ` e
  `Qwen/Qwen2.5-7B-Instruct-AWQ` em vLLM num pod RunPod (`scripts/runpod_vllm.py`). Saída JSON restrita por schema
  (`response_format=json_schema`, com fallback `json_object`).
- **Juiz** audita cada nota em 6 critérios (prescrição, diagnóstico, decisão automática, rede de proteção, linguagem,
  privacidade) com exigência de trecho literal; sempre num modelo de família diferente da geradora.
- **RAG:** 11 documentos (Lei 11.340/2006 e resumo, tipos de violência, medidas protetivas, canais nacionais e do DF,
  rede de proteção do DF, LGPD, norma técnica MS/MJ de violência sexual) em Chroma com embeddings multilíngues;
  a nota cita os títulos usados e a interface mostra quais foram recuperados e quais foram citados.
- **Guardrails:** blocklist de frases proibidas, aviso fixo de apoio à decisão, juiz independente, hash do relato na
  auditoria (texto nunca persiste).
- **Rastreabilidade:** LangSmith (`LANGCHAIN_TRACING_V2=true`) rastreia cada nó e chamada.

## Como executar

```bash
make install            # uv sync (Python 3.12)
make env                # cria .env; preencha LLM_BASE_URL/LLM_MODEL/LLM_API_KEY e JUDGE_* (ou OPENAI_API_KEY)
make data               # SINAN bruto → data/processed (precisa de data/raw/VIOLBR24.parquet; ver abaixo)
make train GA=1         # treina, calibra, SHAP, figuras → models/ e docs/figures/
make serve              # http://localhost:8000
make test               # pytest
```

Dados brutos: `data/raw/VIOLBR24.parquet/` é a extração do DATASUS feita no Tech Challenge 1 (`pysus`). Copie de lá ou
rode o notebook `01_extract_sinan_data.ipynb` daquele repositório.

LLM no RunPod (créditos do grupo): `make pod-create` cria um pod com os dois servidores e imprime as variáveis
para o `.env`; `make pod-status`; `make pod-stop` quando não estiver em uso. Sem pod e sem OpenAI a aplicação continua
funcionando: a nota vem só da engine de regras e do modelo, com a falha registrada na auditoria.

### Docker

```bash
make docker-build
make docker-run         # usa o .env; http://localhost:8000
```

A imagem traz o modelo treinado, os protocolos e o índice RAG; a LLM é remota (`LLM_BASE_URL`).

### Scripts úteis

| Comando | Para quê |
|---|---|
| `make synth` | gera relatos sintéticos a partir de fichas reais (rótulos de graça) → `data/synthetic/relatos.jsonl` |
| `make eval` | avalia F2 (concordância por campo) e F1 (juiz, latência, citação) → `docs/eval_pipeline.json` |
| `make publish-hf` | publica modelo, métricas e figuras em `huggingface.co/fiap-ssp-2025/tc5-guardia-encaminhamento` |
| `uv run guardia demo` | roda o grafo num caso de exemplo no terminal |

## Estrutura

```
src/guardia/
  data/features.py      SINAN → tabela; row_from_case (formulário → linha do modelo)
  ml/train.py           LR · RF (+GA) · HistGB · calibração · limiar · SHAP · figuras
  ml/predict.py         carrega o artefato, SHAP por caso
  graph/flow.py         LangGraph do F1
  llm/                  client (geradora/juiz), prompts e schemas, structurer (F2), judge
  rag/                  retriever (Chroma) · chat (F3)
  guardrails/ audit/    blocklist · SQLite
  web/                  FastAPI, templates Jinja (pages/partials), CSS
data/protocols/         11 documentos da base RAG
models/                 encaminhamento.joblib · metrics.json · model_card.json
notebooks/              02_preprocessing · 03_model_comparison (jupytext)
scripts/                runpod_vllm · gerar_relatos_sinteticos · avaliar_pipeline · publish_hf
tests/                  features · regras do grafo · juiz · rotas web
```

## Limitações

- O alvo é a prática registrada pelos serviços em 2024, não um padrão-ouro de "quem deveria ser encaminhada".
- Transversal: o modelo não prevê desfecho futuro; apoia a leitura do caso.
- A base RAG é pequena e majoritariamente do DF; respostas fora dela são sinalizadas como sem cobertura.
- Relatos sintéticos (gerados pela LLM a partir de fichas reais) substituem relatos reais, que não existem no SINAN.
- A LLM principal depende de um pod alugado; sem ele a nota degrada para regras + modelo, com registro na auditoria.

## Integrantes

Marcelo Arruda de Siqueira · Leonardo Barbosa Nogueira · Jose Flavio Neto · Pedro Matias dos Santos · Wellington Oliveira de Andrade

## Deploy público

| Componente | Onde roda | Custo |
|---|---|---|
| App (web, modelo de ML, RAG, histórico) | Pod de CPU no RunPod (2 vCPU, 4 GB) que clona este repositório num volume de 20 GB; o histórico (SQLite) sobrevive a stop/start | US$ 0,066/h ligado, zero parado |
| LLM geradora | OpenRouter · `openai/gpt-4o-mini` | ~US$ 0,001 por nota |
| Juiz independente | OpenRouter · `mistralai/mistral-small-3.2-24b-instruct` | ~US$ 0,0005 por veredito |
| Extração ao vivo (F2) | OpenRouter · `google/gemini-2.5-flash-lite` | ~US$ 0,0001 por chamada |
| Voz | Web Speech API no navegador; fallback OpenRouter · `google/gemini-2.5-flash` (whisper `base` de reserva no pod) | ~US$ 0,005 por relato ditado |
| Alternativa auto-hospedada | vLLM no RunPod (Qwen3.6-27B + Qwen2.5-7B), `make pod-create` | US$ 0,59/h |

Preços do OpenRouter em 2026-10-10, por milhão de tokens (entrada / saída): gpt-4o-mini 0,15 / 0,60; Mistral Small 3.2 0,094 / 0,25;
Gemini 2.5 Flash Lite 0,10 / 0,40; Gemini 2.5 Flash 0,30 / 2,50 (áudio 1,00). Uma análise completa usa cerca de 6 mil tokens;
a avaliação de 30 relatos custou menos de US$ 0,50.

**URL atual:** https://y5iscqd4fej0cn-8000.proxy.runpod.net (muda se o pod for recriado; `make app-status` mostra a vigente).
Medido pela URL pública em 2026-10-10: páginas em < 1 s, análise completa em 18 s, transcrição de 10 s de áudio em 2,3 s.

Três famílias de modelo distintas (OpenAI, Mistral, Google) garantem juiz ≠ geradora. O pod clona este repositório num volume
persistente ao subir (`scripts/runpod_app.py`, sem registro de imagens); `make app-create` cria e imprime a URL, `make app-status`,
`make app-stop` (custo zero), `make app-start` (religa em ~2 min) e `make app-redeploy` (publica a `main`: o pod refaz o clone ao subir). Alternativa com a imagem Docker, publicada (privada, 5,5 GB) em `ghcr.io/fiap-ssp-2025/guardia-ai:latest`:
`make docker-push` + `app-create --image` (exige token do GitHub com `write:packages`; o RunPod guarda a credencial de registro). O Hugging Face
Spaces foi descartado porque Spaces com Docker passaram a exigir plano pago (Team na org, PRO na conta pessoal); o script
`scripts/deploy_space.py` fica no repositório caso o grupo assine. Detalhes e decisões em
[`specs/002-deploy-hf-spaces-openrouter.md`](specs/002-deploy-hf-spaces-openrouter.md).
