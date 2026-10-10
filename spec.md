# Guardiã AI — Especificação (v1.1, 2026-10-10)

Tech Challenge 5 · FIAP Pós IA para Devs · grupo fiap-ssp-2025. Substitui as decisões do [analise-base-tc5.md](analise-base-tc5.md) onde houver conflito.

## 1. Decisões fechadas

| # | Decisão | Motivo |
|---|---|---|
| D1 | **Escopo único: segurança da mulher** (violência doméstica e familiar). Sem fluxos de saúde. | O roteiro pede um cenário principal com jornada completa. Dois escopos dobram teste, demo e relatório. A dimensão de saúde entra pelos encaminhamentos e protocolos do MS dentro da jornada. |
| D2 | **Base de código: TC3** (LangGraph, RAG, guardrails, auditoria, Streamlit) + ML/SHAP do TC2 + extração/pré-processamento do TC1. | Cobre 7 de 10 requisitos técnicos sem retrabalho. |
| D3 | **Alvo do ML: encaminhamento à rede de proteção**, não "recorrência". `OUT_VEZES` vira feature. | `OUT_VEZES` ("ocorreu outras vezes?") é um fato relatado na mesma ficha. Não há ordem temporal, e a atendente pode simplesmente perguntar. Ver §3. |
| D4 | **LLM geradora: modelo aberto de ~27B servido por vLLM no RunPod**, endpoint compatível com OpenAI. Fallback: gpt-4o-mini. | O grupo já operou `Qwen3.6-27B-AWQ` em A40 no repositório hackathon. O `llm_client` do TC3 só precisa de `LOCAL_LLM_BASE_URL` e `LOCAL_LLM_MODEL`. |
| D5 | **Juiz ≠ geradora.** O LLM-as-judge continua em outro provedor (gpt-4o-mini ou Claude). | Princípio já adotado no TC3; evita o modelo auditar a si mesmo. |
| D6 | **Sem novo fine-tuning.** O LoRA do TC3 fica documentado como histórico. | O roteiro não exige FT. A avaliação do TC3 mostrou falhas de segurança no 3B fine-tuned; um 27B base com RAG e prompt por fluxo rende mais com zero treino. |
| D7 | Repositório `hackathon` (qualificação documental SSP-DF) é arquivado como rascunho pré-edital. | Não atende ao roteiro publicado; dependia de dado privado. |
| D8 | **Interface: HTML renderizado no servidor** (FastAPI + Jinja2) com **htmx 4.0.0, Tailwind CSS 4 e Basecoat 1.0.2** por CDN com versão fixa. Sem Streamlit, React, Vue, Svelte, bundler ou SPA. | Pedido do grupo (ver `AGENTS.md`): servidor como única fonte de verdade, fragmentos HTML, 422 com formulário re-renderizado, tema claro/escuro, responsivo a partir de 360 px. |
| D9 | **Juiz = `Qwen/Qwen2.5-7B-Instruct-AWQ`** num segundo servidor vLLM do mesmo pod (porta 8001). | A chave OpenAI herdada do TC3 está inválida e o Hugging Face Inference Router está sem créditos. Um segundo modelo no mesmo pod mantém juiz ≠ geradora sem custo extra. |
| D11 | **Triagem no estilo chat com extração ao vivo e progresso real.** O relato é uma caixa única (texto ou voz); a cada pausa o modelo rápido (`FAST_*`, por padrão o servidor do juiz) preenche a ficha; a análise roda como job e a página consulta o progresso por nó. | Pedido do grupo após o primeiro teste: formulário longo cansava e a espera de ~50 s sem feedback era ruim. Pesquisa de UX: revelação progressiva, espera honesta por etapas, voz com estados claros e sem envio automático. |
| D12 | **Status em cache.** Sondagens da LLM/juiz/RAG rodam numa thread a cada 20 s; indicador no cabeçalho (popover nativo, abre no hover/clique) e página `/status` leem o cache. Assets com `?v=mtime`. | A página de status travava 10 s sondando o pod na requisição; navegador segurava CSS antigo. |
| D13 | **Identidade visual = Viva Flor.** Roxo #925C98 (medido no app do grupo) como cor primária, cabeçalho roxo, flor no logo; tema escuro em lavanda sobre roxo profundo. Cores de risco continuam semânticas. | Pedido do grupo: app sem cor; alinhar com o produto Viva Flor. |
| D14 | **Voz com gravação contínua.** Web Speech API quando existe; fallback grava sem parar, corta nos silêncios (RMS) e transcreve cada trecho em fila ordenada no servidor (faster-whisper `small`, CPU, ffmpeg), com `skip_ms` para o cabeçalho do webm. | O corte fixo a cada 6 s travava entre pausas. |
| D15 | **LLM padrão via OpenRouter**, três famílias: geradora `openai/gpt-4o-mini`, juiz `mistralai/mistral-small-3.2-24b-instruct`, extração ao vivo `google/gemini-2.5-flash-lite`. O pod vLLM (Qwen) vira alternativa auto-hospedada documentada. | Latência: 26 s por análise contra 52 s no pod; custo por token irrisório; nada a manter ligado. Exigiu liberar todos os provedores em openrouter.ai/settings/privacy (com a lista restrita tudo ia para a DeepInfra, 13–30 s). |
| D16 | **Extração ao vivo com schema esparso** (`marcados: [...]` em vez de 25 booleanos). | ~4x menos tokens de saída; 1,7 s no Gemini Flash Lite. O botão "Reextrair" usa o schema completo. |
| D17 | **App público num pod de CPU do RunPod** (imagem `ghcr.io/fiap-ssp-2025/guardia-ai`, cpu3c 2 vCPU/4 GB, US$ 0,12/h), ligado só quando necessário. HF Spaces descartado. | HF passou a cobrar Spaces com Docker (Team na org, PRO na conta pessoal). RunPod usa créditos já existentes e a API já integrada (`scripts/runpod_app.py`). |
| D10 | **Pacote novo `guardia`** (não `saude_mulher`): os módulos do TC3 foram portados e enxugados, não copiados. | A interface mudou inteira (D8) e o grafo perdeu os fluxos de saúde; portar foi mais barato que limpar. |

## 2. Produto

**Para quem:** profissionais que acolhem mulheres em situação de violência (assistência social, enfermagem, psicologia, atendimento em DEAM/CREAS/Casa da Mulher Brasileira).

**O que faz:** recebe o relato e os dados do atendimento, organiza as informações, estima se o caso tem perfil de encaminhamento à rede de proteção, consulta protocolos e leis cadastrados e devolve uma nota de apoio com fontes citadas. A decisão é sempre da profissional.

**O que não faz:** não decide encaminhamento, não classifica "culpa", não substitui o registro oficial, não faz diagnóstico nem prescreve.

## 3. Modelo de Machine Learning

### 3.1 Por que o alvo mudou

Verificação feita em 2026-10-09 sobre o SINAN 2024 (notificações de mulheres, violência não autoprovocada: 327.752 linhas):

- `OUT_VEZES` é o campo 53 da ficha: "Ocorreu outras vezes? Informar se a violência é de repetição". Preenchido junto com todas as features. É um estudo transversal, não predição (ver `tech-challenge-1/docs/revisao_critica.md`, §1).
- O TC2 reutilizou o mesmo parquet, o mesmo alvo e o mesmo `LabelEncoder` em colunas nominais. Não corrigiu a premissa.
- Campos de desfecho (`EVOLUCAO`, `CLASSI_FIN`, `DT_OBITO`) estão vazios. Não há identificador para ligar notificações da mesma mulher.
- Os campos de encaminhamento da ficha atual estão preenchidos em ~94% das notificações: `DELEG_MULH` (49.118 sim), `ATEND_MULH` (41.150), `DEFEN_PUBL` (8.602), `MPU` (7.583), `DELEG` (47.850), `ASSIST_SOC` (57.536), `REDE_SAU` (187.005).

### 3.2 Definição do alvo

`ENCAMINHA_PROTECAO = 1` se qualquer um de `DELEG_MULH`, `ATEND_MULH`, `MPU`, `DEFEN_PUBL` for "1" (Sim). Linhas em que todos os quatro estão em branco ou "9" (ignorado) saem do treino. Decisão a validar no notebook: incluir `DELEG` (delegacia comum) no alvo ou deixá-lo de fora.

Prevalência estimada: 20 a 25% (a confirmar no notebook 02).

**Leitura correta do modelo:** ele aprende, a partir das notificações de 2024, quais perfis de caso os serviços encaminharam à rede de proteção, e sugere esse encaminhamento à atendente com os fatores que pesaram. Ele reproduz a prática, inclusive sub-encaminhamento. Isso vai explícito no relatório e na interface.

### 3.3 Features

- Mantidas do TC1: `VIOL_*` (10), `AG_*` (9), `REL_PARCEIRO_INTIMO / FAMILIAR / CONHECIDO / INSTITUCIONAL / OUTROS`, `AUTOR_SEXO`, `AUTOR_ALCO`, `CS_RACA`, `ESCOLARIDADE` (ordinal), `SIT_CONJUG`, `CICL_VID`, `DEF_TRANS`, `ORIENT_SEX`, `IDENT_GEN`, `SG_UF`.
- Novas: `OUT_VEZES` (agora feature), `CS_GESTANT`, `LOCAL_OCOR`, `NUM_ENVOLV`, `VIOL_MOTIV`, idade em anos derivada de `NU_IDADE_N`.
- Removidas: `DIA_SEMANA_OCOR`, `MES_OCOR` (sem mecanismo), `CIRC_LESAO` (60% vazio; foi o que derrubou o TC1 para 57 mil linhas).
- Ausentes: categoria "ignorado/branco" explícita em vez de `dropna()`. Meta: treinar com mais de 250 mil linhas, não 57 mil.

### 3.4 Modelos e avaliação

| Modelo | Papel | Origem |
|---|---|---|
| Regressão Logística (OneHot + `class_weight`) | baseline interpretável, odds ratios no relatório | TC1, recodificado |
| Random Forest + algoritmo genético | melhor desempenho esperado; reaproveita `optimize_ga.py` trocando a coluna alvo | TC2 |
| HistGradientBoosting | terceiro modelo, barato, lida bem com categóricas | novo |

- Pipeline: `ColumnTransformer` (OneHot nominais, ordinal para escolaridade) dentro de `Pipeline`, ajustado só no treino. Split 70/30 estratificado + validação cruzada.
- Métricas: recall e precisão da classe positiva, F1, PR-AUC, ROC-AUC, especificidade. Baseline de classe majoritária declarada.
- Ponto de operação: limiar escolhido pelo custo assimétrico (deixar de sugerir encaminhamento custa mais que sugerir a mais). Curva de calibração para que a probabilidade mostrada tenha significado.
- Equidade: recall por `CS_RACA` (função `evaluate_by_group` do TC2).
- Interpretabilidade: SHAP (TreeExplainer) global e por caso; coeficientes da LR.
- Artefato final: um `.pkl` abaixo de 30 MB, versionado no git.

## 4. Fluxos da aplicação

| Fluxo | Status | Descrição | O que reaproveita |
|---|---|---|---|
| **F1 · Triagem do relato** (principal) | obrigatório | Formulário + relato livre → risco por regras → modelo de encaminhamento + SHAP → RAG → ações → LLM → juiz → auditoria → nota de apoio com fontes. | Grafo do TC3 (nós de violência), ML/SHAP do TC2, formulário do TC2 |
| **F2 · Estruturar relato** | obrigatório | A LLM extrai do relato livre os campos do formulário (tipo de violência, relação, arma, filhos, local, recorrência) com saída JSON restrita por schema. A atendente confere e corrige antes de rodar o F1. Atende "organizar informações importantes" do roteiro. | `VLLMProvider` com schema do repositório hackathon; prompt novo |
| **F3 · Consultar protocolos** | obrigatório | Chat com memória sobre a base RAG: "como solicitar medida protetiva?", "quais canais no DF?". Sempre cita o arquivo usado. | `assistant/medical_assistant.py` do TC3 (LangChain), restrito aos protocolos de segurança |
| **F4 · Histórico** | obrigatório | Atendimentos anteriores em cards e, no detalhe, a linha do tempo digerida das 9 etapas (duração e resumo de cada nó), nota, modelo, fontes e validador. Substitui a tela de auditoria crua; os dados continuam no SQLite com hash do relato. | `audit/store.py` (+ colunas steps/total_ms) e `web/jobs.py` |
| **F5 · Relato por voz** | implementado | Botão de ditado na caixa do relato: Web Speech API no navegador (pt-BR, texto provisório em cinza) com fallback de gravação em blocos de 6 s transcritos no servidor por faster-whisper (`/transcrever`). O texto alimenta o F2 ao vivo. | `web/static/voice.js`, `llm/transcribe.py` (ideia do módulo A1 do TC4) |
| **F6 · Painel do SINAN** | opcional | Página com o que o dataset mostra: prevalência de encaminhamento por UF, tipo de violência, relação. Contextualiza o modelo. | Notebooks de EDA do TC1 |

Ordem de implementação: F1 → F2 → F4 → F3 → F5 → F6. F5 e F6 só se sobrar tempo após relatório e vídeo.

### 4.1 Grafo LangGraph do F1 (nós)

```
estruturar_relato (F2, opcional) → avaliar_risco_regras → modelo_encaminhamento (RF + SHAP)
→ consultar_protocolos (RAG) → definir_acoes → gerar_explicacao → preparar_auditoria
→ gerar_resposta (LLM RunPod / fallback) → juiz_guardrails → persistir_auditoria → END
```

Remover do TC3: `classify_flow` (não há mais escolha de fluxo), os ramos de triagem/obstétrico/prevenção em `branch_specific_processing`, as `FLOW_INSTRUCTIONS` de saúde, os 9 protocolos de saúde sem relação com violência, os formulários de saúde no Streamlit.

Manter na base RAG (11 arquivos): protocolo de violência, Lei Maria da Penha (integral e resumo), tipos de violência, medidas protetivas, canais nacionais, canais e rede de proteção do DF, protocolo da rede DF, LGPD, protocolo do MS sobre violência sexual. Avaliar manter `ms_saude_mental_mulher.txt` para o encaminhamento em saúde mental.

## 5. LLM no RunPod

### 5.1 Arquitetura

```
Streamlit (Docker) ──HTTPS──▶ RunPod pod · vllm/vllm-openai · Qwen3.6-27B-AWQ · A40 48 GB
        │                           (endpoint /v1 compatível com OpenAI)
        └──fallback────────────────▶ OpenAI gpt-4o-mini
Juiz: gpt-4o-mini (ou Claude) — sempre em provedor diferente da geradora
```

- Configuração no `.env`: `LOCAL_LLM_BASE_URL=https://<pod-id>-8000.proxy.runpod.net/v1`, `LOCAL_LLM_MODEL=QuantTrio/Qwen3.6-27B-AWQ`, `LOCAL_LLM_API_KEY` (vLLM aceita `--api-key`). Nenhuma mudança de código no `llm_client.py` além de ler a chave.
- O proxy do RunPod rejeita o User-Agent padrão do `urllib` (403). O SDK da OpenAI usa `httpx` com UA próprio; testar no primeiro dia e, se falhar, enviar `default_headers={"User-Agent": "guardia-ai/1.0"}`.
- Saída estruturada: usar `extra_body={"guided_json": schema}` do vLLM nos nós F2 (extração) e na resposta do F1 (campos `nivel_risco`, `resumo`, `acoes`, `fontes`, `aviso`). Elimina parsing de texto livre.

### 5.2 Custo e operação

| Item | Valor |
|---|---|
| A40 48 GB | US$ 0,44/h (preço registrado no plan do hackathon) |
| Uso estimado: desenvolvimento (12 h) + testes (4 h) + gravação (2 h) | ~US$ 8 |
| Latência esperada por resposta curta (300 tokens) | 3 a 8 s |
| Pod desligado | custo zero; o app cai no fallback OpenAI sem quebrar |

Alternativa se quiser demo sem manter pod ligado: endpoint **serverless** vLLM do RunPod (cobra por segundo, cold start de 1 a 2 min). Decidir na semana da gravação.

### 5.3 Vale fine-tuning no RunPod?

Não para esta entrega. Custo: curadoria + treino + avaliação de segurança (o TC3 gastou dias nisso e o modelo 3B reprovou 10 de 22 casos). Ganho: nenhum requisito do roteiro. Uso melhor dos créditos: gerar relatos sintéticos para demo e conjunto de avaliação, e rodar avaliações em lote (juiz sobre 50 casos) para o relatório.

## 6. Ferramentas de baixo custo e alto ganho

Ordenadas por relação ganho/trabalho. "Must" entra no plano; "nice" só após F1 a F4 prontos.

| Prioridade | Ferramenta | Trabalho | O que agrega |
|---|---|---|---|
| must | **Saída JSON restrita (vLLM guided_json)** | ~2 h, código do hackathon | F2 confiável; campos da resposta sem parsing frágil |
| must | **LangSmith tracing** | 0 h, variáveis já no `.env.example` do TC3 | Rastro de cada nó, prompt, latência e custo; ótimo no vídeo e no relatório |
| must | **Calibração + limiar por custo** (`CalibratedClassifierCV`, curva PR) | ~3 h | Responde à crítica do TC1; probabilidade exibida passa a ter significado |
| must | **Relatos sintéticos** (50 casos gerados pelo 27B, revisados) | ~2 h | Demo, testes do grafo, conjunto de avaliação do juiz |
| must | **GitHub Actions**: `pytest` + `docker build` | ~1 h | Prova de que o container sobe num clone limpo |
| nice | **MLflow** no treino dos 3 modelos + GA | ~2 h | Tabela de métricas e artefatos rastreáveis para o relatório |
| nice | **Busca híbrida + reranker** (`rank_bm25` + `bge-reranker-v2-m3`) | ~3 h | Citação mais precisa em perguntas com termos jurídicos (número de artigo, nome de serviço) |
| nice | **faster-whisper** (F5) | ~4 h, módulo A1 do TC4 | Relato por áudio; liga TC4 ao TC5 e o roteiro cita isso como evolução |
| nice | **RAGAS** (faithfulness, answer relevancy) sobre 20 perguntas | ~4 h | Número para a seção "funcionamento do RAG" do relatório |
| nice | **Laya** (modelo de decisão open source, Apache 2.0, roda em CPU) fine-tuned em 500 relatos sintéticos para arma / relação / recorrência, comparado ao F2 do 27B | ~1 dia | Seção de relatório "classificador local e rápido vs LLM"; argumento LGPD (texto não sai da máquina). Zero-shot é inútil (0,36 vs 0,46 da classe majoritária): só com fine-tuning e recalibração |
| não | **Jev** (TypeSafe AI) | — | Proprietário, acesso antecipado, sem pesos; envia o relato a terceiro; paper JevOut (arXiv 2609.30243) mostra 61–73% das decisões corretas viradas por contexto benigno |
| não | Multi-agente, novo fine-tuning da LLM, MongoDB | — | Sem retorno para o roteiro |

## 7. Entregáveis e responsáveis

| Entregável | Base | Responsável |
|---|---|---|
| Repositório `tech-challenge-5` | clone do TC3, limpeza dos fluxos de saúde | definir |
| Notebooks 01-02 (extração, pré-processamento com novo alvo) | TC1 | definir |
| Notebook 03 + `train.py` (LR, RF+GA, HGB, calibração, SHAP, equidade) | TC1 + TC2 | definir |
| Nó `modelo_encaminhamento` + formulário + F2 | novo | definir |
| RunPod + `.env` + Dockerfile + CI | novo | Pedro (créditos RunPod) |
| Relatório PDF | evolui `docs/relatorio_tecnico.md` do TC3 + §3 e §7 do TC2 | definir |
| Vídeo ≤ 15 min: F2 → F1 → F3 → F4 | — | definir |

## 8. Riscos

- **Prevalência do alvo baixa (~20%)**: usar `class_weight`, limiar por custo e PR-AUC em vez de acurácia.
- **O modelo aprende sub-encaminhamento regional**: mostrar recall por UF e por raça no relatório; dizer na UI que é "prática observada", não "o que deveria ser".
- **Pod do RunPod fora do ar na banca**: fallback OpenAI automático; gravar o vídeo com o pod ligado.
- **Imagem Docker grande** (torch + sentence-transformers): aceitável; opção de trocar embeddings por API se passar de 6 GB.
- **Escopo crescer**: F5 e F6 só depois de relatório e vídeo prontos.

## 10. Estado da implementação (v1.0, 2026-10-10)

Repositório: https://github.com/fiap-ssp-2025/tech-challenge-5 (CI verde: ruff, pytest, docker build). Modelo publicado:
https://huggingface.co/fiap-ssp-2025/tc5-guardia-encaminhamento. Apresentação: artefato "Guardiã AI" (claude.ai).

**Entregue e testado localmente**

| Item | Estado |
|---|---|
| Dados | `guardia.data.features`: SINAN 2024 → 280.757 linhas, 42 features, alvo `ENCAMINHA_PROTECAO` (28,3%). Notebook 02 executado. |
| Modelos | LR, RF, RF+GA, HistGB. Implantado HistGB calibrado (0,5 MB): PR-AUC 0,674, ROC-AUC 0,816, recall 0,923 / precisão 0,392 no limiar 0,15; gap de recall por raça 0,18. SHAP global e por caso. Notebook 03 executado; figuras em `docs/figures`. |
| Grafo | LangGraph de 9 nós (`guardia.graph.flow`), rastreado no LangSmith (projeto `guardia-ai`). |
| LLM | Geradora Qwen3.6-27B-AWQ e juiz Qwen2.5-7B-AWQ em vLLM no RunPod (`scripts/runpod_vllm.py`: create/update/stop/start/delete; dois servidores no mesmo pod; 8k ctx; sem eager). JSON por schema em F1, F2 e F3. Modelo rápido (`FAST_*`, por padrão o juiz) para a extração ao vivo. |
| RAG | Chroma (cosseno) + MiniLM multilíngue sobre 11 documentos, 169 trechos, com citação de fonte. |
| Guardrails | Blocklist + aviso fixo + juiz com 6 critérios e defesas contra reprovação sem trecho literal. Auditoria SQLite com hash do relato, nota, ficha, fontes, veredito e linha do tempo por nó. |
| Interface | FastAPI + Jinja2 + htmx 4.0.0 + Tailwind 4 + Basecoat 1.0.2. F1 (caixa única, voz, extração ao vivo, ficha com revelação progressiva, análise como job com progresso real), F3 (chat com memória), F4 (histórico com linha do tempo), Modelo, Status (cache + indicador no cabeçalho). Tema claro/escuro, 360 px sem scroll horizontal. |
| Voz | Web Speech API + fallback faster-whisper no servidor (gravação contínua, corte por silêncio, fila). |
| Avaliação | 60 relatos sintéticos (`data/synthetic`); sobre 30: F2 precisão 0,916 / recall 0,932 por campo, `OUT_VEZES` 100%; F1 30/30 notas, juiz aprovou 29, 1 bloqueada; LLM 25 s mediana, 52 s total; 2,8 fontes recuperadas, 2,1 citadas; concordância modelo × encaminhamento real 0,767 (`docs/eval_pipeline.json`). |
| Qualidade | 22 testes, ruff limpo, Docker 4,6 GB (torch CPU, ffmpeg, whisper) testado em todas as rotas. |
| Docs | README, `docs/relatorio_tecnico.md` (falta exportar PDF), `docs/roteiro_video.md`, AGENTS.md, esta spec. |

**Pendências de entrega (humanas):** PDF do relatório, vídeo, revisão final pelo grupo.
**Deploy (spec 002, executada em 2026-10-10):** LLM via OpenRouter (D15), app no RunPod CPU (D17); URL e custos no README.

**Operação do pod (enquanto não há deploy):** `make pod-status`; se `pod-start` falhar por "not enough free GPUs", `make pod-create` (Secure Cloud é mais estável que Community) e atualizar `LLM_BASE_URL`/`JUDGE_BASE_URL` no `.env`; `make pod-stop` ao terminar (US$ 0,59/h ligado; volume parado custa US$ 0,017/h). Saldo em 2026-10-10: ~US$ 11.
