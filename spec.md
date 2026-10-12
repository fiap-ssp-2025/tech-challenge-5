# Guardiã AI — Especificação (v1.1, 2026-10-11)

Tech Challenge 5 · FIAP Pós IA para Devs · grupo fiap-ssp-2025.

Onde duas linhas tratam do mesmo ponto, vale a de número maior.

## 1. Decisões

| # | Decisão | Motivo |
|---|---|---|
| D1 | **Escopo único: segurança da mulher** (violência doméstica e familiar). A dimensão de saúde entra pelos encaminhamentos e pelos protocolos do Ministério da Saúde. | Um cenário, com uma jornada completa. |
| D2 | **Arquitetura herdada das fases anteriores:** LangGraph, RAG, guardrails e auditoria, com o modelo e o SHAP. A interface foi reescrita (D8) e o pacote passou a ser `guardia` (D10). | O fluxo de triagem, a consulta a protocolos e o registro das análises já existiam. |
| D3 | **Alvo do ML: encaminhamento à rede de proteção.** `OUT_VEZES` é feature. | `OUT_VEZES` ("ocorreu outras vezes?") é um fato da mesma ficha. Não há ordem temporal, e a profissional pode perguntar. Ver §3. |
| D4 | **Alternativa auto-hospedada:** modelo aberto de ~27B (`QuantTrio/Qwen3.6-27B-AWQ`) em vLLM no RunPod, endpoint compatível com OpenAI. O padrão é o da D15. | O grupo já operou esse modelo. O cliente de LLM só precisa de URL, modelo e chave. |
| D5 | **Juiz ≠ geradora.** O LLM-as-judge usa outro provedor e outra família de modelo. | O modelo não audita a si mesmo. |
| D6 | **Sem fine-tuning novo.** | O modelo de 3B com fine-tuning falhou em casos de segurança. Um modelo base maior, com RAG e prompt por fluxo, gera a nota sem treino adicional. |
| D7 | O repositório de qualificação documental (hackathon SSP-DF) ficou de fora desta aplicação. | Tema diferente e dado privado. |
| D8 | **Interface: HTML renderizado no servidor** (FastAPI + Jinja2) com **htmx 4.0.0, Tailwind CSS 4 e Basecoat 1.0.2** por CDN com versão fixa. | O servidor é a única fonte de verdade: fragmentos HTML, erro 422 com o formulário re-renderizado, tema claro e escuro, layout a partir de 360 px. |
| D9 | **Juiz auto-hospedado:** `Qwen/Qwen2.5-7B-Instruct-AWQ` num segundo servidor vLLM do mesmo pod (porta 8001). O juiz padrão é o da D15. | Um segundo modelo no mesmo pod mantém juiz ≠ geradora quando a aplicação roda sem OpenRouter. |
| D10 | **Pacote `guardia`.** Os módulos anteriores foram portados e enxugados. | A interface mudou inteira (D8) e o grafo ficou só com o fluxo de violência. |
| D11 | **Triagem em caixa única, com extração ao vivo e progresso real.** Texto ou voz; a cada pausa o modelo rápido (`FAST_*`) preenche a ficha; a análise roda como job e a página consulta o progresso por nó. | O formulário longo cansava, e a espera de ~50 s sem feedback não mostrava em que etapa a análise estava. |
| D12 | **Status em cache.** Sondagens da LLM, do juiz e do RAG rodam numa thread a cada 20 s. O indicador do cabeçalho lê o cache. Métricas do modelo ficam no relatório. Assets com `?v=mtime`. | Sondar o provedor na requisição travava a página; o navegador segurava CSS antigo. |
| D13 | **Identidade visual alinhada ao Viva Flor.** Roxo #925C98 como cor primária, cabeçalho roxo, flor no logo; tema escuro em lavanda sobre roxo profundo. Cores de risco continuam semânticas. | A interface segue a identidade do produto de referência do grupo. |
| D14 | **Voz com gravação contínua.** Web Speech API quando existe; o fallback grava sem parar, corta nos silêncios (RMS) e transcreve cada trecho em fila ordenada. O modelo de transcrição é o da D18. | O corte fixo a cada 6 s travava entre pausas. |
| D15 | **LLM padrão via OpenRouter**, três famílias: geradora `openai/gpt-4o-mini`, juiz `mistralai/mistral-small-3.2-24b-instruct`, extração ao vivo `google/gemini-2.5-flash-lite`. O pod vLLM (D4, D9) é a alternativa auto-hospedada. | 26 s por análise, contra 52 s no pod. Três famílias mantêm juiz ≠ geradora. |
| D16 | **Extração ao vivo com schema esparso** (`marcados: [...]`). O botão "Reextrair" usa o schema completo. | Menos tokens de saída: 1,7 s no Gemini Flash Lite. |
| D17 | **App público num pod de CPU do RunPod** (2 vCPU, 4 GB, US$ 0,066/h). O pod clona o repositório num volume de 20 GB, onde o histórico persiste. A imagem `ghcr.io/fiap-ssp-2025/guardia-ai` é a alternativa. Ligado só quando necessário. | Spaces com Docker passou a exigir plano pago. O RunPod já tinha créditos e API no repositório (`scripts/runpod_app.py`). |
| D18 | **Transcrição no servidor por `google/gemini-2.5-flash`** (papel `transcribe`, temperatura 0, JSON `{text}`). faster-whisper fica de reserva. Trechos de até 15 s; ao parar, uma passada sobre o áudio inteiro substitui o texto ditado se ele não foi editado. | Flash transcreveu 19 s de áudio em 2,3 s. O whisper `base` no pod de CPU errava muito. Custa menos de US$ 0,01 por relato e dispensa GPU. |
| D19 | **Protocolos (F3) num painel lateral**, aberto por botão flutuante em todas as páginas (`<dialog>` não modal; em telas ≥ 1100 px a página encolhe e o relato fica visível). Conversa contínua em balões, Recomeçar, aviso fixo no rodapé. A busca de uma pergunta de seguimento inclui a pergunta anterior. | A profissional consulta um protocolo no meio do atendimento, sem trocar de tela. |
| D20 | **Guia de entrevista ao vivo:** checklist fixo de 11 tópicos (Formulário Nacional de Avaliação de Risco) sob o relato, mostrando as 3 próximas perguntas. Um tópico fecha pela ficha, pelo relato (`topicos_abordados`, cada um com trecho literal conferido no servidor, inclusive resposta negativa) ou à mão. | Lista fixa é previsível e auditável. A LLM só indica o que o relato já cobriu. |
| D21 | **Protocolos com o caso em andamento.** Na triagem, o painel envia os campos marcados da ficha e, se houver, risco, sugestão do modelo e encaminhamentos da última análise. O texto do relato não sai da triagem. Os fatos usados aparecem sob a pergunta e ficam em `chat_turns.case_context`. | A resposta vale para o caso, sem redigitar a ficha e sem persistir o relato. |
| D22 | **RAG híbrido.** Texto normalizado (hífen de quebra, linhas partidas, número de página; formulários, sumário e trechos curtos descartados), trechos de 450 caracteres, ranking vetorial + BM25 fundidos por RRF, no máximo 2 trechos por documento, e trecho-chave por tema (`THEMES`: violência sexual → atendimento em até 72 h). `scripts/avaliar_rag.py` mede a recuperação sem LLM. | A busca só semântica não achava o prazo de 72 h no meio da norma longa. No conjunto de avaliação, 18 de 18 perguntas ficam no top-6. |

## 2. Produto

**Para quem:** profissionais que acolhem mulheres em situação de violência (assistência social, enfermagem, psicologia, atendimento em DEAM, CREAS e Casa da Mulher Brasileira).

**O que faz:** recebe o relato e os dados do atendimento, organiza as informações, estima se o caso tem perfil de encaminhamento à rede de proteção, consulta protocolos e leis cadastrados e devolve uma nota de apoio com fontes citadas. A decisão é sempre da profissional.

**O que não faz:** não decide encaminhamento, não classifica culpa, não substitui o registro oficial, não faz diagnóstico e não prescreve.

## 3. Modelo de machine learning

### 3.1 Por que o alvo é o encaminhamento

Verificação em 2026-10-09 sobre o SINAN 2024 (mulheres, violência não autoprovocada: 327.752 linhas):

- `OUT_VEZES` é o campo 53 da ficha: "Ocorreu outras vezes?". Preenchido junto com as features. É um estudo transversal.
- Campos de desfecho (`EVOLUCAO`, `CLASSI_FIN`, `DT_OBITO`) estão vazios. Não há identificador para ligar notificações da mesma mulher.
- Os campos de encaminhamento da ficha estão preenchidos em cerca de 94% das notificações.

### 3.2 Definição do alvo

`ENCAMINHA_PROTECAO = 1` se qualquer um de `DELEG_MULH`, `ATEND_MULH`, `MPU`, `DEFEN_PUBL` for "1" (Sim). Linhas em que os quatro estão em branco ou "9" (ignorado) saem do treino. `DELEG` (delegacia comum) fica de fora.

Conjunto de treino: 280.757 notificações, prevalência 28,3%.

O modelo aprende quais perfis de caso os serviços encaminharam à rede de proteção em 2024, inclusive o sub-encaminhamento. A interface diz "perfil encaminhado em 2024".

### 3.3 Features

42 features: 10 tipos de violência (`VIOL_*`), 9 meios de agressão (`AG_*`), 5 grupos de relação com o autor, sexo e álcool do autor, raça/cor, escolaridade ordinal, situação conjugal, ciclo de vida, deficiência, orientação sexual, identidade de gênero, UF, `OUT_VEZES`, gestação, local, número de envolvidos, motivação e idade em anos (`NU_IDADE_N`).

Dia da semana, mês e circunstância CID ficam de fora. Ausentes viram a categoria `IGNORADO`. Os quatro campos do alvo não entram nas features.

### 3.4 Modelos e avaliação

Regressão logística, Random Forest, Random Forest com algoritmo genético e HistGradientBoosting, no mesmo `Pipeline` (one-hot nas nominais, mediana nas numéricas), ajustado só no treino. Divisão 70/30 estratificada. 15% do treino escolhe o limiar (máximo F2), a calibração isotônica e o fitness do algoritmo genético. O teste é usado uma vez.

Métricas: recall e precisão da classe encaminhada, F1, PR-AUC, ROC-AUC, especificidade, e a acurácia da classe majoritária como baseline. Equidade: recall por raça/cor. Explicabilidade: SHAP global e por caso, e os coeficientes da regressão logística.

Modelo implantado: HistGradientBoosting calibrado, em `models/encaminhamento.joblib` (0,5 MB). Números no README e em `docs/relatorio_tecnico.md`.

## 4. Estado da implementação (2026-10-11)

Repositório: https://github.com/fiap-ssp-2025/tech-challenge-5 (CI: ruff, pytest, build da imagem). Modelo publicado: https://huggingface.co/fiap-ssp-2025/tc5-guardia-encaminhamento.

| Item | Estado |
|---|---|
| Dados | `guardia.data.features`: SINAN 2024 → 280.757 linhas, 42 features, alvo `ENCAMINHA_PROTECAO` (28,3%). Notebook 02 executado. |
| Modelos | LR, RF, RF+GA, HistGB. Implantado HistGB calibrado (0,5 MB): PR-AUC 0,674, ROC-AUC 0,816, recall 0,923 / precisão 0,392 no limiar 0,15; gap de recall por raça 0,18. SHAP global e por caso. Notebook 03 executado; figuras em `docs/figures`. |
| Grafo | LangGraph de 9 nós (`guardia.graph.flow`), rastreado no LangSmith (projeto `guardia-ai`). |
| LLM | Padrão via OpenRouter (D15): geradora `openai/gpt-4o-mini`, juiz `mistralai/mistral-small-3.2-24b-instruct`, extração ao vivo `google/gemini-2.5-flash-lite`. JSON por schema. Alternativa auto-hospedada: Qwen3.6-27B-AWQ e Qwen2.5-7B-AWQ em vLLM no RunPod (`scripts/runpod_vllm.py`). |
| RAG | Chroma (cosseno) + MiniLM multilíngue e BM25, fundidos por RRF, sobre 11 documentos, 227 trechos normalizados, com citação de fonte e trecho-chave por tema (D22). Um único cliente Chroma por processo, criado sob lock. |
| Guardrails | Blocklist, aviso fixo e juiz com 6 critérios. Reprovação exige trecho literal presente na nota. Auditoria SQLite com hash do relato, nota, ficha, fontes, veredito e linha do tempo por nó. |
| Interface | FastAPI + Jinja2 + htmx 4.0.0 + Tailwind 4 + Basecoat 1.0.2. Triagem com voz, extração ao vivo e progresso por nó; guia de entrevista (D20); protocolos em painel lateral com o caso em andamento (D19, D21); histórico com linha do tempo; indicador de status no cabeçalho. Tema claro e escuro, 360 px sem scroll horizontal. |
| Voz | Web Speech API e fallback no servidor: Gemini 2.5 Flash via OpenRouter, whisper de reserva (D18). |
| Avaliação | 60 relatos sintéticos. Sobre 30, com OpenRouter: F2 precisão 0,902 / recall 0,949 por campo; F1 30/30 notas, juiz aprovou 30; LLM 5 s de mediana, 20 s no total; concordância do modelo com o encaminhamento da ficha 0,80 (`docs/eval_pipeline.json`). No pod vLLM: F2 0,916 / 0,932, juiz aprovou 29 de 30, 52 s no total (`docs/eval_pipeline_runpod.json`). Recuperação do RAG: 18/18 perguntas (`docs/eval_rag.json`). |
| Qualidade | 44 testes (pytest), ruff limpo em `src`, `tests` e `scripts`. Imagem Docker de 4,6 GB (torch CPU, ffmpeg, whisper). |
| Docs | `README.md`, `docs/relatorio_tecnico.md` e o PDF (`make relatorio`), `docs/roteiro_video.md`, esta spec. |

**App público:** `make app-status` mostra estado e URL (muda se o pod for recriado); `make app-start` religa em cerca de 2 min, `make app-stop` para (custo zero), `make app-redeploy` publica a `main`. US$ 0,066/h ligado. URL, custos e preços de tokens no README.

**Pod vLLM (opcional):** `make pod-status`; se `pod-start` falhar por falta de GPU livre, `make pod-create` e apontar `LLM_BASE_URL` e `JUDGE_BASE_URL` do `.env` para ele; `make pod-stop` ao terminar (US$ 0,59/h ligado; volume parado custa US$ 0,017/h).
