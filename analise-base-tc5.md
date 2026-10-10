# Tech Challenge 5 (Guardiã AI) — qual trabalho usar como base

Data da análise: 2026-10-09. Fonte: `POSTECH - HACKA IADT - Secretaria - Fase 5.pdf` + repositórios locais em `~/projetos/fiap/`.

> **Atualização 2026-10-09:** decisões revisadas em [spec.md](spec.md). Mudanças: escopo único de segurança (sem fluxos de saúde), alvo do ML trocado de `OUT_VEZES` para encaminhamento à rede de proteção (a premissa de "recorrência" era retrospectiva e o TC2 não corrigiu), LLM geradora em vLLM no RunPod. O veredito de usar o TC3 como base continua.

## Veredito

**Base: Tech Challenge 3** (`fiap-ssp-2025/tech-challenge-3`), **enxertando o módulo de ML + SHAP do Tech Challenge 2** e o notebook de comparação de modelos do Tech Challenge 1.

O TC3 já entrega, pronto e funcionando, 7 dos 10 requisitos técnicos do roteiro. O que falta (modelo de ML dentro do fluxo, SHAP, Dockerfile) é exatamente o que o TC2 já tem ou é trivial de adicionar. Nenhum outro repositório chega perto.

## O que o roteiro exige

| # | Requisito do roteiro (obrigatório) | TC1 | TC2 | TC3 | TC4 | trabalho-final |
|---|---|---|---|---|---|---|
| 1 | Dataset público/anonimizado relacionado ao problema | ✅ SINAN 2024 | ✅ mesmo parquet | ⚠️ só Q&A p/ fine-tune | ✅ RAVDESS etc. | ⚠️ sentenças TJDFT |
| 2 | ≥ 2 modelos de ML treinados e comparados (acc/prec/recall/F1) | ✅ LR, KNN, RF | ⚠️ só RF (4 variantes GA) | ❌ | ⚠️ GradientBoosting (V2) | ❌ |
| 3 | Discussão de métricas e impacto dos erros (recall vs FN) | ✅ | ✅ relatório §3 | ❌ | ⚠️ | ❌ |
| 4 | Interpretabilidade (SHAP / feature importance) | ⚠️ feature importance | ✅ SHAP TreeExplainer + UI | ❌ | ❌ | ❌ |
| 5 | Integração com LLM | ❌ | ✅ OpenAI explica SHAP | ✅ Llama FT local + fallback gpt-4o-mini | ❌ | ✅ Qwen 27B via vLLM (RunPod) |
| 6 | RAG com base de conhecimento **e indicação da fonte** | ❌ | ❌ | ✅ Chroma, 20 protocolos, `protocol_sources` já exibido na UI | ❌ | ❌ |
| 7 | LangChain ou LangGraph no fluxo principal | ❌ | ❌ | ✅ LangGraph 10 nós + assistente LangChain | ❌ | ❌ |
| 8 | Guardrails: apoio ao profissional, sem diagnóstico definitivo | ❌ | ⚠️ só no prompt | ✅ blocklist + LLM-as-judge + aviso | ✅ "decisão humana" | ✅ revisão humana |
| 9 | Registro/auditoria das análises | ❌ | ⚠️ Mongo p/ fine-tune | ✅ SQLite com hash SHA256 (LGPD) | ❌ | ✅ `execucoes.jsonl` |
| 10 | Interface simples (Streamlit/Gradio/API) | ❌ notebooks | ✅ Streamlit | ✅ Streamlit (553 linhas, 4 fluxos) | ❌ CLI | ⚠️ `fila_app.py` |
| 11 | Dockerfile | ❌ | ❌ (só compose do Mongo) | ❌ (só compose do Mongo) | ❌ | ❌ |
| 12 | README + notebooks/scripts de treino + relatório PDF + vídeo ≤ 15 min | relatório | relatório + vídeo | relatório + diagramas | relatório + vídeo | parcial |

Cobertura dos itens técnicos 1 a 11: **TC3 = 7**, TC2 = 5, TC1 = 3, TC4 = 3, trabalho-final = 3.

## Por que o TC3

- **Tema idêntico.** O roteiro pede "saúde **e** segurança da mulher" com um cenário principal. O TC3 já tem os 4 fluxos (Triagem Ginecológica, Violência Doméstica, Obstétrico, Prevenção), cobrindo os dois contextos sem precisar inventar nada.
- **Pipeline do roteiro já existe.** O fluxo sugerido no PDF ("receber dados → executar modelo → consultar documentos → gerar explicação → retornar resultado") é literalmente o grafo `womens_safety_graph.py`, faltando só o nó "executar modelo".
- **RAG com citação de fonte já pronto.** `retrieve_protocol_context` devolve `protocol_sources` e a UI já mostra na aba RAG. O roteiro diz "o ideal é que a aplicação indique qual documento foi utilizado" — está feito.
- **Segurança e auditoria já prontas.** O roteiro cita explicitamente "limites de atuação, logging, auditoria e explicabilidade" da Fase 3. O TC3 foi construído em cima disso.
- **Fallback de LLM já resolve o Docker.** `llm_client.probe_backend` usa o mlx server local se existir e cai para `gpt-4o-mini` via `OPENAI_API_KEY`. No container basta usar o fallback (ou apontar `LOCAL_LLM_BASE_URL` para um Ollama). O fine-tuning vira "diferencial" documentado, não dependência.
- **Infra pronta.** Makefile, pyproject, pytest, diagramas auto-gerados, relatório técnico em Markdown (180 linhas) para evoluir.

## Por que não os outros

- **TC2 como base:** teria que construir RAG, LangGraph, auditoria e guardrails do zero. É mais trabalho que o inverso (trazer RF + SHAP para o TC3).
- **TC1:** só notebooks. Serve para fornecer o notebook de comparação de 3 modelos (LR, KNN, RF) e o relatório de EDA.
- **TC4:** multimodal (áudio/vídeo). O roteiro diz que áudio/imagem/vídeo é "evolução adicional, não obrigatória". Sem RAG, sem LangGraph, sem Streamlit, modelos privados no Hugging Face de ~1,5 GB. Não vale a pena como base; no máximo um bônus se sobrar tempo.
- **trabalho-final (repo `hackathon`):** foi um rascunho feito antes do edital sair, com tema diferente (qualificação documental de feminicídios da SSP-DF). Não tem ML clássico, RAG, LangGraph nem Streamlit; depende de GPU A40 no RunPod e tem restrição de dados privados. **Não se encaixa no roteiro publicado.** Recomendo arquivar e pivotar.

## Plano de menor esforço (TC3 + enxerto do TC2)

1. **Criar o repo `tech-challenge-5` a partir do TC3** (clone + reset de histórico ou fork). Renomear no README para "Guardiã AI". Não renomear o pacote `saude_mulher` para economizar trabalho.
2. **Enxertar o ML no fluxo de violência.**
   - Copiar do TC2: `models/random_forest_exp2_moderado.pkl` (6 MB, recall 0,823, praticamente igual ao exp3 de 28 MB; os outros `.pkl` têm 100 a 195 MB e não cabem no git), `data/processed/df_preprocessed.parquet` (766 KB), `src/explain/shap_explainer.py`, o encoding de features de `optimize_ga.load_data` e o formulário `form_simulacao` com `CATEGORICAL_OPTIONS` do `app/streamlit_app.py`.
   - Criar um nó `run_risk_model` no grafo, antes de `retrieve_protocol_context`, que roda `predict_proba` + SHAP top-N e grava no state (`risk_probability`, `top_features`).
   - Em `generate_response`, incluir probabilidade e top features no prompt (o TC2 já tem esse prompt em `llm_explainer.py`, só adaptar).
   - Expandir o `form_violencia` com os campos do RF (ou sub-aba "Avaliação de risco de recorrência").
3. **Cumprir "≥ 2 modelos comparados".** Copiar para `notebooks/` o `04_model_comparison.ipynb` do TC1 (LR, KNN, RF com accuracy/recall/F1/ROC-AUC) e o `src/optimize_ga.py` + `results/comparison.json` do TC2. Reaproveitar a seção 3 do relatório do TC2 (recall vs especificidade, custo do falso negativo).
4. **Dockerfile.** `python:3.12-slim` + `requirements.txt` + `streamlit run app/streamlit_app.py`. O `requirements.txt` do TC3 não inclui mlx (só o Makefile instala), então funciona em Linux. Imagem fica grande (torch + sentence-transformers); aceitável. Se quiser enxugar, trocar embeddings por `text-embedding-3-small` da OpenAI.
5. **Pequenos ajustes de produto.** Banner "sistema de apoio, decisão é do profissional" (já existe o `add_safety_notice`), página de auditoria com `fetch_recent` (já existe `make audit-recent`, só expor na UI).
6. **Entregáveis novos:** README (adaptar), relatório PDF (evoluir `docs/relatorio_tecnico.md` do TC3 + seções de ML do TC2), vídeo ≤ 15 min.

Estimativa: itens 1 a 5 são 2 a 4 dias de trabalho para uma pessoa; relatório e vídeo mais 2 a 3 dias. Dá para fechar em 1 a 2 semanas com o grupo.

## Pontos de atenção no TC3

- `docs/relatorio_tecnico.md` tem alteração local não commitada. Verificar antes de clonar.
- O RAG reconstrói o Chroma por arquivo a cada consulta (`build_vectorstore` dentro do loop). Funciona, mas é lento. Opcional: construir uma vez e cachear.
- Só 2 testes (`test_graph.py`, `test_rag.py`). Suficiente para o roteiro, mas vale adicionar 1 teste para o nó de ML.
- O `.env.example` traz chaves LangSmith e Mongo legadas. Limpar.
- `models/fused/` tem 20 MB de artefatos do Llama que não são usados pelo app. Não copiar.
