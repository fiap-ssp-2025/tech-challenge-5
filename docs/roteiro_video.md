# Roteiro do vídeo (≤ 15 min)

Objetivo do roteiro do TC5: "acompanhar uma jornada completa, desde a entrada das informações até a análise realizada pela IA e a apresentação do resultado". Gravar com o app público ligado (`make app-start` + `make app-status` mostra a URL) ou localmente com `make serve`; a LLM vai pelo OpenRouter, sem pod de GPU.

| Tempo | Cena | O que mostrar / dizer |
|---|---|---|
| 0:00–1:00 | Abertura | Problema: profissionais recebem relatos e precisam decidir rápido. Guardiã AI apoia, não decide. Mostrar `spec.md` §1 (decisões) em 20 s. |
| 1:00–2:30 | Dados e alvo | Relatório §3 e `docs/figures`: 280 mil notificações SINAN 2024, prevalência 28%. Explicar por que o alvo é encaminhamento (não "recorrência"): `OUT_VEZES` é campo da mesma ficha. Mostrar tabela de comparação dos 3 modelos, curvas PR e calibração. Métrica prioritária = recall da classe encaminhada; impacto do falso negativo. |
| 2:30–3:30 | Explicabilidade e equidade | Figuras do SHAP global, recall por raça/cor, limites declarados. |
| 3:30–7:30 | **Jornada F2 → F1** | `/`: ditar o relato pelo botão **Falar** (ou colar um de `data/synthetic/relatos.jsonl`). Mostrar a ficha se preenchendo sozinha enquanto se fala, os chips "detectado no relato" e "conferir"; mostrar o guia sob o relato: as perguntas somem à medida que o relato as responde (inclusive com "não"), clicar numa para marcar como perguntada e abrir o contador "n de 11"; abrir um grupo e corrigir um campo. Clicar **Analisar**: mostrar o progresso real etapa por etapa (regras → modelo → protocolos → LLM → juiz → gravar). Na nota: risco, probabilidade com os fatores SHAP, resumo, ações, encaminhamentos, perguntas pendentes, protocolos recuperados vs citados, aviso; abrir "Etapas" e "Rastro técnico". |
| 7:30–9:00 | Guardrails na prática | Mostrar um caso em que o juiz reprova (usar `docs/eval_pipeline.json` ou forçar via `scripts/`): a nota é bloqueada e o motivo aparece. Explicar juiz ≠ geradora e exigência de trecho literal. |
| 9:00–10:30 | **F3 Protocolos** | Botão **Protocolos** no canto (o relato continua visível ao lado): sugestão "Como solicitar medida protetiva de urgência?" e "O que fazer nas primeiras 72 h em violência sexual?". Mostrar citações e cobertura. Segunda pergunta usando a memória ("e no DF, quais serviços?"). Mostrar **Recomeçar**. |
| 10:30–11:30 | **F4 Histórico** | `/historico`: cards dos atendimentos; abrir o detalhe da análise feita: linha do tempo das 9 etapas com duração e resumo, nota, modelo, fontes, validador; hash do relato (texto não gravado). |
| 11:30–12:30 | Arquitetura e LangGraph | `README.md` (diagrama do grafo) e LangSmith com o trace da análise: 9 nós, latências. |
| 12:30–13:30 | Execução e Docker | `make docker-build` já feito; `make docker-run`; abrir o indicador de status no cabeçalho: LLM remota, modelo e RAG dentro da imagem. `make test` (18+ testes). |
| 13:30–14:30 | Avaliação | `docs/eval_pipeline.json`: concordância do F2 por campo, aprovação do juiz, latência mediana, fontes citadas. Modelo publicado no Hugging Face (`fiap-ssp-2025/tc5-guardia-encaminhamento`). |
| 14:30–15:00 | Limites e fechamento | O modelo aprende a prática de 2024 (inclusive sub-encaminhamento); base RAG pequena; decisão humana. Próximos passos: Laya como classificador local, whisper para relato por áudio. |

Checklist antes de gravar: `make app-status` READY (ou `make serve` local), créditos no OpenRouter, `make serve` sem `--reload`, auditoria com 3 a 5 análises prévias, relatos sintéticos à mão, tema claro para contraste na gravação.
