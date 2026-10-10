# AGENTS.md — Guardiã AI

Instruções para qualquer agente de IA que trabalhe neste repositório.

## Fonte da verdade
- `spec.md` — decisões fechadas (D1–D14) e, no §10, o estado da implementação v1.0.
- `specs/NNN-*.md` — specs das etapas (002 deploy: done; `003-interface-drawer-protocolos-guia-entrevista.md` e `004-protocolos-com-contexto-do-caso.md`: done). Ao retomar, leia a spec ativa, execute as tarefas marcando `[x]` e registre as decisões novas no `spec.md`.
- `analise-base-tc5.md` — por que o TC3 é a base e o que veio de cada Tech Challenge anterior.
- Este arquivo — convenções de código e interface.

## Mapa do código
| Pasta | Papel |
|---|---|
| `src/guardia/data/features.py` | SINAN bruto → tabela de treino; mesma transformação para um caso do formulário (`row_from_case`) |
| `src/guardia/ml/train.py`, `predict.py` | LR, RF (+GA), HistGB; calibração; limiar por F2; SHAP; artefato `models/encaminhamento.joblib` |
| `src/guardia/graph/flow.py` | LangGraph do F1 (9 nós) |
| `src/guardia/llm/` | cliente OpenAI-compatível (geradora + juiz), prompts/schemas, F2 (`structurer.py`), juiz |
| `src/guardia/rag/` | Chroma + MiniLM multilíngue sobre `data/protocols` (F1) e chat com memória (F3) |
| `src/guardia/guardrails/`, `audit/` | blocklist + aviso; SQLite com hash do relato |
| `src/guardia/web/` | FastAPI + Jinja2 + htmx 4 + Tailwind 4 + Basecoat (páginas e partials); `jobs.py` roda a análise em background com progresso por nó; `static/voice.js` faz o ditado |
| `src/guardia/llm/transcribe.py` | fallback de voz no servidor: modelo de áudio via OpenRouter (papel `transcribe`, Gemini 2.5 Flash) com faster-whisper de reserva (grupo `audio` do uv); ffmpeg obrigatório |
| `scripts/` | RunPod (vLLM), relatos sintéticos, avaliação em lote, publicação no Hugging Face |

## Regras
- **Alvo do ML é `ENCAMINHA_PROTECAO`.** Nunca reintroduzir `OUT_VEZES` como alvo (ver spec §3.1).
- **Dois provedores de LLM distintos:** geradora (`LLM_*`) e juiz (`JUDGE_*`). O juiz nunca audita a si mesmo. Papéis auxiliares: `FAST_*` (extração ao vivo) e `TRANSCRIBE_*` (voz), ambos com padrão no OpenRouter.
- **Nada de dado pessoal persistido.** O relato vira hash; a nota gerada e os campos da ficha podem ser gravados.
- **Interface:** HTML renderizado no servidor. htmx 4.0.0 fixo (sintaxe 4: herança explícita `:inherited`, eventos `htmx:fase:ação`, `hx-status:422`). Tailwind CSS 4 e Basecoat 1.0.2 por CDN com versão fixa. Sem React/Vue/Svelte/bundler/SPA. HTML nativo antes de JS (`<dialog>`, `<details>`); Alpine só se o nativo não resolver (hoje não é usado).
- Mesma URL devolve página completa no acesso direto e fragmento quando `HX-Request: true` (função `render` em `web/app.py`). Erro de formulário = 422 com a ficha re-renderizada. Toda interação tem estado carregando (`.htmx-indicator`), vazio, erro e sucesso.
- **Triagem:** a análise nunca bloqueia a requisição: `POST /analisar` inicia um job e devolve o fragmento de progresso, que consulta `GET /analisar/{id}` a cada 2 s até a nota. A ficha é preenchida ao vivo por `POST /estruturar` com `live=1` (modelo rápido) e só é sobrescrita enquanto a profissional não a editou (`ficha_editada`). Não mexer em arquivos do projeto durante uma análise em desenvolvimento: o `--reload` mata o job.
- **Guia de entrevista** (`guardia/interview.py`, `partials/guia.html`): 11 tópicos fixos; volta como OOB de `/estruturar` e via `POST /guia` quando a ficha muda ou uma pergunta é marcada. Estado nos campos ocultos `guia_mencionados`/`guia_feitos`. Perguntas novas entram em `TOPICS`, não no prompt.
- **Protocolos (F3) = painel lateral** (`partials/chat_drawer.html`, `<dialog>` não modal, em `base.html`); não há página nem item de menu. `/protocolos` com `HX-Request` devolve o fio; acesso direto abre a triagem com o painel aberto. `is_htmx` ignora `HX-Request-Type: full` (navegação com `hx-boost`). O chat recebe o caso por `hx-include="#ficha, #analise_id"` (`_chat_case` em `app.py`): só campos estruturados, nunca `relato`.
- `topicos_abordados` na extração é lista de strings `"id: trecho"`, conferida contra o relato em `structurer._quoted_topics`. Não usar objeto aninhado nem `maxItems` nesse schema: travam Gemini e gpt-4o-mini via OpenRouter.
- `htmx.config.defaultTimeout` é 60 s no htmx 4; o `base.html` sobe para 240 s via meta `htmx-config`.
- **Status nunca sonda na requisição.** `web/status.py` mantém um cache atualizado por thread a cada 20 s; o indicador do cabeçalho (`/status/pill`, popover nativo) lê o cache. Não existem páginas `/status` nem `/modelo`: métricas do modelo ficam no relatório e no README. Assets estáticos saem com `?v=mtime` (`asset_version()`), senão o navegador segura CSS antigo.
- Componentes = partials Jinja em `web/templates/partials/`. Não repetir marcação.
- Código em inglês, textos de interface e docs em português. Commits só quando o humano pedir.

## Comandos
`make help` lista tudo. Caminho do zero: `make install env data train serve`. Pod de LLM: `make pod-create`, `make pod-status`, `make pod-stop`. Testes: `make test`.
