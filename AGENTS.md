# AGENTS.md — Guardiã AI

Instruções para qualquer agente de IA que trabalhe neste repositório.

## Fonte da verdade
- `spec.md` — decisões fechadas (escopo só segurança, alvo do ML, LLM no RunPod, fluxos F1–F6).
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
| `src/guardia/llm/transcribe.py` | fallback de voz no servidor (faster-whisper + ffmpeg); grupo `audio` do uv |
| `scripts/` | RunPod (vLLM), relatos sintéticos, avaliação em lote, publicação no Hugging Face |

## Regras
- **Alvo do ML é `ENCAMINHA_PROTECAO`.** Nunca reintroduzir `OUT_VEZES` como alvo (ver spec §3.1).
- **Dois provedores de LLM distintos:** geradora (`LLM_*`) e juiz (`JUDGE_*`). O juiz nunca audita a si mesmo.
- **Nada de dado pessoal persistido.** O relato vira hash; a nota gerada e os campos da ficha podem ser gravados.
- **Interface:** HTML renderizado no servidor. htmx 4.0.0 fixo (sintaxe 4: herança explícita `:inherited`, eventos `htmx:fase:ação`, `hx-status:422`). Tailwind CSS 4 e Basecoat 1.0.2 por CDN com versão fixa. Sem React/Vue/Svelte/bundler/SPA. HTML nativo antes de JS (`<dialog>`, `<details>`); Alpine só se o nativo não resolver (hoje não é usado).
- Mesma URL devolve página completa no acesso direto e fragmento quando `HX-Request: true` (função `render` em `web/app.py`). Erro de formulário = 422 com a ficha re-renderizada. Toda interação tem estado carregando (`.htmx-indicator`), vazio, erro e sucesso.
- **Triagem:** a análise nunca bloqueia a requisição: `POST /analisar` inicia um job e devolve o fragmento de progresso, que consulta `GET /analisar/{id}` a cada 2 s até a nota. A ficha é preenchida ao vivo por `POST /estruturar` com `live=1` (modelo rápido) e só é sobrescrita enquanto a profissional não a editou (`ficha_editada`). Não mexer em arquivos do projeto durante uma análise em desenvolvimento: o `--reload` mata o job.
- `htmx.config.defaultTimeout` é 60 s no htmx 4; o `base.html` sobe para 240 s via meta `htmx-config`.
- Componentes = partials Jinja em `web/templates/partials/`. Não repetir marcação.
- Código em inglês, textos de interface e docs em português. Commits só quando o humano pedir.

## Comandos
`make help` lista tudo. Caminho do zero: `make install env data train serve`. Pod de LLM: `make pod-create`, `make pod-status`, `make pod-stop`. Testes: `make test`.
