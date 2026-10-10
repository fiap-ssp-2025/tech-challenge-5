# Spec 004 — Protocolos com o contexto do caso em andamento

Status: **done** (2026-10-10) · Base: spec 003 (painel de protocolos e guia de entrevista, done).

## Objetivo

No painel de protocolos aberto durante uma triagem, a profissional pergunta "que medidas cabem aqui?" ou "para onde encaminho?"
e a resposta aplica os protocolos **a este caso**, sem que ela precise redigitar o que já está na ficha.

## Decisões

| # | Decisão | Motivo |
|---|---|---|
| C1 | **O contexto é a ficha, nunca o relato.** Vão para a LLM os campos marcados da ficha (tipos de violência, meio, relação, sinais de risco, contexto SINAN informado) e, se a triagem já foi analisada, o risco, a sugestão do modelo e até 3 encaminhamentos da nota. | As conversas ficam gravadas no SQLite; a regra do projeto é que o relato não é persistido e a ficha e a nota podem ser. Campos estruturados também são curtos e sem nomes. |
| C2 | **Opção visível e ligada por padrão:** caixa "Considerar o caso em andamento" no rodapé do painel, só nas páginas que têm ficha. Cada pergunta feita com o caso mostra, sob o balão, o que foi usado ("Com o caso: …"). | A profissional sabe o que a Guardiã viu e pode desligar para perguntas gerais. Transparência depois do envio basta; um resumo ao vivo duplicaria a ficha que está ao lado. |
| C3 | O contexto vai em cada pergunta com o estado **atual** da ficha (não fica preso ao primeiro turno); a memória continua só com perguntas e respostas. Os fatos entram também na busca do RAG (a pergunta e a anterior vêm primeiro). | A ficha muda durante o atendimento. "Violência sexual" na busca traz a norma certa mesmo para "o que fazer agora?". |
| C4 | O prompt manda aplicar os protocolos ao caso **sem inventar fatos fora da ficha** e lembrar que a decisão é da profissional; os fatos usados são gravados na coluna nova `chat_turns.case_context` (migração por `ALTER TABLE` ao iniciar). | Mantém as regras de D1/D5 e permite reabrir o fio com a indicação "Com o caso". |
| C5 | Sugestão extra no estado vazio, só com ficha na página: "O que os protocolos indicam para este caso?". | Mostra o recurso sem explicar. |
| C6 | **Correção no guia da spec 003 (achada neste teste):** o Gemini Flash Lite marcava os 11 tópicos como abordados. Cada tópico agora vem como `"id: trecho literal"` e o servidor descarta o que não aparece no relato; `NAO_SE_APLICA` deixa de contar como resposta. Objeto aninhado (`{id, trecho}`) e `maxItems` travavam os modelos via OpenRouter (Gemini sem JSON válido, gpt-4o-mini sem responder); strings planas mantêm 1,7 s. | Medido em 2026-10-10: os dois relatos de teste passaram de 11/11 para exatamente os tópicos citados. |

## Tarefas

- [x] T1 `web/forms.py`: `case_facts(case, signals)` → lista curta de fatos legíveis.
- [x] T2 `rag/chat.py`: `ask(..., case=[...])`; prompt e busca com o caso; coluna `case_context` com migração; `turns()` devolve os fatos.
- [x] T3 `web/app.py`: `/protocolos/perguntar` monta o contexto a partir de `#ficha` e da última análise (`analise_id` oculto no resultado).
- [x] T4 Templates/CSS: opção no rodapé do painel, linha "Com o caso" sob o balão, sugestão do estado vazio, `analise_id` em `result.html`.
- [x] T5 Testes (fatos, migração, rota com e sem caso) e verificação no navegador; docs (`spec.md` D21, README, AGENTS).

## Critérios de aceite (verificados em 2026-10-10)

- Na triagem com ficha preenchida, a pergunta "Que medidas protetivas cabem neste caso?" recebe resposta que cita os fatos da ficha; sob o balão aparece "Com o caso: …".
- Com a opção desligada, ou fora da triagem, nenhum fato é enviado e nada aparece sob o balão.
- Depois de Analisar, a próxima pergunta inclui risco e encaminhamentos da nota.
- O texto do relato não aparece no banco (`chat_turns`) nem na requisição do chat.

## Observação

A busca semântica não alcança o trecho das 72 h da norma de violência sexual, nem com os fatos do caso na consulta: o trecho está no meio de um documento longo. Melhorar isso é trabalho de RAG (busca híbrida ou trechos marcados por tema), não da interface.

## Fora de escopo

Mandar o relato livre para o chat; resposta em streaming.
