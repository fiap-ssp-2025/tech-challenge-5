# Spec 005 — RAG: busca híbrida e trechos de documentos longos

Status: **done** (2026-10-10) · Base: spec 004 (protocolos com o contexto do caso, done).

## Problema

No teste da spec 004, um caso de violência sexual ocorrido na véspera não trazia o trecho das **72 horas** da norma técnica
MS/MJ (`ms_violencia_sexual.txt`, ~2.000 linhas), que é o ponto central do protocolo, com nenhuma consulta: pergunta,
pergunta + anterior ou só os fatos do caso. Causas encontradas:

1. **Texto extraído de PDF:** hífen de quebra de linha (`agres\xad\nsão`), frase partida em linhas de ~50 caracteres,
   números de página soltos. O embedding e qualquer busca por palavra viam "agres" e "são".
2. **Busca só semântica:** "primeiras horas" ou "72 h" não pesam no MiniLM; um trecho genérico sobre violência sexual
   fica mais perto da consulta que o trecho específico.
3. **Formulários e sumário** do anexo da norma ("Profilaxia HIV Não ( ) Sim ( )", "Atendimento ....... 24") batiam com
   muitas consultas e não respondem nenhuma.
4. **Teto de 2 trechos por documento sobre só `k*3` candidatos:** com a norma ocupando quase todos os candidatos, a
   busca devolvia menos de `k` trechos (no F1, 3,05 fontes distintas em vez de ~4) e às vezes nenhum texto da Lei.
5. **Consulta genérica do caso** ("Violência sexual; Ameaça de morte; Parceiro íntimo — O que os protocolos indicam?")
   não tem nenhuma palavra que leve ao prazo de 72 h; nem a busca híbrida resolve isso sozinha.

## Decisões

| # | Decisão | Motivo |
|---|---|---|
| R1 | **Normalizar o texto antes de dividir:** remove hífen de quebra, junta linha que continua em minúscula, apaga número de página; descarta trechos de formulário (≥ 3 `( )` ou linhas `_____`), de sumário (≥ 3 `....`) e com menos de 40 caracteres. | Corrige a palavra partida, que prejudica as duas buscas; tira cerca de 50 trechos que só faziam ruído. |
| R2 | **Trechos de 450 caracteres (sobreposição 60)**, separando por parágrafo, linha, frase (o ponto fica no fim do trecho, não no início do seguinte). 227 trechos (antes 169). | Um trecho de 700 misturava o prazo de 72 h com outros assuntos; 450 venceu 700, 550 e 350 na medição. |
| R3 | **Busca híbrida:** ranking vetorial (Chroma, cosseno) + BM25 próprio (sem dependência nova: sem acento, dígito separado de letra, prefixo de 6 letras como radical) fundidos por **RRF** (k = 60, 30 candidatos de cada). O `score` exibido continua sendo o cosseno. | BM25 acha termo exato ("72 h", "boletim", "LGPD"); vetor acha paráfrase. Cada um sozinho erra perguntas diferentes (tabela). |
| R4 | **Teto de 2 trechos por documento sobre o corpus inteiro ranqueado** (o corpus tem ~230 trechos); se a fusão não enche as `k` vagas, completa com o ranking vetorial. Subir para 3 não ganhou acerto e reduziu a diversidade de fontes. | Os textos curtos (Lei, medidas protetivas, canais) não são afogados pela norma longa, e a busca não fica sem candidatos. |
| R5 | **Trecho-chave por tema (`THEMES` em `retriever.py`):** se a consulta fala de violência sexual (`violência sexual`, `estupro`, `abuso sexual`, `relação forçada`), o melhor trecho BM25 para "atendimento em até 72 horas, profilaxia, contracepção de emergência, coleta de vestígios" entra na última vaga quando ainda não veio. O gatilho não casa com "heterossexual"/"orientação sexual". | Resolve a consulta genérica do caso (problema 5) sem diluir as perguntas específicas. Alternativas medidas e piores: acrescentar os termos do tema à consulta (15/18, dilui "notificação" e "aborto") e usar o tema como terceira lista do RRF (16/18). Novos temas entram na lista, não no prompt. |
| R6 | `INDEX_VERSION` entra na impressão digital da coleção: mudar o fatiamento cria uma coleção nova no Chroma. | O índice antigo não é reaproveitado por engano; a imagem Docker reconstrói no build. |
| R7 | **Avaliação de recuperação sem LLM** em `scripts/avaliar_rag.py` (`--mode hybrid|vector|bm25`, `--json`): 18 perguntas do chat com documento e padrão esperados (inclui as do roteiro do vídeo) e os 60 relatos sintéticos consultados como o nó do F1. | Medir antes de mexer; repetível em 30 s (`make eval-rag`). Resultado em `docs/eval_rag.json`. |

## Medição (2026-10-10)

Chat: acerto = um trecho do documento esperado **e** com o padrão esperado (ex.: `72 ?h`) entre os 6 primeiros.
F1: consulta = relato + fatores de risco (como `retrieve_protocols`), top-5; acerto = norma MS nos casos de violência
sexual, texto da Lei/medidas/protocolo nos demais.

| Configuração | Chat top-6 (18) | F1 doc. esperado (60) | F1 sexual com trecho das 72 h (10) | Fontes distintas no F1 |
|---|---|---|---|---|
| **Antes:** vetor, 700/80, sem normalizar, teto sobre `k*3` | 12 | 52 | 2 | 3,05 |
| Vetor, texto normalizado, 450/60, sem formulário, teto sobre tudo | 15 | 60 | 0 | 3,93 |
| BM25 sozinho, mesmos trechos | 16 | 60 | 3 | 3,88 |
| Híbrido RRF, sem tema | 17 | 60 | 2 | 3,95 |
| **Depois:** híbrido RRF + trecho-chave por tema | **18** | **60** | **10** | 3,85 |

Por pergunta (rank do primeiro acerto; `—` = fora do top-6):

| Pergunta | Antes | Depois |
|---|---|---|
| Violência sexual; Ameaça de morte; Parceiro íntimo — O que os protocolos indicam para este caso? | — | 6 |
| O que fazer nas primeiras horas após violência sexual? | — | 1 |
| O que fazer nas primeiras 72 h em violência sexual? (roteiro) | — | 1 |
| Como solicitar medida protetiva de urgência? (roteiro) | 1 | 1 |
| … e no DF, quais serviços? (roteiro, com a anterior) | 2 | 1 |
| Quais são os tipos de violência previstos na Lei Maria da Penha? | 1 | 1 |
| O que caracteriza violência patrimonial? | 2 | 1 |
| Qual número ligar para a Central de Atendimento à Mulher? | 2 | 2 |
| Como coletar vestígios biológicos para exame de DNA? | 1 | 1 |
| Quando fazer contracepção de emergência após estupro? | 1 | 1 |
| Profilaxia de HIV e hepatite B após violência sexual | — | 1 |
| A vítima precisa registrar boletim de ocorrência para ser atendida na saúde? | 1 | 1 |
| Como fazer a notificação compulsória do caso de violência sexual? | 2 | 1 |
| A mulher tem direito à interrupção da gravidez resultante de estupro? | — | 1 |
| Posso compartilhar os dados da atendida com outros serviços? | — | 3 |
| Quais medidas protetivas podem ser aplicadas ao agressor? | 1 | 1 |
| Violência física; Objeto cortante; Parceiro íntimo — O que os protocolos indicam para este caso? | 1 | 3 |
| … e se for adolescente, o que muda? (com "primeiras horas" antes) | 1 | 1 |

Outras variantes medidas com o tema (na versão de 251 trechos, antes do filtro de `_____` e trechos curtos) e descartadas: trechos de 700 (16/18), 550 (17/18) e 350 (17/18); título do
documento no início de cada trecho (18/18, sem ganho e exige reindexar por outro motivo); teto de 3 por documento
(18/18, 3,33 fontes distintas no F1); BM25 sem radical (18/18, 3,68 fontes distintas).

**F1 com LLM** (`scripts/avaliar_pipeline.py --limit 10`, OpenRouter gpt-4o-mini + juiz Mistral Small; os 10 primeiros
relatos da rodada de 30 anterior como referência; mesmos modelos):

| | Antes (10 primeiros da rodada de 30) | Depois |
|---|---|---|
| Notas geradas / aprovadas pelo juiz / bloqueadas | 10 / 10 / 0 | 10 / 10 / 0 |
| Documentos recuperados por nota (média) | 3,0 | 3,6 |
| Documentos citados na nota (média) | 2,1 | 2,8 |
| Latência mediana da LLM | 5,3 s | 4,1 s |

Nos dois casos de violência sexual do lote (`syn-42-001`, `syn-42-002`) a nota antes se apoiava só na norma MS; agora
vem também a Lei ou a rede do DF. O "até 72 h" das ações já vinha das regras do nó `define_actions`, não do RAG. A
rodada roda com LLM e varia de uma execução para outra: o número serve para mostrar que o F1 não piorou, não que melhorou.
`docs/eval_pipeline.json` e o bloco do relatório continuam com a rodada de 30.

## Tarefas

- [x] T1 `rag/retriever.py`: normalização, descarte de formulário/sumário, trechos 450/60, `INDEX_VERSION`.
- [x] T2 BM25 + RRF + teto por documento sobre o ranking inteiro; `mode` para avaliação.
- [x] T3 `THEMES` com o trecho-chave de violência sexual.
- [x] T4 `scripts/avaliar_rag.py` e `docs/eval_rag.json`; tabela antes/depois acima.
- [x] T5 Testes (`tests/test_retriever.py`: normalização, tokens, ruído, as duas consultas do caso de teste, textos curtos
  não afogados); `make test`, `ruff`; `avaliar_pipeline.py --limit 10`.
- [x] T6 Docs: `spec.md` (D22 e §10), README, AGENTS.

## Critérios de aceite (verificados em 2026-10-10)

- "Violência sexual; Ameaça de morte; Parceiro íntimo\nO que os protocolos indicam para este caso?" e "O que fazer nas
  primeiras horas após violência sexual?" trazem, entre os 6 primeiros, um trecho de `ms_violencia_sexual.txt` com "72 horas".
- Nenhuma pergunta do conjunto piora de acerto para erro; o F1 continua com o documento esperado em todos os 60 relatos.
- `make test` verde.

## Limites e próximos passos

- O conjunto tem 18 perguntas escritas por quem ajustou o retriever: serve para regressão, não prova generalização.
  RAGAS ou perguntas escritas pelo grupo dariam um número mais honesto.
- O trecho-chave por tema é conhecimento de domínio no código. Vale só para pontos que o protocolo trata como centrais
  e que a consulta genérica não alcança; cada tema novo precisa de uma pergunta no `avaliar_rag.py`.
- Reranker (`bge-reranker-v2-m3`) ficou de fora: ~570 M parâmetros (≈ 2 GB) a mais no pod de CPU de 4 GB e o conjunto já acerta 18/18.
- Com o teto de 2 por documento, uma pergunta só sobre a norma recebe 2 ou 3 trechos dela e completa com textos curtos
  da Lei. Se as respostas sobre violência sexual ficarem rasas, o próximo passo é um teto maior só para o documento que
  domina o ranking, medido no mesmo script.
